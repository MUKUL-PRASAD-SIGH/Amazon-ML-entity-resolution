"""
blocking_multi.py — Multi-Blocker Candidate Generation Suite.

Implements and evaluates:
1. Blocker A: Name-only TF-IDF Cosine (Baseline)
2. Blocker B: Address-only TF-IDF Cosine
3. Blocker C: Combined Name+Address TF-IDF Cosine
4. Blocker D: Exact Normalized Name Inverted Map
5. Blocker E: Rare Token Inverted Index
6. Blocker F: Numeric / House Number Overlap Index

Merges candidate pools via deduplicated score/frequency union.
Reports Candidate Recall, Avg Candidates / S1, Missed Matches Recovered, and Runtime.
"""

import gc
import re
import time
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    K_CANDIDATES,
    MIN_SIM_SCORE,
    S1_BATCH_SIZE,
    S23_CHUNK_SIZE,
    TFIDF_ANALYZER,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
)
from preprocess import extract_numbers

logger = logging.getLogger(__name__)


# ── TF-IDF Helper ─────────────────────────────────────────────────────────────

def _build_vectorizer(texts: List[str]) -> Tuple[TfidfVectorizer, sp.csr_matrix]:
    vec = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents=None,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


def _batched_topk(
    s1_mat: sp.csr_matrix,
    s23_mat: sp.csr_matrix,
    s23_ids: np.ndarray,
    k: int,
    min_score: float = MIN_SIM_SCORE,
) -> List[List[Tuple[str, float]]]:
    n1  = s1_mat.shape[0]
    n23 = s23_mat.shape[0]
    k   = min(k, n23)

    results: List[List[Tuple[str, float]]] = [[] for _ in range(n1)]
    top_ids    = [np.empty(0, dtype=object)   for _ in range(n1)]
    top_scores = [np.empty(0, dtype=np.float32) for _ in range(n1)]

    s1_batch  = S1_BATCH_SIZE
    s23_chunk = S23_CHUNK_SIZE

    try:
        import cupy as cp
        from cupyx.scipy import sparse as cusparse
        HAS_CUPY = True
    except ImportError:
        HAS_CUPY = False

    for s23_start in range(0, n23, s23_chunk):
        s23_end   = min(s23_start + s23_chunk, n23)
        chunk_mat = s23_mat[s23_start:s23_end]
        chunk_ids = s23_ids[s23_start:s23_end]

        if HAS_CUPY:
            chunk_mat_gpu = cusparse.csr_matrix(chunk_mat)

        for s1_start in range(0, n1, s1_batch):
            s1_end = min(s1_start + s1_batch, n1)
            q_mat  = s1_mat[s1_start:s1_end]

            if HAS_CUPY:
                q_mat_dense_gpu = cp.array(q_mat.toarray(), dtype=np.float32)
                sim_block_gpu = chunk_mat_gpu.dot(q_mat_dense_gpu.T).T
                sim_block = cp.asnumpy(sim_block_gpu)
            else:
                sim_block = q_mat.dot(chunk_mat.T)
                if sp.issparse(sim_block):
                    sim_block = sim_block.toarray()
                sim_block = sim_block.astype(np.float32)

            for local_i in range(s1_end - s1_start):
                global_i = s1_start + local_i
                row = sim_block[local_i]
                mask = row >= min_score
                if not mask.any():
                    continue

                new_ids    = chunk_ids[mask]
                new_scores = row[mask]

                merged_ids    = np.concatenate([top_ids[global_i],    new_ids])
                merged_scores = np.concatenate([top_scores[global_i], new_scores])

                if len(merged_ids) > k:
                    idx = np.argpartition(merged_scores, -k)[-k:]
                else:
                    idx = np.arange(len(merged_ids))

                top_ids[global_i]    = merged_ids[idx]
                top_scores[global_i] = merged_scores[idx]

    for i in range(n1):
        if len(top_ids[i]) == 0:
            results[i] = []
            continue
        order = np.argsort(top_scores[i])[::-1]
        results[i] = [(top_ids[i][j], float(top_scores[i][j])) for j in order]
    return results


# ── 1. TF-IDF Cosine Blocker (Name or Address or Combined) ────────────────────

def block_tfidf(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    text_column: str = "name_norm",
    k: int = 30,
) -> Dict[str, List[Tuple[str, float]]]:
    """Country-bucketed TF-IDF cosine blocker on specified text column."""
    all_cands: Dict[str, List[Tuple[str, float]]] = {eid: [] for eid in s1_df["entity_id"]}

    common_countries = set(s1_df["country_norm"].unique()) & set(s23_df["country_norm"].unique())

    for ctry in sorted(common_countries):
        s1_sub  = s1_df[s1_df["country_norm"] == ctry]
        s23_sub = s23_df[s23_df["country_norm"] == ctry]
        if s1_sub.empty or s23_sub.empty:
            continue

        vec, s23_mat = _build_vectorizer(s23_sub[text_column].tolist())
        s1_vecs      = vec.transform(s1_sub[text_column].tolist())
        s23_ids      = s23_sub["entity_id"].values

        hits = _batched_topk(s1_vecs, s23_mat, s23_ids, k=k)
        for sid, hit_list in zip(s1_sub["entity_id"], hits):
            all_cands[sid].extend(hit_list)

        del vec, s23_mat, s1_vecs
        gc.collect()

    return all_cands


# ── 2. Exact Normalized Name Inverted Index ────────────────────────────────────

def block_exact_name(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_matches_per_name: int = 50,
) -> Dict[str, List[Tuple[str, float]]]:
    """Exact normalized name hashtable lookup (O(1) dictionary retrieval)."""
    # Build hashtable: {country_norm: {name_norm: [s23_id, ...]}}
    table = defaultdict(lambda: defaultdict(list))
    for _, row in s23_df.iterrows():
        ctry = row["country_norm"]
        name = row["name_norm"]
        if name:
            table[ctry][name].append(row["entity_id"])

    all_cands = {}
    for _, row in s1_df.iterrows():
        sid  = row["entity_id"]
        ctry = row["country_norm"]
        name = row["name_norm"]
        
        matches = table[ctry].get(name, [])
        if matches:
            # Assign fixed high score 1.0 for exact name match
            all_cands[sid] = [(m, 1.0) for m in matches[:max_matches_per_name]]
        else:
            all_cands[sid] = []

    return all_cands


# ── 3. Rare Token Inverted Index ───────────────────────────────────────────────

def block_rare_tokens(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_token_freq: int = 30,
    max_cands_per_s1: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """Inverted index matching on rare business name/address tokens."""
    # Build document frequency of tokens per country
    df_count = defaultdict(lambda: defaultdict(int))
    token_to_s23 = defaultdict(lambda: defaultdict(list))

    for _, row in s23_df.iterrows():
        ctry = row["country_norm"]
        sid  = row["entity_id"]
        tokens = set((row["name_norm"] + " " + row["addr_norm"]).split())
        for t in tokens:
            if len(t) >= 4:  # ignore short generic words
                df_count[ctry][t] += 1
                token_to_s23[ctry][t].append(sid)

    all_cands = {}
    for _, row in s1_df.iterrows():
        sid  = row["entity_id"]
        ctry = row["country_norm"]
        tokens = set((row["name_norm"] + " " + row["addr_norm"]).split())

        hit_counts = defaultdict(int)
        for t in tokens:
            if len(t) >= 4 and df_count[ctry].get(t, 999999) <= max_token_freq:
                for target_id in token_to_s23[ctry][t]:
                    hit_counts[target_id] += 1

        if hit_counts:
            # Sort by number of rare token overlaps
            sorted_hits = sorted(hit_counts.items(), key=lambda x: -x[1])[:max_cands_per_s1]
            all_cands[sid] = [(tid, float(count) * 0.2) for tid, count in sorted_hits]
        else:
            all_cands[sid] = []

    return all_cands


# ── 4. Numeric / Building Number Index ─────────────────────────────────────────

def block_numeric_address(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    max_cands_per_s1: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """Inverted index matching on address numeric sequences (building numbers/PINs)."""
    table = defaultdict(lambda: defaultdict(list))
    for _, row in s23_df.iterrows():
        ctry = row["country_norm"]
        sid  = row["entity_id"]
        nums = set(extract_numbers(row["business_address"]))
        for n in nums:
            if len(n) >= 2:  # ignore single digits
                table[ctry][n].append(sid)

    all_cands = {}
    for _, row in s1_df.iterrows():
        sid  = row["entity_id"]
        ctry = row["country_norm"]
        nums = set(extract_numbers(row["business_address"]))

        hit_counts = defaultdict(int)
        for n in nums:
            if len(n) >= 2 and len(table[ctry].get(n, [])) <= 100:
                for target_id in table[ctry][n]:
                    hit_counts[target_id] += 1

        if hit_counts:
            sorted_hits = sorted(hit_counts.items(), key=lambda x: -x[1])[:max_cands_per_s1]
            all_cands[sid] = [(tid, float(count) * 0.15) for tid, count in sorted_hits]
        else:
            all_cands[sid] = []

    return all_cands


# ── Multi-Blocker Union Strategy ───────────────────────────────────────────────

def generate_multi_blocker_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    blocker_configs: List[dict],
    k_final: int = K_CANDIDATES,
) -> Tuple[Dict[str, List[str]], Dict[str, dict]]:
    """
    Executes multiple blockers and merges candidate pools.

    Returns:
      final_candidates: {s1_id: [candidate_s23_ids]}
      diagnostics: breakdown of candidates recovered per blocker
    """
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    all_s1_ids = s1_df["entity_id"].tolist()

    blocker_results = {}
    blocker_runtimes = {}

    for cfg in blocker_configs:
        b_type = cfg["type"]
        b_name = cfg["name"]
        t0 = time.time()
        logger.info(f"Running Blocker [{b_name}] (type={b_type})...")

        if b_type == "name_tfidf":
            res = block_tfidf(s1_df, s23_df, text_column="name_norm", k=cfg.get("k", 30))
        elif b_type == "addr_tfidf":
            res = block_tfidf(s1_df, s23_df, text_column="addr_norm", k=cfg.get("k", 20))
        elif b_type == "combo_tfidf":
            s1_sub = s1_df.copy()
            s23_sub = s23_df.copy()
            s1_sub["combo"] = s1_sub["name_norm"] + " " + s1_sub["addr_norm"]
            s23_sub["combo"] = s23_sub["name_norm"] + " " + s23_sub["addr_norm"]
            res = block_tfidf(s1_sub, s23_sub, text_column="combo", k=cfg.get("k", 20))
        elif b_type == "exact_name":
            res = block_exact_name(s1_df, s23_df, max_matches_per_name=cfg.get("max_matches", 30))
        elif b_type == "rare_tokens":
            res = block_rare_tokens(s1_df, s23_df, max_cands_per_s1=cfg.get("max_cands", 15))
        elif b_type == "numeric_addr":
            res = block_numeric_address(s1_df, s23_df, max_cands_per_s1=cfg.get("max_cands", 15))
        else:
            raise ValueError(f"Unknown blocker type: {b_type}")

        elapsed = time.time() - t0
        blocker_results[b_name] = res
        blocker_runtimes[b_name] = round(elapsed, 2)
        logger.info(f"  Finished Blocker [{b_name}] in {elapsed:.2f}s")

    # Merge candidates for each S1 entity
    final_candidates = {}
    recovery_stats = {b_name: 0 for b_name in blocker_results}

    for sid in all_s1_ids:
        score_map = defaultdict(float)
        blocker_origins = defaultdict(set)

        for b_name, b_dict in blocker_results.items():
            hits = b_dict.get(sid, [])
            for cid, score in hits:
                if score > score_map[cid]:
                    score_map[cid] = score
                blocker_origins[cid].add(b_name)

        sorted_cands = sorted(score_map.items(), key=lambda x: -x[1])[:k_final]
        final_cands_list = [cid for cid, _ in sorted_cands]
        final_candidates[sid] = final_cands_list

        for cid in final_cands_list:
            for b_name in blocker_origins[cid]:
                recovery_stats[b_name] += 1

    diagnostics = {
        "runtimes": blocker_runtimes,
        "contributions": recovery_stats,
    }

    return final_candidates, diagnostics
