"""
blocking.py — Candidate generation via country-bucketed TF-IDF retrieval.

Algorithm
---------
For each country bucket:
  1. Fit a char-n-gram TF-IDF vectorizer on [S2 ∪ S3] names (l2-normalised).
  2. Query every S1 entity in that bucket against the joint S2+S3 index:
       - Cosine similarity = dot product (vectors are l2-normalised).
       - Memory-safe: outer loops over S1-batches × S23-chunks.
  3. Keep top-K hits per source (S2 and S3 separately).
  4. Return {s1_id: [s23_id, ...]} dict for every S1 entity.

Country fallback: if a test country never appeared in training (e.g. France),
the TF-IDF index is built from the test data itself — no hard-coded country list.
"""

import gc
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

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
    MODEL_DIR,
)
from preprocess import preprocess_df

logger = logging.getLogger(__name__)


# ── TF-IDF helpers ────────────────────────────────────────────────────────────

def build_vectorizer(texts: List[str]) -> Tuple[TfidfVectorizer, sp.csr_matrix]:
    """Fit a char-ngram TF-IDF vectorizer and return (vectorizer, matrix)."""
    vec = TfidfVectorizer(
        analyzer=TFIDF_ANALYZER,
        ngram_range=TFIDF_NGRAM_RANGE,
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
        strip_accents=None,   # keep Unicode chars (Devanagari etc.)
        norm="l2",            # l2-norm → dot product == cosine
        dtype=np.float32,
    )
    matrix = vec.fit_transform(texts)
    return vec, matrix


# ── Memory-safe batched top-K cosine ─────────────────────────────────────────

def _batched_topk_cosine(
    s1_matrix: sp.csr_matrix,
    s23_matrix: sp.csr_matrix,
    s23_ids: np.ndarray,
    k: int,
    s1_batch: int = S1_BATCH_SIZE,
    s23_chunk: int = S23_CHUNK_SIZE,
    min_score: float = MIN_SIM_SCORE,
) -> List[List[Tuple[str, float]]]:
    """
    For each S1 row return a list of up to `k` (s23_id, cosine_score) tuples,
    sorted by score descending.

    Uses nested loops (S1-batch × S23-chunk) so the peak-memory dense block
    is at most  s1_batch × s23_chunk × 4 bytes.

    With defaults (300 × 400K) that is ~480 MB per iteration — manageable.
    """
    n1   = s1_matrix.shape[0]
    n23  = s23_matrix.shape[0]
    k    = min(k, n23)

    results: List[List[Tuple[str, float]]] = [[] for _ in range(n1)]

    # Running top-K per S1 (stored as parallel arrays for speed)
    top_ids    = [np.empty(0, dtype=object)   for _ in range(n1)]
    top_scores = [np.empty(0, dtype=np.float32) for _ in range(n1)]

    n_s23_chunks = (n23 + s23_chunk - 1) // s23_chunk
    n_s1_batches = (n1  + s1_batch  - 1) // s1_batch

    for ci, s23_start in enumerate(range(0, n23, s23_chunk)):
        s23_end   = min(s23_start + s23_chunk, n23)
        chunk_mat = s23_matrix[s23_start:s23_end]          # (chunk, F)
        chunk_ids = s23_ids[s23_start:s23_end]

        for bi, s1_start in enumerate(range(0, n1, s1_batch)):
            s1_end   = min(s1_start + s1_batch, n1)
            q_mat    = s1_matrix[s1_start:s1_end]           # (batch, F)

            # cosine = dot(q_mat, chunk_mat.T)  [both l2-normalised]
            # Result shape: (batch, chunk) — dense
            sim_block = q_mat.dot(chunk_mat.T)
            if sp.issparse(sim_block):
                sim_block = sim_block.toarray()
            sim_block = sim_block.astype(np.float32)

            # Update running top-K for each S1 in this batch
            for local_i in range(s1_end - s1_start):
                global_i = s1_start + local_i
                row = sim_block[local_i]                    # (chunk,)

                # Filter by min score
                mask = row >= min_score
                if not mask.any():
                    continue

                new_ids    = chunk_ids[mask]
                new_scores = row[mask]

                # Merge with running top-K
                merged_ids    = np.concatenate([top_ids[global_i],    new_ids])
                merged_scores = np.concatenate([top_scores[global_i], new_scores])

                if len(merged_ids) > k:
                    # Partial sort: keep top-K
                    idx = np.argpartition(merged_scores, -k)[-k:]
                else:
                    idx = np.arange(len(merged_ids))

                top_ids[global_i]    = merged_ids[idx]
                top_scores[global_i] = merged_scores[idx]

        if (ci + 1) % 5 == 0 or ci == n_s23_chunks - 1:
            logger.info(
                f"  Blocking progress: S23 chunk {ci+1}/{n_s23_chunks} done"
            )

    # Build final sorted lists
    for i in range(n1):
        if len(top_ids[i]) == 0:
            results[i] = []
            continue
        order = np.argsort(top_scores[i])[::-1]
        results[i] = [
            (top_ids[i][j], float(top_scores[i][j])) for j in order
        ]
    return results


# ── Per-source blocking ───────────────────────────────────────────────────────

def _block_one_source(
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    source_tag: str,
    k: int = K_CANDIDATES,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Produce candidate (entity_id, score) lists for every S1 entity.
    Works country-by-country; falls back to global index for unknown countries.
    """
    all_candidates: Dict[str, List[Tuple[str, float]]] = {
        eid: [] for eid in s1_df["entity_id"]
    }

    s1_countries  = set(s1_df["country_norm"].unique())
    s23_countries = set(s23_df["country_norm"].unique())
    common        = s1_countries & s23_countries
    only_s1       = s1_countries - s23_countries  # fallback → all of s23

    logger.info(
        f"[{source_tag}] Countries in S1: {s1_countries} | "
        f"Countries in {source_tag}: {s23_countries}"
    )

    def _process_country_pair(s1_sub, s23_sub, label):
        logger.info(
            f"  [{source_tag}] Country '{label}': "
            f"{len(s1_sub)} S1 entities vs {len(s23_sub)} candidates"
        )
        vec, s23_mat = build_vectorizer(s23_sub["blocking_text"].tolist())
        s1_vecs      = vec.transform(s1_sub["blocking_text"].tolist())
        s23_ids      = s23_sub["entity_id"].values

        hits = _batched_topk_cosine(s1_vecs, s23_mat, s23_ids, k=k)
        for idx, (s1_id, hit_list) in enumerate(zip(s1_sub["entity_id"], hits)):
            all_candidates[s1_id].extend(hit_list)

        del vec, s23_mat, s1_vecs
        gc.collect()

    for country in sorted(common):
        s1_sub  = s1_df[s1_df["country_norm"] == country]
        s23_sub = s23_df[s23_df["country_norm"] == country]
        _process_country_pair(s1_sub, s23_sub, country)

    if only_s1:
        logger.info(
            f"  [{source_tag}] Fallback to global index for countries: {only_s1}"
        )
        s1_sub  = s1_df[s1_df["country_norm"].isin(only_s1)]
        _process_country_pair(s1_sub, s23_df, "GLOBAL_FALLBACK")

    return all_candidates


# ── Public API ────────────────────────────────────────────────────────────────

def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k: int = K_CANDIDATES,
) -> Dict[str, List[str]]:
    """
    Full blocking pipeline.  Returns:
        { s1_entity_id: [candidate_s23_entity_id, ...] }
    The candidate lists are deduplicated and sorted by blocking score.
    """
    logger.info("=== Blocking: S1 vs S2 ===")
    cands_s2 = _block_one_source(s1_df, s2_df, "S2", k=k)

    logger.info("=== Blocking: S1 vs S3 ===")
    cands_s3 = _block_one_source(s1_df, s3_df, "S3", k=k)

    # Merge, dedup, keep by score
    final: Dict[str, List[str]] = {}
    for s1_id in s1_df["entity_id"]:
        # Pool (id, score) from both sources and deduplicate by id (keep max)
        score_map: Dict[str, float] = {}
        for eid, score in cands_s2.get(s1_id, []):
            if score > score_map.get(eid, -1):
                score_map[eid] = score
        for eid, score in cands_s3.get(s1_id, []):
            if score > score_map.get(eid, -1):
                score_map[eid] = score

        # Sort by score and return id list
        sorted_cands = sorted(score_map.items(), key=lambda x: -x[1])
        final[s1_id] = [eid for eid, _ in sorted_cands]

    return final


def save_candidate_pairs(
    candidates: Dict[str, List[str]],
    s1_ids: List[str],
    output_path: Path,
) -> None:
    """Write candidate_pairs.tsv in the required format."""
    rows = [
        {
            "source1_entity_id":    s1_id,
            "candidate_entity_ids": ",".join(candidates.get(s1_id, [])),
        }
        for s1_id in s1_ids
    ]
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    n_with = sum(1 for r in rows if r["candidate_entity_ids"])
    logger.info(
        f"Saved {len(rows)} rows → {output_path}  "
        f"({n_with} with candidates, {len(rows)-n_with} singletons)"
    )
