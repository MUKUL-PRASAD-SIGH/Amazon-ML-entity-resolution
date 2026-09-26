"""Memory-efficient blocking and candidate-pair generation.

The implementation keeps Source 2 and Source 3 separate and never creates a
combined reference dataframe or a corpus-sized TF-IDF matrix. Inverted indexes
first identify bounded name blocks. Character TF-IDF is then fitted only for
those block rows, in configurable reference chunks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)
STOPWORDS = frozenset(
    "and the of ltd limited llc inc incorporated company co corporation pvt private "
    "enterprises enterprise group international global store shop trading".split()
)
_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")
DEFAULT_OUTPUT = Path("output/candidate_pairs.tsv")
DEFAULT_PROGRESS = Path("output/progress.json")
Posting = Tuple[int, int]


def normalize_name(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with Unicode-normalized ``name_norm`` values."""
    result = df.copy()
    source = result["name_norm"] if "name_norm" in result else result["business_name"]

    def clean(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        text = unicodedata.normalize("NFKC", text).casefold()
        text = _NON_WORD.sub(" ", text)
        return _SPACES.sub(" ", text).strip()

    result["name_norm"] = source.map(clean)
    return result


def _tokens(name: str) -> List[str]:
    return [token for token in name.split() if len(token) > 1 and token not in STOPWORDS]


def build_blocks(
    df: pd.DataFrame,
    source_number: int = 0,
    prefix_length: int = 4,
    rare_token_max_frequency: int = 200,
    max_postings: int = 5000,
) -> Dict[str, Mapping[str, Sequence[Posting]]]:
    """Build bounded exact-name, prefix, and rare-token posting lists.

    A posting is ``(source_number, row_number)`` so Source 2 and Source 3 can
    be queried independently without concatenating their dataframes.
    """
    if prefix_length < 1 or max_postings < 1:
        raise ValueError("prefix_length and max_postings must be positive")
    names = df["name_norm"].fillna("").astype(str)
    exact: DefaultDict[str, List[Posting]] = defaultdict(list)
    prefixes: DefaultDict[str, List[Posting]] = defaultdict(list)
    token_frequency: Counter[str] = Counter()

    for name in names:
        token_frequency.update(set(_tokens(name)))
    rare_tokens: DefaultDict[str, List[Posting]] = defaultdict(list)
    for row_number, name in enumerate(names):
        posting = (source_number, row_number)
        if name:
            if len(exact[name]) < max_postings:
                exact[name].append(posting)
            prefix = name[:prefix_length]
            if len(prefixes[prefix]) < max_postings:
                prefixes[prefix].append(posting)
        for token in set(_tokens(name)):
            if token_frequency[token] <= rare_token_max_frequency and len(rare_tokens[token]) < max_postings:
                rare_tokens[token].append(posting)
    return {"exact": dict(exact), "prefix": dict(prefixes), "rare": dict(rare_tokens)}


def _memory_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except ImportError:
        return 0.0


def _id_column(df: pd.DataFrame) -> str:
    for column in ("entity_id", "source1_entity_id", "source2_entity_id", "source3_entity_id"):
        if column in df.columns:
            return column
    raise KeyError("Input dataframe must contain entity_id or a source-specific entity ID column")


def _log_progress(batch: int, total_batches: int, started: float, pairs: int) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    completed = batch / max(total_batches, 1)
    eta_seconds = elapsed / max(completed, 1e-9) * (1 - completed)
    disk_gb = shutil.disk_usage(Path.cwd()).used / 1024**3
    LOGGER.info(
        "Batch %d/%d | %.1f%% | ETA %.1f min | RAM %.2f GB | disk %.2f GB | pairs %d",
        batch, total_batches, completed * 100, eta_seconds / 60, _memory_gb(), disk_gb, pairs,
    )


def save_candidate_pairs(
    rows: Iterable[Tuple[str, str]],
    output_path: str | Path = DEFAULT_OUTPUT,
    append: bool = True,
) -> int:
    """Append ``(source1_id, candidate_ids)`` rows to the required TSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_rows = list(rows)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if write_header:
            writer.writerow(("source1_entity_id", "candidate_entity_ids"))
        writer.writerows(pair_rows)
    return len(pair_rows)


def resume_progress(
    progress_path: str | Path = DEFAULT_PROGRESS,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> int:
    """Return the next Source 1 batch index from a matching checkpoint."""
    path = Path(progress_path)
    if not path.exists():
        return 0
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("output_path") == str(Path(output_path)):
            return int(state.get("last_processed_batch", -1)) + 1
    except (OSError, ValueError, TypeError):
        LOGGER.warning("Ignoring unreadable progress file: %s", path)
    return 0


def _write_progress(path: Path, batch: int, output_path: Path, pairs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "last_processed_batch": batch,
        "output_path": str(output_path),
        "candidate_pairs_written": pairs,
    }, indent=2), encoding="utf-8")
    temporary.replace(path)


def _tfidf_candidates(
    query_matrix: np.ndarray,
    references: Mapping[int, pd.DataFrame],
    postings: Sequence[Posting],
    top_k: int,
    chunk_size: int,
    vectorizer: TfidfVectorizer,
) -> List[Posting]:
    """Score only block postings, using pre-fitted sparse TF-IDF per reference chunk."""
    scored: List[Tuple[float, Posting]] = []
    for source_number in (0, 1):
        source_positions = [row for source, row in postings if source == source_number]
        for offset in range(0, len(source_positions), chunk_size):
            positions = source_positions[offset:offset + chunk_size]
            if not positions:
                continue
            names = references[source_number].iloc[positions]["name_norm"].fillna("").astype(str).tolist()
            ref_matrix = vectorizer.transform(names)
            scores = cosine_similarity(query_matrix, ref_matrix).ravel()
            scored.extend(
                (float(score), (source_number, position))
                for position, score in zip(positions, scores)
                if score > 0
            )
            del names, ref_matrix, scores
            gc.collect()
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [posting for _, posting in scored[:top_k]]


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    batch_size: int = 1000,
    chunk_size: int = 50000,
    top_k: int = 50,
    output_path: str | Path = DEFAULT_OUTPUT,
    progress_path: str | Path = DEFAULT_PROGRESS,
    limit_source1: int | None = None,
) -> None:
    """Generate at most ``top_k`` unique candidates per Source 1 row."""
    if batch_size < 1 or not 25000 <= chunk_size <= 50000 or top_k < 1:
        raise ValueError("batch_size/top_k must be positive and chunk_size must be 25000-50000")
    
    if limit_source1 and limit_source1 > 0:
        s1_df = s1_df.head(limit_source1)

    s1, s2, s3 = (normalize_name(frame) for frame in (s1_df, s2_df, s3_df))
    
    LOGGER.info("Fitting global TF-IDF on Source 1 names...")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
    vectorizer.fit(s1["name_norm"].fillna("").astype(str))
    LOGGER.info("Global TF-IDF fitted.")

    references = {0: s2, 1: s3}
    reference_ids = {
        source: references[source][_id_column(references[source])].astype(str).tolist()
        for source in references
    }
    blocks = {
        source: build_blocks(references[source], source_number=source)
        for source in references
    }
    s1_ids = s1[_id_column(s1)].astype(str)
    total_batches = (len(s1) + batch_size - 1) // batch_size
    output = Path(output_path)
    progress = Path(progress_path)
    start_batch = resume_progress(progress, output)
    if start_batch == 0 and output.exists() and output.stat().st_size:
        raise FileExistsError(f"{output} exists without a matching checkpoint; remove it to restart")

    started = time.monotonic()
    pairs_written = 0
    for batch_number in range(start_batch, total_batches):
        begin = batch_number * batch_size
        end = min(begin + batch_size, len(s1))
        rows: List[Tuple[str, str]] = []
        s1_batch_names = s1.iloc[begin:end]["name_norm"].fillna("").astype(str).tolist()
        s1_batch_matrices = vectorizer.transform(s1_batch_names)
        
        for idx, row_number in enumerate(range(begin, end)):
            name = s1_batch_names[idx]
            postings: set[Posting] = set()
            for source in references:
                source_blocks = blocks[source]
                postings.update(source_blocks["exact"].get(name, ()))
                postings.update(source_blocks["prefix"].get(name[:4], ()))
                for token in set(_tokens(name)):
                    postings.update(source_blocks["rare"].get(token, ()))
            ranked = _tfidf_candidates(s1_batch_matrices[idx], references, sorted(postings), top_k, chunk_size, vectorizer)
            candidates = list(dict.fromkeys(ranked))
            source_id = s1_ids.iloc[row_number]
            candidate_ids = [reference_ids[source][position] for source, position in candidates[:top_k]]
            rows.append((source_id, ",".join(candidate_ids)))
        pairs_written += save_candidate_pairs(rows, output, append=True)
        _write_progress(progress, batch_number, output, pairs_written)
        _log_progress(batch_number + 1, total_batches, started, pairs_written)
        del rows
        gc.collect()


def validate_output(output_path: Path, limit: int, elapsed: float, gt_path: Path = None):
    LOGGER.info("--- VALIDATION REPORT ---")
    LOGGER.info(f"Runtime: {elapsed:.2f} seconds")
    df = pd.read_csv(output_path, sep="\t", dtype=str)
    
    # Requirements
    num_rows = len(df)
    unique_s1 = df["source1_entity_id"].nunique()
    LOGGER.info(f"Number of output rows: {num_rows}")
    LOGGER.info(f"Exactly one output row per Source1 entity: {num_rows == unique_s1 and num_rows == limit}")
    
    cand_counts = []
    has_duplicates = False
    valid_ids = True
    
    for cands_str in df["candidate_entity_ids"].fillna(""):
        if not cands_str:
            cand_counts.append(0)
            continue
        cands = cands_str.split(",")
        cand_counts.append(len(cands))
        if len(cands) != len(set(cands)):
            has_duplicates = True
        for c in cands:
            if not c.startswith("S2-") and not c.startswith("S3-"):
                valid_ids = False
                
    cand_counts = np.array(cand_counts)
    LOGGER.info(f"No duplicate candidate IDs: {not has_duplicates}")
    LOGGER.info(f"Candidates are only S2/S3 IDs: {valid_ids}")
    LOGGER.info(f"Candidate count <= 50 for all: {(cand_counts <= 50).all()}")
    LOGGER.info(f"Average candidates per Source1: {cand_counts.mean():.2f}")
    LOGGER.info(f"Min/Max candidates: {cand_counts.min()} / {cand_counts.max()}")
    
    if gt_path and gt_path.exists():
        gt = pd.read_csv(gt_path)
        # Check recall
        hits = 0
        total = 0
        s1_to_cands = dict(zip(df["source1_entity_id"], df["candidate_entity_ids"].fillna("")))
        for _, row in gt.iterrows():
            s1_id = str(row.get("source1_entity_id", ""))
            s2_id = str(row.get("source2_entity_id", ""))
            if s1_id in s1_to_cands:
                total += 1
                if s2_id in s1_to_cands[s1_id].split(","):
                    hits += 1
        if total > 0:
            LOGGER.info(f"Candidate recall against ground truth (in limited set): {hits/total*100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", type=Path, default=Path("preprocessed/source1.parquet"))
    parser.add_argument("--source2", type=Path, default=Path("preprocessed/source2.parquet"))
    parser.add_argument("--source3", type=Path, default=Path("preprocessed/source3.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit-source1", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s1 = pd.read_parquet(args.source1)
    s2 = pd.read_parquet(args.source2)
    s3 = pd.read_parquet(args.source3)
    
    if args.limit_source1:
        args.output = Path(f"output/candidate_pairs_{args.limit_source1}.tsv")
        args.progress = Path(f"output/progress_{args.limit_source1}.json")
        
    start_time = time.monotonic()
    generate_candidate_pairs(
        s1, s2, s3, args.batch_size, args.chunk_size, args.top_k, args.output, args.progress, args.limit_source1
    )
    
    if args.limit_source1:
        validate_output(args.output, args.limit_source1, time.monotonic() - start_time, Path("train_ground_truth.tsv"))


if __name__ == "__main__":
    main()
