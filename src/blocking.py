"""Memory-conscious candidate generation for the entity-resolution pipeline.

The module never builds a TF-IDF matrix for the complete reference corpus.
Cheap inverted indexes produce bounded name blocks first; character TF-IDF is
then evaluated only inside those blocks, one Source 1 batch at a time.
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
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence

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
DEFAULT_OUTPUT = Path("matching_candidates.csv")
DEFAULT_PROGRESS = Path("progress.json")


def normalize_name(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic Unicode-normalized ``name_norm`` values."""
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
    prefix_lengths: Sequence[int] = (3, 4, 5),
    rare_token_max_frequency: int = 200,
    max_postings: int = 5000,
) -> Dict[str, Mapping[str, Sequence[int]]]:
    """Build bounded exact-name, prefix, and rare-token inverted indexes."""
    names = df["name_norm"].fillna("").astype(str).tolist()
    exact: DefaultDict[str, List[int]] = defaultdict(list)
    prefixes: DefaultDict[str, List[int]] = defaultdict(list)
    token_frequency: Counter[str] = Counter()
    row_tokens: List[List[str]] = []
    for name in names:
        tokens = _tokens(name)
        row_tokens.append(tokens)
        token_frequency.update(set(tokens))
    rare_tokens: DefaultDict[str, List[int]] = defaultdict(list)
    for row_number, name in enumerate(names):
        if name:
            if len(exact[name]) < max_postings:
                exact[name].append(row_number)
            for length in prefix_lengths:
                prefix = name[:length]
                if len(prefixes[prefix]) < max_postings:
                    prefixes[prefix].append(row_number)
        for token in set(row_tokens[row_number]):
            if token_frequency[token] <= rare_token_max_frequency and len(rare_tokens[token]) < max_postings:
                rare_tokens[token].append(row_number)
    return {"exact": dict(exact), "prefix": dict(prefixes), "rare": dict(rare_tokens)}


def _memory_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**3
    except ImportError:
        return 0.0


def _log_progress(batch: int, total_batches: int, started: float, pairs: int) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    completed = batch / max(total_batches, 1)
    eta = elapsed / max(completed, 1e-9) * (1 - completed)
    disk_gb = shutil.disk_usage(Path.cwd()).used / 1024**3
    LOGGER.info(
        "Batch %d/%d (%.1f%%), ETA %.1f min, RAM %.2f GB, disk %.2f GB, candidate pairs %d",
        batch, total_batches, completed * 100, eta / 60, _memory_gb(), disk_gb, pairs,
    )


def save_candidate_pairs(
    rows: Iterable[Mapping[str, object]],
    output_path: str | Path = DEFAULT_OUTPUT,
    append: bool = True,
) -> int:
    """Append candidate rows with correct CSV quoting and return rows written."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    mode = "a" if append else "w"
    write_header = not append or not path.exists() or path.stat().st_size == 0
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for row in rows:
            writer.writerow([row["source1_entity_id"], row["candidate_entity_ids"]])
    return len(rows)


def resume_progress(
    progress_path: str | Path = DEFAULT_PROGRESS,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> int:
    """Return the next batch index, or zero when no compatible checkpoint exists."""
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "last_processed_batch": batch,
        "output_path": str(output_path),
        "candidate_pairs": pairs,
    }, indent=2), encoding="utf-8")
    temporary.replace(path)


def _tfidf_block_candidates(
    query_name: str,
    reference: pd.DataFrame,
    positions: Sequence[int],
    top_k: int,
    chunk_size: int,
) -> List[int]:
    if not query_name or not positions:
        return []
    scored: List[tuple[float, int]] = []
    for offset in range(0, len(positions), chunk_size):
        chunk_positions = positions[offset:offset + chunk_size]
        names = reference.iloc[list(chunk_positions)]["name_norm"].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), dtype=np.float32)
        matrix = vectorizer.fit_transform([query_name, *names])
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
        scored.extend(
            (float(score), chunk_positions[index])
            for index, score in enumerate(scores)
            if score > 0
        )
        del matrix, vectorizer, names, scores
        gc.collect()
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [position for _, position in scored[:top_k]]


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    batch_size: int = 1000,
    chunk_size: int = 50000,
    top_k: int = 50,
    output_path: str | Path = DEFAULT_OUTPUT,
    progress_path: str | Path = DEFAULT_PROGRESS,
) -> None:
    """Generate and checkpoint up to ``top_k`` unioned candidates per S1 row."""
    if batch_size < 1 or chunk_size < 1 or top_k < 1:
        raise ValueError("batch_size, chunk_size, and top_k must be positive")
    s1, s2, s3 = (normalize_name(frame) for frame in (s1_df, s2_df, s3_df))
    references = pd.concat([s2, s3], ignore_index=True, copy=False)
    ids = references["entity_id"].astype(str).tolist()
    blocks = build_blocks(references)
    total_batches = (len(s1) + batch_size - 1) // batch_size
    output = Path(output_path)
    progress = Path(progress_path)
    start_batch = resume_progress(progress, output)
    if start_batch == 0 and output.exists() and output.stat().st_size:
        raise FileExistsError(f"{output} exists without a matching checkpoint; remove it to restart")
    started = time.monotonic()
    total_pairs = 0
    for batch_number in range(start_batch, total_batches):
        batch = s1.iloc[batch_number * batch_size:(batch_number + 1) * batch_size]
        rows = []
        for source_id, name in zip(batch["entity_id"].astype(str), batch["name_norm"].astype(str)):
            positions = set(blocks["exact"].get(name, ()))
            for length in (3, 4, 5):
                positions.update(blocks["prefix"].get(name[:length], ()))
            for token in set(_tokens(name)):
                positions.update(blocks["rare"].get(token, ()))
            ranked = _tfidf_block_candidates(name, references, sorted(positions), top_k, chunk_size)
            candidates = list(dict.fromkeys(ids[position] for position in ranked))
            for position in sorted(positions):
                candidate_id = ids[position]
                if candidate_id not in candidates:
                    candidates.append(candidate_id)
                if len(candidates) == top_k:
                    break
            rows.append({"source1_entity_id": source_id, "candidate_entity_ids": ",".join(candidates[:top_k])})
        total_pairs += sum(len(str(row["candidate_entity_ids"]).split(",")) for row in rows if row["candidate_entity_ids"])
        save_candidate_pairs(rows, output, append=True)
        _write_progress(progress, batch_number, output, total_pairs)
        _log_progress(batch_number + 1, total_batches, started, total_pairs)
        del batch, rows
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source1", required=True, type=Path)
    parser.add_argument("--source2", required=True, type=Path)
    parser.add_argument("--source3", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s1 = pd.read_parquet(args.source1)
    s2 = pd.read_parquet(args.source2)
    s3 = pd.read_parquet(args.source3)
    generate_candidates(s1, s2, s3, args.batch_size, args.chunk_size, args.top_k, args.output, args.progress)


if __name__ == "__main__":
    main()