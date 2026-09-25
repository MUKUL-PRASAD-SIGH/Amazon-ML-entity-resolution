"""
preprocess.py — Text normalization for business names and addresses.

Key operations
--------------
1. Unicode NFKC normalization  (handles Devanagari, Kannada, accented chars)
2. Lowercase + punctuation removal
3. Abbreviation expansion      (Corp→Corporation, Rd→Road, etc.)
4. Whitespace collapsing
5. Numeric token extraction    (for address number overlap feature)
"""

import re
import unicodedata
import pandas as pd

_NAME_ABBREVS_RAW = {
    # Legal suffixes
    r"\bcorp\b":        "corporation",
    r"\bco\b":          "company",
    r"\binc\b":         "incorporated",
    r"\bltd\b":         "limited",
    r"\bllc\b":         "limited liability company",
    r"\bllp\b":         "limited liability partnership",
    r"\bpvt\b":         "private",
    r"\bpvt ltd\b":     "private limited",
    r"\bpte\b":         "private",
    r"\bpte ltd\b":     "private limited",
    r"\blp\b":          "limited partnership",
    r"\bplc\b":         "public limited company",
    r"\bsas\b":         "societe par actions simplifiee",
    r"\bsarl\b":        "societe a responsabilite limitee",
    r"\bsa\b":          "societe anonyme",
    # Common business words
    r"\b&\b":           "and",
    r"\bsvc\b":         "services",
    r"\bsvcs\b":        "services",
    r"\btech\b":        "technology",
    r"\btechnol\b":     "technology",
    r"\bintl\b":        "international",
    r"\bmfg\b":         "manufacturing",
    r"\bmgmt\b":        "management",
    r"\bassoc\b":       "associates",
    r"\bassn\b":        "association",
    r"\bnatl\b":        "national",
    r"\bgrp\b":         "group",
    r"\bent\b":         "enterprises",
    r"\bhldgs\b":       "holdings",
    r"\binds\b":        "industries",
    r"\bdist\b":        "distributors",
    r"\bdistrib\b":     "distributors",
    r"\bsol\b":         "solutions",
    r"\bsoln\b":        "solutions",
    r"\bsolns\b":       "solutions",
    r"\bmkt\b":         "marketing",
    r"\bmktg\b":        "marketing",
    r"\badv\b":         "advertising",
    r"\bconst\b":       "construction",
    r"\bconsult\b":     "consulting",
    r"\bdev\b":         "development",
    r"\bfinl\b":        "financial",
    r"\bfin\b":         "finance",
    r"\binsur\b":       "insurance",
    r"\bhosp\b":        "hospital",
    r"\bmedl\b":        "medical",
    r"\bmed\b":         "medical",
    r"\bpharm\b":       "pharmaceutical",
    r"\brealty\b":      "realty",
    r"\brestnt\b":      "restaurant",
    r"\brest\b":        "restaurant",
    r"\brestaurant\b":  "restaurant",
}

_ADDR_ABBREVS_RAW = {
    # Street types
    r"\brd\b":      "road",
    r"\bst\b":      "street",
    r"\bave\b":     "avenue",
    r"\bav\b":      "avenue",
    r"\bblvd\b":    "boulevard",
    r"\bdr\b":      "drive",
    r"\bln\b":      "lane",
    r"\bct\b":      "court",
    r"\bpl\b":      "place",
    r"\bsq\b":      "square",
    r"\bhwy\b":     "highway",
    r"\bfwy\b":     "freeway",
    r"\bpkwy\b":    "parkway",
    r"\bexpy\b":    "expressway",
    r"\bxing\b":    "crossing",
    r"\btrl\b":     "trail",
    r"\bcir\b":     "circle",
    # Unit designators
    r"\bste\b":     "suite",
    r"\bapt\b":     "apartment",
    r"\bflr\b":     "floor",
    r"\bfl\b":      "floor",
    r"\bbldg\b":    "building",
    r"\bunit\b":    "unit",
    # Directions
    r"\bnorth\b":   "north",
    r"\bsouth\b":   "south",
    r"\beast\b":    "east",
    r"\bwest\b":    "west",
    r"\bnw\b":      "northwest",
    r"\bne\b":      "northeast",
    r"\bsw\b":      "southwest",
    r"\bse\b":      "southeast",
}


def _compile_abbrevs(raw: dict) -> list:
    return [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in raw.items()]


_NAME_ABBREVS  = _compile_abbrevs(_NAME_ABBREVS_RAW)
_ADDR_ABBREVS  = _compile_abbrevs(_ADDR_ABBREVS_RAW)

_PUNCT_RE      = re.compile(r"[^\w\s]")
_SPACE_RE      = re.compile(r"\s+")
_DIGIT_RE      = re.compile(r"\d+")


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return ""
        return str(value)
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    return s


def _unicode_norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _apply_abbrevs(text: str, abbrevs: list) -> str:
    for pattern, replacement in abbrevs:
        text = pattern.sub(replacement, text)
    return text


def normalize_name(text, expand_abbrevs: bool = True) -> str:
    text = _unicode_norm(_safe_str(text))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if expand_abbrevs:
        text = _apply_abbrevs(text, _NAME_ABBREVS)
        text = _SPACE_RE.sub(" ", text).strip()
    return text


def normalize_address(text, expand_abbrevs: bool = True) -> str:
    text = _unicode_norm(_safe_str(text))
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if expand_abbrevs:
        text = _apply_abbrevs(text, _ADDR_ABBREVS)
        text = _SPACE_RE.sub(" ", text).strip()
    return text


def extract_numbers(text) -> list[str]:
    return _DIGIT_RE.findall(_safe_str(text))


def preprocess_df(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Parallelized preprocessing pipeline for business names and addresses.
    Uses multi-core parallel worker chunking for 10M+ row scalability.
    """
    df = df.copy()
    for col in ("business_name", "business_address", "country"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            df.loc[df[col].str.lower() == "nan", col] = ""

    n_rows = len(df)
    if n_rows < 5000:
        df["name_norm"]  = df["business_name"].apply(normalize_name)
        df["addr_norm"]  = df["business_address"].apply(normalize_address)
        df["country_norm"] = df["country"].apply(lambda x: _safe_str(x).lower().strip())
        df["addr_nums"]  = df["business_address"].apply(lambda x: " ".join(extract_numbers(x)))
        df["blocking_text"] = df["name_norm"]
        return df

    # Parallel chunking for large DataFrames (10M+ rows)
    try:
        from joblib import Parallel, delayed
    except ImportError:
        df["name_norm"]  = df["business_name"].apply(normalize_name)
        df["addr_norm"]  = df["business_address"].apply(normalize_address)
        df["country_norm"] = df["country"].apply(lambda x: _safe_str(x).lower().strip())
        df["addr_nums"]  = df["business_address"].apply(lambda x: " ".join(extract_numbers(x)))
        df["blocking_text"] = df["name_norm"]
        return df

    n_workers = __import__("os").cpu_count() or 4
    chunk_size = max(5000, n_rows // (4 * n_workers))

    def _process_chunk(b_names, b_addrs, ctrys):
        name_out = [normalize_name(x) for x in b_names]
        addr_out = [normalize_address(x) for x in b_addrs]
        ctry_out = [_safe_str(x).lower().strip() for x in ctrys]
        nums_out = [" ".join(extract_numbers(x)) for x in b_addrs]
        return name_out, addr_out, ctry_out, nums_out

    b_names = df.get("business_name", pd.Series([""]*n_rows)).values
    b_addrs = df.get("business_address", pd.Series([""]*n_rows)).values
    ctrys   = df.get("country", pd.Series([""]*n_rows)).values

    chunks = [
        (b_names[i:i+chunk_size], b_addrs[i:i+chunk_size], ctrys[i:i+chunk_size])
        for i in range(0, n_rows, chunk_size)
    ]

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_chunk)(bn, ba, ct) for bn, ba, ct in chunks
    )

    names_flat = [x for r in results for x in r[0]]
    addrs_flat = [x for r in results for x in r[1]]
    ctrys_flat = [x for r in results for x in r[2]]
    nums_flat  = [x for r in results for x in r[3]]

    df["name_norm"]    = names_flat
    df["addr_norm"]    = addrs_flat
    df["country_norm"] = ctrys_flat
    df["addr_nums"]    = nums_flat
    df["blocking_text"] = df["name_norm"]

    return df
