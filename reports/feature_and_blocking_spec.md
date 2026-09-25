# Feature Specification & Candidate Generation (Blocking) Roadmap

**Document Purpose**: 
1. **For Person 3 (Preprocessing & Feature Engineer)**: Complete reference specification of all 28 features, exact mathematical/string definitions, required normalization primitives, underlying dependencies, and feature importance rankings to guide target preprocessing pipeline development.
2. **For Person 2 (Blocking & Candidate Generation Engineer)**: Comprehensive analysis of true matches missed by the current Name TF-IDF blocker, and a step-by-step experiment roadmap to evaluate additional blockers (Address TF-IDF, Exact Name Hashtable, Rare-Token Index, and Numeric/Address Overlap Index) to push Candidate Recall from **88.7% $\rightarrow$ 95%+**.

---

## Part 1: Specification for Person 3 (Features & Preprocessing Targets)

### 1.1 Complete 28-Feature Reference Table

| # | Feature Name | Category | Primary Library / Method | Gain Importance | Target Preprocessing Needs |
|---|--------------|----------|--------------------------|-----------------|----------------------------|
| 1 | `addr_token_set` | Address Similarity | `rapidfuzz.fuzz.token_set_ratio` | **450,449 (Rank 1)** | Expand address abbrevs (`Rd` $\rightarrow$ `Road`), remove punct, lowercase |
| 2 | `addr_jaccard_token` | Address Similarity | Set intersection / union of words | **88,220 (Rank 2)** | Word tokenization, stop-word clean |
| 3 | `candidate_rank` | Candidate Context | 1-based index from Blocker | **24,745 (Rank 3)** | Candidate score ordering |
| 4 | `conflicting_digits` | Digit Analysis | `re.findall(r'\d+', addr)` | **5,824 (Rank 4)** | Extract digit sequences from address |
| 5 | `name_jaro_winkler` | Name Similarity | `rapidfuzz.distance.JaroWinkler` | **3,248 (Rank 5)** | NFKC Unicode normalization, prefix clean |
| 6 | `addr_ratio` | Address Similarity | `rapidfuzz.fuzz.ratio` | **2,388 (Rank 6)** | Levenshtein character alignment |
| 7 | `addr_len_ratio` | Structural | $\min(l_1, l_2) / \max(l_1, l_2)$ | **2,149 (Rank 7)** | Safe non-zero string length calculation |
| 8 | `addr_token_sort` | Address Similarity | `rapidfuzz.fuzz.token_sort_ratio` | 1,233 (Rank 8) | Sort tokens alphabetically |
| 9 | `name_partial` | Name Similarity | `rapidfuzz.fuzz.partial_ratio` | 689 (Rank 9) | Catch legal suffix & prefix truncation |
| 10 | `max_name_address` | Interaction | $\max(\text{name\_tok\_set}, \text{addr\_tok\_set})$ | 550 (Rank 10) | Combined similarity upper bound |
| 11 | `min_name_address` | Interaction | $\min(\text{name\_tok\_set}, \text{addr\_tok\_set})$ | 455 (Rank 11) | Conservative similarity floor |
| 12 | `addr_num_overlap` | Digit Analysis | Digit set Jaccard | 442 (Rank 12) | Extract building #, PIN/ZIP codes |
| 13 | `name_x_address` | Interaction | $\text{name\_jw} \times \text{addr\_tok\_set}$ | 390 (Rank 13) | Multiplicative similarity term |
| 14 | `name_token_set` | Name Similarity | `rapidfuzz.fuzz.token_set_ratio` | 228 (Rank 14) | Handles subset business names |
| 15 | `name_jaccard_token` | Name Similarity | Word token set Jaccard | 201 (Rank 15) | Token set intersection |
| 16 | `name_token_sort` | Name Similarity | `rapidfuzz.fuzz.token_sort_ratio` | 177 (Rank 16) | Word order permutation |
| 17 | `name_len_s23` | Structural | `len(s23_name_norm)` | 144 (Rank 17) | Character length of candidate name |
| 18 | `name_ratio` | Name Similarity | `rapidfuzz.fuzz.ratio` | 105 (Rank 18) | Character Levenshtein ratio |
| 19 | `name_len_ratio` | Structural | $\min(l_1, l_2) / \max(l_1, l_2)$ | 49 (Rank 19) | Name length ratio |
| 20 | `candidate_is_s3` | Source Feature | Binary (`s23_id.startswith("S3-")`) | 10.3 | Identify source origin ($S_2$ vs $S_3$) |
| 21 | `missing_address_asymmetry` | Missingness | Binary ($l_1 \ge 10 \land l_2 < 3$) | 8.2 | Flag asymmetric missing addresses |
| 22 | `strong_name_but_number_conflict` | Evidence Conflict | Binary ($\text{name} \ge 0.85 \land \text{conflict} = 1$) | 3.5 | Flag name match with address number mismatch |
| 23 | `exact_name_match` | Exact Match | Binary ($\text{s1\_name} == \text{s23\_name}$) | 0.0002 | Exact normalized name equality |
| 24 | `country_match` | Country | Binary ($\text{country}_1 == \text{country}_2$) | 0.0 | Open-set country string equality |
| 25 | `name_len_s1` | Structural | `len(s1_name_norm)` | 0.0 | Length of $S_1$ name |
| 26 | `blocking_score` | Blocker Score | Cosine from Blocker | 0.0 | Cosine hit score |
| 27 | `has_both_addresses` | Missingness | Binary ($l_1 \ge 5 \land l_2 \ge 5$) | 0.0 | Non-empty address flag |
| 28 | `strong_name_and_number_match` | Evidence Match | Binary ($\text{name} \ge 0.85 \land \text{num} \ge 0.5$) | 0.0 | Dual high-confidence evidence |

---

### 1.2 Required Preprocessing Pipeline Components for Person 3

To maximize the discriminatory power of the top features above, Person 3 must construct a robust preprocessing pipeline:

1. **NFKC Unicode Normalization**: Converts Devanagari, Kannada, and accented characters to standard forms (`unicodedata.normalize('NFKC', text)`).
2. **NaN & String Literal Safeguards**: Convert `None`, `np.nan`, and string literals `'nan'` / `'NaN'` / `'NAN'` directly to empty string `""`.
3. **Legal Suffix Harmonization**: Standardize legal suffixes (e.g. `Corp` $\rightarrow$ `Corporation`, `Pvt Ltd` $\rightarrow$ `Private Limited`, `LLC` $\rightarrow$ `Limited Liability Company`, French `SAS`/`SARL`/`SA`).
4. **Address Abbreviation Expansion**: Standardize street designators (`Rd` $\rightarrow$ `Road`, `St` $\rightarrow$ `Street`, `Ave` $\rightarrow$ `Avenue`, `Ste` $\rightarrow$ `Suite`, `Bldg` $\rightarrow$ `Building`).
5. **Digit Extraction Primitive**: Function returning ordered numeric sequences (`re.findall(r'\d+', address)`) used by `addr_num_overlap` and `conflicting_digits`.
6. **Open-Set Country Cleaning**: Lowercase string cleaning (`country_norm`) without fixed one-hot encoding, so unseen test countries (e.g., France) process seamlessly.

---

## Part 2: Blocking & Candidate Generation Roadmap for Person 2 (You)

### 2.1 Analysis of Missed Matches in Current Blocker

The current candidate generation blocker uses **Character 2–4 N-gram TF-IDF Cosine Similarity** on **Business Names** only (`name_norm`), bucketed by country.

#### Diagnostic Results on Baseline Blocker
- **Current Candidate Recall**: **88.7% – 89.5%** at $K=50$ candidates per source (up to 100 per $S_1$).
- **Missed True Matches**: **~10.5% – 11.3%** of true entity matches never enter the candidate set!
- **Classifier AUC**: **0.9999+** on candidate pairs.
- **Key Diagnosis**: The LightGBM classifier achieves near-perfect classification on pairs it receives. The overall pipeline $F_{0.5}$ ceiling is set by **Blocking Recall**.

#### Why Name-Only TF-IDF Misses ~11% of True Matches
1. **DBA (Doing Business As) & Brand Name Divergence**: An entity operates under a trade name in $S_1$ (e.g. `Historical Fund Care`) but a legal/parent name in $S_2/S_3$ (e.g. `Historical Fund III`). Their character n-grams differ significantly, so Name TF-IDF assigns a low cosine score.
2. **Missing or Truncated Business Names**: Some records in $S_2/S_3$ have truncated names or extreme typos (e.g. `smt mogal consmlntcy` vs `mogal consultancy`), while their street address is virtually identical.
3. **Common Name Words & High Document Frequency**: Generic business terms (`group`, `services`, `enterprises`, `solutions`, `limited`) dilute TF-IDF weights, causing rare entity names with common words to drop below top-$K$.

---

### 2.2 Proposed Multi-Blocker Architecture for Person 2

To push Candidate Recall from **88.7% $\rightarrow$ 95%+**, Person 2 should test a **Multi-Blocker Union Strategy**:

```
                              ┌─────────────────────────────────────────┐
                              │           Source-1 Entity               │
                              └────────────────────┬────────────────────┘
                                                   │
        ┌──────────────────────┬───────────────────┼───────────────────┬──────────────────────┐
        ▼                      ▼                   ▼                   ▼                      ▼
┌───────────────┐     ┌────────────────┐  ┌─────────────────┐  ┌───────────────┐     ┌─────────────────┐
│ Blocker 1:    │     │ Blocker 2:     │  │ Blocker 3:      │  │ Blocker 4:    │     │ Blocker 5:      │
│ Name TF-IDF   │     │ Address TF-IDF │  │ Exact Name Map  │  │ Rare Token    │     │ Numeric Address │
│ (Char 2-4)    │     │ (Char 2-4)     │  │ (Hash Inverted) │  │ Inverted Index│     │ Overlap Index   │
└───────┬───────┘     └───────┬────────┘  └────────┬────────┘  └───────┬───────┘     └────────┬────────┘
        │                     │                    │                   │                      │
        └─────────────────────┴─────────┬──────────┴───────────────────┴──────────────────────┘
                                        ▼
                       ┌──────────────────────────────────┐
                       │  Deduplicated Candidate Pool     │
                       │  (Target: <100 candidates / S1)  │
                       └──────────────────────────────────┘
```

---

### 2.3 Step-by-Step Implementation Experiments for Person 2

#### Experiment 2.1: Address TF-IDF Blocker (`blocker_addr_tfidf`)
- **Concept**: Fit a character 2–4 n-gram TF-IDF vectorizer on `addr_norm` (or `name_norm + " " + addr_norm`).
- **Target**: Recovers true matches where names diverge but street/building addresses match (`1600 Massachusetts Ave`).
- **Parameters**: `ngram_range=(2, 4)`, `max_features=50,000`, retrieve top-$K_{addr}=15$ per source.

#### Experiment 2.2: Exact Normalized Name Inverted Index (`blocker_exact_name`)
- **Concept**: Build an $O(1)$ Python dictionary mapping `name_norm` $\rightarrow$ `List[s23_entity_id]`.
- **Target**: Instant retrieval of exact name matches, capturing matches where address text is formatted differently or missing. Zero cosine overhead.

#### Experiment 2.3: Rare Token Inverted Index (`blocker_rare_tokens`)
- **Concept**: Identify rare tokens across name and address (IDF score $> 7.0$ or document frequency $< 20$). Index $S_2/S_3$ entities by these rare tokens.
- **Target**: Recovers entities sharing unique brand/location words (e.g. `titaniumcitycenter`, `himagirinagar`, `kovaithirungr`) even if surrounding text has heavy noise.

#### Experiment 2.4: Numeric / Building Number Overlap Index (`blocker_num_addr`)
- **Concept**: For each $S_1$ entity with a building/PIN number, query $S_2/S_3$ entities in the same country sharing the exact numeric sequence.
- **Target**: Recovers matches with heavy typos in street names but identical house/building numbers.

---

### 2.4 Evaluation Matrix Template for Person 2

Person 2 should execute each blocker individually and in combination on the validation split, recording metrics in the following format:

| Experiment ID | Candidate Blocker Strategy | $K$ per Source | Candidate Recall (%) | Avg Candidates / $S_1$ | Missed Matches Recovered | Blocker Runtime | Resulting Model $F_{0.5}$ |
|---|---|---|---|---|---|---|---|
| `BLK-01` (Current) | Name TF-IDF only | $K=50$ | 88.7% | 60 | Baseline | 3.4 min | 0.9461 |
| `BLK-02` | Address TF-IDF only | $K=30$ | TBD | TBD | TBD | TBD | TBD |
| `BLK-03` | Exact Name Inverted Map | - | TBD | TBD | TBD | TBD | TBD |
| `BLK-04` | Union: Name TF-IDF ($K=30$) + Addr TF-IDF ($K=15$) | 45 | TBD | TBD | TBD | TBD | TBD |
| `BLK-05` | Multi-Union: Name TF-IDF + Addr TF-IDF + Exact + Rare Token | 50 | Target $>95.0\%$ | $<100$ | TBD | TBD | TBD |

---

### 2.5 Summary Checklist for Person 2 (Blocking Lead)

- [ ] Measure baseline missed matches on full candidate set.
- [ ] Implement `src/blocking_address.py` (Address TF-IDF cosine blocker).
- [ ] Implement `src/blocking_exact.py` (Exact name hash map).
- [ ] Implement `src/blocking_rare.py` (Rare-token inverted index).
- [ ] Run candidate union merger and report Candidate Recall vs Average Candidates per $S_1$.
- [ ] Re-run LightGBM training on merged candidate pool to measure final pipeline $F_{0.5}$ gain.
