# Learnings & Technical Insights Log

**Project**: Amazon ML Challenge 2026 — Business Entity Resolution  
**Focus**: Feature Engineering, Error Analysis, Decision Policy Optimization, Candidate Generation / Blocking  

---

## Executive Summary & Core Insights

> This document tracks key technical findings, feature importances, false positive/negative failure modes, decision policy breakthroughs, and candidate generation / blocking improvements discovered on the project codebase.

---

## 1. Candidate Generation & Blocking Breakthroughs (Person 2 Results)

Full experiment results documented in [reports/blocking_experiment_report.md](file:///e:/ML_Challenge/reports/blocking_experiment_report.md):

| Exp ID | Blocker Strategy | Candidate Recall (%) | GT Hits Found | Avg Cands / $S_1$ | Blocker Runtime | Pipeline Val Macro $F_{0.5}$ | Key Diagnostic Insight |
|---|---|---|---|---|---|---|---|
| `BLK-001` | **Name TF-IDF (Baseline)** | **87.91%** | 2,333 / 2,654 | 50.0 | 30.8s | **0.9394** | Baseline blocker misses ~12.1% of true matches (DBA trade name divergence & name typos). |
| `BLK-002` | **Address TF-IDF only** | **93.75%** | 2,488 / 2,654 | 50.0 | 33.6s | **0.9662** | **Major Breakthrough (+5.84% Recall)**. Recovers true matches where names diverge but street/building address matches! |
| `BLK-003` | **Exact Name Hashtable** | **23.02%** | 611 / 2,654 | 0.9 | 13.1s | **0.4561** | Ultra-fast $O(1)$ dictionary lookup for exact normalized names. |
| `BLK-004` | **Rare Token Index** | **88.66%** | 2,353 / 2,654 | 21.4 | 21.5s | **0.9246** | Compact candidate generator (21.4 cands/$S_1$) matching on unique brand/location words. |
| `BLK-005` | **Numeric Address Index** | **58.25%** | 1,546 / 2,654 | 16.2 | 12.9s | **0.6935** | Matches building numbers / PIN codes within the same country. |
| `BLK-006` | **Multi-Blocker Union (Name+Addr+Exact+Rare)** | **99.25%** | **2,634 / 2,654** | **45.7** | 77.5s | **0.9874** | **Ultimate Breakthrough (+11.34% Recall Boost!)**. Pushed Candidate Recall to **99.25%** and Pipeline Macro $F_{0.5}$ to **0.9874**! |

---

## 2. Baseline Audit & Evaluation Protocol

- **Metric Alignment**: The contest metric is **Macro-Averaged $F_{0.5}$** across $S_1$ entities, with $\beta = 0.5$. Because $F_{0.5}$ penalizes False Positives far more heavily than False Negatives, **Precision** is the primary driver of performance.
- **Singleton Handling**: Singletons ($S_1$ entities with 0 ground-truth matches) represent ~5.6% of entities. Predicting *any* false match on a singleton yields a score of 0.0 for that entity, whereas predicting empty yields 1.0. High confidence thresholding on singleton candidates directly protects macro score.
- **Validation Protocol**: 3-way split (Train 70% / Val 15% / Holdout 15%) partitioned strictly by `source1_entity_id`.

---

## 3. Feature Importance & Group Ablation Insights

### Baseline 17 Feature Importance Ranking (`reports/top_features.md`)
1. `addr_token_set` (Gain: 494,297): Dominant feature.
2. `addr_ratio` (Gain: 64,300): Address string Levenshtein.
3. `addr_jaccard_token` (Gain: 42,740): Address word token Jaccard.
4. `name_jaro_winkler` (Gain: 22,827): Top name feature.
5. `addr_token_sort` (Gain: 11,445): Address word reordering.

### Controlled Group Ablation Results (`reports/feature_ablation.csv`)

| Ablation Group | Features Active | Val Precision | Val Recall | Val Macro $F_{0.5}$ | FP Count | FN Count | Insight / Impact |
|----------------|-----------------|---------------|------------|---------------------|----------|----------|------------------|
| `all_features` | 17 | 0.9966 | 0.8855 | **0.9449** | 8 | 304 | Baseline reference |
| `minus_address_features` | 12 | 0.9556 | 0.7950 | **0.8855** | **98** | **544** | **Catastrophic Drop (-0.0594)**. Address features prevent false merges. |
| `minus_name_features` | 11 | 0.9947 | 0.8451 | **0.9277** | 12 | 411 | **Large Drop (-0.0172)**. Name features recover 107 missed true matches. |
| `minus_number_features` | 16 | 0.9953 | 0.8824 | **0.9435** | 11 | 312 | Minor drop (-0.0014). |
| `minus_fuzzy` | 13 | 0.9970 | 0.8828 | **0.9442** | 7 | 311 | Minor drop (-0.0007). |

---

## 4. Advanced 28 Feature Engineering & Intra-Entity Candidate Rank

Adding 11 new domain & interaction features (`src/advanced_features.py`) produced significant new feature importances (`reports/advanced_feature_importance.csv`):

| Rank | Feature | Gain Importance | Description |
|------|---------|-----------------|-------------|
| 1 | `addr_token_set` | 450,449 | Address token set similarity |
| 2 | `addr_jaccard_token` | 88,220 | Address Jaccard similarity |
| **3** | **`candidate_rank`** | **24,745** | **Intra-entity blocking rank (1, 2, 3...)**. Crucial context! |
| **4** | **`conflicting_digits`** | **5,824** | **Binary flag for conflicting street/building numbers**. Reduces FP! |
| 5 | `name_jaro_winkler` | 3,248 | Jaro-Winkler name similarity |
| 6 | `addr_ratio` | 2,388 | Address ratio |

---

## 5. Master Experimentation Log

All experiments logged to `experiments/results.csv`:

| Exp ID | Blocker / Feature Set | Threshold Policy | Val Precision | Val Recall | Val Macro $F_{0.5}$ | Blocker Recall | Key Finding / Impact |
|--------|-----------------------|------------------|---------------|------------|---------------------|----------------|----------------------|
| `EXP-001` | Baseline 17 | Global (0.30) | 0.9634 | 0.8804 | 0.9449 | 87.91% | 3-way split baseline |
| `EXP-003` | Advanced 28 | Global (0.55) | 0.9966 | 0.8870 | 0.9461 | 87.91% | Added `candidate_rank` & `conflicting_digits` |
| `BLK-002` | Addr-TFIDF Blocker | Global (0.35) | 0.9942 | 0.9328 | 0.9662 | 93.75% | Address blocking recovers 155 missed GT matches |
| `BLK-006` | **Multi-Blocker Union** | **Global (0.75)** | **0.9972** | **0.9851** | **0.9874** | **99.25%** | **Best Overall Pipeline**. Pushed Macro $F_{0.5}$ to 0.9874! |

---

## 6. Summary Recommendations

1. **Use Multi-Blocker Union Strategy**: Default `generate_candidates()` in `src/blocking.py` uses Name TF-IDF + Address TF-IDF + Exact Name Map + Rare Token Index.
2. **Adopt Advanced 28 Features**: Retain `candidate_rank`, `conflicting_digits`, `name_x_address`, `min_name_address`, `max_name_address`, and `missing_address_asymmetry` in `src/features.py`.
3. **Threshold Selection**: Use global threshold $t = 0.75$ with the Multi-Blocker Union strategy for optimal precision-weighted $F_{0.5}$ score.
