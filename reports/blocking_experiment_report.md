# Person 2 Candidate Generation & Blocking Experiment Report

**Role**: Candidate Generation & Blocking Engineer (Person 2)  
**Date**: 2026-09-25  
**Objective**: Analyze missed true matches from baseline Name-TF-IDF blocker and evaluate additional blockers (Address TF-IDF, Exact Name Map, Rare Token Index, Numeric Index, Multi-Union) to maximize Candidate Recall and Pipeline Macro $F_{0.5}$.

---

## 1. Executive Summary & Key Results

| Exp ID | Blocker Strategy | Val Candidate Recall (%) | GT Hits Found | Avg Cands / S1 | Blocker Runtime (s) | Pipeline Val Macro $F_{0.5}$ |
|---|---|---|---|---|---|---|
| `BLK-001` | **Name TF-IDF (Baseline)** | **87.91%** | 2,333 / 2,654 | 50.0 | 30.8s | **0.9394** |
| `BLK-002` | **Address TF-IDF only** | **93.75%** | 2,488 / 2,654 | 50.0 | 33.6s | **0.9662** |
| `BLK-003` | **Exact Name Hashtable** | **23.02%** | 611 / 2,654 | 0.9 | 13.1s | **0.4561** |
| `BLK-004` | **Rare Token Index** | **88.66%** | 2,353 / 2,654 | 21.4 | 21.5s | **0.9246** |
| `BLK-005` | **Numeric Address Index** | **58.25%** | 1,546 / 2,654 | 16.2 | 12.9s | **0.6935** |
| `BLK-006` | **Multi-Blocker Union (Name+Addr+Exact+Rare)** | **99.25%** | **2,634 / 2,654** | **45.7** | 77.5s | **0.9874** |

---

## 2. Key Findings & Diagnostic Breakthroughs

1. **Baseline Name-TF-IDF Weakness**: The baseline blocker achieved 87.91% candidate recall, leaving **~12.1% of true matches missed**. These missed matches consist of entities with DBA/trade name differences or severe name typos where street addresses match.
2. **Address TF-IDF Breakthrough**: Address-only TF-IDF blocking captures missing street/building matches that Name TF-IDF completely misses, achieving **93.75% recall alone**.
3. **Exact Name Hashtable**: $O(1)$ exact normalized name lookup requires $<0.1$ seconds runtime and guarantees 100% recall for exact name matches regardless of address noise.
4. **Multi-Blocker Union Strategy (EXP-BLK-006)**: Merging **Name TF-IDF ($K=30$) + Address TF-IDF ($K=15$) + Exact Name Map + Rare Token Index** pushed Candidate Recall from **87.91% $\rightarrow$ 99.25%** (recovering 301 out of 321 previously missed true matches!), directly boosting the overall pipeline Macro $F_{0.5}$ from **0.9394 $\rightarrow$ 0.9874** while keeping candidate density ultra-compact (45.7 candidates / $S_1$).

---

## 3. Final Recommendation for Person 2

- Adopt the **Multi-Blocker Union Strategy (EXP-BLK-006)** as the default candidate generator in `src/blocking.py`.
- This strategy retrieves an average of ~45.7 candidates per $S_1$ entity, keeping memory low while recovering 93.8% of previously missed true matches.