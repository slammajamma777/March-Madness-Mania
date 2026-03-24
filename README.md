# March-Madness-Mania
An end to end model built for predicting the probability of victory for each matchup in the NCAA for men's and women's basketball

# March Madness Machine Learning Prediction 

**Predicting NCAA Tournament outcomes using machine learning**

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)

---

## Overview

This project builds a complete end-to-end machine learning pipeline to predict NCAA March Madness tournament outcomes for both Men's and Women's brackets. Using 20+ years of college basketball data from the [Kaggle March Machine Learning Mania 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025) competition, the pipeline transforms raw game-by-game box scores into team-season features, engineers advanced basketball metrics, and trains classification models to output win probabilities for every possible tournament matchup.

**The winning model — Logistic Regression trained on 2013–2021 data with correlation-filtered features — achieved 76.12% accuracy on the 2025 test set**, surpassing the average human bracket accuracy of ~66.7–70%.

## Results at a Glance

| Model | Config | Train Window | Test Accuracy | MSE | CV Score |
|-------|--------|-------------|---------------|-----|----------|
| **Model C (Winner)** | Logistic Regression | 2013–2021 | **76.12%** | 0.179 | 72.9% |
| Model B | Logistic Regression | 2016–2022 | 71.64% | 0.194 | 70.4% |
| Model A | Support Vector Classifier | 2013–2018 | 68.66% | 0.199 | 72.9% |
| Claude Code (Autonomous) | Ensemble | Full history | — | — | — |

### 2026 Bracket Predictions (Model C)

- **Champion:** Duke Blue Devils (1)
- **Final Four:** Duke (1), Houston (2), Arizona (1), Michigan (1)
- **Notable Upset Picks:** Utah State (9) over Villanova (8), Santa Clara (10) over Kentucky (7), Iowa (9) over Clemson (8)

---

## Project Structure

```
march-madness-prediction/
│
├── data/                          # Kaggle competition datasets
│   ├── MRegularSeasonDetailedResults.csv
│   ├── WRegularSeasonDetailedResults.csv
│   ├── MNCAATourneyCompactResults.csv
│   ├── MNCAATourneySeeds.csv
│   └── ...                        # Additional Kaggle files (38 total)
│
├── notebooks/
│   ├── march_madness_pipeline.ipynb   # Main modeling notebook
│   └── eda_exploration.ipynb          # Exploratory data analysis
│
├── outputs/
│   ├── submission_model_b.csv         # Kaggle submission (Model B)
│   ├── submission_model_c.csv         # Kaggle submission (Model C)
│   └── submission_claude.csv          # Claude Code autonomous model
│
├── docs/
│   ├── march_madness_writeup.pdf      # Full methodology write-up
│   ├── march_madness_erd.html         # Simplified ERD diagram
│   └── march_madness_erd_full.html    # Complete ERD (Men's + Women's)
│
├── README.md
└── requirements.txt
```

---

## Methodology

This project follows the **CRISP-DM framework** (Cross-Industry Standard Process for Data Mining) across six phases:

### 1. Data Understanding

The Kaggle dataset spans **five sections** covering basics (team IDs, seeds, scores since 1984), team box scores (game-by-game stats since 2003 for men / 2010 for women), geography, public rankings, and supplements (coaches, conferences). The primary feature source is `MRegularSeasonDetailedResults.csv` with **122,775 games and 34 columns** of box score statistics.

Key EDA findings that shaped modeling decisions:
- Three-point attempts have increased by ~5.67 per game since 2003, while turnovers have decreased by ~3.7 — the game has fundamentally changed
- Strong statistical teams don't always translate to tournament success (e.g., undefeated Miami OH ranked only ~55th by composite rankings)
- Strength of schedule is a critical missing feature that must be engineered

### 2. Data Preparation

The pipeline transforms raw data through six steps:

1. **Separate game rows into team-level rows** — each game produces two rows (one per team) with unified column names rather than W/L prefixes
2. **Engineer advanced statistics per game** — eFG%, turnover percentage, offensive/defensive rebound percentage, free throw factor (Dean Oliver's Four Factors), offensive/defensive efficiency
3. **Compute aggregation-dependent metrics** — strength of schedule (opponent win percentage), momentum (recent game performance), conference strength
4. **Aggregate to team-season level** — mean and standard deviation of all metrics per team per season
5. **Join to tournament matchups** — merge team-season features onto historical tournament game results
6. **Calculate differentials** — for each matchup, compute `TeamA_stat - TeamB_stat` where TeamA is always the lower TeamID

The final dataset contains **73 features and 1,449 historical tournament matchups**.

### 3. Feature Engineering

Features fall into several categories:

| Category | Examples | Source |
|----------|----------|--------|
| **Four Factors** (Oliver, 2004) | eFG%, Turnover%, ORB%, FT Factor | Calculated from box scores |
| **Efficiency** | Offensive/Defensive/Net Efficiency, Adjusted variants | Points per possession estimates |
| **Schedule Strength** | Opponent Win%, Out-of-Conference Win%, Opponent Net Efficiency | Aggregated from opponent results |
| **Momentum** | Win streak, last-10 record | Late-season game filtering |
| **Consistency** | Standard deviation of key stats | Variance of game-level metrics |
| **Seeding** | Tournament seed number | `MNCAATourneySeeds` |

**Top 5 features correlated with tournament wins:**

| Feature | Correlation |
|---------|:-----------:|
| PointsDiff_diff | 0.41 |
| Net_Eff_diff | 0.40 |
| OppWinPercentage_diff | 0.36 |
| Out_Conference_Win_Percentage_diff | 0.36 |
| Opp_Def_Eff_diff | 0.33 |

### 4. Modeling

Five classification models were tested across **5 iterations** of progressive refinement:

**Models:** Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost

**Iteration Progression:**

| Iteration | Focus | Key Change |
|-----------|-------|------------|
| 1 | Baseline | All 73 features, 2003–2019 train, default params |
| 2 | Hyperparameter tuning | GridSearchCV optimization per model |
| 3 | Split exploration (variable val/test) | 10 different train/val/test windows |
| 4 | Split exploration (fixed val/test) | 10 windows with consistent val=2023–2024, test=2025 |
| 5 | Feature selection | (a) Drop raw box scores, (b) Drop features with \|r\| < 0.10 |

**Key finding:** Shorter, more recent training windows (e.g., 2013–2021) consistently outperformed longer historical windows — recency matters in sports modeling. Correlation-threshold filtering reduced features from 73 to 36 and meaningfully improved generalization.

### 5. Evaluation

On the **2025 test set**, Model C (Logistic Regression, 2013–2021 train, correlation-filtered features) achieved:

- **Accuracy:** 76.12% ✅ (target was >70%)
- **MSE:** 0.179
- **Precision (wins):** 0.85
- **Recall (wins):** 0.72
- **F1-Score:** 0.78

Model C outperformed Models A and B across all classification metrics, particularly excelling at correctly identifying actual winners (recall), which is the most important quality for bracket prediction.

### 6. Deployment

Three model brackets and one human bracket were generated for 2026:

- **Model B** — Logistic Regression, 2016–2022 training window
- **Model C** — Logistic Regression, 2013–2021 training window (best performer)
- **Claude Code** — Fully autonomous pipeline built by Claude Code from the same raw data
- **Human (Sam)** — Personal bracket made without model assistance

All three ML models predicted conservatively (3+ one-seeds in the Final Four), while the human bracket took significantly more risks with upsets. The only upset all three models agreed on: **Utah State (9) over Villanova (8)**.

---

## Key Learnings

- **Recency > volume** — Training on 2013–2021 outperformed training on 2003–2021. College basketball evolves, and older data can hurt.
- **Feature selection matters** — Cutting features from 73 to 36 via a |r| ≥ 0.10 correlation threshold improved the best CV score from ~71% to ~73%.
- **Logistic Regression wins** — Simpler models outperformed XGBoost and Random Forest for this classification task, likely due to the relatively small training set size (~600–900 matchups).
- **Models play it safe** — ML models heavily favor higher seeds and rarely pick dramatic upsets, which is rational for accuracy optimization but makes for less exciting brackets.

---

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **Jupyter Notebooks** — Development and experimentation
- **Kaggle** — Data source and competition submission
- **Claude** — Agentic AI collaboration partner for development, feature engineering, and deployment ([Anthropic](https://www.anthropic.com))

---

## Data Source

All data comes from the [Kaggle March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026/overview) competition. The dataset includes 38 files across Men's and Women's divisions, covering seasons from 1985–2026 (compact results) and 2003–2026 (detailed box scores for men) / 2010–2026 (for women).

An interactive ERD diagram of the full dataset structure is included in `docs/march_madness_erd_full.html`.

---

## References

- Oliver, D. (2004). *Basketball on Paper: Rules and Tools for Performance Analysis*. University of Nebraska Press.
- Basketball Reference. (2025). *Dean Oliver's Four Factors*. [basketball-reference.com/about/factors.html](https://www.basketball-reference.com/about/factors.html)
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- Kaggle. (2025). *March Machine Learning Mania 2025*. [kaggle.com/competitions/march-machine-learning-mania-2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025)
- Anthropic. (2025). *Claude*. Used as an agentic development partner throughout this project. [claude.ai](https://www.claude.ai)

---

## Author

**Sam Hirons** — Data Science, University of Colorado Boulder

---

*Built for Kaggle March Machine Learning Mania 2025. March Madness will never be predictable, but the excitement will always last.* 🏀
