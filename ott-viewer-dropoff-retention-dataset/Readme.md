# Early Warning and Intervention Recommendation System for OTT Show Retention

A machine learning system that predicts, from a show's first three episodes, whether it is likely to lose viewers later in the season — and automatically recommends specific content-level fixes for shows flagged as high-risk.

## Overview

Streaming platforms lose viewers gradually across a season, but by the time drop-off is obvious in late-episode data, it's too late to intervene. This project asks a narrower, more useful question: **can early-episode engagement signals alone predict which shows will fail to retain viewers later on** — and if so, can that prediction be turned into an actionable recommendation rather than just a risk score?

The system has three parts:
1. An **XGBoost classifier** trained on engagement signals from episodes 1–3 to predict late-season failure
2. A **validation layer** that checks the model against a simpler baseline and cross-validation, rather than trusting a single train/test split
3. A **diagnostic and intervention engine** that compares each high-risk show's early engagement profile against successful shows and surfaces the specific dimensions (pacing, hook strength, etc.) driving its risk

## Dataset

This project uses the [OTT Viewer Drop-Off & Retention Risk Dataset (v1.0)](#), which combines real show metadata with synthetic engagement signals.

**What's real:** show titles, platforms, genres, and release years, sourced from the TMDB API.

**What's synthetic:** all viewer engagement and retention signals — `pacing_score`, `hook_strength`, `avg_watch_percentage`, `drop_off`, `cognitive_load`, etc. — are synthetically generated, grounded in realistic OTT viewing-behavior assumptions (e.g., strong hooks reduce early drop-off, high cognitive load increases churn). **No real user viewing data is included.**

This is disclosed here deliberately: the project is built and evaluated as a modeling exercise on realistic-but-synthetic data, not as a claim about real streaming platform behavior. The methodology (feature engineering, validation approach, evaluation rigor) is the actual point of the project, and would transfer directly to real engagement data if it were available.

| Property | Value |
|---|---|
| Raw records | 33,171 episode-level rows |
| Unique shows (after feature engineering) | 478 |
| Class balance | 367 "failed" vs. 111 "not failed" shows (~3.4:1) |
| Early-episode window used | Episodes 1–3 |
| Failure label | `avg_watch_percentage < 60%` in episodes after episode 3 |

## Methodology

1. **Feature engineering** — early-episode data (episodes 1–3) is aggregated per show into 10 features: pacing score, hook strength, visual intensity, average watch percentage, pause count, rewind count, cognitive load, episode duration, intro-skip rate, and night-watch-safe rate.
2. **Label construction** — a show is labeled "failed" if its average watch percentage in episodes after episode 3 drops below 60%.
3. **Model** — `XGBClassifier` (300 estimators, max depth 4, learning rate 0.05), chosen for its ability to capture non-linear feature interactions and to directly provide feature-importance rankings used downstream.
4. **Validation** — rather than trusting a single 80/20 split, the model is benchmarked against a Logistic Regression baseline and re-evaluated with 5-fold stratified cross-validation (see [Results](#results) — this step materially changed the reported performance).
5. **Risk output** — predictions are probabilistic, not hard-classified, and bucketed into Low / Medium / High risk tiers to reflect model confidence rather than force a binary call under class imbalance.
6. **Intervention engine** — for each high-risk show, its early engagement profile is z-score-normalized against the mean profile of successful (low-risk, non-failed) shows. The most underperforming dimensions are mapped to specific, human-readable content recommendations (e.g., low `hook_strength` → "Strengthen early narrative hooks to capture viewer attention").

## Results

| Metric | Single 80/20 split | 5-fold cross-validation |
|---|---|---|
| XGBoost ROC-AUC | 0.77 | **0.735** (± 0.021) |
| Logistic Regression ROC-AUC (baseline) | 0.68 | **0.728** (± 0.019) |

The single-split comparison suggested XGBoost had a large ~13% edge over the Logistic Regression baseline. Cross-validation showed this was largely an artifact of a small test set (96 shows) — the real, more reliable gap between the two models is modest, not dramatic. XGBoost was still kept as the final model because it performs at least as well as the baseline across every fold, captures non-linear interactions a linear model cannot, and directly powers the feature-importance ranking the intervention engine depends on.

At the operating threshold used for risk classification (0.4), the model prioritizes **recall over precision** by design — this is an early-warning system, where missing a genuinely at-risk show is more costly than a false alarm:

| | Precision | Recall | F1 |
|---|---|---|---|
| Failed shows (class 1) | 0.83 | **0.97** | 0.90 |
| Non-failed shows (class 0) | 0.75–0.88 | 0.27–0.32 | 0.40–0.47 |

**88 shows** in the flagged set received automated, targeted intervention recommendations from the diagnostic engine.

## Limitations

- **Synthetic engagement data.** The core predictive signals are simulated, not real user telemetry — results demonstrate methodology, not a validated real-world retention model.
- **Small sample size.** 478 shows total (96 in any given test split) is modest for a gradient-boosted model; cross-validation was added specifically to guard against overfitting conclusions to one split.
- **Proxy labels.** "Failure" is defined as a watch-percentage threshold, not explicit user churn or cancellation — a reasonable proxy, but not ground truth.
- **Recall-optimized by design.** The chosen threshold accepts more false positives in exchange for catching more true failures, which is appropriate for an early-warning/decision-support tool but would need re-tuning for a fully automated, high-stakes use case.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `matplotlib` / `seaborn`

## Project Structure

```
├── Early_Warning_and_Intervention_Recommendation_System_for_OTT_Show_Retention.ipynb
├── ott_viewer_dropoff_retention_us_v1.0.csv
└── README.md
```

## Running Locally

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
jupyter notebook Early_Warning_and_Intervention_Recommendation_System_for_OTT_Show_Retention.ipynb
```

---

Built by Shreya Gupta — BITS Pilani, Hyderabad Campus.
