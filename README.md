# Early Warning and Intervention Recommendation System for OTT Show Retention

## Overview
This project builds an early-warning system to identify OTT shows at risk of viewer drop-off using engagement signals from initial episodes. The model outputs probabilistic risk scores rather than hard classifications to handle uncertainty and class imbalance realistically.

## Approach
- Extract early engagement features from initial episodes
- Train a probabilistic risk model to estimate failure likelihood
- Convert risk scores into interpretable risk buckets
- Diagnose high-risk shows by comparing them against successful baselines
- Generate content-level intervention recommendations

## Key Insights
- Early engagement signals provide limited but useful information for risk prioritization
- Risk scoring and bucketing are more practical than hard failure prediction
- High recall is preferred to avoid missing potentially risky shows
- Diagnostic recommendations improve interpretability and actionability

## Limitations
- Failure labels are proxy indicators derived from engagement metrics
- Early predictions are inherently uncertain
- The system is designed for decision support, not automated actions

## How to Run
Open the provided Colab notebook and run cells sequentially.
