# Brazilian Houses – Rent Prediction & Market Segmentation

This project analyzes the **Brazilian Houses rental dataset** (10,962 properties across major Brazilian cities) to identify the key drivers of rent prices and segment the rental market into meaningful groups.

## Task 1 — Rent Prediction (Regression)
Multiple models were trained and compared, including:
- Stepwise Linear Regression (AIC/BIC)
- Lasso & Group Lasso
- GAM (Generalized Additive Model)
- XGBoost (best-performing)

**Final model:** XGBoost  
**Performance:** R² ≈ 0.992 with low prediction error on unseen test data.

## Task 2 — Market Clustering
Unsupervised clustering was applied using:
- K-Means
- Hierarchical Clustering

**Selected method:** K-Means (k = 2), producing two clear segments:
- **Affordable housing cluster**
- **High-end housing cluster**

## Key Insight
The strongest predictors of rent were **fire insurance cost** and **city/location**, with additional influence from HOA fees and floor level.
