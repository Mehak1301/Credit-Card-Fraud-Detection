# Credit Risk Modelling — PD, LGD, EAD & Expected Loss

An end-to-end credit risk pipeline built in Python: data cleaning and feature engineering, Weight-of-Evidence (WOE) binning, a logistic regression Probability-of-Default (PD) model, and a portfolio-level Expected Loss (EL) calculation that mirrors the Basel II/III regulatory framework.

## Overview

Given a portfolio of loan applications, the notebook:
1. Cleans and engineers features from raw applicant/loan data
2. Bins every feature using Weight of Evidence and ranks them by Information Value (IV)
3. Trains a logistic regression model on the WOE-transformed features to estimate PD
4. Evaluates the model with AUC-ROC, the KS statistic, and the Gini coefficient
5. Applies Loss-Given-Default (LGD) and Exposure-at-Default (EAD) assumptions
6. Computes portfolio-level Expected Loss: **EL = PD × LGD × EAD**

## Dataset

[`credit_risk_dataset.csv`](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle) — 32,581 loan records with applicant demographics, income, loan grade/amount/interest rate, and a binary default flag (`loan_status`).

| Field | Description |
|---|---|
| `person_age`, `person_income`, `person_emp_length` | Applicant demographics |
| `person_home_ownership` | RENT / OWN / MORTGAGE / OTHER |
| `loan_intent`, `loan_grade` | Purpose and internal risk grade |
| `loan_amnt`, `loan_int_rate`, `loan_percent_income` | Loan terms |
| `cb_person_default_on_file`, `cb_person_cred_hist_length` | Credit bureau history |
| `loan_status` | Target — 1 = default, 0 = fully repaid |

## Methodology

**Cleaning**: dropped known data-entry outliers (age > 100, employment length > 60 years); median/mode imputation for the remaining nulls.

**Feature engineering**: added `loan_to_income` and `income_per_year_employed` ratios on top of the raw fields.

**WOE / IV**: numeric features are binned into 8 quantile buckets, categorical features use their existing levels. WOE = ln(% good / % bad) per bin; features below an IV of 0.02 (no predictive power) are dropped before modelling.

**Model**: `sklearn.linear_model.LogisticRegression` (`class_weight='balanced'`) trained on WOE-transformed features, on a 75/25 stratified train/test split.

**LGD**: not observed in this dataset (no recovery/collections data), so it's an assumption rather than a fitted quantity — grade-adjusted, from 30% (grade A) up to 75% (grade G), based on the Basel II F-IRB convention that lower-quality exposures carry higher LGD.

**EAD**: `loan_amnt` — these are fixed-term loans, not revolving lines, so no credit-conversion-factor adjustment is needed.

## Results (held-out test set, 7,920 loans)

| Metric | Value |
|---|---|
| AUC-ROC | **0.87** |
| Gini coefficient | **0.75** |
| KS statistic | **62%** |
| Portfolio Expected Loss | **16.83% of exposure** |
| Actual observed default rate | 21.54% |

**Top predictive features by Information Value**: `loan_percent_income` (0.95), `loan_to_income` (0.93), `loan_grade` (0.89), `loan_int_rate` (0.64), `person_income` (0.45).

### A caveat worth knowing

An IV above ~0.5 is generally treated as a red flag for leakage rather than a genuine signal — and `loan_percent_income` and `loan_to_income` both land well above that here, which is most of why AUC/Gini/KS come out unusually strong for a plain logistic regression (real-world retail PD models typically sit around 0.70–0.80 AUC). This dataset is known to have unusually clean separability. In a production setting, an IV this high would warrant investigating the feature for leakage before trusting it — noted here rather than hidden.

## Tech Stack

Python · pandas · NumPy · scikit-learn · SciPy (KS statistic) · Matplotlib

## How to Run

1. Open `credit_risk_model.ipynb` in Google Colab
2. Run cells top to bottom — the first Kaggle download cell will prompt for API credentials (kaggle.com → Account → Create New API Token), or use the manual-upload fallback cell instead
3. The final cell prints an auto-generated summary with the run's actual AUC/KS/Gini/EL — no need to read them off a chart

## Possible Extensions

- Fit LGD from an actual recoveries dataset (e.g. Lending Club) instead of assuming it
- Add a credit scorecard (points-based) on top of the WOE bins for interpretability
- Compare logistic regression against a gradient-boosted model as a benchmark
- Stress-test EL under adverse macroeconomic PD/LGD scenarios (Basel III style)
