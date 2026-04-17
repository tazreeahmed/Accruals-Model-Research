# Accruals-Based Earnings Quality Analysis — Modified Jones Model

Replication and extension of the Modified Jones Model (Dechow et al., 1995) using Compustat firm-level panel data (4,953 firm-year observations). Estimates discretionary accruals as a proxy for earnings management and tests model robustness through a series of augmented specifications and manipulation checks.

> **Data note:** Raw data sourced from Compustat Capital IQ. Data files are not included in this repository due to licensing restrictions.

---

## Dataset

- Source: Compustat Capital IQ annual firm-level financial data
- 4,953 firm-year observations (identified by gvkey)
- Key variables: Net Income, Net Cash Flow from Operations, Total Revenue, Total Trade Receivables, Gross Property Plant and Equipment, Total Assets, Accumulated Depreciation, Total Current Assets, Total Current Liabilities, Total Long-Term Debt, R&D Expense, SG&A Expense

---

## Project Structure

| Notebook | Description |
|---|---|
| `Jones_Dataset_Evaluation.ipynb` | Initial inspection: shape, descriptive statistics, column inventory, null check |
| `Jones_Dataset_Cleaning.ipynb` | Missing data handling, receivables imputation, R&D pre-screening, asset filtering |
| `Jones_ML.ipynb` | Modified Jones Model estimation and robustness tests |
| `Jones_ML_Predictions.ipynb` | Firm-level earnings management estimation using discretionary accruals |

---

## Data Cleaning

- Dropped observations with missing Net Income, operating cash flow, trade receivables, PPE, R&D expense, and long-term debt
- **Receivables imputation:** For firms where Total Trade Receivables was missing but Total Receivables was available, imputed using Total Receivables only where firm-level percentage difference between the two was under 10% — preserving data integrity while maximizing usable observations
- Pre-screened R&D-intensive pre-revenue firms (negative net income, zero revenue, R&D intensity below 75th percentile) to assess overlap with PPE-missing observations
- Excluded zero-asset firms
- Exported cleaned dataset as `FinRecord.csv`

---

## Model Specifications

All models estimated with **HC3 heteroskedasticity-robust standard errors**. All variables scaled by lagged total assets (A_{t-1}) following standard accounting research methodology. Variables winsorized at the 1st and 99th percentiles.

**Dependent variable:** Total accruals scaled by lagged assets
`Accruals_scaled = (Net Income − Operating Cash Flow) / Lag_Assets`

**Core regressors:**
- `Cash_Rev_Growth = (ΔRevenue − ΔTrade Receivables) / Lag_Assets`
- `PPE_scaled = Gross PPE / Lag_Assets`

### 1. Baseline Modified Jones Model
```
Accruals_scaled ~ Cash_Rev_Growth + PPE_scaled
```

### 2. Performance-Matched Jones (Kothari et al.)
```
Accruals_scaled ~ Cash_Rev_Growth + PPE_scaled + ROA
```

### 3. R&D-Augmented Jones
```
Accruals_scaled ~ Cash_Rev_Growth + PPE_scaled + RD_scaled
```

### 4. SG&A-Augmented Jones
```
Accruals_scaled ~ Cash_Rev_Growth + PPE_scaled + SGA_scaled
```
Auxiliary test: Jones residuals regressed on SG&A intensity to assess whether selling and administrative expenditure explains unexplained accrual variation.

### 5. Working Capital vs Long-Term Accruals Decomposition
```
WCA_scaled ~ Cash_Rev_Growth + PPE_scaled
LTA_scaled ~ Cash_Rev_Growth + PPE_scaled
```

### 6. Receivables Manipulation Check
```
Jones_residual ~ Delta_AR_scaled
```

---

## Results

### Baseline Modified Jones Model
N = 4,426 | R² = 0.336 | HC3 robust standard errors

- **PPE_scaled:** coef = −17.65, p < 0.001 — highly significant negative relationship between capital intensity and accruals, consistent with depreciation-driven accrual reduction
- **Cash_Rev_Growth:** coef = 2.28, p = 0.662 — not statistically significant in the baseline specification
- The baseline model explains 33.6% of accrual variation, consistent with prior literature

### Performance-Matched Jones (Kothari et al.)
N = 4,426 | R² = 0.942 | HC3 robust standard errors

- **ROA:** coef = 0.548, p < 0.001 — dominant predictor; adding ROA as a performance control increases R² from 0.336 to 0.942, indicating accruals are strongly driven by firm performance
- PPE_scaled and Cash_Rev_Growth become statistically insignificant once ROA is included, suggesting the baseline model partially captures performance effects rather than pure earnings management
- Consistent with Kothari et al. (2005) — performance-matching is critical for valid earnings management inference

### R&D-Augmented Jones
N = 4,426 | R² = 0.563 | HC3 robust standard errors

- **RD_scaled:** coef = −3.05, p < 0.001 — R&D intensity is a significant negative predictor of accruals, confirming intangible-intensive firms exhibit systematically lower accruals
- **PPE_scaled:** coef = −7.60, p < 0.001 — remains significant with reduced magnitude relative to baseline, indicating R&D absorbs part of what the baseline PPE coefficient captures
- R² improves from 0.336 to 0.563, suggesting R&D omission in the standard Jones specification introduces upward bias in discretionary accrual estimates for R&D-intensive firms

### SG&A-Augmented Jones
N = 2,760 | R² = 0.797 | HC3 robust standard errors

- **SGA_scaled:** coef = −0.759, p < 0.001 — SG&A intensity is a highly significant negative predictor of accruals
- Jones residuals regressed on SG&A intensity yield R² = 0.641 (coef = −0.216, p < 0.001) — confirming that 64.1% of unexplained accrual variation in the baseline model is attributable to SG&A, indicating systematic omitted variable bias in the standard Jones specification

### Working Capital vs Long-Term Accruals Decomposition

- **Long-Term Accruals (LTA_scaled):** R² = 0.937 — PPE_scaled alone drives nearly all long-term accrual variation (coef = −0.666, p < 0.001), confirming depreciation is the primary long-term accrual component
- **Working Capital Accruals (WCA_scaled):** R² = 0.021 — Jones regressors explain very little working capital accrual variation; Cash_Rev_Growth is significant (coef = −15.15, p < 0.01) but model fit is low, consistent with working capital accruals being the primary channel for discretionary manipulation

### Receivables Manipulation Check

- Correlation between Jones residuals and scaled ΔAR = −0.017
- Regression of Jones residuals on Delta_AR_scaled: coef = −47.43, p = 0.689 — not statistically significant
- Jones residuals are not systematically correlated with changes in accounts receivable, suggesting receivables-based revenue manipulation is not a dominant driver of residual accruals in this sample

### Firm-Level Earnings Management Estimation
N = 4,573 | R² = 0.008 (unscaled, pre-winsorization baseline on `Accruals EM.csv`)

- OLS residuals serve as firm-year discretionary accrual estimates for named Compustat firms
- Low R² confirms that after removing non-discretionary components, residual variation represents firm-specific discretionary behavior unexplained by economic fundamentals

---

## Tools & Libraries

Python, pandas, numpy, statsmodels
