# Intercity Rail Choice Modelling — Stated Preference Survey (Munich)

Analysis of mode-choice behaviour for intercity (100 km+) trips, using a stated-preference (SP) survey distributed via LimeSurvey and binary logit models estimated in Biogeme. Built as part of a Master's research project at the Technical University of Munich (TUM).

## Problem

Travellers choosing between intercity rail and competing modes (car, coach, flight) trade off travel time, cost and reliability (delay risk). This project designs an SP experiment around that trade-off for two real Munich routes (Munich–Stuttgart, Munich–Berlin), collects ~200 responses, and estimates discrete choice models to quantify how strongly each attribute — and each socio-demographic segment — drives the choice.

## Data

`data/200_Complete_final_data2process.csv` is the raw LimeSurvey export (one row per respondent). It is fully anonymized: no names, emails, or exact addresses — only bucketed demographics (age range, income range, city-size category, education/employment category) and the SP choice responses themselves. `data/final_biogeme_Data_200.csv` is the cleaned, one-hot-encoded dataset used directly for estimation.

## Notebooks (run in this order)

1. **`Final_Processing_clean_time.ipynb`** — cleans the raw LimeSurvey export, parses the SP scenario tokens (e.g. delay-risk/level combinations) embedded in column headers, and reshapes the wide survey format into a long choice dataset.
2. **`Sociodemographics.ipynb`** — exploratory analysis of respondent demographics and their relationship to mode-use frequency.
3. **`Biogeme_Cleaned_Final_5Nov.ipynb`** — feature engineering (dummy variables, collinearity/VIF checks), then estimation of binary logit models in Biogeme, with robust standard errors and exported LaTeX coefficient tables.
4. **`SP_Choice_Analysis.ipynb`** — diagnostics and visualization of the SP choice patterns (choice shares, heatmaps by attribute) and cohort comparisons.

## Method

- Time, cost and reliability sensitivities estimated via binary logit (Biogeme), separately by socio-demographic segment.
- Variance-inflation-factor checks to screen correlated socio-demographic predictors before model estimation.
- Cohort and confusion-matrix diagnostics to assess model fit beyond raw log-likelihood.

## Running locally

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

These notebooks were developed locally with absolute file paths pointing at the original working machine. Before re-running, update the path variables at the top of each notebook (`CSV_PATH`, `BASE_DIR`, `os.chdir(...)`, etc.) to point at this repo's `data/` folder. Intermediate output paths referenced later in some notebooks (e.g. exported `.tex` tables, regenerated CSVs under a local `Data/` working folder) are not tracked here — re-running a notebook from the top regenerates them locally.

## Tools

LimeSurvey, MS Excel · Python (pandas, NumPy, SciPy, matplotlib, seaborn, statsmodels) · Biogeme (binary logit, robust SE) · LaTeX/Overleaf
