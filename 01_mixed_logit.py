"""
Mixed Logit Extension — Intercity Rail SP Study
================================================
Extends the master's thesis Binary Logit (V_A3 / V_B3 specification) to
a Panel Mixed Logit with random parameters on travel time and cost.

Data: ../ToSend/Biogeme_Cleaned/final_biogeme_Data_200.csv
      197 respondents × 10 SP scenarios = 1970 observations

Run:
    python 01_mixed_logit.py

Or run individual sections by copying them into a Jupyter cell.
"""

import math
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import (
    Beta, Variable,
    bioDraws, MonteCarlo, PanelLikelihoodTrajectory,
    log, exp,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "../ToSend/Biogeme_Cleaned/final_biogeme_Data_200.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {df['Response_ID'].nunique()} respondents")

# Compute time_SC (travel time in minutes) if not already present
if "time_SC" not in df.columns:
    df["time_SC"] = df["time_STR_BER"] * 60

# Biogeme needs a single numeric choice column: 1 = Train A, 2 = Train B
df["choice_numeric"] = df["Choice_ZugA"].map({1: 1, 0: 2})

# Reconstruct per-alternative delay dummies (delay_A_HL, delay_A_HS, ...) from
# the raw risk/duration columns — final_biogeme_Data_200.csv only has these as
# delta_risk_zugA/B (0.10 or 0.40) and delta_delay_zugA/B (1h or 2h), matching
# the reference notebook's `make_delay_dummies` logic:
#   HL: risk=0.40, dur=2h   HS: risk=0.40, dur=1h
#   LL: risk=0.10, dur=2h   LS: risk=0.10, dur=1h (reference, not a model term)
if "delay_A_HL" not in df.columns:
    def _isclose(x, val, tol=1e-9):
        return np.isclose(x, val, atol=tol)

    def _make_delay_dummies(frame, risk_col, dur_col, prefix):
        frame[f"{prefix}_HL"] = (_isclose(frame[risk_col], 0.40) & _isclose(frame[dur_col], 2)).astype(int)
        frame[f"{prefix}_HS"] = (_isclose(frame[risk_col], 0.40) & _isclose(frame[dur_col], 1)).astype(int)
        frame[f"{prefix}_LL"] = (_isclose(frame[risk_col], 0.10) & _isclose(frame[dur_col], 2)).astype(int)
        frame[f"{prefix}_LS"] = (_isclose(frame[risk_col], 0.10) & _isclose(frame[dur_col], 1)).astype(int)
        return frame

    df = _make_delay_dummies(df, "delta_risk_zugA", "delta_delay_zugA", "delay_A")
    df = _make_delay_dummies(df, "delta_risk_zugB", "delta_delay_zugB", "delay_B")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — BIOGEME DATABASE + PANEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

database = db.Database("intercity_rail", df)

# Binary Logit checkpoint uses a flat (non-panel) database — matches the
# thesis's original estimation, which treated the 1970 rows as independent
# observations. A SEPARATE Database instance is required because Biogeme
# requires every variable used with a panel-activated database to be
# wrapped in PanelLikelihoodTrajectory.
database_bl = db.Database("intercity_rail_flat", df)

# Panel Mixed Logit database — CRITICAL: activate panel structure so draws
# are fixed within each person across their 10 scenarios.
database_ml = db.Database("intercity_rail_panel", df)
database_ml.panel("Response_ID")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — VARIABLE DECLARATIONS (matches thesis notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────

cost_zug_A = Variable("cost_zug_A")
cost_zug_B = Variable("cost_zug_B")
time_SC    = Variable("time_SC")

# Delay dummies (LS = reference; not declared)
delay_A_HL = Variable("delay_A_HL")
delay_A_HS = Variable("delay_A_HS")
delay_A_LL = Variable("delay_A_LL")
delay_B_HS = Variable("delay_B_HS")
delay_B_LL = Variable("delay_B_LL")

# Sociodemographics
Age_Old                         = Variable("Age_Old")
Age_Young                       = Variable("Age_Young")
Income_High                     = Variable("Income_High")
Income_Low                      = Variable("Income_Low")
Gender_Female                   = Variable("Gender_Female")
Location_Large_City             = Variable("Location_Large_City")
Location_Medium_City            = Variable("Location_Medium_City")
Education_University            = Variable("Education_University")
Employment_Employed             = Variable("Employment_Employed")
Car_License_Yes                 = Variable("Car_License_Yes")
Car_Yes                         = Variable("Car_Yes")
PT_Ticket_Yes_DT                = Variable("PT_Ticket_Yes_DT")
TripPurpose_Non_Time_Critical   = Variable("TripPurpose_Non_Time_Critical")
PT_Intercity_freq_Frequent_User = Variable("PT_Intercity_freq_Frequent_User")
Coach_Frequent                  = Variable("Coach_Frequent")
PrivateCar_Frequent             = Variable("PrivateCar_Frequent")
Flight_Infrequent               = Variable("Flight_Infrequent")
On_Time_Expectancy_High         = Variable("On_Time_Expectancy_High")
Pre_Book_Time_Early_Booking     = Variable("Pre_Book_Time_Early_Booking")

Choice = Variable("choice_numeric")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BINARY LOGIT CHECKPOINT (reproduces thesis V_A3 / V_B3)
# Run this first to confirm baseline before adding random parameters.
# ─────────────────────────────────────────────────────────────────────────────

def run_binary_logit(database):
    """
    Re-estimates the Binary Logit (V_A3 / V_B3 specification).
    Use this as a checkpoint: if LL differs from thesis results, fix data first.
    """
    # Fixed parameters
    ASC_ZUGB_BL     = Beta("ASC_ZUGB", 0, None, None, 0)
    B_TIME_A_BL     = Beta("B_TIME_A", 0, None, None, 1)   # fixed to 0
    B_TIME_B_BL     = Beta("B_TIME_B", 0, None, None, 0)
    B_COST_A_BL     = Beta("B_COST_A", 0, None, None, 0)
    B_COST_B_BL     = Beta("B_COST_B", 0, None, None, 0)
    B_DELAY_HL_A_BL = Beta("B_DELAY_HL_A", 0, None, None, 0)
    B_DELAY_HS_A_BL = Beta("B_DELAY_HS_A", 0, None, None, 0)
    B_DELAY_LL_A_BL = Beta("B_DELAY_LL_A", 0, None, None, 0)
    B_DELAY_HS_B_BL = Beta("B_DELAY_HS_B", 0, None, None, 0)
    B_DELAY_LL_B_BL = Beta("B_DELAY_LL_B", 0, None, None, 0)

    # Sociodemographic betas (B-side only; A-side fixed at 0 for identification)
    B_AGE_OLD_B_BL                         = Beta("B_AGE_OLD_B", 0, None, None, 0)
    B_AGE_YOUNG_B_BL                       = Beta("B_AGE_YOUNG_B", 0, None, None, 0)
    B_INCOME_HIGH_B_BL                     = Beta("B_INCOME_HIGH_B", 0, None, None, 0)
    B_INCOME_LOW_B_BL                      = Beta("B_INCOME_LOW_B", 0, None, None, 0)
    B_GENDER_FEMALE_B_BL                   = Beta("B_GENDER_FEMALE_B", 0, None, None, 0)
    B_LOCATION_LARGE_CITY_B_BL             = Beta("B_LOCATION_LARGE_CITY_B", 0, None, None, 0)
    B_LOCATION_MEDIUM_CITY_B_BL            = Beta("B_LOCATION_MEDIUM_CITY_B", 0, None, None, 0)
    B_EDUCATION_UNIVERSITY_B_BL            = Beta("B_EDUCATION_UNIVERSITY_B", 0, None, None, 0)
    B_EMPLOYED_B_BL                        = Beta("B_EMPLOYED_B", 0, None, None, 0)
    B_CAR_LICENSE_YES_B_BL                 = Beta("B_CAR_LICENSE_YES_B", 0, None, None, 0)
    B_CAR_YES_B_BL                         = Beta("B_CAR_YES_B", 0, None, None, 0)
    B_PT_TICKET_YES_DT_B_BL               = Beta("B_PT_TICKET_YES_DT_B", 0, None, None, 0)
    B_TRIP_PURPOSE_NON_TIME_CRITICAL_B_BL  = Beta("B_TRIP_PURPOSE_NON_TIME_CRITICAL_B", 0, None, None, 0)
    B_PT_Intercity_freq_Frequent_User_B_BL = Beta("B_PT_Intercity_freq_Frequent_User_B", 0, None, None, 0)
    B_Coach_Frequent_B_BL                  = Beta("B_Coach_Frequent_B", 0, None, None, 0)
    B_PrivateCar_Frequent_B_BL             = Beta("B_PrivateCar_Frequent_B", 0, None, None, 0)
    B_Flight_Infrequent_B_BL               = Beta("B_Flight_Infrequent_B", 0, None, None, 0)
    B_On_Time_Expectancy_High_B_BL         = Beta("B_On_Time_Expectancy_High_B", 0, None, None, 0)
    B_Pre_Book_Time_Early_Booking_B_BL     = Beta("B_Pre_Book_Time_Early_Booking_B", 0, None, None, 0)

    V_A_BL = (
        B_TIME_A_BL * time_SC
        + B_COST_A_BL * cost_zug_A
        + B_DELAY_HL_A_BL * delay_A_HL
        + B_DELAY_HS_A_BL * delay_A_HS
        + B_DELAY_LL_A_BL * delay_A_LL
    )

    V_B_BL = (
        ASC_ZUGB_BL
        + B_TIME_B_BL * time_SC
        + B_COST_B_BL * cost_zug_B
        + B_DELAY_HS_B_BL * delay_B_HS
        + B_DELAY_LL_B_BL * delay_B_LL
        + B_AGE_OLD_B_BL                         * Age_Old
        + B_AGE_YOUNG_B_BL                       * Age_Young
        + B_INCOME_HIGH_B_BL                     * Income_High
        + B_INCOME_LOW_B_BL                      * Income_Low
        + B_GENDER_FEMALE_B_BL                   * Gender_Female
        + B_LOCATION_LARGE_CITY_B_BL             * Location_Large_City
        + B_LOCATION_MEDIUM_CITY_B_BL            * Location_Medium_City
        + B_EDUCATION_UNIVERSITY_B_BL            * Education_University
        + B_EMPLOYED_B_BL                        * Employment_Employed
        + B_CAR_LICENSE_YES_B_BL                 * Car_License_Yes
        + B_CAR_YES_B_BL                         * Car_Yes
        + B_PT_TICKET_YES_DT_B_BL               * PT_Ticket_Yes_DT
        + B_TRIP_PURPOSE_NON_TIME_CRITICAL_B_BL  * TripPurpose_Non_Time_Critical
        + B_PT_Intercity_freq_Frequent_User_B_BL * PT_Intercity_freq_Frequent_User
        + B_Coach_Frequent_B_BL                  * Coach_Frequent
        + B_PrivateCar_Frequent_B_BL             * PrivateCar_Frequent
        + B_Flight_Infrequent_B_BL               * Flight_Infrequent
        + B_On_Time_Expectancy_High_B_BL         * On_Time_Expectancy_High
        + B_Pre_Book_Time_Early_Booking_B_BL     * Pre_Book_Time_Early_Booking
    )

    V = {1: V_A_BL, 2: V_B_BL}
    av = {1: 1, 2: 1}

    logprob_BL = models.loglogit(V, av, Choice)
    m = bio.BIOGEME(database, logprob_BL)
    m.modelName = "binary_logit_v3"
    m.calculateNullLoglikelihood(av)

    results = m.estimate()
    print("\n=== BINARY LOGIT CHECKPOINT ===")
    print(results.shortSummary())
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PANEL MIXED LOGIT (random TIME and COST)
# ─────────────────────────────────────────────────────────────────────────────

def run_mixed_logit(database, n_draws=500):
    """
    Panel Mixed Logit with:
    - Random B_TIME_B ~ Normal(MU_TIME_B, SIGMA_TIME_B)
    - Fixed B_COST_A, B_COST_B (SIGMA_COST_A/B were not significant at 200
      draws — p=0.675 and p=0.486 — so cost sensitivity is estimated as a
      single population value, same as in the Binary Logit)
    - All delay dummies and sociodemographics kept FIXED

    Panel structure: 197 respondents × 10 scenarios each.
    Draws are held constant within each respondent across all their scenarios.
    """

    # --- Fixed parameters (same as BL) ---
    ASC_ZUGB      = Beta("ASC_ZUGB", 0, None, None, 0)
    B_TIME_A_FIXED = Beta("B_TIME_A", 0, None, None, 1)   # fixed to 0
    B_COST_A      = Beta("B_COST_A", 0, None, None, 0)
    B_COST_B      = Beta("B_COST_B", 0, None, None, 0)

    B_DELAY_HL_A = Beta("B_DELAY_HL_A", 0, None, None, 0)
    B_DELAY_HS_A = Beta("B_DELAY_HS_A", 0, None, None, 0)
    B_DELAY_LL_A = Beta("B_DELAY_LL_A", 0, None, None, 0)
    B_DELAY_HS_B = Beta("B_DELAY_HS_B", 0, None, None, 0)
    B_DELAY_LL_B = Beta("B_DELAY_LL_B", 0, None, None, 0)

    B_AGE_OLD_B                         = Beta("B_AGE_OLD_B", 0, None, None, 0)
    B_AGE_YOUNG_B                       = Beta("B_AGE_YOUNG_B", 0, None, None, 0)
    B_INCOME_HIGH_B                     = Beta("B_INCOME_HIGH_B", 0, None, None, 0)
    B_INCOME_LOW_B                      = Beta("B_INCOME_LOW_B", 0, None, None, 0)
    B_GENDER_FEMALE_B                   = Beta("B_GENDER_FEMALE_B", 0, None, None, 0)
    B_LOCATION_LARGE_CITY_B             = Beta("B_LOCATION_LARGE_CITY_B", 0, None, None, 0)
    B_LOCATION_MEDIUM_CITY_B            = Beta("B_LOCATION_MEDIUM_CITY_B", 0, None, None, 0)
    B_EDUCATION_UNIVERSITY_B            = Beta("B_EDUCATION_UNIVERSITY_B", 0, None, None, 0)
    B_EMPLOYED_B                        = Beta("B_EMPLOYED_B", 0, None, None, 0)
    B_CAR_LICENSE_YES_B                 = Beta("B_CAR_LICENSE_YES_B", 0, None, None, 0)
    B_CAR_YES_B                         = Beta("B_CAR_YES_B", 0, None, None, 0)
    B_PT_TICKET_YES_DT_B               = Beta("B_PT_TICKET_YES_DT_B", 0, None, None, 0)
    B_TRIP_PURPOSE_NON_TIME_CRITICAL_B  = Beta("B_TRIP_PURPOSE_NON_TIME_CRITICAL_B", 0, None, None, 0)
    B_PT_Intercity_freq_Frequent_User_B = Beta("B_PT_Intercity_freq_Frequent_User_B", 0, None, None, 0)
    B_Coach_Frequent_B                  = Beta("B_Coach_Frequent_B", 0, None, None, 0)
    B_PrivateCar_Frequent_B             = Beta("B_PrivateCar_Frequent_B", 0, None, None, 0)
    B_Flight_Infrequent_B               = Beta("B_Flight_Infrequent_B", 0, None, None, 0)
    B_On_Time_Expectancy_High_B         = Beta("B_On_Time_Expectancy_High_B", 0, None, None, 0)
    B_Pre_Book_Time_Early_Booking_B     = Beta("B_Pre_Book_Time_Early_Booking_B", 0, None, None, 0)

    # --- Random parameter definition (mean + std dev) ---
    MU_TIME_B    = Beta("MU_TIME_B",    0,    None, None, 0)
    SIGMA_TIME_B = Beta("SIGMA_TIME_B", 0.01, None, None, 0)

    # --- Random draws ---
    # NORMAL_MLHS_ANTI: Modified Latin Hypercube Sampling with antithetics
    B_TIME_B_RND = MU_TIME_B + SIGMA_TIME_B * bioDraws("B_TIME_B_RND", "NORMAL_MLHS_ANTI")

    # --- Utilities ---
    V_A_ML = (
        B_TIME_A_FIXED * time_SC
        + B_COST_A * cost_zug_A
        + B_DELAY_HL_A * delay_A_HL
        + B_DELAY_HS_A * delay_A_HS
        + B_DELAY_LL_A * delay_A_LL
    )

    V_B_ML = (
        ASC_ZUGB
        + B_TIME_B_RND * time_SC
        + B_COST_B * cost_zug_B
        + B_DELAY_HS_B * delay_B_HS
        + B_DELAY_LL_B * delay_B_LL
        + B_AGE_OLD_B                         * Age_Old
        + B_AGE_YOUNG_B                       * Age_Young
        + B_INCOME_HIGH_B                     * Income_High
        + B_INCOME_LOW_B                      * Income_Low
        + B_GENDER_FEMALE_B                   * Gender_Female
        + B_LOCATION_LARGE_CITY_B             * Location_Large_City
        + B_LOCATION_MEDIUM_CITY_B            * Location_Medium_City
        + B_EDUCATION_UNIVERSITY_B            * Education_University
        + B_EMPLOYED_B                        * Employment_Employed
        + B_CAR_LICENSE_YES_B                 * Car_License_Yes
        + B_CAR_YES_B                         * Car_Yes
        + B_PT_TICKET_YES_DT_B               * PT_Ticket_Yes_DT
        + B_TRIP_PURPOSE_NON_TIME_CRITICAL_B  * TripPurpose_Non_Time_Critical
        + B_PT_Intercity_freq_Frequent_User_B * PT_Intercity_freq_Frequent_User
        + B_Coach_Frequent_B                  * Coach_Frequent
        + B_PrivateCar_Frequent_B             * PrivateCar_Frequent
        + B_Flight_Infrequent_B               * Flight_Infrequent
        + B_On_Time_Expectancy_High_B         * On_Time_Expectancy_High
        + B_Pre_Book_Time_Early_Booking_B     * Pre_Book_Time_Early_Booking
    )

    V = {1: V_A_ML, 2: V_B_ML}
    av = {1: 1, 2: 1}

    # Logit prob for one observation (conditional on draw)
    prob_chosen = models.logit(V, av, Choice)

    # Product over all t observations for the same person (panel likelihood)
    panel_prob = PanelLikelihoodTrajectory(prob_chosen)

    # Simulate the integral (average over R draws)
    simulated_prob = MonteCarlo(panel_prob)

    logprob_ML = log(simulated_prob)

    m = bio.BIOGEME(database, logprob_ML)
    m.modelName = "panel_mixed_logit"
    m.number_of_draws = n_draws
    m.calculateNullLoglikelihood(av)

    results = m.estimate()
    print("\n=== PANEL MIXED LOGIT ===")
    print(results.shortSummary())
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results_BL, results_ML):
    """Prints a model comparison table and LRT between BL and ML."""

    def model_stats(res, label):
        ll   = res.data.logLike
        k    = len(res.data.betas)   # betas list already contains only estimated (free) parameters
        n    = 1970   # number of observations
        ll0  = n * math.log(0.5)
        rho2 = 1 - ll / ll0
        rho2a = 1 - (ll - k) / ll0
        aic  = -2 * ll + 2 * k
        bic  = -2 * ll + k * math.log(n)
        return {"Model": label, "K": k, "LL*": round(ll, 2),
                "rho2": round(rho2, 4), "rho2_adj": round(rho2a, 4),
                "AIC": round(aic, 2), "BIC": round(bic, 2)}

    stats_BL = model_stats(results_BL, "Binary Logit")
    stats_ML = model_stats(results_ML, "Mixed Logit (Panel)")

    df_cmp = pd.DataFrame([stats_BL, stats_ML])
    print("\n=== MODEL COMPARISON ===")
    print(df_cmp.to_string(index=False))

    # Likelihood Ratio Test (BL is nested in ML when SIGMA params = 0)
    ll_BL = results_BL.data.logLike
    ll_ML = results_ML.data.logLike
    K_extra = 1   # SIGMA_TIME_B (only random parameter; costs are fixed)
    lrt = -2 * (ll_BL - ll_ML)
    p = 1 - stats.chi2.cdf(lrt, df=K_extra)
    print(f"\nLRT (BL vs ML): χ²({K_extra}) = {lrt:.2f}, p = {p:.4f}")
    print("→ ML significantly better than BL" if p < 0.05 else "→ ML NOT significantly better")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — WTP COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_wtp(results_ML, n_sim=10_000):
    """
    Derives WTP distributions from Mixed Logit results.
    B_COST_B is fixed (population-constant; SIGMA_COST_B was not significant
    at the 200-draw test, p=0.486), so only B_TIME_B varies across the
    simulated population — VTTS heterogeneity comes entirely from the
    random time taste.
    Returns a summary DataFrame and generates a VTTS histogram.
    """
    params = results_ML.getEstimatedParameters()

    mu_time    = params.loc["MU_TIME_B",    "Value"]
    sigma_time = params.loc["SIGMA_TIME_B", "Value"]
    b_cost_fixed_A = params.loc["B_COST_A", "Value"]
    b_cost_fixed_B = params.loc["B_COST_B", "Value"]

    np.random.seed(42)
    b_time = np.random.normal(mu_time, abs(sigma_time), n_sim)
    b_cost = np.full(n_sim, b_cost_fixed_B)

    # Filter extreme draws (numerical stability)
    mask = (b_cost != 0) & (np.abs(b_cost) > 1e-6)
    # Both B_TIME_B and B_COST_B carry their natural (negative) utility
    # signs, so their ratio is already the positive €/hr a traveller would
    # pay to save time — no extra leading minus (that would flip the sign).
    vtts = b_time[mask] / b_cost[mask] * 60   # €/hour

    # Clip outliers for display (|VTTS| < 200 €/hr)
    vtts_clipped = vtts[np.abs(vtts) < 200]

    summary = {
        "Attribute": ["VTTS (€/hr)"],
        "Mean":  [round(np.mean(vtts_clipped), 2)],
        "Median":[round(np.median(vtts_clipped), 2)],
        "StdDev":[round(np.std(vtts_clipped), 2)],
        "P10":   [round(np.percentile(vtts_clipped, 10), 2)],
        "P90":   [round(np.percentile(vtts_clipped, 90), 2)],
        "pct_negative": [round((vtts_clipped < 0).mean() * 100, 1)],
    }
    df_wtp = pd.DataFrame(summary)
    print("\n=== WTP DISTRIBUTIONS ===")
    print(df_wtp.to_string(index=False))

    # Delay WTP (from fixed parameters). Uses B_COST_A since B_DELAY_HS_A is
    # a Train-A attribute — dividing by the wrong alternative's cost
    # coefficient (B_COST_B) would mix two different marginal utilities of
    # money.
    try:
        b_delay_hs = params.loc["B_DELAY_HS_A", "Value"]
        delay_hs_p = params.loc["B_DELAY_HS_A", "Rob. p-value"] if "Rob. p-value" in params.columns else None
        if b_cost_fixed_A != 0:
            wtp_hs = -b_delay_hs / b_cost_fixed_A
            sig_note = f" (NOT significant, p={delay_hs_p:.3f})" if delay_hs_p is not None and delay_hs_p > 0.05 else ""
            print(f"\nWTP to avoid HS delay vs LS (point estimate): {wtp_hs:.2f} €{sig_note}")
    except KeyError:
        pass

    # Plot VTTS distribution
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(vtts_clipped, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(vtts_clipped), color="red", ls="--",
               label=f"Mean = {np.mean(vtts_clipped):.1f} €/hr")
    ax.axvline(np.median(vtts_clipped), color="orange", ls="--",
               label=f"Median = {np.median(vtts_clipped):.1f} €/hr")
    ax.set_xlabel("Value of Travel Time Savings (€/hour)")
    ax.set_ylabel("Frequency")
    ax.set_title("WTP Distribution — Mixed Logit Panel (Intercity Rail)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("wtp_vtts_intercity.pdf", dpi=200)
    plt.show()
    print("Saved: wtp_vtts_intercity.pdf")

    return df_wtp


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 1: Binary Logit checkpoint")
    print("=" * 60)
    results_BL = run_binary_logit(database_bl)

    print("\n" + "=" * 60)
    print("STEP 2: Panel Mixed Logit — random B_TIME_B only, fixed B_COST_A/B (1000 draws)")
    print("  This takes ~10–20 minutes.")
    print("=" * 60)
    results_ML = run_mixed_logit(database_ml, n_draws=1000)

    print("\n" + "=" * 60)
    print("STEP 3: Model comparison")
    print("=" * 60)
    compare_models(results_BL, results_ML)

    print("\n" + "=" * 60)
    print("STEP 4: WTP analysis")
    print("=" * 60)
    compute_wtp(results_ML)

    print("\nDone. Check biogeme_output/ folder for estimation reports.")
