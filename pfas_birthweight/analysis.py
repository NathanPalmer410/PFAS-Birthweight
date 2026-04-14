import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pfas_birthweight import build_pfas_birthweight_panel

def plot_eda(panel):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("PFAS & Birth Weight — Exploratory Analysis", fontsize=15, fontweight="bold")
 
    states = sorted(panel["STATE"].unique())
 
    # --- 1. Scatter: PFAS vs birth weight ---
    ax1.scatter(panel["PFAS_county"], panel["avg_birth_weight"],
                alpha=0.3, s=12, color="#2980B9")
    m, b = np.polyfit(panel["PFAS_county"].dropna(), panel["avg_birth_weight"].dropna(), 1)
    x = np.linspace(panel["PFAS_county"].min(), panel["PFAS_county"].max(), 100)
    ax1.plot(x, m * x + b, color="#E74C3C", linewidth=2)
    ax1.set_title("PFAS vs Birth Weight")
    ax1.set_xlabel("PFAS Concentration (ng/L)")
    ax1.set_ylabel("Avg Birth Weight (grams)")
 
    # --- 2. Birth weight trend over time ---
    yearly = panel.groupby(["year", "STATE"])["avg_birth_weight"].mean().reset_index()
    colors = plt.cm.tab10.colors
    for i, state in enumerate(states):
        sub = yearly[yearly["STATE"] == state]
        ax2.plot(sub["year"], sub["avg_birth_weight"],
                 marker="o", markersize=4, label=state, color=colors[i])
    ax2.set_title("Avg Birth Weight Over Time")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Avg Birth Weight (grams)")
    ax2.legend(fontsize=8)
 
    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_regression(panel):
    # Year fixed effects via dummies, cluster SEs by county
    df = panel.copy().dropna(subset=["avg_birth_weight", "PFAS_county"])

    year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True)
    X = pd.concat([df["PFAS_county"], year_dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    y = df["avg_birth_weight"].astype(float)

    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["FIPS"]}
    )

    print("OLS REGRESSION: Avg Birth Weight ~ PFAS + Year FE")
    print(f"Observations:  {int(model.nobs)}")
    print(f"R-squared:     {model.rsquared:.4f}")
    print(f"Adj R-squared: {model.rsquared_adj:.4f}")
    print(f"{'Variable':<20} {'Coef':>8} {'Std Err':>10} {'p-value':>10}")

    for var in ["const", "PFAS_county"]:
        coef = model.params[var]
        se   = model.bse[var]
        pval = model.pvalues[var]
        print(f"{var:<20} {coef:>8.3f} {se:>10.3f} {pval:>10.3f}")

    return model

panel = build_pfas_birthweight_panel()
plot_eda(panel)