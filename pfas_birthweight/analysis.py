import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pfas_birthweight import build_pfas_birthweight_panel, build_extended_panel

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


def plot_extended_eda(panel):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PFAS & Birth Outcomes — Extended Exploratory Analysis",
                 fontsize=15, fontweight="bold")

    states = sorted(panel["STATE"].unique())
    colors = plt.cm.tab10.colors

    outcomes = [
        ("avg_birth_weight", "Avg Birth Weight (grams)", "PFAS vs Birth Weight"),
        ("lbw_rate",         "Low Birth Weight Rate",    "PFAS vs LBW Rate"),
        ("preterm_rate",     "Preterm Rate",             "PFAS vs Preterm Rate"),
    ]

    # --- Row 1: scatter plots ---
    for col, (outcome, ylabel, title) in enumerate(outcomes):
        ax = axes[0, col]
        valid = panel.dropna(subset=["PFAS_county", outcome])
        ax.scatter(valid["PFAS_county"], valid[outcome],
                   alpha=0.3, s=12, color="#2980B9")
        m, b = np.polyfit(valid["PFAS_county"], valid[outcome], 1)
        x = np.linspace(valid["PFAS_county"].min(), valid["PFAS_county"].max(), 100)
        ax.plot(x, m * x + b, color="#E74C3C", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("PFAS Concentration (ng/L)")
        ax.set_ylabel(ylabel)

    # --- Row 2: time trends ---
    for col, (outcome, ylabel, _) in enumerate(outcomes):
        ax = axes[1, col]
        yearly = panel.groupby(["year", "STATE"])[outcome].mean().reset_index()
        for i, state in enumerate(states):
            sub = yearly[yearly["STATE"] == state]
            ax.plot(sub["year"], sub[outcome],
                    marker="o", markersize=4, label=state, color=colors[i])
        ax.set_title(f"{ylabel} Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig("eda_extended_plots.png", dpi=150, bbox_inches="tight")
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




def run_extended_regression(panel):
    outcomes = [
        ("avg_birth_weight", "Avg Birth Weight"),
        ("lbw_rate",         "Low Birth Weight Rate"),
        ("preterm_rate",     "Preterm Rate"),
    ]

    models = {}

    for outcome, label in outcomes:
        df = panel.copy().dropna(subset=[outcome, "PFAS_county"])

        year_dummies = pd.get_dummies(df["year"], prefix="year", drop_first=True)
        X = pd.concat([df["PFAS_county"], year_dummies], axis=1).astype(float)
        X = sm.add_constant(X)
        y = df[outcome].astype(float)

        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["FIPS"]}
        )

        print(f"\nOLS REGRESSION: {label} ~ PFAS + Year FE")
        print(f"Observations:  {int(model.nobs)}")
        print(f"R-squared:     {model.rsquared:.4f}")
        print(f"Adj R-squared: {model.rsquared_adj:.4f}")
        print(f"{'Variable':<20} {'Coef':>8} {'Std Err':>10} {'p-value':>10}")

        for var in ["const", "PFAS_county"]:
            coef = model.params[var]
            se   = model.bse[var]
            pval = model.pvalues[var]
            print(f"{var:<20} {coef:>8.3f} {se:>10.3f} {pval:>10.3f}")

        models[outcome] = model

    return models

def plot_compound_detection_rates(ucmr5_df):
    # Exclude lithium - not a PFAS
    pfas_df = ucmr5_df[ucmr5_df["Contaminant"] != "lithium"].copy()

    # Detection = result above MRL
    pfas_df["detected"] = pfas_df["result"] > pfas_df["MRL"]

    detection_rates = (
        pfas_df.groupby("Contaminant")["detected"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    detection_rates["detected"] *= 100  # convert to percentage

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(detection_rates["Contaminant"], detection_rates["detected"],
                  color="#2980B9", edgecolor="white")
    ax.set_title("PFAS Detection Rates by Compound (UCMR5, 2023–2025)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Detection Rate (%)")
    ax.tick_params(axis="x", rotation=45)

    # Label each bar
    for bar, val in zip(bars, detection_rates["detected"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig("pfas_detection_rates.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_compound_concentrations(ucmr5_df):
    # Exclude lithium and only show detected samples
    pfas_df = ucmr5_df[
        (ucmr5_df["Contaminant"] != "lithium") &
        (ucmr5_df["result"] > ucmr5_df["MRL"])
    ].copy()

    # Order compounds by median concentration
    order = (
        pfas_df.groupby("Contaminant")["result"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    data = [pfas_df[pfas_df["Contaminant"] == c]["result"].values for c in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)

    for patch in bp["boxes"]:
        patch.set_facecolor("#2980B9")
        patch.set_alpha(0.6)

    ax.set_title("PFAS Concentration Distribution by Compound (detected samples only)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Compound")
    ax.set_ylabel("Concentration (ng/L)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig("pfas_concentrations.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_pfas_choropleth(panel):
    try:
        import geopandas as gpd
    except ImportError:
        print("geopandas is required for choropleth maps")
        return

    # Download US counties shapefile from Census
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    import urllib.request, json
    with urllib.request.urlopen(url) as r:
        counties = json.load(r)

    import plotly.express as px

    latest = panel[panel["year"] == panel["year"].max()].copy()
    latest["FIPS"] = latest["FIPS"].astype(str).str.zfill(5)

    fig = px.choropleth(
        latest,
        geojson=counties,
        locations="FIPS",
        color="PFAS_county",
        color_continuous_scale="Reds",
        scope="usa",
        labels={"PFAS_county": "PFAS (ng/L)"},
        title=f"County-Level PFAS Concentration ({int(panel['year'].max())})"
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_image("pfas_choropleth.png", scale=2)
    fig.show()

if __name__ == "__main__":
    print("Building extended panel...")
    panel = build_extended_panel()

    print("\nRunning EDA plots...")
    plot_eda(panel)
    plot_extended_eda(panel)

    print("\nRunning original regression...")
    run_regression(panel)

    print("\nRunning extended regressions...")
    run_extended_regression(panel)

    print("\nRunning UCMR5 compound plots...")
    from pfas_birthweight.pipeline import load_ucmr5
    ucmr5_df = load_ucmr5()
    plot_compound_detection_rates(ucmr5_df)
    plot_compound_concentrations(ucmr5_df)

    print("\nGenerating choropleth...")
    plot_pfas_choropleth(panel)