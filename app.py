import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from pfas_birthweight import build_extended_panel

st.set_page_config(page_title="PFAS & Birth Weight Explorer", layout="wide")

st.title("PFAS & Birth Weight Explorer")
st.markdown("Explore the relationship between PFAS contamination in drinking water and birth outcomes across counties.")

@st.cache_data
def load_data():
    return build_extended_panel()

panel = load_data()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("Controls")

# View toggle
view_mode = st.sidebar.radio(
    "View mode:",
    options=["Single State", "All States"]
)

# Outcome selector
outcome_options = {
    "Avg Birth Weight (g)": "avg_birth_weight",
    "Low Birth Weight Rate": "lbw_rate",
    "Preterm Rate": "preterm_rate"
}
selected_outcome_label = st.sidebar.selectbox(
    "Select outcome:",
    options=list(outcome_options.keys())
)
selected_outcome = outcome_options[selected_outcome_label]

available_years = sorted(panel["year"].unique())
selected_year = st.sidebar.selectbox(
    "Select a year:",
    options=available_years,
    index=len(available_years) - 1
)

if view_mode == "Single State":
    selected_state = st.sidebar.selectbox(
        "Select a state:",
        options=sorted(panel["STATE"].unique())
    )
else:
    selected_state = None

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Time Trend", "Regression Results"])

# =====================================================
# TAB 1 — SCATTER
# =====================================================
with tab1:
    if view_mode == "Single State":
        filtered = panel[
            (panel["STATE"] == selected_state) &
            (panel["year"] == selected_year)
        ].copy()
    else:
        filtered = panel[panel["year"] == selected_year].copy()

    if view_mode == "Single State":
        st.sidebar.info(f"{len(filtered)} counties shown")

    if filtered.empty:
        st.warning("No data available for this selection.")
    else:
        title = (
            f"{selected_state} Counties — PFAS vs {selected_outcome_label} ({int(selected_year)})"
            if view_mode == "Single State"
            else f"All States — PFAS vs {selected_outcome_label} ({int(selected_year)})"
        )

        fig = px.scatter(
            filtered,
            x="PFAS_county",
            y=selected_outcome,
            hover_name="COUNTY_SERVED",
            hover_data={
                "PFAS_county": ":.2f",
                selected_outcome: ":.4f",
                "births": True,
                "STATE": True,
                "COUNTY_SERVED": False
            },
            size="births",
            size_max=30,
            color="STATE" if view_mode == "All States" else selected_outcome,
            color_continuous_scale="RdYlGn" if view_mode == "Single State" else None,
            labels={
                "PFAS_county": "PFAS Concentration (ng/L)",
                selected_outcome: selected_outcome_label,
                "births": "Number of Births",
                "STATE": "State"
            },
            title=title,
            height=600
        )

        # Add trend line
        valid = filtered.dropna(subset=["PFAS_county", selected_outcome])
        if len(valid) > 1:
            m, b = np.polyfit(valid["PFAS_county"], valid[selected_outcome], 1)
            x_range = np.linspace(valid["PFAS_county"].min(), valid["PFAS_county"].max(), 100)
            fig.add_scatter(
                x=x_range,
                y=m * x_range + b,
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Trend",
                showlegend=True
            )

        fig.update_layout(
            hovermode="closest",
            plot_bgcolor="rgba(240, 240, 240, 0.5)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics
        label = selected_state if view_mode == "Single State" else "All States"
        st.subheader(f"Summary — {label} ({int(selected_year)})")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Counties", len(filtered))
        col2.metric("Avg PFAS (ng/L)", f"{filtered['PFAS_county'].mean():.2f}")
        col3.metric(selected_outcome_label, f"{filtered[selected_outcome].mean():.4f}")
        col4.metric("Total Births", f"{filtered['births'].sum():,.0f}")

        # Data table
        with st.expander("View county data"):
            st.dataframe(
                filtered[["STATE", "COUNTY_SERVED", "PFAS_county",
                           selected_outcome, "births"]]
                .rename(columns={
                    "STATE": "State",
                    "COUNTY_SERVED": "County",
                    "PFAS_county": "PFAS (ng/L)",
                    selected_outcome: selected_outcome_label,
                    "births": "Births"
                })
                .sort_values("PFAS (ng/L)", ascending=False)
                .reset_index(drop=True),
                use_container_width=True
            )

# =====================================================
# TAB 2 — TIME TREND
# =====================================================
with tab2:
    if view_mode == "Single State":
        trend_data = panel[panel["STATE"] == selected_state].copy()
        trend_data = trend_data.groupby("year")[selected_outcome].mean().reset_index()
        fig2 = px.line(
            trend_data,
            x="year",
            y=selected_outcome,
            markers=True,
            labels={
                "year": "Year",
                selected_outcome: selected_outcome_label
            },
            title=f"{selected_state} — {selected_outcome_label} Over Time",
            height=500
        )
    else:
        trend_data = panel.groupby(["year", "STATE"])[selected_outcome].mean().reset_index()
        fig2 = px.line(
            trend_data,
            x="year",
            y=selected_outcome,
            color="STATE",
            markers=True,
            labels={
                "year": "Year",
                selected_outcome: selected_outcome_label,
                "STATE": "State"
            },
            title=f"All States — {selected_outcome_label} Over Time",
            height=500
        )

    fig2.update_layout(plot_bgcolor="rgba(240, 240, 240, 0.5)")
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================
# TAB 3 — REGRESSION RESULTS
# =====================================================
with tab3:
    st.subheader("OLS Regression Results")
    st.markdown("PFAS concentration regressed on each birth outcome with year fixed effects and standard errors clustered by county.")

    @st.cache_data
    def run_all_regressions(panel_hash):
        outcomes = [
            ("avg_birth_weight", "Avg Birth Weight (g)"),
            ("lbw_rate",         "Low Birth Weight Rate"),
            ("preterm_rate",     "Preterm Rate"),
        ]
        rows = []
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
            coef  = model.params["PFAS_county"]
            se    = model.bse["PFAS_county"]
            pval  = model.pvalues["PFAS_county"]
            ci    = model.conf_int().loc["PFAS_county"]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            rows.append({
                "Outcome": label,
                "PFAS Coef": f"{coef:.4f}{stars}",
                "Std Err": f"{se:.4f}",
                "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                "p-value": f"{pval:.3f}",
                "R²": f"{model.rsquared:.4f}",
                "N": int(model.nobs)
            })
        return pd.DataFrame(rows)

    results_df = run_all_regressions(len(panel))
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.caption("*** p<0.01, ** p<0.05, * p<0.1. Year fixed effects included in all models.")