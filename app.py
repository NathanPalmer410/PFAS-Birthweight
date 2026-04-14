import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pfas_birthweight import build_pfas_birthweight_panel

st.set_page_config(page_title="PFAS & Birth Weight Explorer", layout="wide")

st.title("PFAS & Birth Weight Explorer")
st.markdown("Explore the relationship between PFAS contamination in drinking water and average birth weight across counties.")

@st.cache_data
def load_data():
    return build_pfas_birthweight_panel()

panel = load_data()

# =====================================================
# SIDEBAR CONTROLS
# =====================================================
st.sidebar.header("Controls")

selected_state = st.sidebar.selectbox(
    "Select a state:",
    options=sorted(panel["STATE"].unique())
)

available_years = sorted(panel["year"].unique())
selected_year = st.sidebar.selectbox(
    "Select a year:",
    options=available_years,
    index=len(available_years) - 1
)

# =====================================================
# FILTER DATA
# =====================================================
filtered = panel[
    (panel["STATE"] == selected_state) &
    (panel["year"] == selected_year)
].copy()

st.sidebar.info(f"{len(filtered)} counties shown")

# =====================================================
# SCATTERPLOT
# =====================================================
if filtered.empty:
    st.warning("No data available for this state and year.")
else:
    fig = px.scatter(
        filtered,
        x="PFAS_county",
        y="avg_birth_weight",
        hover_name="COUNTY_SERVED",
        hover_data={
            "PFAS_county": ":.2f",
            "avg_birth_weight": ":.1f",
            "births": True,
            "COUNTY_SERVED": False
        },
        size="births",
        size_max=30,
        color="avg_birth_weight",
        color_continuous_scale="RdYlGn",
        labels={
            "PFAS_county": "PFAS Concentration (ng/L)",
            "avg_birth_weight": "Avg Birth Weight (grams)",
            "births": "Number of Births"
        },
        title=f"{selected_state} Counties — PFAS vs Birth Weight ({selected_year})",
        height=600
    )

    # Add trend line
    if len(filtered) > 1:
        m, b = np.polyfit(filtered["PFAS_county"], filtered["avg_birth_weight"], 1)
        x_range = np.linspace(filtered["PFAS_county"].min(), filtered["PFAS_county"].max(), 100)
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

    # =====================================================
    # SUMMARY STATS
    # =====================================================
    st.subheader(f"Summary — {selected_state} ({selected_year})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Counties", len(filtered))
    col2.metric("Avg PFAS (ng/L)", f"{filtered['PFAS_county'].mean():.2f}")
    col3.metric("Avg Birth Weight (g)", f"{filtered['avg_birth_weight'].mean():.1f}")
    col4.metric("Total Births", f"{filtered['births'].sum():,.0f}")

    # =====================================================
    # DATA TABLE
    # =====================================================
    with st.expander("View county data"):
        st.dataframe(
            filtered[["COUNTY_SERVED", "PFAS_county", "avg_birth_weight", "births"]]
            .rename(columns={
                "COUNTY_SERVED": "County",
                "PFAS_county": "PFAS (ng/L)",
                "avg_birth_weight": "Avg Birth Weight (g)",
                "births": "Births"
            })
            .sort_values("PFAS (ng/L)", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )