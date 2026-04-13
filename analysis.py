import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pfas_birthweight import build_pfas_birthweight_panel

panel = build_pfas_birthweight_panel()

# Download Census county shapefile (no file needed, pulls from Census directly)
counties = gpd.read_file(
    "https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_county_20m.zip"
)

counties["FIPS"] = counties["STATEFP"] + counties["COUNTYFP"]

# Filter to your 6 states
state_fips = {"01": "AL", "05": "AR", "29": "MO", "34": "NJ", "39": "OH", "51": "VA"}
counties = counties[counties["STATEFP"].isin(state_fips.keys())].copy()

# Merge with your panel (collapse to one row per county first)
county_level = panel.groupby("FIPS")["PFAS_county"].first().reset_index()
counties = counties.merge(county_level, on="FIPS", how="left")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

counties.plot(
    column="PFAS_county",
    ax=ax,
    legend=True,
    cmap="YlOrRd",
    missing_kwds={"color": "lightgrey", "label": "No data"},
    legend_kwds={"label": "PFAS Concentration (ng/L)", "shrink": 0.6}
)

ax.set_title("County-Level PFAS Concentration in Drinking Water", fontsize=14, fontweight="bold")
ax.axis("off")

plt.tight_layout()
plt.savefig("pfas_choropleth.png", dpi=150, bbox_inches="tight")
plt.show()