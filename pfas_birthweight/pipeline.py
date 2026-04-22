import pandas as pd
import numpy as np
from importlib.resources import files


def _data_path(filename):
    return str(files("pfas_birthweight.data").joinpath(filename))


def load_pfas():
    df = pd.read_excel(_data_path("PFAS.xlsx"))

    df["PWS ID"] = df["PWS ID"].astype(str).str.strip().str.upper()

    df["result"] = pd.to_numeric(df["Analytical Result Value (ng/L)"], errors="coerce")
    df["MRL"] = pd.to_numeric(df["Minimum Reporting Level (ng/L)"], errors="coerce")

    # Replace non-detects with MRL/2
    df["result"] = np.where(df["result"].isna(), df["MRL"] / 2, df["result"])
    df = df[df["result"].notna()].copy()

    return df


def collapse_pfas_to_pws(pfas_df):
    pws = pfas_df.groupby("PWS ID").agg(
        PFAS_mean=("result", "mean"),
        population_served=("Population Served", "first")
    ).reset_index()
    return pws


def load_geo():
    df = pd.read_csv(_data_path("SDWA_GEOGRAPHIC_AREAS.csv"), low_memory=False)
    df = df[df["AREA_TYPE_CODE"] == "CN"].copy()
    df = df[["PWSID", "COUNTY_SERVED"]].copy()

    df["PWSID"] = df["PWSID"].astype(str).str.strip().str.upper()
    df["COUNTY_SERVED"] = (
        df["COUNTY_SERVED"]
        .astype(str)
        .str.upper()
        .str.replace(" COUNTY", "", regex=False)
        .str.strip()
    )
    df = df[
        df["COUNTY_SERVED"].notna() &
        (df["COUNTY_SERVED"] != "") &
        (df["COUNTY_SERVED"] != "NAN")
    ].copy()

    return df


def build_county_pfas(pws_pfas, geo_df):
    merged = pws_pfas.merge(geo_df, left_on="PWS ID", right_on="PWSID", how="inner")

    # Extract state from PWS ID prefix (e.g. "AL0000001" → "AL")
    merged["STATE"] = merged["PWS ID"].str[:2]

    # Deduplicate: one row per PWS-county pair
    merged = merged.drop_duplicates(subset=["PWS ID", "STATE", "COUNTY_SERVED"]).copy()

    merged["population_served"] = pd.to_numeric(merged["population_served"], errors="coerce")
    merged = merged[
        merged["population_served"].notna() &
        (merged["population_served"] > 0)
    ].copy()

    # Population-weighted average PFAS per county
    merged["weighted_PFAS"] = merged["PFAS_mean"] * merged["population_served"]

    county_pfas = (
        merged
        .groupby(["STATE", "COUNTY_SERVED"], as_index=False)
        .agg(
            weighted_PFAS_sum=("weighted_PFAS", "sum"),
            population_total=("population_served", "sum")
        )
    )
    county_pfas["PFAS_county"] = county_pfas["weighted_PFAS_sum"] / county_pfas["population_total"]
    county_pfas = county_pfas.drop(columns=["weighted_PFAS_sum", "population_total"])

    return county_pfas


def load_crosswalk():
    xwalk = pd.read_csv(
        _data_path("national_county.txt"),
        header=None,
        names=["STATE", "STATE_FIPS", "COUNTY_FIPS", "COUNTY_NAME", "CLASS_CODE"],
        dtype=str
    )
    xwalk["FIPS"] = xwalk["STATE_FIPS"] + xwalk["COUNTY_FIPS"]
    xwalk["COUNTY_CLEAN"] = (
        xwalk["COUNTY_NAME"]
        .str.upper()
        .str.strip()
        .str.replace(r'(?i)\s*county$', '', regex=True)
        .str.replace(r'(?i)\s*city$', '', regex=True)
        .str.replace(r'(?i)\s*parish$', '', regex=True)
        .str.strip()
    )
    return xwalk


def attach_fips_to_pfas(county_pfas, crosswalk):
    county_pfas["COUNTY_CLEAN"] = (
        county_pfas["COUNTY_SERVED"]
        .str.upper()
        .str.strip()
        .str.replace(r'(?i)\s*county$', '', regex=True)
        .str.strip()
    )

    pfas_fips = county_pfas.merge(
        crosswalk[["STATE", "COUNTY_CLEAN", "FIPS"]],
        on=["STATE", "COUNTY_CLEAN"],
        how="left"
    )

    return pfas_fips


def load_birth_weights():
    df = pd.read_csv(
        _data_path("birth_weights.csv"),
        dtype={"County of Residence Code": str}
    )

    # CDC WONDER puts footnotes and total rows in the Notes column
    df = df[df["Notes"].isna()].copy()

    df["Average Birth Weight (grams)"] = pd.to_numeric(
        df["Average Birth Weight (grams)"], errors="coerce"
    )
    df["Births"] = pd.to_numeric(df["Births"], errors="coerce")
    df = df[
        df["Average Birth Weight (grams)"].notna() &
        df["Births"].notna()
    ].copy()

    df = df.rename(columns={
        "County of Residence Code": "FIPS",
        "Year": "year",
        "Births": "births",
        "Average Birth Weight (grams)": "avg_birth_weight"
    })
    df["FIPS"] = df["FIPS"].str.zfill(5)
    
    return df[["FIPS", "year", "births", "avg_birth_weight"]].copy()


def build_panel(bw_df, pfas_fips):
    panel = bw_df.merge(
        pfas_fips[["FIPS", "STATE", "COUNTY_SERVED", "PFAS_county"]],
        on="FIPS",
        how="inner"
    )

    panel = panel[[
        "FIPS", "STATE", "COUNTY_SERVED", "year",
        "births", "avg_birth_weight", "PFAS_county"
    ]].sort_values(["FIPS", "year"]).reset_index(drop=True)

    return panel


def build_pfas_birthweight_panel():
    pfas_df = load_pfas()
    pws_pfas = collapse_pfas_to_pws(pfas_df)

    geo_df = load_geo()

    county_pfas = build_county_pfas(pws_pfas, geo_df)

    crosswalk = load_crosswalk()

    pfas_fips = attach_fips_to_pfas(county_pfas, crosswalk)

    bw_df = load_birth_weights()

    panel = build_panel(bw_df, pfas_fips)

    return panel



def load_ucmr5():
    df = pd.read_excel(_data_path("UCMR5.xlsx"), sheet_name=0,
                       dtype={"PWS ID": str})
    df["PWS ID"] = df["PWS ID"].astype(str).str.strip().str.upper()

    # Convert µg/L → ng/L to match existing pipeline
    df["result"] = pd.to_numeric(df["Result (µg/L)"], errors="coerce") * 1000

    # Load MRL lookup from sheet 2 and convert units
    mrl_df = pd.read_excel(_data_path("UCMR_min_reporting_level.xlsx"))
    mrl_df["MRL"] = pd.to_numeric(
        mrl_df["UCMR Minimum Reporting Level (MRL, µg/L)"], errors="coerce"
    ) * 1000

    # Join MRLs onto results by contaminant
    df = df.merge(mrl_df[["Contaminant", "MRL"]], on="Contaminant", how="left")

    # Replace non-detects with MRL/2, same convention as load_pfas()
    df["result"] = np.where(df["result"].isna(), df["MRL"] / 2, df["result"])
    df = df[df["result"].notna()].copy()

    return df


def collapse_ucmr5_to_pws(ucmr5_df):
    # Mirror collapse_pfas_to_pws() but without population_served
    # (not in UCMR5 export — joined later from SDWA)
    pws = ucmr5_df.groupby("PWS ID").agg(
        PFAS_mean=("result", "mean")
    ).reset_index()
    return pws


def load_ucmr5_population():
    # Reuse population data already in PFAS.xlsx
    df = pd.read_excel(_data_path("PFAS.xlsx"))
    df["PWS ID"] = df["PWS ID"].astype(str).str.strip().str.upper()
    df["population_served"] = pd.to_numeric(
        df["Population Served"], errors="coerce"
    )
    pop = df[["PWS ID", "population_served"]].drop_duplicates(subset="PWS ID")
    pop = pop[pop["population_served"].notna() & (pop["population_served"] > 0)]
    return pop


def attach_population_to_ucmr5(pws_ucmr5, pop_df):
    merged = pws_ucmr5.merge(
        pop_df, on="PWS ID", how="left"
    )
    merged["population_served"] = merged["population_served"].fillna(1)
    return merged


# ── NEW: Birth weight count loader (LBW rate) ─────────────────────────────────

def load_birthweight_counts():
    df = pd.read_csv(
        _data_path("Birthweight_Counts.xls"),
        sep="\t",
        dtype={"County of Residence Code": str}
    )

    df = df[df["Notes"].isna()].copy()

    df["Births"] = pd.to_numeric(df["Births"], errors="coerce")
    df = df[df["Births"].notna() & (df["Births"] > 0)].copy()

    df = df.rename(columns={
        "County of Residence Code": "FIPS",
        "Year": "year",
        "Births": "births",
        "Infant Birth Weight 12": "birth_weight_category"
    })

    df["FIPS"] = df["FIPS"].str.zfill(5)

    LBW_CODES = ["500 - 999 grams", "1000 - 1499 grams",
                 "1500 - 1999 grams", "2000 - 2499 grams"]

    # Sum all LBW categories into one row per county-year
    lbw = (
        df[df["birth_weight_category"].isin(LBW_CODES)]
        .groupby(["FIPS", "year"])["births"]
        .sum()
        .reset_index()
        .rename(columns={"births": "lbw_births"})
    )

    total = (
        df.groupby(["FIPS", "year"])["births"]
        .sum()
        .reset_index()
        .rename(columns={"births": "total_births"})
    )

    result = lbw.merge(total, on=["FIPS", "year"], how="inner")
    result["lbw_rate"] = result["lbw_births"] / result["total_births"]

    return result[["FIPS", "year", "lbw_births", "total_births", "lbw_rate"]].copy()


# ── NEW: Gestational age loader (preterm rate) ────────────────────────────────

def load_gestational_age():
    df = pd.read_csv(
        _data_path("Gestational_age.xls"),
        sep="\t",
        dtype={"County of Residence Code": str}
    )

    df = df[df["Notes"].isna()].copy()

    df["Births"] = pd.to_numeric(df["Births"], errors="coerce")
    df = df[df["Births"].notna() & (df["Births"] > 0)].copy()

    df = df.rename(columns={
        "County of Residence Code": "FIPS",
        "Year": "year",
        "Births": "births",
        "OE Gestational Age Recode 10": "gestational_age_category"
    })

    df["FIPS"] = df["FIPS"].str.zfill(5)

    PRETERM_CODES = ["Under 20 weeks", "20 - 27 weeks", "28 - 31 weeks",
                     "32 - 35 weeks", "36 weeks"]

    # Sum all preterm categories into one row per county-year
    preterm = (
        df[df["gestational_age_category"].isin(PRETERM_CODES)]
        .groupby(["FIPS", "year"])["births"]
        .sum()
        .reset_index()
        .rename(columns={"births": "preterm_births"})
    )

    total = (
        df.groupby(["FIPS", "year"])["births"]
        .sum()
        .reset_index()
        .rename(columns={"births": "total_births"})
    )

    result = preterm.merge(total, on=["FIPS", "year"], how="inner")
    result["preterm_rate"] = result["preterm_births"] / result["total_births"]

    return result[["FIPS", "year", "preterm_births", "total_births", "preterm_rate"]].copy()

# ── NEW: Extended panel builder ───────────────────────────────────────────────

def build_extended_panel():
    # --- existing PFAS + birth weight path (unchanged) ---
    pfas_df     = load_pfas()
    pws_pfas    = collapse_pfas_to_pws(pfas_df)
    geo_df      = load_geo()
    county_pfas = build_county_pfas(pws_pfas, geo_df)
    crosswalk   = load_crosswalk()
    pfas_fips   = attach_fips_to_pfas(county_pfas, crosswalk)
    bw_df       = load_birth_weights()
    panel       = build_panel(bw_df, pfas_fips)

    # --- add LBW rate ---
    lbw_df = load_birthweight_counts()
    panel = panel.merge(lbw_df, on=["FIPS", "year"], how="left")

    # --- add preterm rate ---
    ga_df = load_gestational_age()
    panel = panel.merge(ga_df[["FIPS", "year", "preterm_births",
                                "total_births", "preterm_rate"]],
                        on=["FIPS", "year"], how="left")

    panel = panel.sort_values(["FIPS", "year"]).reset_index(drop=True)

    return panel


def build_ucmr5_panel():
    # --- UCMR5 PFAS path ---
    ucmr5_df  = load_ucmr5()
    pws_ucmr5 = collapse_ucmr5_to_pws(ucmr5_df)
    pop_df    = load_ucmr5_population()
    pws_ucmr5 = attach_population_to_ucmr5(pws_ucmr5, pop_df)

    # Reuse existing geo + crosswalk helpers
    geo_df      = load_geo()
    county_pfas = build_county_pfas(pws_ucmr5, geo_df)
    crosswalk   = load_crosswalk()
    pfas_fips   = attach_fips_to_pfas(county_pfas, crosswalk)

    # Birth weight outcomes — filter to 2023-2024 to match UCMR5
    bw_df  = load_birth_weights()
    bw_df  = bw_df[bw_df["year"].isin([2023, 2024])].copy()
    lbw_df = load_birthweight_counts()
    lbw_df = lbw_df[lbw_df["year"].isin([2023, 2024])].copy()
    ga_df  = load_gestational_age()
    ga_df  = ga_df[ga_df["year"].isin([2023, 2024])].copy()

    panel = build_panel(bw_df, pfas_fips)
    panel = panel.merge(lbw_df, on=["FIPS", "year"], how="left")
    panel = panel.merge(ga_df[["FIPS", "year", "preterm_births",
                                "total_births", "preterm_rate"]],
                        on=["FIPS", "year"], how="left")

    panel = panel.sort_values(["FIPS", "year"]).reset_index(drop=True)

    return panel

if __name__ == "__main__":
    print("Testing original pipeline...")
    panel = build_pfas_birthweight_panel()
    print("Original:", panel.shape)
    print(panel.head())

    print("\nTesting extended panel...")
    extended = build_extended_panel()
    print("Extended:", extended.shape)
    print(extended.head())

    print("\nTesting UCMR5 panel...")
    ucmr5 = build_ucmr5_panel()
    print("UCMR5:", ucmr5.shape)
    print(ucmr5.head())