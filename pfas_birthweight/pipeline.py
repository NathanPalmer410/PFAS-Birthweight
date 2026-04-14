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
