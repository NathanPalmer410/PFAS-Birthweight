"""
pfas_birthweight
================
A Python package for building a county-year panel dataset linking
PFAS contamination in drinking water to birth weight outcomes.

Data sources:
    - EPA UCMR PFAS monitoring data (PFAS.xlsx)
    - EPA SDWA Geographic Areas (SDWA_GEOGRAPHIC_AREAS.csv)
    - Census national county FIPS crosswalk (national_county.txt)
    - CDC WONDER birth weight data (birth_weights.csv)

Basic usage::

    from pfas_birthweight import build_pfas_birthweight_panel

    panel = build_pfas_birthweight_panel(
        pfas_path="PFAS.xlsx",
        geo_path="SDWA_GEOGRAPHIC_AREAS.csv",
        crosswalk_path="national_county.txt",
        birthweight_path="birth_weights.csv",
    )
"""

from pfas_birthweight.pipeline import (
    load_pfas,
    collapse_pfas_to_pws,
    load_geo,
    build_county_pfas,
    load_crosswalk,
    attach_fips_to_pfas,
    load_birth_weights,
    build_panel,
    build_pfas_birthweight_panel,
)

__all__ = [
    "load_pfas",
    "collapse_pfas_to_pws",
    "load_geo",
    "build_county_pfas",
    "load_crosswalk",
    "attach_fips_to_pfas",
    "load_birth_weights",
    "build_panel",
    "build_pfas_birthweight_panel",
]

