# pfas-birthweight

A Python package for building a county-year panel dataset linking PFAS contamination in drinking water to birth weight outcomes.

All data is bundled with the package — no external downloads needed.

## Installation

```bash
pip install pfas-birthweight
```

## Usage

```python
from pfas_birthweight import build_pfas_birthweight_panel

panel = build_pfas_birthweight_panel()
```

This returns a pandas DataFrame with the following columns:

| Column | Description |
|---|---|
| `FIPS` | 5-digit county FIPS code |
| `STATE` | 2-letter state abbreviation |
| `COUNTY_SERVED` | County name |
| `year` | Year (2016–2024) |
| `births` | Number of births |
| `avg_birth_weight` | Average birth weight (grams) |
| `PFAS_county` | Population-weighted mean PFAS concentration (ng/L) |

## Individual Functions

```python
from pfas_birthweight import (
    load_pfas,
    collapse_pfas_to_pws,
    load_geo,
    build_county_pfas,
    load_crosswalk,
    attach_fips_to_pfas,
    load_birth_weights,
    build_panel,
)
```

## Data Sources

- EPA UCMR PFAS monitoring data
- EPA SDWA Geographic Areas
- Census national county FIPS crosswalk
- CDC WONDER birth weight data (2016–2024)
