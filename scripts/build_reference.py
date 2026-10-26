"""Build data/reference/power_plants_highveld.csv from the WRI Global Power Plant Database.

    python scripts/build_reference.py

Source: World Resources Institute, Global Power Plant Database v1.3.0 (CC BY 4.0)
https://github.com/wri/global-power-plant-database
Only fossil plants >= 100 MW inside the study box (plus a 0.5 degree buffer) are kept.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from highveld_co2.config import BBOX, REF_DIR  # noqa: E402

URL = ("https://raw.githubusercontent.com/wri/global-power-plant-database/"
       "master/output_database/global_power_plant_database.csv")
FUELS = {"Coal", "Gas", "Oil"}
MIN_MW = 100
BUFFER = 0.5


def main():
    print("Downloading", URL)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)

    w, s, e, n = BBOX
    keep = (
        (df.country == "ZAF")
        & df.primary_fuel.isin(FUELS)
        & (df.capacity_mw >= MIN_MW)
        & df.longitude.between(w - BUFFER, e + BUFFER)
        & df.latitude.between(s - BUFFER, n + BUFFER)
    )
    out = df.loc[keep, ["name", "primary_fuel", "capacity_mw", "latitude", "longitude",
                        "gppd_idnr", "geolocation_source"]].rename(columns={"primary_fuel": "fuel"})
    out = out.sort_values("capacity_mw", ascending=False)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    path = REF_DIR / "power_plants_highveld.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} plants to {path}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
