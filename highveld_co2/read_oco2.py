"""Read an OCO-2 Lite (L2 bias-corrected XCO2) NetCDF4 file into a pandas DataFrame.

Layout of a Lite file (each variable has one value per sounding):
  root:      xco2, xco2_uncertainty, xco2_quality_flag, latitude, longitude, time, sounding_id
  /Sounding: operation_mode, land_water_indicator, solar_zenith_angle, ...

If a variable name differs in your file version, run
    python -m highveld_co2.inspect_file path/to/file.nc4
and edit the names in REQUIRED / OPTIONAL below. That is the single place to fix.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from .config import BBOX

# canonical column  ->  path inside the file
REQUIRED = {
    "xco2": "xco2",
    "xco2_unc": "xco2_uncertainty",
    "qf": "xco2_quality_flag",
    "lat": "latitude",
    "lon": "longitude",
    "t_s": "time",                       # seconds since 1970-01-01 UTC
}
OPTIONAL = {
    "sounding_id": "sounding_id",
    "op_mode": "Sounding/operation_mode",         # 0 nadir, 1 glint, 2 target, 3 transition
    "land_water": "Sounding/land_water_indicator",  # 0 land, 1 water
    "sza": "solar_zenith_angle",
}


def _get(ds: Dataset, path: str):
    """Fetch one variable as a float array with NaN for fill values; None if absent."""
    node = ds
    *groups, var = path.split("/")
    for g in groups:
        if g not in node.groups:
            return None
        node = node.groups[g]
    if var not in node.variables:
        return None
    arr = node.variables[var][:]
    return np.ma.filled(np.ma.asarray(arr).astype("float64"), np.nan)


def read_lite_file(path, bbox=BBOX) -> pd.DataFrame:
    """Return the soundings inside `bbox` as a DataFrame (no quality filtering yet).

    Reading everything and masking afterwards is fine: a daily global file holds on the
    order of 10^5 soundings, so memory is not a concern.
    """
    path = Path(path)
    with Dataset(path) as ds:
        cols = {}
        for name, var in REQUIRED.items():
            a = _get(ds, var)
            if a is None:
                raise KeyError(f"{path.name}: required variable '{var}' not found. "
                               f"Run `python -m highveld_co2.inspect_file {path}` and update REQUIRED.")
            cols[name] = a
        for name, var in OPTIONAL.items():
            a = _get(ds, var)
            cols[name] = a if a is not None else np.full(len(cols["xco2"]), np.nan)
        synthetic = str(getattr(ds, "synthetic", "false")).lower() == "true"

    df = pd.DataFrame(cols)
    df["sounding_id"] = df["sounding_id"].round().astype("Int64")   # 16-digit ids fit exactly in float64
    w, s, e, n = bbox
    keep = df["lon"].between(w, e) & df["lat"].between(s, n) & df["t_s"].notna() & df["xco2"].notna()
    df = df.loc[keep].copy()
    df["time"] = pd.to_datetime(df["t_s"], unit="s", utc=True)
    df["file"] = path.name
    df.attrs["synthetic"] = synthetic
    return df.reset_index(drop=True)


def read_many(paths, bbox=BBOX) -> pd.DataFrame:
    """Read several files and concatenate. Files with no soundings in the box are skipped."""
    frames, synthetic_flags = [], []
    for p in sorted(paths):
        df = read_lite_file(p, bbox)
        synthetic_flags.append(df.attrs.get("synthetic", False))
        if len(df):
            frames.append(df)
    if not frames:
        raise ValueError("No soundings inside the study box in any file. Check BBOX and the file dates.")
    out = pd.concat(frames, ignore_index=True)
    # Mixing synthetic and real data would silently corrupt results, so refuse.
    if any(synthetic_flags) and not all(synthetic_flags):
        raise ValueError("Refusing to mix synthetic and real files in one run.")
    out.attrs["synthetic"] = bool(synthetic_flags and all(synthetic_flags))
    return out
