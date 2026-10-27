"""Detecting enhancement events: bin along track, flag significant bins, merge into events.

Single soundings are too noisy (~0.7 ppm) to flag one by one. Averaging the ~10-40 soundings in a
5 km along-track bin cuts the noise by sqrt(n), so a bin is flagged only when its mean anomaly
is both large in ppm (>= min_enh_ppm) and large relative to its standard error (>= z_bin).
Adjacent flagged bins in one pass are merged into one event.
"""
import numpy as np
import pandas as pd

from .config import DEFAULT_PARAMS, Params
from .sources import SourceIndex


def bin_passes(df: pd.DataFrame, index: SourceIndex, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """One row per (pass, along-track bin) with mean anomaly, standard error, z, distance to source."""
    d = df.copy()
    d["bin_idx"] = np.floor(d["s_km"] / params.bin_km).astype(int)
    g = d.groupby(["pass_id", "bin_idx"], sort=True).agg(
        n=("anomaly_ppm", "size"),
        mean_anomaly_ppm=("anomaly_ppm", "mean"),
        mean_xco2_ppm=("xco2", "mean"),
        sigma_ppm=("sigma_ppm", "first"),
        lat=("lat", "mean"), lon=("lon", "mean"),
        x_m=("x_m", "mean"), y_m=("y_m", "mean"),
        obs_date=("time", lambda t: t.iloc[0].date()),
    ).reset_index()
    g = g[g["n"] >= params.min_bin_n].copy()
    g["se_ppm"] = g["sigma_ppm"] / np.sqrt(g["n"])
    g["z"] = g["mean_anomaly_ppm"] / g["se_ppm"]
    dist, idx = index.query(g["x_m"].values, g["y_m"].values)
    g["dist_source_km"] = dist
    g["nearest_source_idx"] = idx
    g["flagged"] = (g["mean_anomaly_ppm"] >= params.min_enh_ppm) & (g["z"] >= params.z_bin)
    return g.reset_index(drop=True)


def group_events(bins: pd.DataFrame, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Merge runs of adjacent flagged bins (same pass, consecutive bin_idx) into events."""
    f = bins[bins["flagged"]].sort_values(["pass_id", "bin_idx"]).copy()
    if f.empty:
        return pd.DataFrame(columns=[
            "event_id", "pass_id", "obs_date", "n_bins", "length_km", "n_soundings", "peak_anomaly_ppm",
            "mean_anomaly_ppm", "peak_z", "lat", "lon", "x_m", "y_m"])
    new_run = (f["pass_id"] != f["pass_id"].shift()) | (f["bin_idx"] - f["bin_idx"].shift() != 1)
    f["run"] = new_run.cumsum() - 1
    rows = []
    for run, g in f.groupby("run"):
        peak = g.loc[g["mean_anomaly_ppm"].idxmax()]
        w = g["mean_anomaly_ppm"].clip(lower=1e-6)       # weight positions by the size of the anomaly
        rows.append({
            "event_id": int(run), "pass_id": peak["pass_id"], "obs_date": peak["obs_date"],
            "n_bins": len(g), "length_km": len(g) * params.bin_km, "n_soundings": int(g["n"].sum()),
            "peak_anomaly_ppm": float(peak["mean_anomaly_ppm"]),
            "mean_anomaly_ppm": float(np.average(g["mean_anomaly_ppm"], weights=g["n"])),
            "peak_z": float(g["z"].max()),
            "lat": float(np.average(g["lat"], weights=w)), "lon": float(np.average(g["lon"], weights=w)),
            "x_m": float(np.average(g["x_m"], weights=w)), "y_m": float(np.average(g["y_m"], weights=w)),
        })
    return pd.DataFrame(rows)
