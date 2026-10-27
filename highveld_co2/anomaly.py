"""From soundings to anomalies: segment overpasses, estimate a background, subtract it.

Why anomalies? Atmospheric CO2 is ~415 ppm everywhere and a power-plant plume adds only
~1-2 ppm. So the quantity of interest is not XCO2 itself but XCO2 minus what XCO2 *would* have
been without the local source. We estimate that 'what it would have been' (the background)
from soundings of the same overpass that are far from any known source.
"""
import numpy as np
import pandas as pd

from .config import DEFAULT_PARAMS, Params
from .geo import to_metric
from .sources import SourceIndex

MAD_TO_SIGMA = 1.4826   # for Gaussian data, sigma = 1.4826 * median absolute deviation


def add_metric(df: pd.DataFrame) -> pd.DataFrame:
    x, y = to_metric(df["lon"].values, df["lat"].values)
    df = df.copy()
    df["x_m"], df["y_m"] = x, y
    return df


def assign_passes(df: pd.DataFrame, gap_s: float = DEFAULT_PARAMS.pass_gap_s) -> pd.DataFrame:
    """Label each sounding with the overpass it belongs to.

    Soundings are sorted by time; a gap longer than `gap_s` starts a new pass. One pass over the
    box lasts about a minute and consecutive orbits are ~99 minutes apart, so any gap between
    ~2 minutes and ~90 minutes separates passes cleanly. pass_id = '<start date>_<n>'.
    """
    df = df.sort_values("t_s").reset_index(drop=True)
    new = np.r_[True, np.diff(df["t_s"].values) > gap_s]
    seg = np.cumsum(new) - 1
    start_date = pd.Series(df["time"].values[new]).dt.strftime("%Y-%m-%d").values   # one per segment
    # number segments that share a start date: 0, 1, ...
    k = pd.Series(start_date).groupby(start_date).cumcount().values
    ids = np.array([f"{d}_{n}" for d, n in zip(start_date, k)])
    df["pass_id"] = ids[seg]
    return df


def along_track_km(x_m: np.ndarray, y_m: np.ndarray, t_s: np.ndarray | None = None) -> np.ndarray:
    """Distance along the swath (km, starting at 0) for each sounding.

    The footprints of one pass lie on a nearly straight strip, so the first principal axis of
    their (x, y) coordinates *is* the along-track direction. Projecting each sounding onto it
    gives a 1-D coordinate. (Sorting by time instead would zigzag across the 8 footprints.)
    The axis is oriented so it increases with time, making results reproducible.
    """
    xy = np.c_[x_m - x_m.mean(), y_m - y_m.mean()]
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    s = xy @ vt[0] / 1000.0
    if t_s is not None and np.corrcoef(s, t_s)[0, 1] < 0:
        s = -s
    return s - s.min()


def estimate_background(s_km, xco2, far_mask, method: str, window_km: float, sigma_floor: float):
    """Background value at every sounding of one pass, plus a robust scatter.

    method="median":  one number, the median XCO2 of the far soundings.
    method="running": the median in consecutive `window_km` windows along the track, linearly
                      interpolated. This follows slow along-track trends (a real gradient or a
                      retrieval artefact) that a single number would leave in the anomaly.
    Scatter sigma = 1.4826 * MAD of the far soundings about the background (robust to the odd
    plume that leaks into the far field), floored at `sigma_floor`.
    """
    far_x = xco2[far_mask]
    bg_med = np.median(far_x)
    bg = np.full(len(xco2), bg_med)
    if method == "running":
        edges = np.arange(s_km.min(), s_km.max() + window_km, window_km)
        centers, meds = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = far_mask & (s_km >= lo) & (s_km < hi)
            if m.sum() >= 10:
                centers.append(s_km[m].mean())
                meds.append(np.median(xco2[m]))
        if len(centers) >= 2:
            bg = np.interp(s_km, centers, meds)   # flat beyond the first/last centre
    sigma = max(sigma_floor, MAD_TO_SIGMA * np.median(np.abs(far_x - bg[far_mask])))
    return bg, float(sigma)


def compute_anomalies(df: pd.DataFrame, index: SourceIndex, params: Params = DEFAULT_PARAMS):
    """Add pass_id, s_km, dist_src_km, background_ppm, sigma_ppm, anomaly_ppm to each sounding.

    Returns (soundings of valid passes only, table with one row per pass incl. invalid ones).
    A pass is valid when it has enough good soundings, enough far-field soundings to define a
    background, and a long enough along-track extent.
    """
    df = assign_passes(add_metric(df), params.pass_gap_s)
    d_km, _ = index.query(df["x_m"].values, df["y_m"].values)
    df["dist_src_km"] = d_km
    df["s_km"] = np.nan
    df["background_ppm"] = np.nan
    df["sigma_ppm"] = np.nan

    rows, valid_ids = [], []
    for pid, g in df.groupby("pass_id", sort=False):
        s = along_track_km(g["x_m"].values, g["y_m"].values, g["t_s"].values)
        far = g["dist_src_km"].values >= params.exclude_km
        extent = float(s.max() - s.min())
        row = {"pass_id": pid, "obs_date": g["time"].iloc[0].date(), "t_start": g["time"].min(),
               "n_good": len(g), "n_far": int(far.sum()), "extent_km": extent,
               "unc_median_ppm": float(g["xco2_unc"].median()),
               "background_ppm": np.nan, "sigma_ppm": np.nan, "is_valid": False}
        if len(g) >= params.min_soundings and far.sum() >= params.min_far_soundings \
                and extent >= params.min_extent_km:
            bg, sigma = estimate_background(s, g["xco2"].values, far, params.background_method,
                                            params.running_window_km, params.sigma_floor_ppm)
            df.loc[g.index, "s_km"] = s
            df.loc[g.index, "background_ppm"] = bg
            df.loc[g.index, "sigma_ppm"] = sigma
            row.update(background_ppm=float(np.median(bg)), sigma_ppm=sigma, is_valid=True)
            valid_ids.append(pid)
        rows.append(row)

    passes = pd.DataFrame(rows)
    out = df[df["pass_id"].isin(valid_ids)].copy()
    out["anomaly_ppm"] = out["xco2"] - out["background_ppm"]
    out.attrs.update(df.attrs)
    return out.reset_index(drop=True), passes
