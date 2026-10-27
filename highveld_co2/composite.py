"""Compositing: average many sparse overpasses into a map and a distance profile.

One overpass shows almost nothing; hundreds averaged can reveal a persistent source.
"""
import numpy as np
import pandas as pd

from .config import BBOX


def grid_mean(bins: pd.DataFrame, res_deg: float = 0.1, min_count: int = 5, bbox=BBOX) -> pd.DataFrame:
    """Mean bin anomaly in res_deg cells (0.1 degrees ~ 10 km). Cells with < min_count bins are dropped."""
    w, s, e, n = bbox
    ix = np.floor((bins["lon"].values - w) / res_deg).astype(int)
    iy = np.floor((bins["lat"].values - s) / res_deg).astype(int)
    g = bins.assign(ix=ix, iy=iy).groupby(["ix", "iy"]).agg(
        mean_anomaly_ppm=("mean_anomaly_ppm", "mean"), n_bins=("mean_anomaly_ppm", "size"),
        n_passes=("pass_id", "nunique")).reset_index()
    g = g[g["n_bins"] >= min_count].copy()
    g["lon"] = w + (g["ix"] + 0.5) * res_deg
    g["lat"] = s + (g["iy"] + 0.5) * res_deg
    return g.reset_index(drop=True)


def radial_profile(bins: pd.DataFrame, edges_km=(0, 10, 20, 30, 40, 60, 80, 100), n_boot: int = 2000, seed: int = 0):
    """Mean anomaly vs distance to the nearest source, with a bootstrap-over-passes 95 % CI.

    Honest reading: the background was built from soundings > exclude_km from any source, so
    rings beyond that distance are ~0 BY CONSTRUCTION. The information is in the inner rings.
    """
    edges = np.asarray(edges_km, dtype=float)
    ring = np.digitize(bins["dist_source_km"].values, edges) - 1
    ok = (ring >= 0) & (ring < len(edges) - 1)
    b = bins.loc[ok].assign(ring=ring[ok])
    passes = b["pass_id"].unique()
    pidx = {p: i for i, p in enumerate(passes)}
    n_r = len(edges) - 1
    sums = np.zeros((len(passes), n_r))
    cnts = np.zeros((len(passes), n_r))
    for (p, r), g in b.groupby(["pass_id", "ring"]):
        sums[pidx[p], r] = g["mean_anomaly_ppm"].sum()
        cnts[pidx[p], r] = len(g)
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, n_r), np.nan)
    for k in range(n_boot):
        sel = rng.integers(0, len(passes), len(passes))
        s, c = sums[sel].sum(axis=0), cnts[sel].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots[k] = np.where(c > 0, s / c, np.nan)
    tot_c = cnts.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(tot_c > 0, sums.sum(axis=0) / tot_c, np.nan)
    return pd.DataFrame({
        "r_lo_km": edges[:-1], "r_hi_km": edges[1:], "n_bins": tot_c.astype(int),
        "n_passes": [(cnts[:, r] > 0).sum() for r in range(n_r)],
        "mean_anomaly_ppm": mean,
        "ci_low": np.nanpercentile(boots, 2.5, axis=0), "ci_high": np.nanpercentile(boots, 97.5, axis=0)})
