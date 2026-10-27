"""Linking events to sources, finding hotspots, and the statistics that keep us honest.

The key idea: because we hold ALL soundings (not just detections) we know how often each
source was actually *observed*, so we can report a detection frequency (events / observed passes)
instead of a bare count, and we can measure the false-alarm rate with control bins far from sources.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import fisher_exact
from sklearn.cluster import DBSCAN

from .config import DEFAULT_PARAMS, Params
from .geo import to_lonlat
from .sources import SourceIndex


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95 % Wilson score interval for a proportion k/n. Sensible even for small n or k=0."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def link_events(events: pd.DataFrame, index: SourceIndex, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Attach nearest source, its distance, its complex, and whether it is within attrib_km."""
    ev = events.copy()
    if ev.empty:
        for c in ("nearest_source", "dist_source_km", "complex_id", "attributed"):
            ev[c] = pd.Series(dtype="object")
        return ev
    d, i = index.query(ev["x_m"].values, ev["y_m"].values)
    ev["nearest_source"] = index.names(i)
    ev["dist_source_km"] = d
    ev["complex_id"] = index.complex_ids(i)
    ev["attributed"] = d <= params.attrib_km
    return ev


def cluster_hotspots(events: pd.DataFrame, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Group events into spatial hotspots and say whether each is explained by a known source.

    A hotspot with many events on different days and NO source nearby is the interesting case:
    here it is the Johannesburg/Pretoria urban area, which is not in the power-plant list.
    Returns the hotspots table; also adds 'hotspot_id' to `events` in place.
    """
    cols = ["hotspot_id", "n_events", "n_distinct_days", "lat", "lon", "mean_peak_ppm",
            "median_dist_source_km", "explained_by_source"]
    if events.empty:
        events["hotspot_id"] = pd.Series(dtype="int64")
        return pd.DataFrame(columns=cols)
    lab = DBSCAN(eps=params.hotspot_eps_km * 1000.0, min_samples=1).fit_predict(events[["x_m", "y_m"]].values)
    events["hotspot_id"] = lab
    rows = []
    for h, g in events.groupby("hotspot_id"):
        lon, lat = to_lonlat(g["x_m"].mean(), g["y_m"].mean())
        med = float(g["dist_source_km"].median())
        rows.append({"hotspot_id": int(h), "n_events": len(g), "n_distinct_days": g["obs_date"].nunique(),
                     "lat": float(lat), "lon": float(lon), "mean_peak_ppm": float(g["peak_anomaly_ppm"].mean()),
                     "median_dist_source_km": med, "explained_by_source": med <= params.attrib_km})
    return pd.DataFrame(rows).sort_values("n_distinct_days", ascending=False).reset_index(drop=True)


def complex_stats(bins: pd.DataFrame, sources, complexes, params: Params = DEFAULT_PARAMS) -> pd.DataFrame:
    """Per source complex: how many passes observed it, how many had an event, with a 95 % CI.

    'Observed' = the pass has >= 2 valid bins within cover_radius_km of any source in the complex.
    'Event'    = the pass has >= 1 flagged bin within that radius.
    """
    rows = []
    for _, c in complexes.iterrows():
        members = sources[sources["complex_id"] == c["complex_id"]]
        tree = cKDTree(members[["x_m", "y_m"]].values)
        d, _ = tree.query(bins[["x_m", "y_m"]].values)
        near = bins.loc[d / 1000.0 <= params.cover_radius_km]
        per_pass = near.groupby("pass_id").agg(n_bins=("flagged", "size"), n_flag=("flagged", "sum"))
        observed = int((per_pass["n_bins"] >= 2).sum())
        event_passes = int(((per_pass["n_bins"] >= 2) & (per_pass["n_flag"] >= 1)).sum())
        lo, hi = wilson_ci(event_passes, observed)
        rows.append({"complex_id": int(c["complex_id"]), "name": c["name"], "n_sources": int(c["n_sources"]),
                     "total_capacity_mw": float(c["total_capacity_mw"]),
                     "observed_passes": observed, "event_passes": event_passes,
                     "frequency": event_passes / observed if observed else np.nan,
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def control_test(bins: pd.DataFrame, params: Params = DEFAULT_PARAMS, n_boot: int = 2000, seed: int = 0) -> dict:
    """Is the flagged-bin rate higher near sources than far from them?

    'Far' bins are the control: any flags there are false alarms (noise, retrieval artefacts,
    unlisted sources). If near and far flag equally often, our detections mean nothing.
    The CI comes from a bootstrap over PASSES, because bins in one pass are not independent
    (they share a background, a sigma and slowly varying errors), so a plain binomial CI would
    be too narrow.
    """
    near = bins["dist_source_km"] <= params.near_km
    far = bins["dist_source_km"] >= params.far_km
    nn, nf = int(near.sum()), int(bins.loc[near, "flagged"].sum())
    fn, ff = int(far.sum()), int(bins.loc[far, "flagged"].sum())
    out = {"near_bins": nn, "near_flagged": nf, "far_bins": fn, "far_flagged": ff,
           "near_rate": nf / nn if nn else np.nan, "far_rate": ff / fn if fn else np.nan}
    out["rate_ratio"] = out["near_rate"] / out["far_rate"] if ff > 0 and nn else np.nan
    if nn and fn:
        # One-sided Fisher exact test (is the near rate higher?). With thousands of bins the p-value can
        # underflow to exactly 0.0, which would read as "impossible"; clamp to the smallest honest bound.
        p = float(fisher_exact([[nf, nn - nf], [ff, fn - ff]], alternative="greater")[1])
        out["fisher_p"] = max(p, 1e-300)

    # per-pass counts: summing booleans counts the True values
    cnt = pd.DataFrame({"pass_id": bins["pass_id"], "nn": near, "nf": near & bins["flagged"],
                        "fn": far, "ff": far & bins["flagged"]}).groupby("pass_id").sum()
    rng = np.random.default_rng(seed)
    arr = cnt.values.astype(float)
    ratios = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        if s[0] > 0 and s[2] > 0 and s[3] > 0:
            ratios.append((s[1] / s[0]) / (s[3] / s[2]))
    if ratios:
        out["rate_ratio_ci"] = (float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5)))
    return out
