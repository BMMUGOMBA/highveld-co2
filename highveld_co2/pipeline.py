"""Run the whole pipeline: files -> QC -> anomalies -> events -> attribution -> figures -> (PostGIS).

    python -m highveld_co2.pipeline --raw data/raw/oco2 --out results            # real data
    python -m highveld_co2.pipeline --raw data/raw/synthetic --out results_demo  # synthetic demo
    add --db to load PostGIS (needs docker compose up -d)
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import mapping
from .anomaly import compute_anomalies
from .attribute import cluster_hotspots, complex_stats, control_test, link_events
from .composite import grid_mean, radial_profile
from .config import DEFAULT_PARAMS, RAW_DIR, RESULTS_DIR, Params
from .events import bin_passes, group_events
from .qc import apply_qc
from .read_oco2 import read_many
from .sources import SourceIndex, group_complexes, load_sources


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def run(raw_dir=RAW_DIR, out_dir=RESULTS_DIR, params: Params = DEFAULT_PARAMS, use_db=False,
        make_figures=True, verbose=True) -> dict:
    """Execute every stage and return all intermediate tables (the notebook uses this too)."""
    log = print if verbose else (lambda *a, **k: None)
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(raw_dir.glob("*.nc4")) + list(raw_dir.glob("*.nc")))
    if not files:
        raise SystemExit(f"No .nc/.nc4 files in {raw_dir}. See README step 3 (download) or use the synthetic demo.")
    log(f"[1/8] reading {len(files)} files from {raw_dir}")
    df = read_many(files)
    synthetic = bool(df.attrs.get("synthetic"))
    if synthetic:
        log("      *** SYNTHETIC DATA: results below are a software test, NOT observations ***")

    log("[2/8] quality control")
    clean, qc_report = apply_qc(df, params)
    log(f"      kept {qc_report['n_out']:,} of {qc_report['n_in']:,} soundings ({qc_report['fraction_kept']:.0%})")

    log("[3/8] sources")
    sources, complexes = group_complexes(load_sources(), params.complex_eps_km)
    index = SourceIndex(sources)
    log(f"      {len(sources)} sources in {len(complexes)} complexes")

    log("[4/8] overpasses, background, anomalies")
    soundings, passes = compute_anomalies(clean, index, params)
    n_valid = int(passes["is_valid"].sum())
    log(f"      {len(passes)} overpasses, {n_valid} valid")
    if n_valid == 0:
        raise SystemExit("No valid overpasses. Loosen min_soundings / min_extent_km in config.Params.")

    log("[5/8] binning and event detection")
    bins = bin_passes(soundings, index, params)
    events = link_events(group_events(bins, params), index, params)
    hotspots = cluster_hotspots(events, params)
    log(f"      {len(bins):,} bins, {int(bins['flagged'].sum())} flagged, {len(events)} events, {len(hotspots)} hotspots")

    log("[6/8] statistics")
    cstats = complex_stats(bins, sources, complexes, params)
    control = control_test(bins, params)
    profile = radial_profile(bins)
    grid = grid_mean(bins)
    log(f"      flag rate near sources {control['near_rate']:.1%} vs far {control['far_rate']:.1%}; "
        f"ratio {control.get('rate_ratio', float('nan')):.1f}")

    log("[7/8] writing tables" + (" and figures" if make_figures else ""))
    passes.to_csv(out_dir / "passes.csv", index=False)
    bins.drop(columns=["x_m", "y_m", "nearest_source_idx"]).to_csv(out_dir / "bins.csv.gz", index=False)
    events.drop(columns=["x_m", "y_m"]).to_csv(out_dir / "events.csv", index=False)
    hotspots.to_csv(out_dir / "hotspots.csv", index=False)
    cstats.to_csv(out_dir / "complex_stats.csv", index=False)
    profile.to_csv(out_dir / "radial_profile.csv", index=False)

    if make_figures:
        mapping.fig_anomaly_map(grid, sources, complexes, out_dir / "fig_anomaly_map.png",
                                n_passes=n_valid, synthetic=synthetic)
        mapping.fig_radial_profile(profile, out_dir / "fig_radial_profile.png", synthetic=synthetic)
        mapping.fig_control(control, out_dir / "fig_control_test.png", synthetic=synthetic)
        if len(events):
            pid = events.sort_values("peak_anomaly_ppm", ascending=False)["pass_id"].iloc[0]
        else:
            pid = passes.loc[passes["is_valid"], "pass_id"].iloc[0]
        mapping.fig_example_pass(soundings, bins, pid, out_dir / "fig_example_pass.png", params, synthetic=synthetic)
        mapping.interactive_map(sources, complexes, events, hotspots, out_dir / "map.html")

    summary = {
        "synthetic_data": synthetic,
        "n_files": len(files),
        "date_range": [str(df["time"].min()), str(df["time"].max())],
        "qc": qc_report,
        "n_overpasses": len(passes), "n_valid_overpasses": n_valid,
        "n_bins": len(bins), "n_flagged_bins": int(bins["flagged"].sum()),
        "n_events": len(events), "n_hotspots": len(hotspots),
        "noise": {"median_sigma_ppm": float(passes.loc[passes.is_valid, "sigma_ppm"].median()),
                  "median_reported_uncertainty_ppm": float(passes.loc[passes.is_valid, "unc_median_ppm"].median())},
        "control_test": control,
        "params": dataclasses.asdict(params),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    if use_db:
        log("[8/8] loading PostGIS")
        from .db import save_all
        counts = save_all(sources, complexes, passes, bins, events, cstats)
        log(f"      rows: {counts}")
    else:
        log("[8/8] database step skipped (use --db)")
    log(f"done -> {out_dir}")
    return dict(df=df, clean=clean, soundings=soundings, passes=passes, bins=bins, events=events,
                hotspots=hotspots, sources=sources, complexes=complexes, cstats=cstats, control=control,
                profile=profile, grid=grid, summary=summary, qc=qc_report, index=index, params=params)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(RAW_DIR))
    ap.add_argument("--out", default=str(RESULTS_DIR))
    ap.add_argument("--db", action="store_true")
    ap.add_argument("--exclude-km", type=float)
    ap.add_argument("--min-enh", type=float)
    ap.add_argument("--z-bin", type=float)
    a = ap.parse_args()
    over = {k: v for k, v in {"exclude_km": a.exclude_km, "min_enh_ppm": a.min_enh, "z_bin": a.z_bin}.items()
            if v is not None}
    run(a.raw, a.out, dataclasses.replace(DEFAULT_PARAMS, **over), use_db=a.db)


if __name__ == "__main__":
    main()
