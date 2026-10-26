"""Synthetic OCO-2-like files, so the whole pipeline can be built and tested without NASA access.

THESE ARE NOT OBSERVATIONS. Files carry the global attribute synthetic="true"; the reader
propagates it, the pipeline prints a banner and stamps every figure "SYNTHETIC DATA", and it
refuses to mix synthetic and real files.

What is simulated (so we know the right answer):
  * the real OCO-2 sampling pattern: a 16-day repeat, 3 tracks crossing the box, 8 footprints
    across a ~10 km swath, a sounding every ~2.25 km along track;
  * a background (trend + seasonal cycle + per-pass offset + along-track gradient + spatially
    correlated error + white noise matching the reported uncertainty);
  * cloud: patchy bad-quality soundings, and a few almost fully clouded passes;
  * CO2 plumes from the REAL power-plant locations, blown downwind with random wind;
  * two urban 'domes' (Johannesburg, Pretoria) that are NOT power plants but sit close to small
    ones, which makes them a deliberate attribution trap;
  * occasional localised retrieval artefacts that are not sources at all (false-alarm generators).
A truth table is written alongside the files so detection performance can be scored.

    python -m highveld_co2.synthetic --days 730 --out data/raw/synthetic
"""
import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from .config import SYNTH_DIR
from .geo import to_lonlat, to_metric
from .sources import load_sources

TRACKS = {1: 27.9, 6: 29.4, 12: 30.9}   # day offset in the 16-day cycle -> longitude at lat -26.25
HEADING = np.radians(352.0)             # sun-synchronous ground track runs ~8 degrees west of north
URBAN = [("Johannesburg", 28.04, -26.20, 1.5, 18.0),   # name, lon, lat, peak ppm, sigma km
         ("Pretoria", 28.19, -25.75, 1.2, 12.0)]
FRAME_STEP_KM = 2.25
N_FOOTPRINTS = 8
FOOTPRINT_SPACING_KM = 1.3


def _smooth_noise(rng, n, step_km, corr_km, scale):
    """Zero-mean noise with an exponential correlation length (km), rescaled to std=scale."""
    k = max(2, int(5 * corr_km / step_km))
    kernel = np.exp(-np.arange(k) * step_km / corr_km)
    white = rng.normal(size=n + 2 * k)
    sm = np.convolve(white, kernel, mode="full")[k:k + n]
    sm = (sm - sm.mean()) / (sm.std() + 1e-12)
    return sm * scale


def _plume_enhancement(x, y, src, wind_dir_deg, wind_ms, rng, amplitude_scale):
    """Sum of Gaussian-like plumes (ppm) at points (x, y) in metres; returns (enh, truth rows)."""
    enh = np.zeros_like(x)
    truth = []
    for r in src.itertuples():
        cap = 4000.0 if pd.isna(r.capacity_mw) else float(r.capacity_mw)
        operating = rng.uniform(0.5, 1.0)
        theta = np.radians(wind_dir_deg + rng.normal(0, 15))
        amp = min(4.0, 2.0 * (cap / 4000.0) ** 0.8 * operating * (4.0 / wind_ms) * amplitude_scale)
        dx, dy = x - r.x_m, y - r.y_m
        wx, wy = np.sin(theta), np.cos(theta)
        d = (dx * wx + dy * wy) / 1000.0        # km downwind
        c = (dx * wy - dy * wx) / 1000.0        # km crosswind
        sy = 2.5 + 0.12 * np.clip(d, 0, None)
        shape = np.where(d > 0, np.exp(-0.5 * (c / sy) ** 2) * np.exp(-d / 40.0) * (1 - np.exp(-d / 4.0)), 0.0)
        enh += amp * shape
        truth.append({"source": r.name, "amplitude_ppm": round(amp, 3),
                      "wind_dir_deg": round(float(np.degrees(theta)) % 360, 1), "wind_ms": round(wind_ms, 2)})
    return enh, truth


def _urban(x, y):
    out = np.zeros_like(x)
    for _, lon, lat, peak, sig in URBAN:
        ux, uy = to_metric(lon, lat)
        r2 = ((x - ux) ** 2 + (y - uy) ** 2) / 1000.0 ** 2
        out += peak * np.exp(-0.5 * r2 / sig ** 2)
    return out


def _simulate_pass(day: date, cycle_index: int, lon0: float, src, rng, amplitude_scale):
    """Return (DataFrame of every sounding on this pass, truth rows)."""
    n_frames = int(660 / FRAME_STEP_KM)
    u = (np.arange(n_frames) - n_frames / 2) * FRAME_STEP_KM          # km along track
    a = (np.arange(N_FOOTPRINTS) - (N_FOOTPRINTS - 1) / 2) * FOOTPRINT_SPACING_KM
    uu, aa = np.meshgrid(u, a, indexing="ij")
    fp = np.tile(np.arange(N_FOOTPRINTS), (n_frames, 1))
    frame = np.repeat(np.arange(n_frames)[:, None], N_FOOTPRINTS, axis=1)

    x0, y0 = to_metric(lon0, -26.25)
    d = np.array([np.sin(HEADING), np.cos(HEADING)])
    p = np.array([d[1], -d[0]])
    x = float(x0) + (uu * d[0] + aa * p[0]) * 1000.0
    y = float(y0) + (uu * d[1] + aa * p[1]) * 1000.0
    lon, lat = to_lonlat(x, y)

    # --- background --------------------------------------------------------------
    t_years = (day - date(2021, 1, 1)).days / 365.25
    base = 413.0 + 2.4 * t_years + 3.0 * np.cos(2 * np.pi * (day.timetuple().tm_yday - 100) / 365.25)
    pass_off = rng.normal(0, 0.4)
    gradient = rng.normal(0, 0.4) * (u / 200.0)
    corr = _smooth_noise(rng, n_frames, FRAME_STEP_KM, 25.0, 0.5)   # spatially correlated retrieval error
    fp_off = rng.normal(0, 0.08, N_FOOTPRINTS)
    unc = rng.uniform(0.45, 0.85, size=x.shape)
    white = rng.normal(size=x.shape) * unc * 1.1
    xco2 = base + pass_off + (gradient + corr)[:, None] + fp_off[None, :] + white

    # --- signal -----------------------------------------------------------------
    wind_dir = rng.uniform(0, 360)
    wind_ms = float(np.clip(rng.lognormal(np.log(4.0), 0.4), 1.5, 10.0))
    plume, truth = _plume_enhancement(x, y, src, wind_dir, wind_ms, rng, amplitude_scale)
    xco2 = xco2 + plume + _urban(x, y)

    # --- retrieval artefacts: ~12 % of passes get one localised +/- bump that is NOT a source.
    # Real retrievals do this (bright/dark surfaces, thin cloud or aerosol); it is what makes
    # false alarms possible, and why the control test exists.
    if rng.random() < 0.12:
        uc = rng.uniform(-150, 150)
        amp = rng.uniform(1.2, 2.0) * rng.choice([-1, 1, 1])
        xco2 = xco2 + amp * np.exp(-0.5 * ((uu - uc) / 8.0) ** 2)

    # --- cloud / quality ------------------------------------------------------------
    cloud_frac = 0.8 if rng.random() < 0.10 else rng.uniform(0.05, 0.55)
    cloud_field = _smooth_noise(rng, n_frames, FRAME_STEP_KM, 25.0, 1.0)
    bad_frame = cloud_field > np.quantile(cloud_field, 1 - cloud_frac)
    qf = (bad_frame[:, None] | (rng.random(x.shape) < 0.05)).astype(np.int16)

    t_start = datetime(day.year, day.month, day.day, 11, 30, tzinfo=timezone.utc).timestamp()
    t = t_start + frame * 0.333
    mode = np.full(x.shape, cycle_index % 2, dtype=np.int8)           # alternate nadir / glint by cycle
    ts = pd.to_datetime(t.ravel(), unit="s", utc=True).strftime("%Y%m%d%H%M%S").astype("int64")
    sid = ts.values * 100 + (frame.ravel() % 10) * 10 + fp.ravel() + 1

    df = pd.DataFrame({
        "latitude": lat.ravel(), "longitude": lon.ravel(), "time": t.ravel(),
        "xco2": xco2.ravel(), "xco2_uncertainty": unc.ravel(), "xco2_quality_flag": qf.ravel(),
        "sounding_id": sid, "operation_mode": mode.ravel(),
        "land_water_indicator": np.zeros(x.size, dtype=np.int8),
        "solar_zenith_angle": np.full(x.size, 40.0),
    })
    for tr in truth:
        tr["date"] = day.isoformat()
    return df, truth


def _write_nc(path: Path, df: pd.DataFrame):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "SYNTHETIC OCO-2-like Lite file for testing. NOT real observations."
        ds.synthetic = "true"
        ds.createDimension("sounding_id", len(df))
        g = ds.createGroup("Sounding")

        def put(node, name, arr, dtype, units=""):
            v = node.createVariable(name, dtype, ("sounding_id",), zlib=True)
            v[:] = arr
            if units:
                v.units = units

        put(ds, "latitude", df["latitude"].values, "f8", "degrees_north")
        put(ds, "longitude", df["longitude"].values, "f8", "degrees_east")
        put(ds, "time", df["time"].values, "f8", "seconds since 1970-01-01 00:00:00 UTC")
        put(ds, "xco2", df["xco2"].values, "f4", "ppm")
        put(ds, "xco2_uncertainty", df["xco2_uncertainty"].values, "f4", "ppm")
        put(ds, "xco2_quality_flag", df["xco2_quality_flag"].values, "i2")
        put(ds, "sounding_id", df["sounding_id"].values, "i8")
        put(ds, "solar_zenith_angle", df["solar_zenith_angle"].values, "f4", "degrees")
        put(g, "operation_mode", df["operation_mode"].values, "i1")
        put(g, "land_water_indicator", df["land_water_indicator"].values, "i1")


def make_files(out_dir=SYNTH_DIR, start="2021-06-01", n_days=730, seed=1, amplitude_scale=1.0):
    """Write one file per overpass day. Returns the list of paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = load_sources()
    d0 = date.fromisoformat(start)
    paths, truth_all = [], []
    for i in range(n_days):
        day = d0 + timedelta(days=i)
        off = i % 16
        if off not in TRACKS:
            continue
        rng = np.random.default_rng(seed * 100003 + i)
        df, truth = _simulate_pass(day, i // 16, TRACKS[off], src, rng, amplitude_scale)
        path = out_dir / f"oco2_LtCO2_{day:%y%m%d}_B11014Ar_synthetic.nc4"
        _write_nc(path, df)
        paths.append(path)
        truth_all.extend(truth)
    pd.DataFrame(truth_all).to_csv(out_dir / "synthetic_truth.csv", index=False)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(SYNTH_DIR))
    ap.add_argument("--start", default="2021-06-01")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amplitude-scale", type=float, default=1.0)
    a = ap.parse_args()
    paths = make_files(a.out, a.start, a.days, a.seed, a.amplitude_scale)
    print(f"Wrote {len(paths)} synthetic overpass files to {a.out}  (NOT real data)")


if __name__ == "__main__":
    main()
