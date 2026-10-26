"""Download OCO-2 Lite files from NASA GES DISC with `earthaccess`.

One-time account setup (this is where most people get stuck):
  1. Free account at https://urs.earthdata.nasa.gov
  2. In your Earthdata profile -> Applications -> Authorized Apps, approve
     "NASA GESDISC DATA ARCHIVE". Without this, downloads fail with 401/403.
  3. Put EARTHDATA_USERNAME / EARTHDATA_PASSWORD in .env (see .env.example)

Why daily files matter: each Lite file covers the whole globe for one day, so a bounding box
cannot shrink the download. OCO-2 repeats its ground track every 16 days and sees the study
box only on ~2-3 days of every 16. `offsets` + `get --offsets` exploit that: download one
16-day window, learn which days cross the box, then fetch only those days in later cycles.

Commands
  python -m highveld_co2.download get --start 2023-06-01 --end 2023-08-31 [--max-files 100]
  python -m highveld_co2.download offsets --raw data/raw/oco2 --cycle-start 2023-06-01
  python -m highveld_co2.download get --start 2023-06-01 --end 2024-05-31 \\
         --cycle-start 2023-06-01 --offsets 2,7,11
"""
import argparse
import os
import re
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .config import BBOX, OCO2_SHORT_NAME, OCO2_VERSION, RAW_DIR


# ---------- pure helpers (unit-tested; no network) ----------------------------------

def parse_version(v: str) -> float:
    """'11.2r' -> 11.2 ; '10r' -> 10.0 ; unparsable -> -1."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else -1.0


def pick_latest_version(granules, wanted=None):
    """Keep only granules of one product version.

    CMR can return the same day under several versions. Mixing them double-counts days and mixes
    retrieval algorithms, so we keep exactly one: `wanted`, or else the numerically newest.
    Returns (granules_kept, version_used).
    """
    def ver(g):
        return g["umm"]["CollectionReference"]["Version"]

    versions = sorted({ver(g) for g in granules}, key=parse_version)
    if not versions:
        return [], None
    use = wanted if wanted else versions[-1]
    return [g for g in granules if ver(g) == use], use


def cycle_days(cycle_start: date, offsets, start: date, end: date, period: int = 16):
    """All dates in [start, end] whose (date - cycle_start) mod `period` is in `offsets`."""
    offs = set(int(o) for o in offsets)
    out, d = [], start
    while d <= end:
        if (d - cycle_start).days % period in offs:
            out.append(d)
        d += timedelta(days=1)
    return out


def date_from_filename(name: str):
    """oco2_LtCO2_230715_B11014Ar_....nc4 -> date(2023, 7, 15); None if no match."""
    m = re.search(r"LtCO2_(\d{2})(\d{2})(\d{2})", name)
    if not m:
        return None
    yy, mm, dd = map(int, m.groups())
    return date(2000 + yy, mm, dd)


def overpass_offsets(raw_dir: Path, cycle_start: date, bbox=BBOX, period: int = 16):
    """Which day-offsets (0..15) of a 16-day cycle have soundings inside the box?

    Reads the files already downloaded for the first cycle. Returns a sorted list.
    """
    from .read_oco2 import read_lite_file

    found = set()
    for f in sorted(Path(raw_dir).glob("*.nc*")):
        d = date_from_filename(f.name)
        if d is None:
            continue
        off = (d - cycle_start).days
        if 0 <= off < period and len(read_lite_file(f, bbox)) > 0:
            found.add(off)
    return sorted(found)


# ---------- network part -------------------------------------------------------------

def search(start: str, end: str, bbox=BBOX, version=OCO2_VERSION):
    import earthaccess

    load_dotenv()
    earthaccess.login(strategy="environment")
    results = earthaccess.search_data(
        short_name=OCO2_SHORT_NAME, bounding_box=tuple(bbox), temporal=(start, end), count=2000)
    kept, used = pick_latest_version(results, version)
    print(f"CMR returned {len(results)} granules; keeping {len(kept)} of version {used}")
    return kept


def get(start, end, out_dir=RAW_DIR, max_files=200, cycle_start=None, offsets=None, version=OCO2_VERSION):
    import earthaccess

    granules = search(start, end, version=version)
    if cycle_start and offsets:
        wanted = set(cycle_days(date.fromisoformat(cycle_start), offsets,
                                date.fromisoformat(start), date.fromisoformat(end)))
        granules = [g for g in granules
                    if date_from_filename(g["umm"]["GranuleUR"]) in wanted]
        print(f"Cycle filter keeps {len(granules)} granules")
    if len(granules) > max_files:
        raise SystemExit(f"{len(granules)} files exceeds --max-files {max_files}. "
                         f"Narrow the dates, use --offsets, or raise --max-files.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = earthaccess.download(granules, str(out_dir))   # skips files already present
    print(f"{len(files)} files in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("--start", required=True)
    g.add_argument("--end", required=True)
    g.add_argument("--out", default=str(RAW_DIR))
    g.add_argument("--max-files", type=int, default=200)
    g.add_argument("--cycle-start")
    g.add_argument("--offsets", type=lambda s: [int(x) for x in s.split(",")])
    o = sub.add_parser("offsets")
    o.add_argument("--raw", default=str(RAW_DIR))
    o.add_argument("--cycle-start", required=True)
    a = ap.parse_args()
    if a.cmd == "get":
        get(a.start, a.end, a.out, a.max_files, a.cycle_start, a.offsets)
    else:
        offs = overpass_offsets(Path(a.raw), date.fromisoformat(a.cycle_start))
        print("Day offsets with soundings in the box:", ",".join(map(str, offs)))
        print("Use:  --cycle-start", a.cycle_start, "--offsets", ",".join(map(str, offs)))


if __name__ == "__main__":
    main()
