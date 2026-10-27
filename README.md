# Highveld CO₂ — OCO-2 XCO₂ anomalies around South Africa's coal-power cluster

An end-to-end remote-sensing pipeline: download NASA OCO-2 column-CO₂ (XCO₂) data, clean it,
estimate a per-overpass background, find localised enhancements, attribute them to known power
plants, quantify how often each source is actually observed, and store everything in PostGIS.

**Study region:** the South African Highveld (27–31°E, 24.5–28°S): Mpumalanga's coal-power
cluster plus Johannesburg/Pretoria. **Instrument:** OCO-2 (NASA), Level 2 Lite bias-corrected XCO₂.

## Status — read this first

| Part | Verified how |
|---|---|
| Reader, QC, anomalies, events, attribution, composites, figures | 31 automated tests on **synthetic** OCO-2-format files with known answers |
| PostGIS schema, loading, example SQL | Run against a real PostGIS 3.4 server; SQL control test reproduces the Python numbers exactly |
| **Real OCO-2 data** | **Not yet run.** The code was written without access to NASA servers. Your first run on real files is the true test (see "Run on real data"). |
| Scientific results | **None yet.** `results/` is empty until you run on real data. Do not present synthetic output as findings. |

Synthetic files carry `synthetic="true"`; the pipeline prints a banner, stamps every figure
`SYNTHETIC DATA`, and refuses to mix synthetic and real files.

## Quick start (synthetic demo, no account needed)

```bash
python -m venv .venv && source .venv/bin/activate        # tested on Python 3.13 (pandas 2.3 and 3.x)
pip install -r requirements.txt
make test                                                # 30 pass, 1 skipped (PostGIS)
make demo                                                # ~10 s; writes results_demo/
```
Open `results_demo/map.html` and the `fig_*.png` files. Then work through
`notebooks/01_walkthrough.ipynb`.

## Run on real data

1. **Earthdata account** (free): https://urs.earthdata.nasa.gov
2. **Authorise the archive** — profile → *Applications → Authorized Apps* → approve
   **NASA GESDISC DATA ARCHIVE**. Skipping this is the most common cause of 401/403 errors.
3. `cp .env.example .env` and fill in `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.
4. **Download.** OCO-2 files are daily and global, and the box is crossed on only ~2–3 days of every
   16, so first learn which days matter:
   ```bash
   python -m highveld_co2.download get --start 2023-06-01 --end 2023-06-16
   python -m highveld_co2.download offsets --cycle-start 2023-06-01      # prints e.g. 3,8,12
   python -m highveld_co2.download get --start 2023-06-01 --end 2024-05-31 \
          --cycle-start 2023-06-01 --offsets 3,8,12
   ```
5. **Inspect one file** and confirm the variable names match `read_oco2.py`:
   `python -m highveld_co2.inspect_file data/raw/oco2/<file>.nc4`
6. **Run:** `make real` (add `--db` for PostGIS: `make db-up`, then
   `python -m highveld_co2.pipeline --raw data/raw/oco2 --out results --db`).
7. Look at `results/`, **sanity-check before believing anything** (see below), then commit
   `results/` and fill in the Results section.

## Method in one paragraph

Soundings are quality-filtered (`xco2_quality_flag == 0`, land, nadir/glint), split into overpasses by
time gaps, and projected onto an along-track coordinate. For each overpass a **background** is estimated
from soundings > 40 km from any known source (running median, 100 km window) together with a robust
scatter (1.4826·MAD). Anomaly = XCO₂ − background. Soundings are averaged in 5 km along-track bins; a bin
is flagged if its mean anomaly is ≥ 1 ppm **and** ≥ 3 standard errors. Adjacent flagged bins form an
**event**. Events are linked to the nearest source (≤ 40 km), clustered into hotspots, and summarised per
source complex as *events ÷ passes that actually observed it*. A **control test** compares the flagged rate
near sources with the rate far from them (the false-alarm baseline), with a bootstrap-over-passes interval.

## Layout

```
highveld_co2/   config · geo · sources · download · inspect_file · read_oco2 · qc · anomaly
                events · attribute · composite · mapping · db · synthetic · pipeline
tests/          unit + integration tests          notebooks/   guided walkthrough
sql/schema.sql  PostGIS schema                    data/reference/  real power-plant locations
scripts/        build_reference.py                docs/        project overview + build tutorial
```

## Limitations (these matter more than the code)

- **Sparse sampling.** One 10 km swath, a 16-day repeat: only ~2–3 passes per 16 days cross the box, and
  clouds remove many. Most plants are seen on few days.
- **Small signal.** Plumes add ~1–2 ppm on ~415 ppm; retrieval noise is ~0.5–1 ppm per sounding, and
  retrieval artefacts over bright/dark surfaces can look like plumes. That is why the control test exists.
- **No wind.** Without wind data, attribution is by proximity only; a plume can drift tens of km. Adding
  ERA5/MERRA-2 winds is the obvious next step.
- **Attribution ≠ emissions.** Detecting an enhancement near a plant does not quantify its emissions, and
  urban CO₂ (Johannesburg/Pretoria) sits next to small plants, so it can masquerade as them.
- **Source list is a snapshot.** The power-plant data is a past vintage; plants have since closed
  (e.g. Komati), changed output, or been added (Kusile). Secunda's coordinates are approximate.
- **Complexes, not plants.** Sources < 30 km apart are merged; the instrument cannot separate them.

## Data and licences

- OCO-2 L2 Lite XCO₂ (`OCO2_L2_Lite_FP`): NASA GES DISC. Cite the DOI on the product page for the version you use.
- Power plants: WRI Global Power Plant Database v1.3.0, CC BY 4.0 — https://github.com/wri/global-power-plant-database
- Secunda: manually added, approximate (see `data/reference/extra_sources.csv`).

## Results (fill in after your real run)

> _Not yet run on real data._ After step 6, replace this block with: the date range and number of valid
> overpasses; the control-test numbers (`results/summary.json`); `results/fig_anomaly_map.png` and
> `fig_radial_profile.png`; and two honest sentences on what the data does and does not show.
