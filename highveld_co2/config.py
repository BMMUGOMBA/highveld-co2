"""Central configuration: paths, study region and every tunable analysis parameter.

Everything that is a *choice* rather than a fact lives here, so that sensitivity
tests can change one number and re-run (see `Params` and `dataclasses.replace`).
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "oco2"          # real OCO-2 Lite files go here
SYNTH_DIR = ROOT / "data" / "raw" / "synthetic"   # synthetic test files go here
REF_DIR = ROOT / "data" / "reference"
PROC_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Study region: South African Highveld (Mpumalanga coal-power cluster plus
# Johannesburg/Pretoria). West, South, East, North in degrees (EPSG:4326).
BBOX = (27.0, -28.0, 31.0, -24.5)

# Metric CRS used for every distance calculation. UTM zone 35S (EPSG:32735)
# covers 24E-30E; at the eastern edge of the box (31E) the scale error is
# well under 0.1 %, which is negligible next to the other uncertainties.
METRIC_CRS = "EPSG:32735"

# NASA GES DISC product: OCO-2 Level 2 bias-corrected XCO2 ("Lite"), daily files.
OCO2_SHORT_NAME = "OCO2_L2_Lite_FP"
OCO2_VERSION = None   # None = pick the newest version CMR returns; or e.g. "11.2r"


@dataclass(frozen=True)
class Params:
    """Analysis parameters. Frozen so a run can be reproduced from its Params."""

    # --- quality control -------------------------------------------------
    keep_modes: tuple = (0, 1)        # operation_mode: 0 nadir, 1 glint (2 target, 3 transition dropped)

    # --- segmenting soundings into overpasses ------------------------------
    pass_gap_s: float = 300.0         # time gap that starts a new pass (a pass over the box lasts ~1 min)
    min_soundings: int = 150          # good soundings needed for a usable pass
    min_far_soundings: int = 60       # far-from-source soundings needed to define a background
    min_extent_km: float = 100.0      # along-track length needed (short clips cannot define a background)

    # --- background and anomaly -------------------------------------------
    exclude_km: float = 40.0          # soundings closer than this to a source are not used for the background
    background_method: str = "running"   # "median" (one value per pass) or "running" (along-track median)
    running_window_km: float = 100.0
    sigma_floor_ppm: float = 0.3      # never trust a scatter estimate below this

    # --- binning and detection --------------------------------------------
    bin_km: float = 5.0               # along-track bin length
    min_bin_n: int = 5                # soundings needed in a bin
    min_enh_ppm: float = 1.0          # bin mean must exceed the background by this much...
    z_bin: float = 3.0                # ...and by this many standard errors

    # --- attribution and statistics ------------------------------------------
    # attrib_km, near_km and cover_radius_km are deliberately EQUAL to exclude_km. The background
    # is built from soundings farther than exclude_km from any source, so a plume more than
    # exclude_km downwind is partly absorbed into the background and cannot be detected. Signal
    # therefore lives inside exclude_km; attributing or testing at a smaller radius would orphan
    # real detections (found the hard way: with 30 km, 29 % of synthetic events were unattributed).
    attrib_km: float = 40.0           # an event within this distance of a source is "attributed" to it
    near_km: float = 40.0             # "near" bins for the control test
    far_km: float = 60.0              # "far" (control) bins; the 40-60 km gap is a buffer
    cover_radius_km: float = 40.0     # a pass "observes" a complex if it has data within this radius
    complex_eps_km: float = 30.0      # sources closer than this (chained) form one complex
    hotspot_eps_km: float = 20.0      # events closer than this form one hotspot


DEFAULT_PARAMS = Params()
