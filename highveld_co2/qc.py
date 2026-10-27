"""Quality control: keep only soundings we are willing to interpret."""
import pandas as pd

from .config import DEFAULT_PARAMS, Params


def apply_qc(df: pd.DataFrame, params: Params = DEFAULT_PARAMS):
    """Filter soundings and report how many each rule removed.

    Rules (in order):
      1. xco2_quality_flag == 0   (NASA's own recommended 'good' flag)
      2. land only                (land_water_indicator == 0)
      3. operation_mode in params.keep_modes  (drop target / transition modes)
    A rule is skipped (and noted) when its column is entirely missing, so a file version
    without that variable still runs instead of silently dropping everything.

    Returns (clean DataFrame, report dict).
    """
    n0 = len(df)
    report = {"n_in": n0}
    ok = df["qf"] == 0
    report["removed_quality_flag"] = int((~ok).sum())

    if df["land_water"].notna().any():
        land = df["land_water"].fillna(0) == 0
        report["removed_water"] = int((ok & ~land).sum())
        ok &= land
    else:
        report["land_water_rule"] = "skipped (column missing)"

    if df["op_mode"].notna().any():
        mode_ok = df["op_mode"].isin(params.keep_modes) | df["op_mode"].isna()
        report["removed_operation_mode"] = int((ok & ~mode_ok).sum())
        ok &= mode_ok
    else:
        report["operation_mode_rule"] = "skipped (column missing)"

    out = df.loc[ok].reset_index(drop=True)
    out.attrs.update(df.attrs)
    report["n_out"] = len(out)
    report["fraction_kept"] = round(len(out) / n0, 4) if n0 else 0.0
    return out, report
