"""Figures (matplotlib, saved as PNG) and an interactive folium map.

Design rules followed (from the data-visualisation guidance used for this project):
  * anomaly = polarity  -> diverging blue/red ramp with a NEUTRAL GREY midpoint (never a hue at 0);
  * one series          -> one colour, no legend box; several series -> legend AND a shape difference,
                           so identity never rests on colour alone;
  * thin marks, recessive grid, text in ink colours (never in the series colour);
  * synthetic data      -> every figure is stamped SYNTHETIC DATA.
"""
import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .attribute import wilson_ci  # noqa: E402
from .config import BBOX  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED, NEUTRAL = "#2a78d6", "#eb6834", "#e34948", "#f0efec"
# diverging: three equal steps per arm, grey midpoint
DIV = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#184f95", "#2a78d6", "#9ec5f4", NEUTRAL, "#f3b0ad", "#e34948", "#a02a2a"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _finish(fig, path, synthetic):
    fig.patch.set_facecolor(SURFACE)
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA", rotation=28, fontsize=44, color=INK, alpha=0.10,
                 ha="center", va="center", weight="bold")
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_anomaly_map(grid, sources, complexes, path, bbox=BBOX, res_deg=0.1, vlim=1.5,
                    n_passes=None, synthetic=False):
    """Mean XCO2 anomaly per 0.1 degree cell, with known sources on top."""
    w, s, e, n = bbox
    nx, ny = int(round((e - w) / res_deg)), int(round((n - s) / res_deg))
    z = np.full((ny, nx), np.nan)
    g = grid[(grid["ix"] < nx) & (grid["iy"] < ny)]
    z[g["iy"].values, g["ix"].values] = g["mean_anomaly_ppm"].values

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    _style(ax)
    ax.grid(False)
    mesh = ax.pcolormesh(np.linspace(w, e, nx + 1), np.linspace(s, n, ny + 1), np.ma.masked_invalid(z),
                         cmap=DIV, vmin=-vlim, vmax=vlim, shading="flat")
    ax.scatter(sources["longitude"], sources["latitude"], marker="^", s=24 + sources["capacity_mw"].fillna(3000) / 70,
               c=INK, edgecolors=SURFACE, linewidths=1.2, zorder=5, label="Known source (size = capacity)")
    for _, c in complexes.sort_values("total_capacity_mw", ascending=False).head(4).iterrows():
        ax.annotate(c["name"].replace(" power station", "").replace(" complex", ""),
                    (c["longitude"], c["latitude"]), xytext=(6, 7), textcoords="offset points", fontsize=8.5,
                    color=INK, path_effects=[pe.withStroke(linewidth=2.6, foreground=SURFACE)], zorder=6)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.set_aspect(1 / np.cos(np.radians((s + n) / 2)))
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg)")
    sub = f"{n_passes} valid overpasses; " if n_passes else ""
    ax.set_title(f"Mean XCO2 anomaly per 0.1 degree cell\n{sub}blank = fewer than 5 bins (not sampled)",
                 fontsize=11, color=INK, loc="left")
    cb = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Anomaly vs pass background (ppm)", color=INK2)
    cb.ax.tick_params(colors=INK2, labelsize=9)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2)
    return _finish(fig, path, synthetic)


def fig_radial_profile(profile: pd.DataFrame, path, synthetic=False):
    """Mean anomaly vs distance to nearest source, with the bootstrap interval as a band."""
    p = profile.dropna(subset=["mean_anomaly_ppm"])
    x = (p["r_lo_km"] + p["r_hi_km"]) / 2
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _style(ax)
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.fill_between(x, p["ci_low"], p["ci_high"], color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, p["mean_anomaly_ppm"], color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xlabel("Distance to nearest known source (km)")
    ax.set_ylabel("Mean anomaly (ppm)")
    ax.set_title("XCO2 anomaly falls with distance from known sources\nband = 95 % interval, bootstrap over overpasses",
                 fontsize=11, color=INK, loc="left")
    ax.text(0.99, 0.04, "Beyond the exclusion radius the background was built from these\nsoundings, so ~0 there is by construction.",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    return _finish(fig, path, synthetic)


def fig_example_pass(soundings, bins, pass_id, path, params, synthetic=False):
    """One overpass: sounding anomalies, 5 km bin means, flagged bins, and distance to nearest source."""
    s_ = soundings[soundings["pass_id"] == pass_id]
    b = bins[bins["pass_id"] == pass_id].copy()
    b["s_mid"] = (b["bin_idx"] + 0.5) * params.bin_km
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for a in (ax, ax2):
        _style(a)
    ax.axhline(0, color=INK2, linewidth=0.9)
    ax.scatter(s_["s_km"], s_["anomaly_ppm"], s=3, color=INK2, alpha=0.25, linewidths=0, label="Single soundings")
    ok = b[~b["flagged"]]
    fl = b[b["flagged"]]
    ax.plot(b["s_mid"], b["mean_anomaly_ppm"], color=BLUE, linewidth=1.2, alpha=0.6)
    ax.scatter(ok["s_mid"], ok["mean_anomaly_ppm"], s=22, color=BLUE, edgecolors=SURFACE, linewidths=1, zorder=4,
               label=f"{params.bin_km:g} km bin mean")
    ax.scatter(fl["s_mid"], fl["mean_anomaly_ppm"], s=60, marker="D", color=ORANGE, edgecolors=SURFACE,
               linewidths=1.2, zorder=5, label="Flagged bin (event)")
    ax.axhline(params.min_enh_ppm, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.text(s_["s_km"].max(), params.min_enh_ppm, f" {params.min_enh_ppm:g} ppm threshold", fontsize=8, color=INK2,
            va="bottom", ha="right")
    ax.set_ylabel("Anomaly (ppm)")
    ax.set_title(f"Overpass {pass_id}: anomaly along the track", fontsize=11, color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2)
    ax2.plot(b["s_mid"], b["dist_source_km"], color=INK2, linewidth=1.4)
    ax2.axhline(params.attrib_km, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax2.set_ylabel("Dist. to\nsource (km)")
    ax2.set_xlabel("Distance along track (km)")
    ax2.set_ylim(bottom=0)
    return _finish(fig, path, synthetic)


def fig_control(control: dict, path, synthetic=False):
    """Flagged-bin rate near sources vs far from them (the control test), with 95 % Wilson intervals."""
    vals = [("Near sources", control["near_flagged"], control["near_bins"]),
            ("Far from sources\n(control)", control["far_flagged"], control["far_bins"])]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    _style(ax)
    for i, (lab, k, n) in enumerate(vals):
        r = k / n if n else 0
        lo, hi = wilson_ci(k, n)
        ax.bar(i, r * 100, width=0.5, color=BLUE)
        ax.errorbar(i, r * 100, yerr=[[ (r - lo) * 100], [(hi - r) * 100]], color=INK2, capsize=4, linewidth=1.2)
        ax.text(i, hi * 100 + 0.4, f"{r * 100:.1f} %\n({k}/{n})", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(2))
    ax.set_xticklabels([v[0] for v in vals])
    ax.set_ylabel("Bins flagged (%)")
    top = max(wilson_ci(k, n)[1] * 100 for _, k, n in vals if n) if any(n for _, _, n in vals) else 1.0
    ax.set_ylim(0, top * 1.35)          # headroom so the value labels never hit the title
    ax.set_title("Do detections concentrate near sources?\nerror bars = 95 % Wilson interval", fontsize=11,
                 color=INK, loc="left", pad=12)
    return _finish(fig, path, synthetic)


def interactive_map(sources, complexes, events, hotspots, path, bbox=BBOX):
    """Folium map: sources, events (by peak anomaly), hotspots. Open the HTML in any browser."""
    w, s, e, n = bbox
    m = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=7, tiles="OpenStreetMap")
    folium.Rectangle([[s, w], [n, e]], color="#52514e", weight=1, fill=False).add_to(m)

    fs = folium.FeatureGroup(name="Known sources")
    for _, r in sources.iterrows():
        cap = "n/a" if pd.isna(r["capacity_mw"]) else f"{r['capacity_mw']:,.0f} MW"
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=4 + (0 if pd.isna(r["capacity_mw"]) else r["capacity_mw"] / 800),
                            color="#0b0b0b", weight=1, fill=True, fill_color="#52514e", fill_opacity=0.8,
                            tooltip=f"{r['name']} ({cap})").add_to(fs)
    fs.add_to(m)

    fe = folium.FeatureGroup(name="Events")
    for _, r in events.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=3 + 2 * r["peak_anomaly_ppm"], color="#eb6834", weight=1,
                            fill=True, fill_color="#eb6834", fill_opacity=0.6,
                            tooltip=f"{r['obs_date']} | peak {r['peak_anomaly_ppm']:.2f} ppm | "
                                    f"{r['dist_source_km']:.0f} km from {r['nearest_source']}").add_to(fe)
    fe.add_to(m)

    fh = folium.FeatureGroup(name="Hotspots (>= 3 distinct days)")
    for _, r in hotspots[hotspots["n_distinct_days"] >= 3].iterrows():
        folium.Circle([r["lat"], r["lon"]], radius=20000, color="#e34948", weight=2, fill=False,
                      tooltip=f"hotspot {r['hotspot_id']}: {r['n_events']} events on {r['n_distinct_days']} days; "
                              f"explained by source: {r['explained_by_source']}").add_to(fh)
    fh.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(str(path))
    return path
