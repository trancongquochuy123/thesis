"""
fig36_cycle_boundaries.py
Figure 3.6: Autocorrelation-detected running cycle boundaries overlaid on signal.
Exact contact_diff formula from notebook 06.
Output: figures_new/fig36_cycle_boundaries.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.ndimage import gaussian_filter1d
import json
import pathlib

RAW = pathlib.Path(__file__).parent.parent / "_raw_data" / "autocorr_boundary"
OUT = pathlib.Path(__file__).parent.parent

# ── load ─────────────────────────────────────────────────────────────────────
lm    = np.load(RAW / "DUNG27_lm_filt.npy")   # (410, 33, 4)
times = np.load(RAW / "DUNG27_times.npy")     # (410,) seconds

with open(RAW / "DUNG27_cycles.json", "r", encoding="utf-8-sig") as f:
    cycles_data = json.load(f)
print(f"cycles.json keys: {list(cycles_data.keys())}")
print(f"n_cycles: {cycles_data.get('n_cycles')}")
print(f"cycles sample: {cycles_data.get('cycles', [])[:2]}")

# ── contact_diff from notebook 06 ────────────────────────────────────────────
def build_contact_diff(lm, sigma=1.5):
    def z(x):
        x = np.asarray(x, np.float32)
        return (x - x.mean()) / (x.std() + 1e-6)
    def sc(h, t):
        low = z((h[:, 1] + t[:, 1]) / 2)
        c   = (h + t) / 2
        spd = z(np.linalg.norm(np.diff(c, axis=0, prepend=c[:1]), axis=1))
        return gaussian_filter1d(low - 0.8 * spd, sigma=sigma)
    return sc(lm[:, 29, :2], lm[:, 31, :2]) - sc(lm[:, 30, :2], lm[:, 32, :2])

raw_cd   = build_contact_diff(lm)
smooth_cd = gaussian_filter1d(raw_cd, sigma=2).astype(np.float32)

# ── parse cycle boundaries ────────────────────────────────────────────────────
cycles = cycles_data.get("cycles", [])
# Each entry has f_start, f_end (frame indices)
boundaries = []
for cyc in cycles:
    boundaries.append(cyc["f_start"])
if cycles:
    boundaries.append(cycles[-1]["f_end"])

print(f"Boundary frames: {boundaries}")
boundary_times = [times[f] for f in boundaries if f < len(times)]

# ── plot ─────────────────────────────────────────────────────────────────────
CYCLE_COLORS = ["#DBEAFE", "#EDE9FE", "#D1FAE5", "#FEF3C7", "#FFE4E6", "#F0FDF4"]

fig, ax = plt.subplots(figsize=(10, 3.4))
fig.patch.set_facecolor("white")

# shade alternating cycles
for i, cyc in enumerate(cycles):
    fs, fe = cyc["f_start"], cyc["f_end"]
    if fs < len(times) and fe < len(times):
        t0, t1 = times[fs], times[fe]
        ax.axvspan(t0, t1, alpha=0.25,
                   color=CYCLE_COLORS[i % len(CYCLE_COLORS)], zorder=1)

# raw + smooth signal
t_lo, t_hi = times[max(0, boundaries[0]-5)], times[min(len(times)-1, boundaries[-1]+5)]
mask = (times >= t_lo) & (times <= t_hi)

ax.plot(times[mask], raw_cd[mask],    color="#6B7280", lw=0.7, alpha=0.45, label="Raw signal")
ax.plot(times[mask], smooth_cd[mask], color="#1D4ED8", lw=1.8, alpha=0.9,  label="Smoothed (σ=2)")

# boundary vertical lines
for i, bt in enumerate(boundary_times):
    ax.axvline(bt, color="#DC2626", lw=1.3, ls="--", alpha=0.85,
               label="Cycle boundary" if i == 0 else "_nolegend_")

# phase label in centre of each cycle
for i, cyc in enumerate(cycles):
    fs, fe = cyc["f_start"], cyc["f_end"]
    if fs < len(times) and fe < len(times):
        t_mid = (times[fs] + times[fe]) / 2
        y_top = smooth_cd[mask].max() * 0.80
        ax.text(t_mid, y_top, f"Cycle {i+1}",
                ha="center", va="center", fontsize=7.5,
                color="#374151", fontweight="bold")

ax.set_xlabel("Time (s)", fontsize=9)
ax.set_ylabel("contact_diff\n(a.u.)", fontsize=8.5)
ax.set_title("Autocorrelation Cycle Detection — contact_diff signal (DUNG27)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
ax.grid(True, alpha=0.25, lw=0.6)
ax.axhline(0, color="#9CA3AF", lw=0.7, ls=":", alpha=0.8)
ax.set_xlim(t_lo, t_hi)
ax.tick_params(labelsize=8)

plt.tight_layout(pad=0.6)
plt.savefig(OUT / "fig36_cycle_boundaries.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved: fig36_cycle_boundaries.png")
