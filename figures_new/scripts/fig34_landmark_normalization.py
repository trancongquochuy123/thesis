"""
fig34_landmark_normalization.py
Figure 3.4: Landmark coordinates before vs after hip-centred normalisation.
Uses DUNG27 (410-frame full video) so drift across the image is visible.
Output: figures_new/fig34_landmark_normalization.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib

RAW_SIG  = pathlib.Path(__file__).parent.parent / "_raw_data" / "signal_example"
RAW_NORM = pathlib.Path(__file__).parent.parent / "_raw_data" / "landmark_before_after"
RAW_AUTO = pathlib.Path(__file__).parent.parent / "_raw_data" / "autocorr_boundary"
OUT      = pathlib.Path(__file__).parent.parent

# ── load data ─────────────────────────────────────────────────────────────────
lm_filt = np.load(RAW_SIG  / "DUNG27_lm_filt.npy")   # (410, 33, 4) image coords
lm_norm = np.load(RAW_NORM / "DUNG27_lm_norm.npy")   # (410, 33, 4) hip-centred
times   = np.load(RAW_AUTO / "DUNG27_times.npy")      # (410,) seconds

print(f"lm_filt shape: {lm_filt.shape}, range y: {lm_filt[:,:,1].min():.3f}–{lm_filt[:,:,1].max():.3f}")
print(f"lm_norm shape: {lm_norm.shape}, range y: {lm_norm[:,:,1].min():.3f}–{lm_norm[:,:,1].max():.3f}")

# ── landmark indices (MediaPipe) ──────────────────────────────────────────────
# 0=nose  23=left_hip  24=right_hip  29=left_heel  30=right_heel
COLORS = {"Nose": "#E63946", "Hip (mid)": "#457B9D", "Heel (mid)": "#2A9D8F"}

def mid(lm, i, j, coord): return (lm[:, i, coord] + lm[:, j, coord]) / 2

signals_filt = {
    "Nose":      lm_filt[:, 0, 1],
    "Hip (mid)": mid(lm_filt, 23, 24, 1),
    "Heel (mid)": mid(lm_filt, 29, 30, 1),
}
signals_norm = {
    "Nose":      lm_norm[:, 0, 1],
    "Hip (mid)": mid(lm_norm, 23, 24, 1),
    "Heel (mid)": mid(lm_norm, 29, 30, 1),
}

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
fig.patch.set_facecolor("white")

ALPHA = 0.85
LW    = 1.2

for name, vals in signals_filt.items():
    axes[0].plot(times, vals, label=name, color=COLORS[name],
                 lw=LW, alpha=ALPHA)
axes[0].invert_yaxis()          # image y: 0=top, 1=bottom → invert so "up" is up
axes[0].set_title("Before Normalisation\n(image coordinates)", fontsize=10, fontweight="bold")
axes[0].set_xlabel("Time (s)", fontsize=9)
axes[0].set_ylabel("y  (image fraction, 0 = top)", fontsize=8.5)
axes[0].legend(fontsize=8, loc="upper right")
axes[0].grid(True, alpha=0.3, lw=0.6)

for name, vals in signals_norm.items():
    axes[1].plot(times, vals, label=name, color=COLORS[name],
                 lw=LW, alpha=ALPHA)
axes[1].invert_yaxis()
axes[1].set_title("After Hip-Centred Normalisation\n(torso-scaled coordinates)", fontsize=10, fontweight="bold")
axes[1].set_xlabel("Time (s)", fontsize=9)
axes[1].set_ylabel("y  (torso units, hip = 0)", fontsize=8.5)
axes[1].legend(fontsize=8, loc="upper right")
axes[1].grid(True, alpha=0.3, lw=0.6)
axes[1].axhline(0, color="#457B9D", lw=0.8, ls="--", alpha=0.5)

# shared x zoom to middle portion for clarity
t_lo, t_hi = times[30], times[380]
for ax in axes:
    ax.set_xlim(t_lo, t_hi)
    ax.tick_params(labelsize=8)

plt.tight_layout(pad=0.8)
plt.savefig(OUT / "fig34_landmark_normalization.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved: fig34_landmark_normalization.png")
