"""
fig35_motion_signals.py
Figure 3.5: Motion signal examples — head_y, hip_y, heel_y, contact_diff.
Exact formulas from notebook 06_signal_ablation_cycle_detection.ipynb.
Output: figures_new/fig35_motion_signals.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import pathlib

RAW_SIG  = pathlib.Path(__file__).parent.parent / "_raw_data" / "signal_example"
RAW_AUTO = pathlib.Path(__file__).parent.parent / "_raw_data" / "autocorr_boundary"
OUT      = pathlib.Path(__file__).parent.parent

# ── load ─────────────────────────────────────────────────────────────────────
lm   = np.load(RAW_SIG  / "DUNG27_lm_filt.npy")   # (410, 33, 4)
times = np.load(RAW_AUTO / "DUNG27_times.npy")     # (410,)

PARAMS = {
    "signal_sigma":  2,
    "detrend_sigma": 20,
}

# ── exact formulas from notebook 06 ──────────────────────────────────────────
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


def build_signal(name, lm, p):
    if name == "contact_diff":
        raw = build_contact_diff(lm)
    else:
        hip = (lm[:, 23, :2] + lm[:, 24, :2]) / 2
        raw = {
            "head_y":  lm[:, 0,  1],
            "hip_y":   (lm[:, 23, 1] + lm[:, 24, 1]) / 2,
            "heel_y":  (lm[:, 29, 1] + lm[:, 30, 1]) / 2,
            "toe_y":   (lm[:, 31, 1] + lm[:, 32, 1]) / 2,
            "AP_diff": (lm[:, 27, 0] - hip[:, 0]) - (lm[:, 28, 0] - hip[:, 0]),
            "V_diff":  lm[:, 29, 1] - lm[:, 30, 1],
        }[name].copy()
        raw = raw - gaussian_filter1d(raw, sigma=p["detrend_sigma"])
    raw  = raw.astype(np.float32)
    smooth = gaussian_filter1d(raw, sigma=p["signal_sigma"]).astype(np.float32)
    return raw, smooth


# ── compute signals ───────────────────────────────────────────────────────────
SIGNALS = [
    ("head_y",       "Head vertical (y)",       "#E63946"),
    ("hip_y",        "Hip vertical (y)",         "#457B9D"),
    ("heel_y",       "Heel vertical (y)",        "#2A9D8F"),
    ("contact_diff", "Contact asymmetry\n(contact_diff)", "#F4A261"),
]

results = {}
for name, label, col in SIGNALS:
    raw, smooth = build_signal(name, lm, PARAMS)
    results[name] = (raw, smooth, label, col)
    print(f"{name}: range [{raw.min():.4f}, {raw.max():.4f}]")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(SIGNALS), 1, figsize=(9, 6.5), sharex=True)
fig.patch.set_facecolor("white")

t_lo, t_hi = times[30], times[380]
mask = (times >= t_lo) & (times <= t_hi)

for ax, (name, label, col) in zip(axes, SIGNALS):
    raw, smooth, label, col = results[name]
    ax.plot(times[mask], raw[mask],    color=col, lw=0.6, alpha=0.35)
    ax.plot(times[mask], smooth[mask], color=col, lw=1.5, alpha=0.92, label=label)
    ax.set_ylabel(label, fontsize=8.2, labelpad=4)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.axhline(0, color="#888", lw=0.7, ls="--", alpha=0.6)
    ax.tick_params(labelsize=7.5)

axes[-1].set_xlabel("Time (s)", fontsize=9)
fig.suptitle("Motion Signals Extracted from Landmark Sequence (DUNG27)",
             fontsize=10, fontweight="bold", y=1.01)

plt.tight_layout(pad=0.6, h_pad=0.5)
plt.savefig(OUT / "fig35_motion_signals.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved: fig35_motion_signals.png")
