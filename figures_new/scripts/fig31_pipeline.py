"""
fig31_pipeline.py — System architecture diagram for Figure 3.1.
Uses inch-space coordinates so box sizes and gaps are precise.
Output: figures_new/fig31_pipeline.png and .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pathlib

# ── colours ──────────────────────────────────────────────────────────────────
BLUE_F, BLUE_E    = "#DBEAFE", "#1D4ED8"
AMBER_F, AMBER_E  = "#FEF3C7", "#92400E"
PURPLE_F, PURPLE_E = "#EDE9FE", "#6D28D9"
GREEN_F, GREEN_E  = "#D1FAE5", "#065F46"

# ── figure in inches ──────────────────────────────────────────────────────────
W, H = 5.5, 10.8          # figure size in inches
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")

PAD = 0.06   # FancyBboxPatch pad (inches)


def rbox(ax, cx, cy, w, h, title, sub=None,
         fc=BLUE_F, ec=BLUE_E, lw=1.5, fs=9.5, sub_fs=8.0):
    """
    Rounded box centred at (cx, cy) with visual size (w, h) in data-inches.
    The FancyBboxPatch is drawn at the INNER rect so that the padded result
    exactly matches the specified (cx, cy, w, h).
    """
    inner_w = w - 2 * PAD
    inner_h = h - 2 * PAD
    rect = FancyBboxPatch(
        (cx - inner_w/2, cy - inner_h/2), inner_w, inner_h,
        boxstyle=f"round,pad={PAD}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3, clip_on=False
    )
    ax.add_patch(rect)
    if sub:
        ax.text(cx, cy + h * 0.12, title,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=4)
        ax.text(cx, cy - h * 0.18, sub,
                ha="center", va="center", fontsize=sub_fs,
                color="#555", style="italic", zorder=4)
    else:
        ax.text(cx, cy, title,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", zorder=4)


def varrow(ax, x, y0, y1, color=BLUE_E, lw=1.4, ms=12):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                linewidth=lw, mutation_scale=ms),
                zorder=5)


def hline(ax, x0, x1, y, color, lw=1.0):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=4)


def vline(ax, x, y0, y1, color, lw=1.0):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, zorder=4)


# ── layout constants (inches) ─────────────────────────────────────────────────
CX  = W / 2          # main column centre-x  = 2.75
BW  = 4.8            # main box width
BH  = 0.68           # main box height
ABW = 1.45           # alt-box (cycle detection) width
ABH = 0.60           # alt-box height
RBW = 2.30           # repr-box width
RBH = 0.70           # repr-box height
OUT_H = 0.62         # output box height

VSTEP = BH + 0.28    # centre-to-centre step between main boxes

# ── absolute y positions of box centres (top → bottom) ─────────────────────
Y_VID    = H - BH/2 - 0.15
Y_POSE   = Y_VID   - VSTEP
Y_FILT   = Y_POSE  - VSTEP
Y_SIG    = Y_FILT  - VSTEP

# cycle detection
BUS1_Y   = Y_SIG   - BH/2 - 0.22      # horizontal bus below signal
Y_CYC    = BUS1_Y  - 0.20 - ABH/2     # cycle-box centres

# group box
GRP_PAD  = 0.18
GRP_T    = Y_CYC + ABH/2 + GRP_PAD
GRP_B    = Y_CYC - ABH/2 - GRP_PAD
BUS2_Y   = GRP_B  - 0.20              # merge bus below group

Y_PHASE  = BUS2_Y  - 0.18 - BH/2

# repr boxes
FORK_Y   = Y_PHASE - BH/2 - 0.22
LM_CX    = CX - 1.30
IM_CX    = CX + 1.30
Y_REPR   = FORK_Y  - 0.20 - RBH/2

# classifier & output
MRG2_Y   = Y_REPR  - RBH/2 - 0.20
Y_CLS    = MRG2_Y  - 0.18 - BH/2
Y_OUT    = Y_CLS   - BH/2 - 0.22 - OUT_H/2

AXPOS    = [CX - 1.60, CX, CX + 1.60]   # x-centres of cycle detection boxes

# ── draw main pipeline stages ─────────────────────────────────────────────────
rbox(ax, CX, Y_VID, BW, BH, "Raw Running Video")
rbox(ax, CX, Y_POSE, BW, BH, "Pose Estimation",
     sub="MediaPipe BlazePose-33")
rbox(ax, CX, Y_FILT, BW, BH, "Landmark Filtering & Normalisation",
     sub="visibility ≥ 0.45  ·  hip-centred  ·  torso-scaled",
     fs=9.0, sub_fs=7.8)
rbox(ax, CX, Y_SIG, BW, BH, "Motion-Signal Extraction",
     sub="head_y,  hip_y,  heel_y,  contact_diff,  AP_diff,  V_diff",
     fs=9.0, sub_fs=7.8)

# main arrows
for y0, y1 in [
    (Y_VID  - BH/2, Y_POSE + BH/2),
    (Y_POSE - BH/2, Y_FILT + BH/2),
    (Y_FILT - BH/2, Y_SIG  + BH/2),
]:
    varrow(ax, CX, y0, y1)

# ── cycle detection alternatives ─────────────────────────────────────────────
for cx_a, lbl in zip(AXPOS, ["Manual\nSegmentation",
                               "Autocorrelation\nDetector",
                               "Boundary\nDetector (NN)"]):
    rbox(ax, cx_a, Y_CYC, ABW, ABH, lbl,
         fc=AMBER_F, ec=AMBER_E, lw=1.2, fs=8.5)

# dashed group rectangle
dash_rect = FancyBboxPatch(
    (AXPOS[0]-ABW/2-GRP_PAD+PAD, GRP_B+PAD),
    (AXPOS[2]+ABW/2+GRP_PAD-PAD) - (AXPOS[0]-ABW/2-GRP_PAD+PAD),
    GRP_T - GRP_B - 2*PAD,
    boxstyle=f"round,pad={PAD}",
    facecolor="none", edgecolor=AMBER_E,
    linewidth=1.3, linestyle=(0, (6, 3)), zorder=2
)
ax.add_patch(dash_rect)
ax.text(CX, GRP_T + 0.10,
        "Cycle Detection Stage  (interchangeable)",
        ha="center", va="bottom", fontsize=7.8,
        color=AMBER_E, fontweight="bold", zorder=4)

# fan out: signal → cycle boxes
sig_bot = Y_SIG - BH/2
vline(ax, CX, sig_bot, BUS1_Y, AMBER_E, lw=1.0)
hline(ax, AXPOS[0], AXPOS[2], BUS1_Y, AMBER_E, lw=1.0)
for cx_a in AXPOS:
    varrow(ax, cx_a, BUS1_Y, Y_CYC + ABH/2, color=AMBER_E, lw=1.0, ms=9)

# fan in: cycle boxes → phase
for cx_a in AXPOS:
    vline(ax, cx_a, Y_CYC - ABH/2, BUS2_Y, AMBER_E, lw=1.0)
hline(ax, AXPOS[0], AXPOS[2], BUS2_Y, AMBER_E, lw=1.0)
vline(ax, CX, BUS2_Y, Y_PHASE + BH/2, AMBER_E, lw=1.0)
varrow(ax, CX, BUS2_Y, Y_PHASE + BH/2, color=AMBER_E, lw=1.0, ms=9)

# ── phase box ─────────────────────────────────────────────────────────────────
rbox(ax, CX, Y_PHASE, BW, BH, "8-Phase Cycle Sampling")

# fork: phase → repr boxes
phase_bot = Y_PHASE - BH/2
vline(ax, CX, phase_bot, FORK_Y, PURPLE_E, lw=1.0)
hline(ax, LM_CX, IM_CX, FORK_Y, PURPLE_E, lw=1.0)
for cx_r in [LM_CX, IM_CX]:
    varrow(ax, cx_r, FORK_Y, Y_REPR + RBH/2, color=PURPLE_E, lw=1.0, ms=9)

# ── representation boxes ──────────────────────────────────────────────────────
rbox(ax, LM_CX, Y_REPR, RBW, RBH, "Landmark Representation",
     sub="13 joints × 4 coords × 8 phases",
     fc=PURPLE_F, ec=PURPLE_E, lw=1.2, fs=8.5, sub_fs=6.8)
rbox(ax, IM_CX, Y_REPR, RBW, RBH, "Image Representation",
     sub="8 × 224×224 RGB crops",
     fc=PURPLE_F, ec=PURPLE_E, lw=1.2, fs=8.5, sub_fs=6.8)

# merge: repr boxes → classifier
for cx_r in [LM_CX, IM_CX]:
    vline(ax, cx_r, Y_REPR - RBH/2, MRG2_Y, BLUE_E, lw=1.0)
hline(ax, LM_CX, IM_CX, MRG2_Y, BLUE_E, lw=1.0)
vline(ax, CX, MRG2_Y, Y_CLS + BH/2, BLUE_E, lw=1.0)
varrow(ax, CX, MRG2_Y, Y_CLS + BH/2, color=BLUE_E, lw=1.0, ms=9)

# ── classifier ────────────────────────────────────────────────────────────────
rbox(ax, CX, Y_CLS, BW, BH, "Deep Learning Classifier",
     sub="MLP / 1D-CNN / BiLSTM / CNN-BiLSTM / GRU")
varrow(ax, CX, Y_CLS - BH/2, Y_OUT + OUT_H/2)

# ── output ────────────────────────────────────────────────────────────────────
rbox(ax, CX, Y_OUT, BW + 0.30, OUT_H,
     "Cycle-Level Prediction  →  Video-Level Prediction",
     fc=GREEN_F, ec=GREEN_E, lw=1.5, fs=9.0)

# restrict axes to content bounds to eliminate bottom whitespace
ax.set_ylim(Y_OUT - OUT_H/2 - 0.25, H + 0.10)

# ── save ─────────────────────────────────────────────────────────────────────
out_dir = pathlib.Path(__file__).parent.parent
plt.savefig(out_dir / "fig31_pipeline.png", dpi=200,
            bbox_inches="tight", pad_inches=0.15, facecolor="white")
plt.savefig(out_dir / "fig31_pipeline.pdf",
            bbox_inches="tight", pad_inches=0.15, facecolor="white")
print(f"Saved: {out_dir / 'fig31_pipeline.png'}")
