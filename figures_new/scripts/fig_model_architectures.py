"""
fig_model_architectures.py
Layer-by-layer architecture diagram for the four landmark-based classifiers.
Exact architectures read from notebook 04_model_ablation.ipynb.
Output: figures_new/fig_model_architectures.png and .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pathlib

OUT = pathlib.Path(__file__).parent.parent

# ── colour palette (by layer kind) ───────────────────────────────────────────
C_IN   = ("#E0F2FE", "#0369A1")   # input  – sky blue
C_LIN  = ("#DBEAFE", "#1D4ED8")   # linear – blue
C_CONV = ("#FEF3C7", "#B45309")   # conv   – amber
C_LSTM = ("#EDE9FE", "#7C3AED")   # lstm   – violet
C_NORM = ("#F0FDF4", "#15803D")   # norm / pool – green
C_DROP = ("#F9FAFB", "#6B7280")   # dropout – gray
C_OUT  = ("#D1FAE5", "#065F46")   # output – emerald

PAD = 0.012  # fraction of axis units (xlim=0..1, ylim=0..1)

# ── layer drawing ─────────────────────────────────────────────────────────────
def draw_layer(ax, cx, cy, w, h, text, fc, ec, fs=7.5, bold=False, lw=1.4):
    """
    cx, cy  — centre (data units, xlim=0..1, ylim=0..1 top-down)
    w, h    — box width / height (data units)
    returns (top_y, bot_y)
    """
    inner_w = w - 2*PAD
    inner_h = h - 2*PAD
    rect = FancyBboxPatch(
        (cx - inner_w/2, cy - inner_h/2), inner_w, inner_h,
        boxstyle=f"round,pad={PAD}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3, clip_on=False
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight=("bold" if bold else "normal"),
            zorder=4, multialignment="center")
    return cy - h/2, cy + h/2   # top_y (smaller number), bot_y


def draw_arrow(ax, x, y_from_bot, y_to_top, ec):
    """Arrow from bottom of previous box to top of next box (y-down coords)."""
    ax.annotate("",
                xy=(x, y_to_top), xytext=(x, y_from_bot),
                arrowprops=dict(arrowstyle="-|>", color=ec,
                                linewidth=1.1, mutation_scale=8),
                zorder=5)


# ── column drawing ────────────────────────────────────────────────────────────
def draw_column(ax, title, layers_spec):
    """
    layers_spec: list of (label, (fc,ec), h_weight)
    Axis uses xlim=[0,1], ylim=[0,1] with y=0 at top.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()   # y=0 at top, y=1 at bottom
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)

    CX   = 0.50
    BW   = 0.82
    GAP  = 0.013          # gap between boxes (axis units)
    MARGIN_TOP = 0.020
    MARGIN_BOT = 0.015

    n   = len(layers_spec)
    h_weights = [hw for _, _, hw in layers_spec]
    total_w   = sum(h_weights)
    avail     = 1.0 - MARGIN_TOP - MARGIN_BOT - GAP * (n - 1)
    unit_h    = avail / total_w   # height of one h_weight=1 box

    cy = MARGIN_TOP
    prev_bot = None
    for text, (fc, ec), hw in layers_spec:
        h = hw * unit_h
        cy += h / 2
        top_y, bot_y = draw_layer(ax, CX, cy, BW, h, text,
                                   fc, ec, fs=6.9 if hw < 0.85 else 7.6,
                                   bold=(hw >= 1.0))
        if prev_bot is not None:
            draw_arrow(ax, CX, prev_bot, top_y, ec)
        prev_bot = bot_y
        cy += h / 2 + GAP


# ── figure layout ─────────────────────────────────────────────────────────────
FW, FH = 11.6, 8.6
fig, axes = plt.subplots(1, 4, figsize=(FW, FH))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.04,
                    wspace=0.06)

# ── MLP ──────────────────────────────────────────────────────────────────────
mlp_layers = [
    ("Input\n(B, 8, 78)",             C_IN,   1.10),
    ("Flatten → (B, 624)",             C_DROP, 0.75),
    ("Linear(624 → 256)\n+ ReLU",      C_LIN,  1.00),
    ("Dropout(0.3)",                   C_DROP, 0.65),
    ("Linear(256 → 128)\n+ ReLU",      C_LIN,  1.00),
    ("Dropout(0.3)",                   C_DROP, 0.65),
    ("Linear(128 → 1)",                C_LIN,  0.80),
    ("Output logit\n→ Sigmoid → P",    C_OUT,  1.05),
]
draw_column(axes[0], "MLP", mlp_layers)

# ── 1D-CNN ───────────────────────────────────────────────────────────────────
cnn_layers = [
    ("Input (B, 8, 78)\n→ transpose (B, 78, 8)", C_IN,   1.15),
    ("Conv1d(78→64, k=3)\n+ ReLU + BN(64)",       C_CONV, 1.00),
    ("Conv1d(64→128, k=3)\n+ ReLU + BN(128)",      C_CONV, 1.00),
    ("AdaptiveAvgPool1d(1)\n→ (B, 128)",            C_NORM, 0.85),
    ("Flatten → (B, 128)",             C_DROP, 0.72),
    ("Dropout(0.3)",                   C_DROP, 0.65),
    ("Linear(128 → 1)",                C_LIN,  0.80),
    ("Output logit\n→ Sigmoid → P",    C_OUT,  1.05),
]
draw_column(axes[1], "1D-CNN", cnn_layers)

# ── BiLSTM ───────────────────────────────────────────────────────────────────
bilstm_layers = [
    ("Input\n(B, 8, 78)",              C_IN,   1.10),
    ("BiLSTM\nF=78 → h=64\n2 layers, bidirectional\ndropout=0.3",
                                       C_LSTM, 1.55),
    ("Concat [h_fwd, h_bwd]\n→ (B, 128)",
                                       C_LSTM, 0.90),
    ("LayerNorm(128)",                 C_NORM, 0.70),
    ("Dropout(0.3)",                   C_DROP, 0.65),
    ("Linear(128 → 1)",                C_LIN,  0.80),
    ("Output logit\n→ Sigmoid → P",    C_OUT,  1.05),
]
draw_column(axes[2], "BiLSTM", bilstm_layers)

# ── CNN-BiLSTM ───────────────────────────────────────────────────────────────
cnnbilstm_layers = [
    ("Input (B, 8, 78)\n→ transpose (B, 78, 8)", C_IN,   1.15),
    ("Conv1d(78→64, k=3)\n+ ReLU + BN(64)",       C_CONV, 1.00),
    ("Transpose → (B, 8, 64)",         C_DROP, 0.72),
    ("BiLSTM\nF=64 → h=64\n1 layer, bidirectional",
                                       C_LSTM, 1.25),
    ("Concat [h_fwd, h_bwd]\n→ (B, 128)",
                                       C_LSTM, 0.90),
    ("LayerNorm(128)",                 C_NORM, 0.70),
    ("Dropout(0.3)",                   C_DROP, 0.65),
    ("Linear(128 → 1)",                C_LIN,  0.80),
    ("Output logit\n→ Sigmoid → P",    C_OUT,  1.05),
]
draw_column(axes[3], "CNN-BiLSTM", cnnbilstm_layers)

# ── footer note ───────────────────────────────────────────────────────────────
fig.text(0.50, 0.005,
         "Input: (B, N=8, F=78)  —  "
         "B: batch · N: 8 gait phases · "
         "F: 13 joints × (3 coords + 3 velocity)",
         ha="center", va="bottom", fontsize=8.0, color="#374151",
         style="italic")

plt.savefig(OUT / "fig_model_architectures.png", dpi=200,
            bbox_inches="tight", pad_inches=0.12, facecolor="white")
plt.savefig(OUT / "fig_model_architectures.pdf",
            bbox_inches="tight", pad_inches=0.12, facecolor="white")
print("Saved: fig_model_architectures.png")
