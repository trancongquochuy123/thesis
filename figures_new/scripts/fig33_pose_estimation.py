"""
fig33_pose_estimation.py
Figure 3.3: MediaPipe BlazePose skeleton overlay on sample frame.
Output: figures_new/fig33_pose_estimation.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import pathlib

RAW = pathlib.Path(__file__).parent.parent / "_raw_data" / "pose_comparison"
OUT = pathlib.Path(__file__).parent.parent

img = Image.open(RAW / "DUNG22_mediapipe.jpg")

fig, ax = plt.subplots(figsize=(5.5, 4.0))
ax.imshow(img)
ax.axis("off")

ax.set_title("MediaPipe BlazePose-33", fontsize=10, fontweight="bold", pad=6)

# thin border frame
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color("#333333")

plt.tight_layout(pad=0.3)
plt.savefig(OUT / "fig33_pose_estimation.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved: fig33_pose_estimation.png")
