"""Render Figs 1-2 as classic monochrome paper diagrams in matplotlib.

Canvas coordinates are print pixels (624 px = 6.5 in) with the y-axis pointing
down, so the layout numbers read like the print sheet. 300 dpi output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

INK = "#000000"
PX = 96.0  # canvas px per inch: 624 px spans 6.5 in


def canvas(w_px: int, h_px: int):
    fig, ax = plt.subplots(figsize=(w_px / PX, h_px / PX))
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)  # y grows downward, like the print sheet
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def node(ax, x, y, w, h, title, lines, fill="#ffffff"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK,
                           linewidth=0.9, zorder=2))
    cy = y + h / 2
    ax.text(x + w / 2, cy - 9, title, ha="center", va="center",
            fontsize=8.5, fontweight="bold", zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, cy + 4 + 11 * i, line, ha="center", va="center",
                fontsize=7.5, zorder=3)


def group(ax, x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=INK,
                           linewidth=0.7, linestyle=(0, (4, 2.5)), zorder=1))
    ax.text(x + 7, y + 11, label, ha="left", va="center",
            fontsize=8, fontstyle="italic", zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2))


def wire(ax, points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    ax.plot(xs[:-1] + [xs[-1]], ys[:-1] + [ys[-1]], color=INK, lw=0.9,
            solid_joinstyle="miter", zorder=1)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9,
                                shrinkA=0, shrinkB=0, mutation_scale=8))


# ================= Fig 1: six-step methodology (624 x 116) =================
STEPS = [
    ("1. Dataset", ["Mendeley data", "832,881 rows"]),
    ("2. Ingest", ["HDFS, 9 nodes", "by division"]),
    ("3. Engineer", ["GPS checks", "48 features"]),
    ("4. Model", ["RQ1–RQ4", "four methods"]),
    ("5. Evaluate", ["Optuna, 5-fold", "47 tests"]),
    ("6. Deliver", ["dashboard", "audit lists"]),
]
fig, ax = canvas(624, 116)
sbw, sgap, sy, sh = 93, 12, 14, 88
x = 3
for i, (title, lines) in enumerate(STEPS):
    node(ax, x, sy, sbw, sh, title, lines)
    if i < len(STEPS) - 1:
        wire(ax, [(x + sbw, sy + sh / 2), (x + sbw + sgap - 1, sy + sh / 2)])
    x += sbw + sgap
fig.savefig(FIG / "f_overview.png")
plt.close(fig)
print("wrote f_overview.png")

# ================= Fig 2: two-lane architecture (624 x 296) =================
GRP_X, GRP_W = 118, 396
BOX_H = 74

fig, ax = canvas(624, 296)
group(ax, GRP_X, 16, GRP_W, 106, "Reference: scikit-learn")
group(ax, GRP_X, 176, GRP_W, 106, "Distributed: Hadoop HDFS + Apache Spark")

ref_x, bw, gap = [], 87, 12
x = GRP_X + 6
for title, lines in [("Phase 1", ["cleaning", "48 features"]),
                     ("Phase 2", ["K-Means", "k = 8"]),
                     ("Phases 3–4", ["XGBoost", "weights · SHAP"]),
                     ("Phase 5", ["IForest", "+ LOF"])]:
    node(ax, x, 40, bw, BOX_H, title, lines)
    ref_x.append(x)
    x += bw + gap

dist_x, dw = [], 120
x = GRP_X + 6
for title, lines in [("HDFS warehouse", ["9 datanodes · repl. 2", "division partitions"]),
                     ("Spark SQL", ["distributed cleaning", "grid densities"]),
                     ("Spark MLlib", ["phases 2–5", "regional runs"])]:
    node(ax, x, 200, dw, BOX_H, title, lines)
    dist_x.append(x)
    x += dw + gap

node(ax, 8, 106, 92, BOX_H + 12, "Raw dataset",
     ["3 CSVs · 206 MB", "832,881 rows"], fill="#f2f2f2")
node(ax, 532, 106, 84, BOX_H + 12, "Deliverables",
     ["dashboard · report", "audit lists"], fill="#f2f2f2")

ref_mid, dist_mid = 40 + BOX_H / 2, 200 + BOX_H / 2
# raw -> groups (orthogonal elbows)
wire(ax, [(100, 138), (109, 138), (109, ref_mid), (GRP_X - 1, ref_mid)])
wire(ax, [(100, 158), (109, 158), (109, dist_mid), (GRP_X - 1, dist_mid)])
# between boxes
for i in range(3):
    wire(ax, [(ref_x[i] + bw, ref_mid), (ref_x[i + 1] - 1, ref_mid)])
for i in range(2):
    wire(ax, [(dist_x[i] + dw, dist_mid), (dist_x[i + 1] - 1, dist_mid)])
# groups -> deliverables (orthogonal elbows)
wire(ax, [(GRP_X + GRP_W + 1, ref_mid), (524, ref_mid), (524, 138), (531, 138)])
wire(ax, [(GRP_X + GRP_W + 1, dist_mid), (524, dist_mid), (524, 158), (531, 158)])

fig.savefig(FIG / "f15_architecture.png")
plt.close(fig)
print("wrote f15_architecture.png")
