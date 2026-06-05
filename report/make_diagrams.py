"""Render Figs 1-2 as browser-laid-out HTML diagrams at true print scale (624px = 6.5in)."""

from __future__ import annotations

import base64
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report/figures"
ICON = FIG / "icons"

NAVY = "#33506e"
INKB = "#333333"


def b64(name: str) -> str:
    return base64.b64encode((ICON / f"{name}.png").read_bytes()).decode()


CSS = f"""
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: "Times New Roman", Times, serif; background: white; }}
.canvas {{ position: relative; background: white; }}

.card {{
  position: absolute; border: 1px solid {INKB}; background: #fff;
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 3px; text-align: center; padding: 4px 3px;
}}
.card img {{ height: 15px; margin-bottom: 1px; }}
.card .t {{ font-weight: bold; font-size: 10px; color: {NAVY}; }}
.card .b {{ font-size: 8.5px; color: #222; line-height: 1.35; }}

.lane {{ position: absolute; border: 1px solid #8a8a8a; background: #fff; }}
.lane .hdr {{
  height: 15px; background: {NAVY}; color: #fff;
  font-size: 7.5px; font-variant: small-caps; letter-spacing: 0.6px;
  display: flex; align-items: center; padding: 0 7px;
}}
svg.wires {{ position: absolute; inset: 0; pointer-events: none; }}
"""

ARROW_DEFS = """
<defs><marker id="ah" markerWidth="7" markerHeight="6" refX="5.6" refY="3"
 orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#333"/></marker></defs>
"""


def wire(points):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    return (f'<path d="{d}" fill="none" stroke="#333" stroke-width="1.1" '
            f'marker-end="url(#ah)"/>')


def card(x, y, w, h, icon, title, body, icon_h=15):
    return (f'<div class="card" style="left:{x}px;top:{y}px;width:{w}px;height:{h}px;">'
            f'<img style="height:{icon_h}px" src="data:image/png;base64,{b64(icon)}">'
            f'<div class="t">{title}</div><div class="b">{body}</div></div>')


# ================= Fig 2: two-lane architecture (624 x 296) =================
LANE_X, LANE_W = 118, 396
BOX_H = 74

ref_boxes, ref_x = [], []
bw, gap = 87, 12
x = LANE_X + 6
for icon, t, b in [("filter", "Phase 1", "cleaning<br>48 features"),
                   ("bubbles", "Phase 2", "K-Means<br>k = 8"),
                   ("tree", "Phases 3&ndash;4", "XGBoost<br>weights &middot; SHAP"),
                   ("search", "Phase 5", "IForest<br>+ LOF")]:
    ref_boxes.append(card(x, 40, bw, BOX_H, icon, t, b))
    ref_x.append(x)
    x += bw + gap

dist_boxes, dist_x = [], []
dw = 120
x = LANE_X + 6
for icon, t, b in [("hadoop", "HDFS warehouse", "9 datanodes &middot; repl. 2<br>division partitions"),
                   ("spark", "Spark SQL", "distributed cleaning<br>grid densities"),
                   ("spark", "Spark MLlib", "phases 2&ndash;5<br>regional runs")]:
    dist_boxes.append(card(x, 200, dw, BOX_H, icon, t, b))
    dist_x.append(x)
    x += dw + gap

ref_mid, dist_mid = 40 + BOX_H // 2, 200 + BOX_H // 2

wires = [
    # raw -> lanes (orthogonal elbows)
    wire([(100, 138), (109, 138), (109, ref_mid), (LANE_X + 4, ref_mid)]),
    wire([(100, 158), (109, 158), (109, dist_mid), (LANE_X + 4, dist_mid)]),
    # between boxes
    wire([(ref_x[0] + bw, ref_mid), (ref_x[1] - 1, ref_mid)]),
    wire([(ref_x[1] + bw, ref_mid), (ref_x[2] - 1, ref_mid)]),
    wire([(ref_x[2] + bw, ref_mid), (ref_x[3] - 1, ref_mid)]),
    wire([(dist_x[0] + dw, dist_mid), (dist_x[1] - 1, dist_mid)]),
    wire([(dist_x[1] + dw, dist_mid), (dist_x[2] - 1, dist_mid)]),
    # lanes -> deliverables (orthogonal elbows)
    wire([(LANE_X + LANE_W, ref_mid), (524, ref_mid), (524, 138), (532, 138)]),
    wire([(LANE_X + LANE_W, dist_mid), (524, dist_mid), (524, 158), (532, 158)]),
]

FIG2 = f"""
<div id="fig2" class="canvas" style="width:624px;height:296px;">
  <div class="lane" style="left:{LANE_X}px;top:16px;width:{LANE_W}px;height:106px;">
    <div class="hdr">Reference lane &mdash; scikit-learn (exact proposal algorithms)</div>
  </div>
  <div class="lane" style="left:{LANE_X}px;top:176px;width:{LANE_W}px;height:106px;">
    <div class="hdr">Distributed lane &mdash; Hadoop HDFS + Apache Spark (same phases)</div>
  </div>
  {"".join(ref_boxes)}{"".join(dist_boxes)}
  {card(8, 106, 92, BOX_H + 12, "database", "Raw dataset", "3 CSVs &middot; 206 MB<br>832,881 rows")}
  {card(532, 106, 84, BOX_H + 12, "desktop", "Deliverables", "dashboard &middot; report<br>audit lists")}
  <svg class="wires" width="624" height="296">{ARROW_DEFS}{"".join(wires)}</svg>
</div>
"""

# ================= Fig 1: six-step methodology (624 x 96) =================
steps = [
    ("database", "1. Dataset", "Mendeley data<br>832,881 rows"),
    ("dbimport", "2. Ingest", "HDFS, 9 nodes<br>by division"),
    ("settings", "3. Engineer", "GPS checks<br>48 features"),
    ("tree", "4. Model", "RQ1&ndash;RQ4<br>four methods"),
    ("gauge", "5. Evaluate", "Optuna, 5-fold<br>47 tests"),
    ("desktop", "6. Deliver", "dashboard<br>audit lists"),
]
sbw, sgap = 94, 12  # six boxes across 624px
cells, wires1 = [], []
x = 2
for i, (icon, t, b) in enumerate(steps):
    cells.append(card(x, 4, sbw, 108, icon, t, b, icon_h=20))
    if i < len(steps) - 1:
        wires1.append(wire([(x + sbw, 58), (x + sbw + sgap - 1, 58)]))
    x += sbw + sgap

FIG1 = f"""
<div id="fig1" class="canvas" style="width:624px;height:116px;">
  {"".join(cells)}
  <svg class="wires" width="624" height="116">{ARROW_DEFS}{"".join(wires1)}</svg>
</div>
"""

HTML = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
        f"<body>{FIG1}<div style='height:24px'></div>{FIG2}</body></html>")


def main() -> None:
    tmp = FIG / "_diagrams.html"
    tmp.write_text(HTML)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 700, "height": 900},
                          device_scale_factor=4)
        page.goto(tmp.as_uri())
        page.wait_for_timeout(400)
        page.locator("#fig1").screenshot(path=str(FIG / "f_overview.png"))
        page.locator("#fig2").screenshot(path=str(FIG / "f15_architecture.png"))
        b.close()
    tmp.unlink()
    print("wrote diagrams at print scale")


if __name__ == "__main__":
    main()
