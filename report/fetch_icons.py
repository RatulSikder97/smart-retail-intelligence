"""Fetch real technology icons (Simple Icons brands + Tabler outline set) and rasterize to PNG."""

from __future__ import annotations

import base64
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/figures/icons"
OUT.mkdir(parents=True, exist_ok=True)

ICONS = {
    # brand logos (Simple Icons, CC0) in official brand colours
    "hadoop": "https://cdn.simpleicons.org/apachehadoop/1B7EB8",
    "dbimport": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/database-import.svg",
    "settings": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/settings.svg",
    "spark": "https://cdn.simpleicons.org/apachespark/E25A1C",
    "sklearn": "https://cdn.simpleicons.org/scikitlearn/F7931E",
    "streamlit": "https://cdn.simpleicons.org/streamlit/FF4B4B",
    "pandas": "https://cdn.simpleicons.org/pandas/150458",
    # generic concepts (Tabler outline icons, MIT)
    "database": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/database.svg",
    "filter": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/filter.svg",
    "bubbles": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/chart-bubble.svg",
    "tree": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/binary-tree.svg",
    "gauge": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/gauge.svg",
    "search": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/zoom-scan.svg",
    "desktop": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/device-desktop-analytics.svg",
    "report": "https://raw.githubusercontent.com/tabler/tabler-icons/main/icons/outline/report-analytics.svg",
}

TABLER_STROKE = "#2f3e4e"  # dark slate for the outline icons


def main() -> None:
    svgs = {}
    for name, url in ICONS.items():
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        svg = r.text
        if "tabler" in url:
            svg = svg.replace('stroke="currentColor"', f'stroke="{TABLER_STROKE}"')
        svgs[name] = svg
        print("fetched", name)

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 300, "height": 300})
        for name, svg in svgs.items():
            data = base64.b64encode(svg.encode()).decode()
            page.set_content(
                f'<body style="margin:0"><img id="i" width="256" height="256" '
                f'src="data:image/svg+xml;base64,{data}"></body>'
            )
            page.wait_for_timeout(250)
            page.locator("#i").screenshot(path=str(OUT / f"{name}.png"),
                                          omit_background=True)
            print("rendered", name)
        b.close()


if __name__ == "__main__":
    main()
