"""Reassemble the report from rewritten section chunks, with integrity checks.

Splices each report/_chunks_rewritten/<id>.html back into report_final.html at
the same marker boundaries, after verifying deterministically that (a) every
number of the original chunk survives in the rewrite, (b) tables are
byte-identical, and (c) headings are untouched. Refuses to splice a chunk
that fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report/report_final.html"

MARKERS = [
    ("abstract_intro", '<div class="abstract">', '<h2 class="sec">2. Review of Literature</h2>'),
    ("litreview", '<h2 class="sec">2. Review of Literature</h2>', '<h2 class="sec">3. Data</h2>'),
    ("data_31", '<h2 class="sec">3. Data</h2>', '<h3 class="sub">3.2 Data Transformation and Feature Engineering</h3>'),
    ("data_32_33", '<h3 class="sub">3.2 Data Transformation and Feature Engineering</h3>', '<h3 class="sub">3.4 Data Cleaning and Handling Missing Values</h3>'),
    ("data_34_35", '<h3 class="sub">3.4 Data Cleaning and Handling Missing Values</h3>', '<h2 class="sec">4. Methodologies and Experimental Results</h2>'),
    ("meth_41_42", '<h2 class="sec">4. Methodologies and Experimental Results</h2>', '<h3 class="sub">4.3 Supervised Learning: Sales-Target Regression (RQ2)</h3>'),
    ("meth_43", '<h3 class="sub">4.3 Supervised Learning: Sales-Target Regression (RQ2)</h3>', '<h3 class="sub">4.4 Supervised Learning: Target-Miss Risk Classification (RQ3)</h3>'),
    ("meth_44_45", '<h3 class="sub">4.4 Supervised Learning: Target-Miss Risk Classification (RQ3)</h3>', '<h3 class="sub">4.6 Distributed Execution on the Spark Cluster</h3>'),
    ("meth_46_48", '<h3 class="sub">4.6 Distributed Execution on the Spark Cluster</h3>', '<h2 class="sec">5. Conclusion</h2>'),
    ("conclusion", '<h2 class="sec">5. Conclusion</h2>', '<h2 class="sec">References</h2>'),
]

NUM = re.compile(r"\d(?:,?\d)*(?:\.\d+)?")
TAG = re.compile(r"<table.*?</table>", re.S)
HEAD = re.compile(r"<h[23][^>]*>(.*?)</h[23]>", re.S)


def numbers(text: str) -> list[str]:
    stripped = re.sub(r"<[^>]+>", " ", text)
    return sorted(NUM.findall(stripped))


def check(original: str, rewritten: str, name: str) -> list[str]:
    problems = []
    o_nums, r_nums = numbers(original), numbers(rewritten)
    missing = [n for n in set(o_nums) if o_nums.count(n) > r_nums.count(n)]
    if missing:
        problems.append(f"{name}: numbers missing in rewrite: {sorted(missing)[:10]}")
    o_tables, r_tables = TAG.findall(original), TAG.findall(rewritten)
    if len(o_tables) != len(r_tables):
        problems.append(f"{name}: table count changed {len(o_tables)} -> {len(r_tables)}")
    else:
        for i, (a, b) in enumerate(zip(o_tables, r_tables)):
            if re.sub(r"\s+", "", a) != re.sub(r"\s+", "", b):
                problems.append(f"{name}: table #{i+1} modified")
    if HEAD.findall(original) != HEAD.findall(rewritten):
        problems.append(f"{name}: heading text changed")
    for phrase in ("delve", "It is worth noting", "leverage", "utilize ", "seamlessly"):
        if phrase.lower() in re.sub(r"<[^>]+>", " ", rewritten).lower():
            problems.append(f"{name}: banned phrase present: {phrase!r}")
    return problems


def main() -> None:
    src = SRC.read_text()
    all_problems, spliced = [], 0
    for cid, a, b in MARKERS:
        rew_path = ROOT / f"report/_chunks_rewritten/{cid}.html"
        if not rew_path.exists():
            all_problems.append(f"{cid}: rewrite missing")
            continue
        original = ROOT.joinpath(f"report/_chunks/{cid}.html").read_text()
        rewritten = rew_path.read_text()
        problems = check(original, rewritten, cid)
        if problems:
            all_problems.extend(problems)
            continue
        i, j = src.index(a), src.index(b)
        src = src[:i] + rewritten + src[j:]
        spliced += 1
    if all_problems:
        print("PROBLEMS:")
        for p in all_problems:
            print(" -", p)
    SRC.write_text(src)
    print(f"spliced {spliced}/{len(MARKERS)} chunks")
    sys.exit(1 if all_problems else 0)


if __name__ == "__main__":
    main()
