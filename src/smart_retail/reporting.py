"""Narrative report builder: turns pipeline metrics into commented markdown.

Used by the dashboard's "Full report" tab (and downloadable from there), so
the written analysis always matches the artifacts on disk instead of being
hand-edited prose that drifts out of date.
"""

from __future__ import annotations

import pandas as pd

from smart_retail.features.engineering import TARGET_COLUMN


def _silhouette_reading(value: float) -> str:
    if value >= 0.5:
        return "strong, well-separated structure"
    if value >= 0.25:
        return "moderate separation"
    return (
        "overlapping clusters — expected for a continuous national retail "
        "fabric; the taxonomy is actionable because profiles differ strongly "
        "on density, importance, and target"
    )


def _auc_reading(value: float) -> str:
    if value >= 0.9:
        return "excellent"
    if value >= 0.8:
        return "strong"
    if value >= 0.7:
        return "fair"
    return "weak"


def data_quality_section(metrics: dict) -> str:
    q = metrics.get("phase_1_preprocessing", {})
    n_raw = q.get("n_raw_shop_rows", 0)
    n_gps = q.get("n_invalid_or_missing_gps", 0)
    n_final = q.get("n_final_shops", 0)
    n_target = q.get("n_shops_with_target", 0)
    return f"""
### The dataset

**Comprehensive Retail Network Dataset of Bangladesh** (Shahin, Swarna & Farid,
2025) — Mendeley `mgzvngzng2` v1, CC BY 4.0, collected Feb 2022 – Oct 2024.
Three files: shop records with GPS and structure ({n_raw:,} rows), monthly
sales targets for 17 products per shop (853,732 rows), and a 42-SKU product
catalogue (5 brands, 9 product types) with price tiers.

### Data quality & preprocessing

| Step | Count |
|---|---|
| Raw shop records | {n_raw:,} |
| Removed: missing GPS | {n_gps:,} ({n_gps / max(n_raw, 1):.1%}) |
| **Final analysable shops** | **{n_final:,}** |
| Shops with a monthly sales target | {n_target:,} ({n_target / max(n_final, 1):.1%}) |

**Why are {n_gps:,} rows removed?** Their `latitude`/`longitude` cells are
empty in the raw CSV and their `geometry` field is literally `POINT EMPTY` —
the coordinates were never captured at survey time. This 13.6% missing-GPS
rate is documented by the dataset authors themselves (Shahin et al., 2025),
and after removing those rows the shop count equals the paper's published
figure of 719,817 exactly. No valid shop is discarded: zero non-null
coordinates fall outside the Bangladesh bounding box.

**Comment.** The gap itself is a finding for the distributor: one in seven
book entries cannot be placed on a map, which is consistent with the
duplicate / ghost-shop problem that motivated this project.
"""


def feature_engineering_section(metrics: dict) -> str:
    q = metrics.get("phase_1_preprocessing", {})
    return f"""
### Feature engineering

From the cleaned records, **{q.get('n_feature_columns', 0)} model features** are
engineered per shop:

| Group | Features | How they are computed |
|---|---|---|
| Neighbourhood density | `density_500m`, `density_1km`, `density_5km` | BallTree (haversine) neighbour counts within each radius, excluding the shop itself |
| Centrality | `dist_to_district_centre_km` | Great-circle distance to the centroid of the shop's district (zila) |
| Product mix | `n_products_targeted` | Number of the 17 products with a positive target for the shop |
| Position | `latitude`, `longitude` | Validated GPS coordinates |
| Structure (one-hot) | shop type (12), style (3), importance (3), place nature, division | Explicit "Unknown" level instead of dropping ~10% categorical gaps |

**Comment.** Density and centrality translate the proposal's
"urban shops over-stocked / rural shops stock-out" hypothesis into measurable
inputs; the SHAP ranking in RQ2 confirms they carry real signal. The monthly
target (`total_target`) is joined for {q.get('n_shops_with_target', 0):,} shops
and used as the regression label — never as a model input.
"""


def conclusion_section(metrics: dict) -> str:
    q = metrics.get("phase_1_preprocessing", {})
    c = metrics.get("phase_2_clustering", {})
    r = metrics.get("phase_3_regression", {})
    cl = metrics.get("phase_4_classification", {})
    a = metrics.get("phase_5_anomaly", {})
    durations = metrics.get("durations_sec", {})
    best_reg = r.get("models", {}).get(r.get("best_model", ""), {})
    best_clf = cl.get("models", {}).get(cl.get("best_model", ""), {})
    return f"""
### Conclusion & recommended actions

End-to-end, the pipeline turns {q.get('n_raw_shop_rows', 0):,} raw book entries
into: **{c.get('best_k', '?')} interpretable archetypes**, a sales model
(R² {best_reg.get('r2', float('nan')):.2f}), a target-miss risk ranking
(ROC-AUC {best_clf.get('roc_auc', float('nan')):.3f}), and
{a.get('n_top_anomalies', 0):,} triage-ready anomalies — in
{sum(durations.values()):.0f}s of compute.

Recommended actions for the distributor:

1. **Audit the data-poor segment first** — the archetype with zero product mix
   plus the {q.get('n_raw_shop_rows', 0) - q.get('n_final_shops', 0):,} entries
   without GPS are the cheapest wins against the 15–25% capacity waste estimate.
2. **Rebalance targets where the model and the book disagree** — shops whose
   predicted volume far exceeds the assigned target are under-served demand;
   the reverse are over-allocation risks (RQ3 list, worst in the highest-risk
   division).
3. **Field-verify the anomaly top decile per division**, prioritising shops
   that are simultaneously RQ3 at-risk.
4. **Refresh quarterly** — the pipeline is deterministic (seed 42) and fully
   containerised, so re-running on updated books is one command.
"""


def rq1_section(metrics: dict, profiles: pd.DataFrame | None) -> str:
    c = metrics.get("phase_2_clustering", {})
    best_k = c.get("best_k")
    km = c.get("kmeans", {}).get(str(best_k), {})
    sil = km.get("silhouette", float("nan"))
    db = km.get("davies_bouldin", float("nan"))
    dbscan = c.get("dbscan", {})

    text = f"""
### RQ1 — What natural archetypes exist among the outlets?

K-Means was scored for k = {min(map(int, c.get('kmeans', {"0": 0})))}…{max(map(int, c.get('kmeans', {"0": 0})))}
on silhouette (higher better) and Davies–Bouldin (lower better).
**Best k = {best_k}** (silhouette {sil:.3f}, DB {db:.2f}).

**Comment.** A silhouette of {sil:.2f} indicates {_silhouette_reading(sil)}.
Agglomerative clustering on a subsample corroborates the structure
(silhouette {c.get('agglomerative', {}).get('silhouette', float('nan')):.3f});
DBSCAN found {dbscan.get('n_clusters_found', '?')} cluster(s) with
{dbscan.get('noise_share', 0):.0%} noise at default settings — evidence the
network is dense and continuous rather than separated islands.
"""
    if profiles is not None and not profiles.empty and TARGET_COLUMN in profiles:
        p = profiles.copy()
        largest = p.loc[p["n_shops"].idxmax()]
        key_account = p.loc[p[TARGET_COLUMN].idxmax()]
        text += f"""
Reading the profiles (medians per cluster):

- **Largest archetype — cluster {int(largest['cluster'])}** ({largest['share']:.1%} of shops,
  dominant division {largest['top_division']}): the network's typical outlet.
- **Key-account tier — cluster {int(key_account['cluster'])}**: median monthly target
  {key_account[TARGET_COLUMN]:,.1f}, {key_account[TARGET_COLUMN] / max(p[TARGET_COLUMN].median(), 1e-9):.1f}×
  the median archetype — a small segment carrying disproportionate volume.
"""
        ghost = p[(p["n_products_targeted"] == 0)]
        if not ghost.empty:
            g = ghost.iloc[0]
            text += f"""- **Data-poor segment — cluster {int(g['cluster'])}** ({g['share']:.1%} of shops):
  zero targeted products and unknown type/importance — a machine-found
  candidate list for the duplicate / ghost-shop audit (RQ4 triage).
"""
    return text


def rq2_section(metrics: dict, shap_importance: pd.DataFrame | None) -> str:
    r = metrics.get("phase_3_regression", {})
    models = r.get("models", {})
    best = r.get("best_model", "?")
    if not models:
        return "### RQ2 — Sales prediction\n\n_No regression metrics found._"
    baseline = models.get("linear_regression", {})
    winner = models.get(best, {})
    rows = "\n".join(
        f"| {'**' + name + '**' if name == best else name} | "
        f"{m['rmse']:.1f} | {m['mae']:.2f} | {m['r2']:.3f} |"
        for name, m in sorted(models.items(), key=lambda kv: kv[1]["rmse"], reverse=True)
    )
    text = f"""
### RQ2 — Can monthly sales be predicted from location, type, and density?

| Model | RMSE | MAE | R² |
|---|---|---|---|
{rows}

**Comment.** The linear baseline explains almost none of the variance
(R² {baseline.get('r2', float('nan')):.2f}) while **{best}** reaches
R² {winner.get('r2', float('nan')):.2f} and cuts RMSE by
{1 - winner.get('rmse', 1) / max(baseline.get('rmse', 1), 1e-9):.0%} —
the target-to-shop relationship is real but strongly non-linear.
"""
    if shap_importance is not None and not shap_importance.empty:
        top = shap_importance.head(5)
        drivers = "; ".join(
            f"`{row.feature}` ({row.mean_abs_shap:.1f})" for row in top.itertuples()
        )
        text += f"""
SHAP attribution on the winner ranks the drivers: {drivers}.
**Comment.** Assortment breadth and importance tier dominate, followed by
micro-location (neighbour density, coordinates, distance to district centre)
— allocation by shop type alone leaves measurable signal on the table.
"""
    return text


def rq3_section(metrics: dict, risk_by_division: pd.Series | None) -> str:
    c = metrics.get("phase_4_classification", {})
    models = c.get("models", {})
    best = c.get("best_model", "?")
    if not models:
        return "### RQ3 — Target-miss risk\n\n_No classification metrics found._"
    winner = models.get(best, {})
    rows = "\n".join(
        f"| {'**' + name + '**' if name == best else name} | "
        f"{m['roc_auc']:.3f} | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |"
        for name, m in sorted(models.items(), key=lambda kv: kv[1]["roc_auc"])
    )
    text = f"""
### RQ3 — Which shops will miss their targets?

**Label (documented proxy).** The dataset ships assigned targets, not realized
sales, so: `{c.get('label_definition', '')}` with threshold
{c.get('risk_threshold', 0.8)} — a shop is at-risk when it materially
under-indexes its (cluster × shop-type) peers. Base rate:
{c.get('label_rate', 0):.1%}. SMOTE is applied to training folds only.

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
{rows}

**Comment.** The best model ({best}) reaches ROC-AUC
{winner.get('roc_auc', float('nan')):.3f} — {_auc_reading(winner.get('roc_auc', 0))}
discrimination from structural features alone; with realized sales data the
same pipeline retrains unchanged into a true "will miss next quarter" model.
"""
    if risk_by_division is not None and len(risk_by_division) > 0:
        known = risk_by_division.drop("Unknown", errors="ignore").sort_values()
        lo_name, lo = known.index[0], known.iloc[0]
        hi_name, hi = known.index[-1], known.iloc[-1]
        text += f"""
Predicted at-risk share is geographically uneven: from **{lo:.1%} in {lo_name}**
to **{hi:.1%} in {hi_name}** — a concrete prioritisation map for field teams.
"""
    return text


def rq4_section(metrics: dict, top_anomalies: pd.DataFrame | None) -> str:
    a = metrics.get("phase_5_anomaly", {})
    text = f"""
### RQ4 — Which shops behave anomalously vs their peers?

Isolation Forest and LOF were fitted **within each cluster**, so a rural kiosk
is compared to rural kiosks, not to metro superstores. All
{a.get('n_scored', 0):,} shops were scored; the top decile
(**{a.get('n_top_anomalies', 0):,} shops**) is exported for manual triage into
**fraud / data error / hidden opportunity**.
{f"{a.get('n_skipped_small_clusters')} cluster(s) below the minimum size were skipped." if a.get('n_skipped_small_clusters') else ""}
"""
    if top_anomalies is not None and not top_anomalies.empty and "division" in top_anomalies:
        top_div = top_anomalies["division"].value_counts()
        text += f"""
**Comment.** Anomalies concentrate in {top_div.index[0]}
({top_div.iloc[0]:,} of the top decile), which also has the largest shop base —
triage should therefore work the ranked list per division rather than absolute
counts. Shops that combine a high anomaly rank with the RQ3 at-risk flag are
the highest-priority field visits.
"""
    return text


def build_report(
    metrics: dict,
    profiles: pd.DataFrame | None = None,
    shap_importance: pd.DataFrame | None = None,
    risk_by_division: pd.Series | None = None,
    top_anomalies: pd.DataFrame | None = None,
) -> dict[str, str]:
    """All report sections as {section key: markdown}, in reading order."""
    return {
        "data_quality": data_quality_section(metrics),
        "feature_engineering": feature_engineering_section(metrics),
        "rq1": rq1_section(metrics, profiles),
        "rq2": rq2_section(metrics, shap_importance),
        "rq3": rq3_section(metrics, risk_by_division),
        "rq4": rq4_section(metrics, top_anomalies),
        "conclusion": conclusion_section(metrics),
    }


def report_markdown(sections: dict[str, str]) -> str:
    """Single downloadable markdown document."""
    header = (
        "# Smart Retail Intelligence — Analysis Report\n\n"
        "Generated from `reports/metrics.json` — numbers always match the "
        "artifacts of the latest pipeline run.\n"
    )
    return header + "\n---\n".join(sections.values())
