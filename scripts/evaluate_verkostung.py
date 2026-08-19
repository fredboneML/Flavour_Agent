"""First-glance evaluation of all scoring methods against the 25-06-2026 Verkostung panel.

Recomputes the 15 scoring variants (identical to notebooks/scoring_ms_assessment.ipynb) on the
new all-fruit PDM export, then scores each method against the panel ground truth in
`Ergebnisse Verkostung 25_06_2026.xlsx`, using two truth definitions:
  Method 1 = M1 Label Propagation (column B),  Method 2 = M2 Rule Based (column C).

True-label rule per recipe (Prozent % = column H):
  - H >= 60 : true label = the single value in B (method 1) / C (method 2).
  - H <  60 : acceptable labels = tokens in D (Cluster vorgegeben) and E (Jan Free Sorting)
              that are literally one of the 7 cluster names; if none, the recipe is ignored.
A prediction is correct if it equals ANY label in the recipe's acceptable set (lenient match).

Output: outputs/verkostung_eval_25_06_2026.xlsx

Usage:
    python3 scripts/evaluate_verkostung.py
"""

import re
from pathlib import Path

import openpyxl
import pandas as pd

from scripts.ms_scoring import (
    load_weight_matrix,
    load_recipes,
    load_rohstoffe_weights,
    load_avg_meli,
    compute_scores,
    assign_clusters,
    map_to_panel,
    apply_zscore_quantities,
    central_amount_when_present,
    apply_squared_zscore_quantities,
    rescale_warm_weights,
    compute_tfidf_scores,
    compute_topk_scores,
    compute_cosine_scores,
    ensemble_scores,
)

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "gold"

SCORING_XLSX = GOLD / "Scoring Index_Beispielrechung.xlsx"
PDM_CSV = GOLD / "PDM_Rezepturen_Gesamt_nurP_18_08_2026.csv"
IGNORE_PATH = GOLD / "ignone_substances.csv"
VERK_XLSX = GOLD / "Ergebnisse Verkostung 25_06_2026.xlsx"
OUT_XLSX = ROOT / "outputs" / "verkostung_eval_25_06_2026.xlsx"

# MS cluster name -> panel/expert cluster name (copied from notebook cell 2).
MS_TO_PANEL = {
    "warm": "warm",
    "Unpleasant": "unpleasant",
    "green": "green",
    "floral": "floral",
    "citrus": "fruity",
    # "exotic" and "Outlayer" have no panel equivalent -> map() yields NaN
}

# The 7 expert/panel cluster names (lowercase) used as the truth vocabulary.
SEVEN = {"warm", "floral", "walderdbeere", "green", "dairy", "unpleasant", "fruity"}
# Clusters a z-method can actually output (via MS_TO_PANEL).
REACHABLE = {"warm", "unpleasant", "green", "floral", "fruity"}

Z_METHODS = {
    "MS Z-Score", "Z-Score mean", "Z-Score median", "Warm + Z-Score",
    "Squared Z-Score", "Warm + Sq.Z-Score", "Ensemble orig+Z", "Ensemble warm+Z",
}


def base_id(x: object) -> str:
    """Strip a trailing 'P' variant marker for cross-file recipe matching."""
    return re.sub(r"P$", "", str(x).strip())


def norm_label(v: object) -> str:
    """Lowercase, strip whitespace, drop '(...)' suffixes, map German 'grün' -> 'green'."""
    if v is None:
        return ""
    s = str(v).split("(")[0].strip().lower()
    return "green" if s == "grün" else s


def split_tokens(v: object) -> list[str]:
    """Split a D/E cell on '/' and '-' into normalized tokens."""
    if v is None:
        return []
    return [norm_label(t) for t in re.split(r"[/\-]", str(v)) if t.strip()]


# ── 1. Recompute the 15 method score matrices (mirrors notebook cell 37) ──────
def build_variants():
    weights = load_weight_matrix(SCORING_XLSX)
    recipes_df = load_recipes(PDM_CSV, IGNORE_PATH)
    avg_meli = load_avg_meli(SCORING_XLSX, csv_path=PDM_CSV)
    weights_rohst = load_rohstoffe_weights(SCORING_XLSX)
    weights_warm = rescale_warm_weights(weights)

    scores = compute_scores(recipes_df, weights)
    scores_z = compute_scores(apply_zscore_quantities(recipes_df, avg_meli), weights)
    # Data-derived Z-Scores: divide amounts by the mean / median amount-when-present per CAS,
    # computed from the recipe dataset itself (median is robust to outlier recipes).
    mean_present = central_amount_when_present(recipes_df, "mean")
    median_present = central_amount_when_present(recipes_df, "median")
    scores_z_mean = compute_scores(apply_zscore_quantities(recipes_df, mean_present), weights)
    scores_z_median = compute_scores(apply_zscore_quantities(recipes_df, median_present), weights)
    scores_rohst = compute_scores(recipes_df, weights_rohst)
    sc_warm = compute_scores(recipes_df, weights_warm)
    sc_warm_z = compute_scores(apply_zscore_quantities(recipes_df, avg_meli), weights_warm)
    recipes_z2 = apply_squared_zscore_quantities(recipes_df, avg_meli)
    sc_z2 = compute_scores(recipes_z2, weights)
    sc_warm_z2 = compute_scores(recipes_z2, weights_warm)
    sc_tfidf = compute_tfidf_scores(recipes_df, weights)
    sc_tfidf_warm = compute_tfidf_scores(recipes_df, weights_warm)
    sc_top5 = compute_topk_scores(recipes_df, weights, k=5)
    sc_top5_warm = compute_topk_scores(recipes_df, weights_warm, k=5)
    sc_cosine = compute_cosine_scores(recipes_df, weights)
    sc_cosine_warm = compute_cosine_scores(recipes_df, weights_warm)
    sc_ens_oz = ensemble_scores(scores, scores_z)
    sc_ens_wz = ensemble_scores(sc_warm, sc_warm_z)

    return {
        "MS original": scores,
        "MS Z-Score": scores_z,
        "Z-Score mean": scores_z_mean,
        "Z-Score median": scores_z_median,
        "MS Rohstoffe": scores_rohst,
        "Warm rescale": sc_warm,
        "Warm + Z-Score": sc_warm_z,
        "Squared Z-Score": sc_z2,
        "Warm + Sq.Z-Score": sc_warm_z2,
        "TF-IDF": sc_tfidf,
        "TF-IDF + warm": sc_tfidf_warm,
        "Top-5": sc_top5,
        "Top-5 + warm": sc_top5_warm,
        "Cosine": sc_cosine,
        "Cosine + warm": sc_cosine_warm,
        "Ensemble orig+Z": sc_ens_oz,
        "Ensemble warm+Z": sc_ens_wz,
    }


# ── 2. Build the truth table from the Verkostung file ─────────────────────────
def build_truth() -> pd.DataFrame:
    wb = openpyxl.load_workbook(VERK_XLSX, data_only=True)
    ws = wb["Tabelle1"]
    rows = []
    for r in range(3, ws.max_row + 1):
        rec = ws.cell(r, 1).value
        if rec is None:
            continue
        B, C, D, E, H = (ws.cell(r, c).value for c in (2, 3, 4, 5, 8))
        if H is None or not isinstance(H, (int, float)):
            continue
        if H >= 60:
            m1 = {norm_label(B)} - {""}
            m2 = {norm_label(C)} - {""}
        else:
            de = {t for t in (split_tokens(D) + split_tokens(E)) if t in SEVEN}
            m1 = m2 = de
        rows.append(
            {
                "Recipe": str(rec).strip(),
                "base": base_id(rec),
                "H": H,
                "B_M1": B,
                "C_M2": C,
                "D": D,
                "E": E,
                "M1_set": sorted(m1),
                "M2_set": sorted(m2),
            }
        )
    return pd.DataFrame(rows)


def accuracy(preds_by_method: dict, truth: pd.DataFrame, set_col: str) -> pd.DataFrame:
    """Per-method accuracy: prediction correct if in the recipe's acceptable label set."""
    evalset = truth[truth[set_col].map(len) > 0]
    out = []
    for method, panel_pred in preds_by_method.items():
        corr = tot = r_corr = r_tot = 0
        for _, row in evalset.iterrows():
            pid = row["pdm_id"]
            if pid is None:
                continue
            pred = panel_pred.get(pid)
            pred = None if (pred is None or (isinstance(pred, float))) else str(pred).lower()
            labels = set(row[set_col])
            hit = pred in labels
            tot += 1
            corr += hit
            if labels & REACHABLE:  # reachable-only view
                r_tot += 1
                r_corr += hit
        out.append(
            {
                "Method": method,
                "z_method": method in Z_METHODS,
                "Correct": corr,
                "Total": tot,
                "Accuracy_%": round(100 * corr / tot, 1) if tot else None,
                "Correct_reachable": r_corr,
                "Total_reachable": r_tot,
                "Accuracy_reachable_%": round(100 * r_corr / r_tot, 1) if r_tot else None,
            }
        )
    return pd.DataFrame(out).sort_values("Accuracy_%", ascending=False, ignore_index=True)


def per_cluster_accuracy(
    preds_by_method: dict, truth: pd.DataFrame, set_col: str, reachable_only: bool = False
) -> pd.DataFrame:
    """Accuracy per TRUE cluster × method: read down a method column to see which clusters it
    nails vs misses. Rows = true label (joined ``set_col``), columns = n + one % per method.

    reachable_only: drop recipes whose true cluster has no weight column (i.e. dairy /
    Walderdbeere — clusters the methods can never predict), so scores reflect only reachable cases.
    """
    evalset = truth[truth[set_col].map(len) > 0].copy()
    if reachable_only:
        evalset = evalset[evalset[set_col].map(lambda s: bool(set(s) & REACHABLE))].copy()
    evalset["_group"] = evalset[set_col].map(lambda s: ", ".join(s))
    rows = []
    for group, g in evalset.groupby("_group"):
        row = {"True_cluster": group, "n": len(g)}
        for method, panel_pred in preds_by_method.items():
            corr = 0
            for _, r in g.iterrows():
                pred = panel_pred.get(r["pdm_id"])
                pred = None if (pred is None or isinstance(pred, float)) else str(pred).lower()
                corr += pred in set(r[set_col])
            row[method] = round(100 * corr / len(g), 0)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)
    return df


def main() -> None:
    variants = build_variants()
    n_recipes = len(next(iter(variants.values())))
    print(f"Recomputed {len(variants)} methods over {n_recipes} recipes")

    # Raw MS argmax cluster + panel-mapped prediction per method.
    ms_cluster = {m: assign_clusters(sc) for m, sc in variants.items()}
    panel_pred = {m: map_to_panel(a, MS_TO_PANEL) for m, a in ms_cluster.items()}

    truth = build_truth()
    pdm_base = {base_id(x): str(x).strip() for x in ms_cluster["MS original"].index}
    truth["in_pdm"] = truth["base"].isin(pdm_base)
    truth["pdm_id"] = truth["base"].map(pdm_base)
    truth["evaluable"] = truth["in_pdm"] & (truth["M1_set"].map(len) > 0)

    n_eval = int(truth["evaluable"].sum())
    print(f"Verkostung recipes: {len(truth)}  |  in PDM: {int(truth['in_pdm'].sum())}  |  evaluable subset = {n_eval}")

    eval_truth = truth[truth["evaluable"]].copy()
    acc_m1 = accuracy(panel_pred, eval_truth, "M1_set")
    acc_m2 = accuracy(panel_pred, eval_truth, "M2_set")

    # Recipes whose truth set has no reachable cluster (only dairy/Walderdbeere) — the
    # z-methods can never predict these, so they are excluded from the reachable-only view.
    def _unreachable(labels: list[str]) -> bool:
        return len(labels) > 0 and not (set(labels) & REACHABLE)

    excluded_rows = []
    for _, row in eval_truth.iterrows():
        for m, col in (("M1", "M1_set"), ("M2", "M2_set")):
            if _unreachable(row[col]):
                excluded_rows.append(
                    {"Recipe": row["Recipe"], "truth_method": m,
                     "H": row["H"], "truth": ", ".join(row[col])}
                )
    excluded_df = pd.DataFrame(excluded_rows)
    n_keep_m1 = n_eval - int((excluded_df["truth_method"] == "M1").sum()) if len(excluded_df) else n_eval
    n_keep_m2 = n_eval - int((excluded_df["truth_method"] == "M2").sum()) if len(excluded_df) else n_eval

    # Subset predictions: one row per evaluable recipe, MS cluster + panel pred per method.
    subset_rows = []
    for _, row in eval_truth.iterrows():
        pid = row["pdm_id"]
        rec = {"Recipe": row["Recipe"], "H": row["H"],
               "M1_set": ", ".join(row["M1_set"]), "M2_set": ", ".join(row["M2_set"])}
        for m in variants:
            rec[f"{m} [MS]"] = ms_cluster[m].get(pid)
            rec[f"{m} [panel]"] = panel_pred[m].get(pid)
        subset_rows.append(rec)
    subset_df = pd.DataFrame(subset_rows)
    # Group recipes by their true label so per-cluster performance is scannable at a glance.
    subset_df = subset_df.sort_values(["M1_set", "M2_set", "Recipe"], ignore_index=True)

    # Per-true-cluster accuracy (rows = true cluster, columns = method) for M1 and M2.
    pc_m1 = per_cluster_accuracy(panel_pred, eval_truth, "M1_set")
    pc_m2 = per_cluster_accuracy(panel_pred, eval_truth, "M2_set")
    pc_m1_reach = per_cluster_accuracy(panel_pred, eval_truth, "M1_set", reachable_only=True)
    pc_m2_reach = per_cluster_accuracy(panel_pred, eval_truth, "M2_set", reachable_only=True)

    # All-recipe hand-over: raw MS argmax cluster per method (full info, no mapping loss).
    all_pred = pd.DataFrame({m: ms_cluster[m] for m in variants})
    all_pred.index.name = "Rez.-Nr."

    truth_out = truth.drop(columns=["base"]).copy()
    truth_out["M1_set"] = truth_out["M1_set"].map(lambda s: ", ".join(s))
    truth_out["M2_set"] = truth_out["M2_set"].map(lambda s: ", ".join(s))

    # Readable column labels for the exported sheets (internal keys stay terse).
    READABLE = {
        "H": "Prozent %",
        "B_M1": "M1 Label Propagation",
        "C_M2": "M2 Rule Based",
        "D": "Cluster vorgegeben",
        "E": "Jan Free Sorting",
        "M1_set": "M1 true labels",
        "M2_set": "M2 true labels",
        "in_pdm": "In PDM export",
        "pdm_id": "PDM Rez.-Nr.",
        "evaluable": "Evaluable",
        "truth_method": "Truth method",
        "truth": "True label(s)",
    }
    truth_out = truth_out.rename(columns=READABLE)
    subset_df = subset_df.rename(columns=READABLE)
    excluded_df = excluded_df.rename(columns=READABLE)

    OUT_XLSX.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        acc_m1.to_excel(writer, sheet_name="Accuracy_vs_M1", index=False)
        acc_m2.to_excel(writer, sheet_name="Accuracy_vs_M2", index=False)
        subset_df.to_excel(writer, sheet_name="Subset_Predictions", index=False)
        pc_m1.to_excel(writer, sheet_name="Per_Cluster_vs_M1", index=False)
        pc_m2.to_excel(writer, sheet_name="Per_Cluster_vs_M2", index=False)
        pc_m1_reach.to_excel(writer, sheet_name="Per_Cluster_vs_M1_reachable", index=False)
        pc_m2_reach.to_excel(writer, sheet_name="Per_Cluster_vs_M2_reachable", index=False)
        truth_out.to_excel(writer, sheet_name="Truth_Labels", index=False)
        if len(excluded_df):
            excluded_df.to_excel(writer, sheet_name="Excluded_Recipes", index=False)
        all_pred.reset_index().to_excel(writer, sheet_name="All_Predictions_AllMethods", index=False)

    print(f"\nSaved: {OUT_XLSX}")
    print(f"\n=== Accuracy vs M1 (Label Propagation), {n_eval} recipes ===")
    print(acc_m1[["Method", "z_method", "Accuracy_%", "Accuracy_reachable_%"]].to_string(index=False))
    print(f"\n=== Accuracy vs M2 (Rule Based), {n_eval} recipes ===")
    print(acc_m2[["Method", "z_method", "Accuracy_%", "Accuracy_reachable_%"]].to_string(index=False))
    print(
        f"\nReachable-only (excludes dairy/Walderdbeere truths): "
        f"M1 keeps {n_keep_m1}/{n_eval}, M2 keeps {n_keep_m2}/{n_eval} recipes. "
        f"See Accuracy_reachable_% column and Excluded_Recipes sheet."
    )


if __name__ == "__main__":
    main()
