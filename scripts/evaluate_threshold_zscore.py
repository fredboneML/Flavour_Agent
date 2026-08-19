"""Threshold-gated Z-Score sweep.

Applies a HARD CUTOFF to the z-score (z = amount / typical-amount per ingredient): an ingredient
counts toward cluster scores only if z >= T, else 0. Sweeps T across all denominators to see
whether gating lets a self-computed z-score beat the champion MS Z-Score (Melanie reference).

Baseline (T = 0, no gating) reachable accuracy: MS Z-Score = 64.7% (M1) / 50.0% (M2).

Denominators swept:
  - Melanie ref        : Melanie's AvgMeli (external strawberry reference)
  - Strawberry mean    : central mean amount-when-present over strawberry recipes
  - Strawberry median  : central median amount-when-present over strawberry recipes

Output: console matrix + outputs/threshold_zscore_sweep.xlsx

Usage:
    python3 scripts/evaluate_threshold_zscore.py
"""

from pathlib import Path

import pandas as pd

from scripts.ms_scoring import (
    load_weight_matrix,
    load_recipes,
    load_avg_meli,
    compute_scores,
    assign_clusters,
    map_to_panel,
    apply_zscore_quantities,
    apply_threshold,
    central_amount_when_present,
)
from scripts.evaluate_verkostung import (
    SCORING_XLSX, PDM_CSV, IGNORE_PATH, MS_TO_PANEL,
    build_truth, accuracy, base_id,
)
from scripts.evaluate_strawberry_zscore import strawberry_recipe_ids

OUT_XLSX = Path(__file__).resolve().parent.parent / "outputs" / "threshold_zscore_sweep.xlsx"

# z = amount / typical-amount; T is a multiple of the typical amount. T=0 => no gating (baseline).
THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def main() -> None:
    weights = load_weight_matrix(SCORING_XLSX)
    recipes_df = load_recipes(PDM_CSV, IGNORE_PATH)
    straw_ids = strawberry_recipe_ids()
    straw_df = recipes_df[recipes_df["Rez.-Nr."].astype(str).str.strip().isin(straw_ids)]

    denoms = {
        "Melanie ref": load_avg_meli(SCORING_XLSX, csv_path=PDM_CSV),
        "Strawberry mean": central_amount_when_present(straw_df, "mean"),
        "Strawberry median": central_amount_when_present(straw_df, "median"),
    }

    truth = build_truth()
    pdm_base = {base_id(x): str(x).strip() for x in recipes_df["Rez.-Nr."].unique()}
    truth["pdm_id"] = truth["base"].map(pdm_base)
    truth["evaluable"] = truth["base"].isin(pdm_base) & (truth["M1_set"].map(len) > 0)
    eval_truth = truth[truth["evaluable"]].copy()
    n = len(eval_truth)

    rows = []
    for dname, denom in denoms.items():
        z_df = apply_zscore_quantities(recipes_df, denom)
        for T in THRESHOLDS:
            gated = apply_threshold(z_df, T) if T > 0 else z_df
            # recipes left with no surviving ingredient after gating
            empty = int((gated.groupby("Rez.-Nr.")["Totalmenge"].sum() == 0).sum())
            pred = {"m": map_to_panel(assign_clusters(compute_scores(gated, weights)), MS_TO_PANEL)}
            a1 = accuracy(pred, eval_truth, "M1_set").iloc[0]
            a2 = accuracy(pred, eval_truth, "M2_set").iloc[0]
            rows.append({
                "Denominator": dname,
                "Threshold_T": T,
                "M1_reachable_%": a1["Accuracy_reachable_%"],
                "M1_raw_%": a1["Accuracy_%"],
                "M2_reachable_%": a2["Accuracy_reachable_%"],
                "M2_raw_%": a2["Accuracy_%"],
                "Empty_recipes": empty,
            })
    res = pd.DataFrame(rows)

    OUT_XLSX.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="Threshold_Sweep", index=False)
        for metric in ("M1_reachable_%", "M2_reachable_%"):
            piv = res.pivot(index="Denominator", columns="Threshold_T", values=metric)
            piv.to_excel(w, sheet_name=metric.replace("%", "pct").replace("_", " ").strip())

    base = 64.7  # MS Z-Score baseline (M1 reachable)
    print(f"Evaluable strawberry recipes: {n}")
    print(f"Champion baseline (MS Z-Score, no gating): M1 reachable 64.7% / M2 reachable 50.0%\n")
    for metric in ("M1_reachable_%", "M2_reachable_%"):
        print(f"=== {metric} by threshold ===")
        print(res.pivot(index="Denominator", columns="Threshold_T", values=metric).to_string())
        print()
    best = res.sort_values("M1_reachable_%", ascending=False).head(5)
    print("Top 5 by M1 reachable accuracy:")
    print(best[["Denominator", "Threshold_T", "M1_reachable_%", "M2_reachable_%", "Empty_recipes"]].to_string(index=False))
    print(f"\nSaved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
