"""Strawberry-scoped Z-Score experiment.

The 27 evaluable Verkostung recipes are all strawberry (Erdbeer-Aroma), so filtering the test
set to strawberries is a no-op. The real question is whether computing our own AvgMeli over
STRAWBERRY recipes only (instead of all 3,981 fruits) closes the gap to the champion MS Z-Score
(which uses Melanie's strawberry reference AvgMeli).

Denominators compared, all fed through apply_zscore_quantities -> compute_scores:
  - MS Z-Score            : Melanie's AvgMeli (external strawberry reference)   [champion]
  - Z mean (all fruit)    : central mean amount-when-present over all recipes
  - Z median (all fruit)  : central median amount-when-present over all recipes
  - Z mean (strawberry)   : central mean over strawberry recipes only
  - Z median (strawberry) : central median over strawberry recipes only

Usage:
    python3 scripts/evaluate_strawberry_zscore.py
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
    central_amount_when_present,
)
from scripts.evaluate_verkostung import (
    SCORING_XLSX, PDM_CSV, IGNORE_PATH, GOLD, MS_TO_PANEL,
    build_truth, accuracy, base_id,
)

PDM_XLSX = GOLD / "PDM_Rezepturen_Gesamt_nurP_18_08_2026.xlsx"


def strawberry_recipe_ids() -> set[str]:
    """Rez.-Nr. whose Rezepturbezeichnung marks a strawberry recipe (Erdbeer*)."""
    new = pd.read_excel(PDM_XLSX, sheet_name="Rezept", header=14, dtype=str)
    bez = new["Rezepturbezeichnung"].astype(str)
    straw = new.loc[bez.str.contains("Erdbeer", case=False, na=False), "Rez.-Nr."]
    return set(straw.astype(str).str.strip())


def main() -> None:
    weights = load_weight_matrix(SCORING_XLSX)
    recipes_df = load_recipes(PDM_CSV, IGNORE_PATH)

    straw_ids = strawberry_recipe_ids()
    straw_df = recipes_df[recipes_df["Rez.-Nr."].astype(str).str.strip().isin(straw_ids)]
    print(f"strawberry recipes: {straw_df['Rez.-Nr.'].nunique()} / {recipes_df['Rez.-Nr.'].nunique()}")

    # Denominators
    avg_meli = load_avg_meli(SCORING_XLSX, csv_path=PDM_CSV)
    denoms = {
        "MS Z-Score (Melanie ref)": avg_meli,
        "Z mean (all fruit)": central_amount_when_present(recipes_df, "mean"),
        "Z median (all fruit)": central_amount_when_present(recipes_df, "median"),
        "Z mean (strawberry)": central_amount_when_present(straw_df, "mean"),
        "Z median (strawberry)": central_amount_when_present(straw_df, "median"),
    }

    # Predictions: score ALL recipes with each denominator, map to panel names.
    panel_pred = {}
    for name, denom in denoms.items():
        sc = compute_scores(apply_zscore_quantities(recipes_df, denom), weights)
        panel_pred[name] = map_to_panel(assign_clusters(sc), MS_TO_PANEL)

    # Truth (same rule as the main eval) restricted to the evaluable subset.
    truth = build_truth()
    pdm_base = {base_id(x): str(x).strip() for x in recipes_df["Rez.-Nr."].unique()}
    truth["pdm_id"] = truth["base"].map(pdm_base)
    truth["evaluable"] = truth["base"].isin(pdm_base) & (truth["M1_set"].map(len) > 0)
    eval_truth = truth[truth["evaluable"]].copy()
    n = len(eval_truth)

    acc_m1 = accuracy(panel_pred, eval_truth, "M1_set")
    acc_m2 = accuracy(panel_pred, eval_truth, "M2_set")

    cols = ["Method", "Accuracy_%", "Accuracy_reachable_%"]
    print(f"\n=== vs M1 (Label Propagation), {n} strawberry recipes ===")
    print(acc_m1[cols].to_string(index=False))
    print(f"\n=== vs M2 (Rule Based), {n} strawberry recipes ===")
    print(acc_m2[cols].to_string(index=False))


if __name__ == "__main__":
    main()
