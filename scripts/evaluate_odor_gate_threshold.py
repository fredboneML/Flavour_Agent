"""Combine the odor-type GATE with the odor THRESHOLD (OAV), on top of the z-score.

Two winning-ish ideas stacked:
  - Odor-type gate (evaluate_odor_zscore.py, Method B): zero out any ingredient->cluster weight
    whose cluster disagrees with the ingredient's odor type. Best standalone result: 70.6% (M1) /
    55.0% (M2) reachable — the only method to beat plain MS Z-Score.
  - Odor threshold / OAV (evaluate_oav_zscore.py): scale/gate the z by potency = amount/threshold.

Here every method uses the odor-type MASKED weights (the gate) and varies how the ppm threshold is
folded into the z-frame. Target to beat: odor-gate baseline 70.6% / 55.0% reachable.

Output: console + outputs/odor_gate_threshold_eval.xlsx

Usage:
    python3 scripts/evaluate_odor_gate_threshold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ms_scoring import (
    load_weight_matrix, load_recipes, load_avg_meli,
    compute_scores, assign_clusters, map_to_panel, apply_zscore_quantities,
)
from scripts.evaluate_verkostung import (
    SCORING_XLSX, PDM_CSV, IGNORE_PATH, MS_TO_PANEL,
    build_truth, accuracy, base_id, REACHABLE,
)
from scripts.evaluate_odor_zscore import cas_odor_sets, masked_weights
from scripts.evaluate_oav_zscore import cas_threshold, factor_frame

OUT_XLSX = Path(__file__).resolve().parent.parent / "outputs" / "odor_gate_threshold_eval.xlsx"
ALPHAS = [0.25, 0.5, 1.0]
GATES = [0.5, 1.0, 2.0]


def main() -> None:
    weights = load_weight_matrix(SCORING_XLSX)
    recipes_df = load_recipes(PDM_CSV, IGNORE_PATH)
    avg_meli = load_avg_meli(SCORING_XLSX, csv_path=PDM_CSV)
    odor = cas_odor_sets()
    thr, med = cas_threshold()
    w_masked = masked_weights(weights, odor)          # the odor-type gate
    z_base = apply_zscore_quantities(recipes_df, avg_meli)
    cas = recipes_df["CAS-Nr."].astype(str).str.strip()
    oav_amt = (recipes_df["Totalmenge"] / cas.map(thr)).fillna(recipes_df["Totalmenge"] / med)

    def predict(df):  # scored with the MASKED (odor-gated) weights
        return map_to_panel(assign_clusters(compute_scores(df, w_masked)), MS_TO_PANEL)

    # Threshold-folded z-frames
    oav_replace = recipes_df.copy(); oav_replace["Totalmenge"] = oav_amt.values
    oav_times = z_base.copy(); oav_times["Totalmenge"] = z_base["Totalmenge"] * oav_amt.values

    methods = {"Odor-gate (no threshold)": predict(z_base)}
    methods["Odor-gate + OAV replaces z"] = predict(oav_replace)
    methods["Odor-gate + OAV x z"] = predict(oav_times)
    for a in ALPHAS:
        methods[f"Odor-gate + inv-threshold a={a}"] = predict(factor_frame(z_base, thr, med, "inv", a))
    methods["Odor-gate + log-OAV"] = predict(factor_frame(z_base, thr, med, "logoav", 0))
    for T in GATES:
        methods[f"Odor-gate + OAV gate T={T}"] = predict(factor_frame(z_base, thr, med, "gate", T))

    truth = build_truth()
    pdm_base = {base_id(x): str(x).strip() for x in recipes_df["Rez.-Nr."].unique()}
    truth["pdm_id"] = truth["base"].map(pdm_base)
    truth["evaluable"] = truth["base"].isin(pdm_base) & (truth["M1_set"].map(len) > 0)
    eval_truth = truth[truth["evaluable"]].copy()
    reach = eval_truth[eval_truth["M1_set"].map(lambda s: bool(set(s) & REACHABLE))]

    rows = []
    for name, pred in methods.items():
        a1 = accuracy({"m": pred}, eval_truth, "M1_set").iloc[0]
        a2 = accuracy({"m": pred}, eval_truth, "M2_set").iloc[0]
        rows.append({"Method": name,
                     "M1_reachable_%": a1["Accuracy_reachable_%"], "M1_raw_%": a1["Accuracy_%"],
                     "M2_reachable_%": a2["Accuracy_reachable_%"], "M2_raw_%": a2["Accuracy_%"]})
    res = pd.DataFrame(rows).sort_values("M1_reachable_%", ascending=False, ignore_index=True)

    gate_base = res.loc[res["Method"] == "Odor-gate (no threshold)", "M1_reachable_%"].iloc[0]
    champion = 64.7  # plain MS Z-Score
    best = res.iloc[0]

    det = []
    base_pred = methods["Odor-gate (no threshold)"]
    best_pred = methods[best["Method"]]
    for _, r_ in reach.sort_values("Recipe").iterrows():
        pid = r_["pdm_id"]
        det.append({"Recipe": r_["Recipe"], "M1 true": ", ".join(r_["M1_set"]),
                    "M2 true": ", ".join(r_["M2_set"]),
                    "Odor-gate": base_pred.get(pid), best["Method"]: best_pred.get(pid)})
    det_df = pd.DataFrame(det)

    OUT_XLSX.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="OdorGate_Threshold_Sweep", index=False)
        det_df.to_excel(w, sheet_name="Best_vs_OdorGate_17", index=False)

    print(f"Plain MS Z-Score reachable: 64.7% (M1) / 50.0% (M2)")
    print(f"Odor-gate baseline reachable: M1 {gate_base}% (target to beat)\n")
    print(res.to_string(index=False))
    print()
    win = res[res["M1_reachable_%"] > gate_base]
    if len(win):
        print("Adding threshold BEATS the odor-gate on M1_reachable:")
        print(win.to_string(index=False))
    else:
        print(f"No threshold combination beats the odor-gate alone ({gate_base}%). "
              f"Best: {best['Method']} at {best['M1_reachable_%']}%.")
    print(f"\nSaved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
