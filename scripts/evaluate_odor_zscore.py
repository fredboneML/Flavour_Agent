"""Odor-type + Z-Score experiment.

Each ingredient carries up to 3 odor types (Odour-Type 1/2/3 in the old strawberry CSV, keyed by
CAS), and that vocabulary maps almost 1:1 onto the cluster names. Two ways to combine odor type
with the z-score (z = amount / typical-amount, using Melanie's AvgMeli), each swept over a hard
z-cutoff T (ingredient counts only if z >= T):

  A) Odor-type voting x z-score
     Drop the expert weight matrix. Each ingredient votes for the cluster(s) matching its odor
     types (rank-weighted 1.0 / 0.5 / 0.25 for OT1/2/3), vote strength = its z-score.
     NOTE: odor vocabulary includes `dairy`, which the expert weights lack -> A *can* predict dairy.

  B) Odor-type gate on MS Z-Score
     Keep MS Z-Score (expert weights) but zero out any ingredient->cluster weight whose cluster
     does not agree with the ingredient's odor types. Odor type acts as the gate.

Baseline: MS Z-Score raw accuracy 40.7% (M1) / 37.0% (M2); reachable 64.7% / 50.0%.
Output: console + outputs/odor_zscore_eval.xlsx

Usage:
    python3 scripts/evaluate_odor_zscore.py
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
    CLUSTER_COLS,
)
from scripts.evaluate_verkostung import (
    SCORING_XLSX, PDM_CSV, IGNORE_PATH, GOLD, MS_TO_PANEL,
    build_truth, accuracy, base_id,
)

OLD_CSV = GOLD / "Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv"
OUT_XLSX = Path(__file__).resolve().parent.parent / "outputs" / "odor_zscore_eval.xlsx"

THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
RANK_W = [1.0, 0.5, 0.25]  # OT1 / OT2 / OT3 vote weights

# Odor-type string -> cluster (panel vocabulary). Unmapped odors (fresh, woody, ...) don't vote.
ODOR_TO_CLUSTER = {
    "warm": "warm", "floral": "floral", "green": "green", "dairy": "dairy",
    "unpleasant": "unpleasant", "fruity": "fruity", "exotic": "exotic",
}
# Expert weight column -> odor name, for the Method-B agreement gate (Outlayer has no odor).
COL_TO_ODOR = {
    "Unpleasant": "unpleasant", "warm": "warm", "green": "green",
    "floral": "floral", "citrus": "fruity", "exotic": "exotic", "Outlayer": None,
}


def cas_odor_sets() -> dict[str, list[str]]:
    """CAS -> [OT1, OT2, OT3] mapped to cluster names (None where odor is unmapped/absent)."""
    old = pd.read_csv(OLD_CSV, dtype=str)
    cols = ["Odour-Type 1", "Odour-Type 2", "Odour-Type 3"]
    out: dict[str, list[str]] = {}
    for _, r in old[["CAS-Nr."] + cols].drop_duplicates("CAS-Nr.").iterrows():
        cas = str(r["CAS-Nr."]).strip()
        slots = [ODOR_TO_CLUSTER.get(str(r[c]).strip().lower()) if pd.notna(r[c]) else None for c in cols]
        out[cas] = slots
    return out


def score_odor_voting(z_df: pd.DataFrame, odor: dict) -> pd.Series:
    """Method A: sum z-score votes into the cluster of each ingredient's odor types."""
    votes: dict[str, dict[str, float]] = {}
    for rez, cas, z in zip(z_df["Rez.-Nr."], z_df["CAS-Nr."], z_df["Totalmenge"]):
        if z <= 0:
            continue
        slots = odor.get(str(cas).strip())
        if not slots:
            continue
        bucket = votes.setdefault(str(rez), {})
        for cl, w in zip(slots, RANK_W):
            if cl:
                bucket[cl] = bucket.get(cl, 0.0) + z * w
    rows = {rez: max(b, key=b.get) if b else None for rez, b in votes.items()}
    return pd.Series(rows, name="odor_vote")


def masked_weights(weights: pd.DataFrame, odor: dict) -> pd.DataFrame:
    """Method B: zero each CAS's weight columns whose cluster disagrees with its odor types."""
    m = weights.copy()
    for cas in m.index:
        allowed = {c for c in (odor.get(str(cas).strip()) or []) if c}
        for col in CLUSTER_COLS:
            if COL_TO_ODOR.get(col) not in allowed:
                m.loc[cas, col] = 0.0
    return m


def main() -> None:
    weights = load_weight_matrix(SCORING_XLSX)
    recipes_df = load_recipes(PDM_CSV, IGNORE_PATH)
    avg_meli = load_avg_meli(SCORING_XLSX, csv_path=PDM_CSV)
    odor = cas_odor_sets()
    z_base = apply_zscore_quantities(recipes_df, avg_meli)  # Totalmenge holds z
    w_masked = masked_weights(weights, odor)

    truth = build_truth()
    pdm_base = {base_id(x): str(x).strip() for x in recipes_df["Rez.-Nr."].unique()}
    truth["pdm_id"] = truth["base"].map(pdm_base)
    truth["evaluable"] = truth["base"].isin(pdm_base) & (truth["M1_set"].map(len) > 0)
    eval_truth = truth[truth["evaluable"]].copy()
    n = len(eval_truth)

    rows = []
    for T in THRESHOLDS:
        zg = apply_threshold(z_base, T) if T > 0 else z_base
        preds = {
            "A: Odor-voting x z": score_odor_voting(zg, odor),
            "B: Odor-gate on MS-Z": map_to_panel(assign_clusters(compute_scores(zg, w_masked)), MS_TO_PANEL),
        }
        for name, p in preds.items():
            a1 = accuracy({"m": p}, eval_truth, "M1_set").iloc[0]
            a2 = accuracy({"m": p}, eval_truth, "M2_set").iloc[0]
            rows.append({
                "Method": name, "Threshold_T": T,
                "M1_raw_%": a1["Accuracy_%"], "M1_reachable_%": a1["Accuracy_reachable_%"],
                "M2_raw_%": a2["Accuracy_%"], "M2_reachable_%": a2["Accuracy_reachable_%"],
            })
    res = pd.DataFrame(rows)

    OUT_XLSX.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="Odor_Zscore_Sweep", index=False)
        for metric in ("M1_raw_%", "M2_raw_%"):
            res.pivot(index="Method", columns="Threshold_T", values=metric).to_excel(
                w, sheet_name=metric.replace("%", "pct"))

    print(f"Evaluable strawberry recipes: {n}")
    print("Champion MS Z-Score: raw 40.7% (M1) / 37.0% (M2);  reachable 64.7% / 50.0%\n")
    for metric in ("M1_raw_%", "M2_raw_%"):
        print(f"=== {metric} (all {n} recipes) by threshold ===")
        print(res.pivot(index="Method", columns="Threshold_T", values=metric).to_string())
        print()
    print("Top 5 by M1_raw_%:")
    print(res.sort_values("M1_raw_%", ascending=False)
             .head(5)[["Method", "Threshold_T", "M1_raw_%", "M2_raw_%"]].to_string(index=False))
    print(f"\nSaved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
