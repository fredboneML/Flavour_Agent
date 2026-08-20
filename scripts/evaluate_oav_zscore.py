"""Odor-threshold (ppm) x Z-Score experiment.

Each ingredient has an odor perception threshold in ppm (Threshold column of the old strawberry
CSV, keyed by CAS). Low threshold = potent = perceptible in trace amounts. This is the Odor
Activity Value: OAV = amount / threshold. We combine threshold with the champion z-score
(z = amount / Melanie AvgMeli, x expert weights) several ways and test each on the reachable
strawberry panel (17 recipes for M1, 20 for M2) to see if any beats MS Z-Score (64.7% / 50.0%).

Missing threshold (`k.E.`) -> neutral (never zeroed); one zero-threshold CAS -> clamped.
Scale is normalized by the median threshold (t_norm = threshold / median), since raw 1/threshold
spans ~6 orders of magnitude.

Output: console + outputs/oav_zscore_eval.xlsx

Usage:
    python3 scripts/evaluate_oav_zscore.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ms_scoring import (
    load_weight_matrix, load_recipes, load_avg_meli,
    compute_scores, assign_clusters, map_to_panel, apply_zscore_quantities,
)
from scripts.evaluate_verkostung import (
    SCORING_XLSX, PDM_CSV, IGNORE_PATH, GOLD, MS_TO_PANEL,
    build_truth, accuracy, base_id, REACHABLE,
)

OLD_CSV = GOLD / "Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv"
OUT_XLSX = Path(__file__).resolve().parent.parent / "outputs" / "oav_zscore_eval.xlsx"

ALPHAS = [0.25, 0.5, 1.0]      # inverse-threshold exponent
GATES = [0.5, 1.0, 2.0, 5.0]   # OAV perceptibility gate
EPS = 0.05                      # floor for log-OAV factor


def cas_threshold() -> tuple[dict[str, float], float]:
    """CAS -> odor threshold (ppm). Returns (map, median). German decimals; `k.E.` dropped; 0 clamped."""
    old = pd.read_csv(OLD_CSV, dtype=str)

    def parse(v):
        s = str(v).strip()
        if s in ("", "nan", "k.E.", "kE", "n.a."):
            return np.nan
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return np.nan

    old["th"] = old["Threshold"].apply(parse)
    per = old[["CAS-Nr.", "th"]].dropna().drop_duplicates("CAS-Nr.")
    per["CAS-Nr."] = per["CAS-Nr."].astype(str).str.strip()
    pos_min = per.loc[per["th"] > 0, "th"].min()
    per["th"] = per["th"].clip(lower=pos_min)  # clamp the single zero threshold
    med = per["th"].median()
    return dict(zip(per["CAS-Nr."], per["th"])), med


def factor_frame(z_df: pd.DataFrame, thr: dict, med: float, kind: str, param: float) -> pd.DataFrame:
    """Return a z-frame whose Totalmenge is scaled by a threshold-derived per-ingredient factor.

    z_df.Totalmenge holds the z-score. amount is unavailable post-z, so OAV is expressed relative
    to the median threshold: an ingredient's threshold factor uses t = threshold / median.
    'oav_replace'/'oav_times' need the raw amount, so those are computed from a separate amount map.
    """
    df = z_df.copy()
    cas = df["CAS-Nr."].astype(str).str.strip()
    t = cas.map(thr) / med  # normalized threshold; NaN where missing

    if kind == "inv":            # z * (1/t)^alpha ; missing -> factor 1
        f = np.power(1.0 / t, param)
        f = f.fillna(1.0)
        df["Totalmenge"] = df["Totalmenge"] * f
    elif kind == "logoav":       # z * max(eps, 1 + log10(amount/threshold)); use z as amount proxy
        # OAV proxy = z / t  (z stands in for the normalized amount signal)
        oav = df["Totalmenge"] / t
        f = np.maximum(EPS, 1.0 + np.log10(oav.where(oav > 0)))
        f = f.fillna(1.0)
        df["Totalmenge"] = df["Totalmenge"] * f
    elif kind == "gate":         # keep only if OAV proxy (z/t) >= T ; missing threshold -> keep
        oav = df["Totalmenge"] / t
        drop = (oav < param) & oav.notna()
        df.loc[drop, "Totalmenge"] = 0.0
    else:
        raise ValueError(kind)
    return df


def oav_amount_methods(recipes_df, weights, thr, med):
    """OAV-replace and OAV x z, which need the raw normalized amount (pre-z)."""
    amt = recipes_df.copy()
    cas = amt["CAS-Nr."].astype(str).str.strip()
    t = cas.map(thr)
    oav = amt["Totalmenge"] / t
    oav = oav.fillna(amt["Totalmenge"] / med)      # missing threshold -> use median threshold
    out = {}
    # OAV replaces z
    r = amt.copy(); r["Totalmenge"] = oav
    out["OAV replaces z"] = r
    return out


def main() -> None:
    weights = load_weight_matrix(SCORING_XLSX)
    recipes_df = load_recipes(PDM_CSV, IGNORE_PATH)
    avg_meli = load_avg_meli(SCORING_XLSX, csv_path=PDM_CSV)
    thr, med = cas_threshold()
    z_base = apply_zscore_quantities(recipes_df, avg_meli)

    def predict(df):
        return map_to_panel(assign_clusters(compute_scores(df, weights)), MS_TO_PANEL)

    # OAV x z: multiply z by OAV(amount) — build from raw amount then re-scale the z-frame
    cas = recipes_df["CAS-Nr."].astype(str).str.strip()
    oav_amt = (recipes_df["Totalmenge"] / cas.map(thr)).fillna(recipes_df["Totalmenge"] / med)
    oav_times = z_base.copy(); oav_times["Totalmenge"] = z_base["Totalmenge"] * oav_amt.values

    methods: dict[str, pd.Series] = {"Baseline MS Z-Score": predict(z_base)}
    r = recipes_df.copy(); r["Totalmenge"] = (recipes_df["Totalmenge"] / cas.map(thr)).fillna(
        recipes_df["Totalmenge"] / med)
    methods["OAV replaces z"] = predict(r)
    methods["OAV x z"] = predict(oav_times)
    for a in ALPHAS:
        methods[f"inv-threshold a={a}"] = predict(factor_frame(z_base, thr, med, "inv", a))
    methods["log-OAV weight"] = predict(factor_frame(z_base, thr, med, "logoav", 0))
    for T in GATES:
        methods[f"OAV gate T={T}"] = predict(factor_frame(z_base, thr, med, "gate", T))

    # Truth restricted to reachable subset
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

    base_m1 = res.loc[res["Method"] == "Baseline MS Z-Score", "M1_reachable_%"].iloc[0]
    best = res.iloc[0]
    winner = res[res["M1_reachable_%"] > base_m1]

    # Best config predictions on the 17 reachable, vs baseline
    best_pred = methods[best["Method"]]
    base_pred = methods["Baseline MS Z-Score"]
    det = []
    for _, r_ in reach.sort_values("Recipe").iterrows():
        pid = r_["pdm_id"]
        det.append({"Recipe": r_["Recipe"], "M1 true": ", ".join(r_["M1_set"]),
                    "M2 true": ", ".join(r_["M2_set"]),
                    "Baseline MS Z-Score": base_pred.get(pid),
                    best["Method"]: best_pred.get(pid)})
    det_df = pd.DataFrame(det)

    # Threshold coverage on the evaluated recipes
    ev_cas = recipes_df[recipes_df["Rez.-Nr."].astype(str).str.strip().isin(set(reach["pdm_id"]))]
    ev_cas = ev_cas[ev_cas["Totalmenge"] > 0]
    cov = ev_cas["CAS-Nr."].astype(str).str.strip().isin(thr).mean()

    OUT_XLSX.parent.mkdir(exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="OAV_Sweep", index=False)
        det_df.to_excel(w, sheet_name="Best_vs_MSZ_17", index=False)

    print(f"Reachable recipes: {len(reach)} (M1)  |  threshold row-coverage on them: {100*cov:.0f}%")
    print(f"Baseline MS Z-Score reachable: M1 {base_m1}%  (target 64.7 / 50.0)\n")
    print(res.to_string(index=False))
    print()
    if len(winner):
        print("BEATS baseline on M1_reachable:")
        print(winner.to_string(index=False))
    else:
        print(f"No method beats the baseline on M1_reachable ({base_m1}%). Best: "
              f"{best['Method']} at {best['M1_reachable_%']}%.")
    print(f"\nSaved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
