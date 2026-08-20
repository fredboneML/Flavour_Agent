"""Shared odor-type + odor-threshold helpers for the scoring methods.

Canonical home for the odor-type gate (masked expert weights) and the ppm odor-threshold /
Odor Activity Value (OAV) transforms, so the eval pipeline and the standalone experiment scripts
share one implementation. Auxiliary data (odor types, thresholds) comes from the old strawberry CSV,
keyed by CAS.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ms_scoring import CLUSTER_COLS

GOLD = Path(__file__).resolve().parent.parent / "data" / "gold"
OLD_CSV = GOLD / "Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv"

EPS = 0.05  # floor for log-OAV factor

# Odor-type string -> cluster (panel vocabulary). Unmapped odors (fresh, woody, ...) don't map.
ODOR_TO_CLUSTER = {
    "warm": "warm", "floral": "floral", "green": "green", "dairy": "dairy",
    "unpleasant": "unpleasant", "fruity": "fruity", "exotic": "exotic",
}
# Expert weight column -> odor name, for the odor-type agreement gate (Outlayer has no odor).
COL_TO_ODOR = {
    "Unpleasant": "unpleasant", "warm": "warm", "green": "green",
    "floral": "floral", "citrus": "fruity", "exotic": "exotic", "Outlayer": None,
}


def cas_odor_sets() -> dict[str, list]:
    """CAS -> [OT1, OT2, OT3] mapped to cluster names (None where odor is unmapped/absent)."""
    old = pd.read_csv(OLD_CSV, dtype=str)
    cols = ["Odour-Type 1", "Odour-Type 2", "Odour-Type 3"]
    out: dict[str, list] = {}
    for _, r in old[["CAS-Nr."] + cols].drop_duplicates("CAS-Nr.").iterrows():
        cas = str(r["CAS-Nr."]).strip()
        slots = [ODOR_TO_CLUSTER.get(str(r[c]).strip().lower()) if pd.notna(r[c]) else None for c in cols]
        out[cas] = slots
    return out


def masked_weights(weights: pd.DataFrame, odor: dict, keep_unknown: bool = False) -> pd.DataFrame:
    """Odor-type gate: zero each CAS's weight columns whose cluster disagrees with its odor types.

    keep_unknown: if a CAS has no odor information at all, leave its weights unchanged (so
    non-strawberry recipes remain scoreable). If False, unknown-odor CAS get fully zeroed.
    """
    m = weights.copy()
    for cas in m.index:
        slots = odor.get(str(cas).strip())
        if slots is None and keep_unknown:
            continue
        allowed = {c for c in (slots or []) if c}
        for col in CLUSTER_COLS:
            if COL_TO_ODOR.get(col) not in allowed:
                m.loc[cas, col] = 0.0
    return m


def cas_threshold() -> tuple[dict, float]:
    """CAS -> odor threshold (ppm). German decimals; `k.E.` dropped; single 0 clamped. Returns (map, median)."""
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
    per["th"] = per["th"].clip(lower=pos_min)
    return dict(zip(per["CAS-Nr."], per["th"])), per["th"].median()


def factor_frame(z_df: pd.DataFrame, thr: dict, med: float, kind: str, param: float) -> pd.DataFrame:
    """Scale a z-frame's Totalmenge by a threshold-derived per-ingredient factor.

    kind: 'inv' (z*(1/t_norm)^param), 'logoav' (z*max(EPS,1+log10(z/t_norm))),
    'gate' (drop ingredient if z/t_norm < param). Missing threshold -> neutral / kept.
    """
    df = z_df.copy()
    cas = df["CAS-Nr."].astype(str).str.strip()
    t = cas.map(thr) / med  # normalized threshold; NaN where missing
    if kind == "inv":
        f = np.power(1.0 / t, param).fillna(1.0)
        df["Totalmenge"] = df["Totalmenge"] * f
    elif kind == "logoav":
        oav = df["Totalmenge"] / t
        f = np.maximum(EPS, 1.0 + np.log10(oav.where(oav > 0))).fillna(1.0)
        df["Totalmenge"] = df["Totalmenge"] * f
    elif kind == "gate":
        oav = df["Totalmenge"] / t
        df.loc[(oav < param) & oav.notna(), "Totalmenge"] = 0.0
    else:
        raise ValueError(kind)
    return df


def oav_amount(recipes_df: pd.DataFrame, thr: dict, med: float) -> pd.Series:
    """OAV = normalized amount / threshold; missing threshold -> amount / median threshold."""
    cas = recipes_df["CAS-Nr."].astype(str).str.strip()
    return (recipes_df["Totalmenge"] / cas.map(thr)).fillna(recipes_df["Totalmenge"] / med)


# Threshold parameter grids shared across methods.
ALPHAS = [0.25, 0.5, 1.0]
GATES = [0.5, 1.0, 2.0]


def threshold_variants(recipes_df, weights, z_base, w_scorer, prefix: str) -> dict:
    """All threshold combinations as {name: score_matrix}, scored with ``w_scorer`` weights.

    z_base: z-frame (Totalmenge = z). prefix: label prefix (e.g. "" or "Odor-gate + ").
    """
    from scripts.ms_scoring import compute_scores

    thr, med = cas_threshold()
    oav = oav_amount(recipes_df, thr, med)
    out = {}

    r = recipes_df.copy(); r["Totalmenge"] = oav.values
    out[f"{prefix}OAV replaces z"] = compute_scores(r, w_scorer)
    ox = z_base.copy(); ox["Totalmenge"] = z_base["Totalmenge"] * oav.values
    out[f"{prefix}OAV x z"] = compute_scores(ox, w_scorer)
    for a in ALPHAS:
        out[f"{prefix}inv-threshold a={a}"] = compute_scores(factor_frame(z_base, thr, med, "inv", a), w_scorer)
    out[f"{prefix}log-OAV"] = compute_scores(factor_frame(z_base, thr, med, "logoav", 0), w_scorer)
    for T in GATES:
        out[f"{prefix}OAV gate T={T}"] = compute_scores(factor_frame(z_base, thr, med, "gate", T), w_scorer)
    return out
