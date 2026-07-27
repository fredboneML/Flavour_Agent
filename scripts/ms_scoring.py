"""
MS Scoring: Melanie's weighted cluster scoring method.

Computes Raw_Score[recipe][cluster] = Σ (weight[CAS][cluster] × norm_Totalmenge[CAS])
and assigns each recipe to argmax cluster.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLUSTER_COLS = ["Unpleasant", "warm", "green", "floral", "citrus", "exotic", "Outlayer"]
_REZ_COL = "Rez.-Nr."
_IDENT_COL = "Ident"
_CAS_COL = "CAS-Nr."
_NAME_COL = "Name"
_TOTAL_COL = "Totalmenge"


def load_weight_matrix(scoring_xlsx: Path) -> pd.DataFrame:
    """Return DataFrame indexed by CAS, columns = CLUSTER_COLS, values = weights (float)."""
    raw = pd.read_excel(
        scoring_xlsx,
        sheet_name="Übersicht Score Index",
        header=None,
        skiprows=2,          # row 0 = header labels, row 1 = weight ranges, data starts row 2
    )
    raw.columns = ["CAS", "Name"] + CLUSTER_COLS
    raw = raw[raw["CAS"].notna() & (raw["CAS"] != 0)].copy()
    raw["CAS"] = raw["CAS"].astype(str).str.strip()
    for col in CLUSTER_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0.0)
    return raw.set_index("CAS")[CLUSTER_COLS]


def load_recipes(csv_path: Path, ignore_path: Path) -> pd.DataFrame:
    """
    Load recipe-ingredient rows.
    - Zeros out ignore substances (solvents/carriers).
    - Normalizes Totalmenge per recipe (sum → 1).
    Returns long-format DataFrame with columns: Rez.-Nr., CAS-Nr., Totalmenge (normalized).
    """
    def _to_float(v: object, fallback: float = 0.0) -> float:
        if v is None:
            return fallback
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v).strip().replace(",", "."))
        except ValueError:
            return fallback

    df = pd.read_csv(csv_path, dtype=str)
    df[_TOTAL_COL] = df[_TOTAL_COL].apply(_to_float)
    df = df[df[_REZ_COL].notna()].copy()

    if ignore_path.exists():
        ign = pd.read_csv(ignore_path)
        ign_idents = set(ign[_IDENT_COL].dropna().astype(str).str.strip())
        names_to_ignore = {str(n).lower().strip() for n in ign[_NAME_COL]}
        mask = df[_IDENT_COL].astype(str).str.strip().isin(ign_idents) | \
               df[_NAME_COL].str.lower().str.strip().isin(names_to_ignore)
        cas_to_ignore = set(df.loc[mask, _CAS_COL].dropna().astype(str).str.strip())
        df.loc[df[_CAS_COL].astype(str).str.strip().isin(cas_to_ignore), _TOTAL_COL] = 0.0

    # Per-recipe normalization: amounts sum to 1 (matches M1/M2 preprocessing)
    per_recipe_total = df.groupby(_REZ_COL)[_TOTAL_COL].transform("sum")
    df[_TOTAL_COL] = np.where(per_recipe_total > 0,
                               df[_TOTAL_COL] / per_recipe_total,
                               df[_TOTAL_COL])

    df[_CAS_COL] = df[_CAS_COL].astype(str).str.strip()
    return df[[_REZ_COL, _CAS_COL, _TOTAL_COL]].copy()


def compute_scores(recipes_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted score per recipe per cluster.
    CAS numbers not in the weight matrix contribute 0.

    Returns DataFrame: index = recipe_id, columns = CLUSTER_COLS (raw scores).
    """
    merged = recipes_df.join(weights, on=_CAS_COL, how="left")
    for col in CLUSTER_COLS:
        merged[col] = merged[col].fillna(0.0) * merged[_TOTAL_COL]

    scores = merged.groupby(_REZ_COL)[CLUSTER_COLS].sum()
    return scores


def normalize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Scale each cluster column by its global max across all recipes. Used for visualization only.
    Per-column normalization preserves relative recipe strength within each cluster."""
    col_max = scores.max(axis=0).replace(0, np.nan)
    return scores.div(col_max, axis=1).fillna(0.0)


def assign_clusters(scores: pd.DataFrame) -> pd.Series:
    """Return Series: recipe_id → predicted MS cluster name (argmax of raw scores)."""
    return scores.idxmax(axis=1).rename("MS_cluster")


def map_to_panel(assigned: pd.Series, ms_to_panel: dict[str, str]) -> pd.Series:
    """Remap MS cluster names → panel GT cluster names for accuracy evaluation."""
    return assigned.map(ms_to_panel).rename("MS_panel_cluster")


def load_rohstoffe_weights(scoring_xlsx: Path) -> pd.DataFrame:
    """Load Mittelwert columns from Rohstoffe_im_Cluster_Scoring as a data-driven weight matrix.
    Mittelwert = mean normalized ingredient amount across the cluster's reference recipes."""
    raw = pd.read_excel(scoring_xlsx, sheet_name="Rohstoffe_im_Cluster_Scoring", header=None)
    data = raw.iloc[2:].copy().reset_index(drop=True)
    data = data[data[0].notna() & (data[0] != 0)].copy()
    data[0] = data[0].astype(str).str.strip()
    # Mittelwert column indices (per cluster): Unpleasant=3, warm=6, green=9, floral=12,
    #                                          citrus=15, exotic=18, Outlayer=21
    mittelwert_cols = [3, 6, 9, 12, 15, 18, 21]
    result = data[[0] + mittelwert_cols].copy()
    result.columns = ["CAS"] + CLUSTER_COLS
    for col in CLUSTER_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
    result = result.drop_duplicates(subset="CAS")
    return result.set_index("CAS")[CLUSTER_COLS]


def load_avg_meli(scoring_xlsx: Path) -> pd.Series:
    """Load AvgMeli (col 3) from Rezepte sheet: average normalized amount per CAS when present."""
    raw = pd.read_excel(scoring_xlsx, sheet_name="Rezepte", header=None)
    data = raw.iloc[2:].copy()
    data = data[data[0].notna() & (data[0].astype(str) != "nan")].copy()
    data[0] = data[0].astype(str).str.strip()
    avg = pd.to_numeric(data[3], errors="coerce")
    s = pd.Series(avg.values, index=data[0].values, name="avg_meli").dropna()
    return s[~s.index.duplicated(keep="first")]


def apply_zscore_quantities(recipes_df: pd.DataFrame, avg_meli: pd.Series) -> pd.DataFrame:
    """Replace Totalmenge with Z-Score = Totalmenge / AvgMeli per CAS.
    CAS not in avg_meli keep their raw Totalmenge (no scaling)."""
    df = recipes_df.copy()
    avg_mapped = df[_CAS_COL].map(avg_meli)
    has_avg = avg_mapped.notna() & (avg_mapped > 0)
    df.loc[has_avg, _TOTAL_COL] = df.loc[has_avg, _TOTAL_COL] / avg_mapped[has_avg]
    return df


def load_melanie_scores(scoring_xlsx: Path) -> pd.DataFrame:
    """Load Melanie's pre-computed cluster scores from Rezepte_Scoring (rows 2-10).
    Returns DataFrame indexed by recipe id, columns = CLUSTER_COLS + Predicted."""
    raw = pd.read_excel(scoring_xlsx, sheet_name="Rezepte_Scoring", header=None)
    data = raw.iloc[2:11].copy()
    data.columns = ["Recipe"] + CLUSTER_COLS + ["Predicted"]
    data = data[data["Recipe"].notna()].copy()
    data["Recipe"] = data["Recipe"].astype(str).str.strip()
    for col in CLUSTER_COLS:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)
    return data.set_index("Recipe")
