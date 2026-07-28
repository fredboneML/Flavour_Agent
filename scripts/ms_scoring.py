"""
MS Scoring: Melanie's weighted cluster scoring method.

Computes Raw_Score[recipe][cluster] = Σ (weight[CAS][cluster] × norm_Totalmenge[CAS])
and assigns each recipe to argmax cluster.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Words stripped from ingredient names when building CAS alias lookups.
_NAME_QUALIFIERS = re.compile(
    r"\b(natürlich|natural|halal|kosher|bg|kbg|nf|food\s+grade|ex\s+eugenol|"
    r"ex\s+lignin|ex\s+wood|ex|mild|80%|90%|95%|99%)\b",
    flags=re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """Strip certification/quality qualifiers to get base chemical name."""
    s = _NAME_QUALIFIERS.sub("", name.lower())
    s = re.sub(r"[,;()\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

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


def load_avg_meli(scoring_xlsx: Path, csv_path: Path | None = None) -> pd.Series:
    """Load AvgMeli (col 3) from Rezepte sheet: average normalized amount per CAS when present.

    If csv_path is provided, extend the result with name-based aliases so that CAS variants
    that appear in the recipe CSV under different CAS numbers (e.g. "Cycloten natürlich Halal"
    80-71-7 vs "Cycloten Halal Kosher" 765-70-8) still receive a valid AvgMeli.
    """
    raw = pd.read_excel(scoring_xlsx, sheet_name="Rezepte", header=None)
    data = raw.iloc[2:].copy()
    data = data[data[0].notna() & (data[0].astype(str) != "nan")].copy()
    data[0] = data[0].astype(str).str.strip()
    data[1] = data[1].astype(str).str.strip()
    avg = pd.to_numeric(data[3], errors="coerce")
    s = pd.Series(avg.values, index=data[0].values, name="avg_meli").dropna()
    s = s[~s.index.duplicated(keep="first")]

    if csv_path is None:
        return s

    # Build normalized-name → AvgMeli from Rezepte reference data
    name_map: dict[str, float] = {}
    for cas, am in s.items():
        rows = data[data[0] == cas]
        if rows.empty:
            continue
        norm = _normalize_name(str(rows.iloc[0, 1]))
        if norm and norm not in name_map:
            name_map[norm] = float(am)

    # For each CAS in CSV with no (or zero) AvgMeli, attempt a name-based alias.
    # A CAS may already be in s.index with value 0 (ingredient not in Melanie's reference
    # recipes) but its Halal/Kosher variant may have a valid AvgMeli — allow alias in that case.
    csv_df = pd.read_csv(csv_path, dtype=str)[[_CAS_COL, _NAME_COL]].drop_duplicates()
    csv_df[_CAS_COL] = csv_df[_CAS_COL].astype(str).str.strip()
    extras: dict[str, float] = {}
    for _, row in csv_df.iterrows():
        cas = row[_CAS_COL]
        if not cas or cas.lower() in ("nan", "none", ""):
            continue  # skip rows with no valid CAS
        if cas in s.index and float(s[cas]) > 0:
            continue  # already has a valid AvgMeli
        norm = _normalize_name(str(row[_NAME_COL]))
        if norm in name_map:
            extras[cas] = name_map[norm]

    if extras:
        extras_series = pd.Series(extras, name="avg_meli")
        # Drop zero-AvgMeli entries from s that are being replaced by aliases
        s_clean = s.drop(index=[c for c in extras if c in s.index])
        return pd.concat([s_clean, extras_series])
    return s


def apply_zscore_quantities(recipes_df: pd.DataFrame, avg_meli: pd.Series) -> pd.DataFrame:
    """Replace Totalmenge with Z-Score = Totalmenge / AvgMeli per CAS.
    CAS with no AvgMeli or AvgMeli=0 are zeroed out (matches Melanie's formula exactly)."""
    df = recipes_df.copy()
    avg_mapped = df[_CAS_COL].map(avg_meli)
    has_avg = avg_mapped.notna() & (avg_mapped > 0)
    df.loc[has_avg,  _TOTAL_COL] = df.loc[has_avg, _TOTAL_COL] / avg_mapped[has_avg]
    df.loc[~has_avg, _TOTAL_COL] = 0.0
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


# ── Extended scoring variants ─────────────────────────────────────────────────

def rescale_warm_weights(weights: pd.DataFrame, factor: float = 0.5) -> pd.DataFrame:
    """Halve warm column weights to match the 0-10 scale of other clusters (warm uses 0-20)."""
    w = weights.copy()
    w["warm"] = w["warm"] * factor
    return w


def apply_squared_zscore_quantities(recipes_df: pd.DataFrame, avg_meli: pd.Series) -> pd.DataFrame:
    """Replace Totalmenge with (Totalmenge / AvgMeli)² — amplifies signature ingredients
    quadratically; an ingredient at 3× typical concentration contributes 9× not 3×."""
    df = recipes_df.copy()
    avg_mapped = df[_CAS_COL].map(avg_meli)
    has_avg = avg_mapped.notna() & (avg_mapped > 0)
    df.loc[has_avg, _TOTAL_COL] = (df.loc[has_avg, _TOTAL_COL] / avg_mapped[has_avg]) ** 2
    return df


def compute_tfidf_scores(recipes_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF style scoring: down-weight CAS that appear in many recipes (low cluster signal).
    IDF[CAS] = log(N_recipes / N_recipes_containing_CAS).
    Score = Σ weight[CAS][cluster] × Totalmenge[CAS] × IDF[CAS]."""
    n_recipes = recipes_df[_REZ_COL].nunique()
    present = recipes_df[recipes_df[_TOTAL_COL] > 0]
    cas_doc_count = present.groupby(_CAS_COL)[_REZ_COL].nunique()
    idf = np.log((n_recipes / cas_doc_count.clip(lower=1)) + 1)

    df = recipes_df.copy()
    df[_TOTAL_COL] = df[_TOTAL_COL] * df[_CAS_COL].map(idf).fillna(0.0)
    return compute_scores(df, weights)


def compute_topk_scores(recipes_df: pd.DataFrame, weights: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Score using only the top-k ingredients per recipe by total weight contribution.
    Filters out trace amounts that add noise."""
    merged = recipes_df.join(weights, on=_CAS_COL, how="left")
    for col in CLUSTER_COLS:
        merged[col] = merged[col].fillna(0.0) * merged[_TOTAL_COL]
    merged["_total_contrib"] = merged[CLUSTER_COLS].sum(axis=1)

    def _topk(grp: pd.DataFrame) -> pd.Series:
        return grp.nlargest(k, "_total_contrib")[CLUSTER_COLS].sum()

    return merged.groupby(_REZ_COL).apply(_topk)


def compute_cosine_scores(recipes_df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between each recipe's ingredient vector and each cluster's weight vector.
    Removes bias from recipe size (number of matched CAS)."""
    pivot = recipes_df.pivot_table(
        index=_REZ_COL, columns=_CAS_COL, values=_TOTAL_COL, aggfunc="sum", fill_value=0.0
    )
    common = pivot.columns.intersection(weights.index)
    w_clean = weights[~weights.index.duplicated(keep="first")]
    common = pivot.columns.intersection(w_clean.index)
    R = pivot[common].values.astype(float)
    W = w_clean.loc[common].values.astype(float)

    R_norm = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    W_norm = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)

    sim = R_norm @ W_norm
    return pd.DataFrame(sim, index=pivot.index, columns=CLUSTER_COLS)


def ensemble_scores(*score_dfs: pd.DataFrame) -> pd.DataFrame:
    """Average multiple score matrices after normalizing each to sum=1 per recipe."""
    normed = []
    for df in score_dfs:
        row_sum = df.sum(axis=1).replace(0, np.nan)
        normed.append(df.div(row_sum, axis=0).fillna(0.0))
    combined = normed[0]
    for n in normed[1:]:
        combined = combined.add(n, fill_value=0.0)
    return combined / len(normed)
