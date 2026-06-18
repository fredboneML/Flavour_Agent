"""
PCA within each M1 (Label Propagation) cluster + combined frequency/PCA stats.

Mirrors scripts/pca_per_hdbscan_cluster.py but for the 7 named M1 clusters.

Outputs: outputs/m1_cluster_pca_comparison.xlsx
  - Sheet "All_Ingredients"   : long format, Cluster column, all stats per ingredient
  - Sheet "Top20_Comparison"  : top 20 per cluster side-by-side (wide format)
  - Sheet "Summary"           : PCA metadata per cluster (variance explained, etc.)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE      = Path(".")
DATA_PATH = BASE / "data/gold/Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv"
IGNORE_PATH = BASE / "data/gold/ignone_substances.csv"
CLUSTER_PATH = BASE / "outputs/cluster_assignments_expert_seeded_all_strategies.xlsx"
M1_SHEET  = "M1_label_prop"
OUT_PATH  = BASE / "outputs/m1_cluster_pca_comparison.xlsx"

N_PC = 4   # PCs to use for global importance

# Descriptive cluster names (column -> label). Order defines sheet ordering.
CLUSTER_COLUMNS = [
    ("Cluster_Walderdbeere", "Walderdbeere"),
    ("Cluster_dairy",        "dairy"),
    ("Cluster_floral",       "floral"),
    ("Cluster_fruity",       "fruity"),
    ("Cluster_green",        "green"),
    ("Cluster_unpleasant",   "unpleasant"),
    ("Cluster_warm",         "warm"),
]

# ── 1. Load and preprocess data (identical to PCA notebook) ─────────────────

def parse_de_float(val):
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")

df = pd.read_csv(DATA_PATH, dtype=str)
df["Totalmenge"] = df["Totalmenge"].apply(parse_de_float)
df["Threshold"]  = df["Threshold"].apply(parse_de_float)

if IGNORE_PATH.exists():
    ign = pd.read_csv(IGNORE_PATH)
    ign_idents     = set(ign["Ident"].dropna().astype(str).str.strip())
    names_to_ignore = {str(n).lower().strip() for n in ign["Name"]}
    mask = (
        df["Ident"].astype(str).str.strip().isin(ign_idents) |
        df["Name"].str.lower().str.strip().isin(names_to_ignore)
    )
    cas_to_ignore = set(df.loc[mask, "CAS-Nr."].dropna().astype(str).str.strip())
    df.loc[df["CAS-Nr."].astype(str).str.strip().isin(cas_to_ignore), "Totalmenge"] = 0.0
    print(f"Ignored CAS: {len(cas_to_ignore)}")

df_t = df[df["Totalmenge"] > 0].copy()
per_recipe_sum = df_t.groupby("Rez.-Nr.")["Totalmenge"].transform("sum")
df_t["Norm_Totalmenge"] = df_t["Totalmenge"] / per_recipe_sum

pivot = df_t.pivot_table(
    index="Rez.-Nr.", columns="CAS-Nr.",
    values="Norm_Totalmenge", aggfunc="sum", fill_value=0
)
print(f"Full pivot: {pivot.shape}")

cas_name = df.groupby("CAS-Nr.")["Name"].first().to_dict()

# ── 2. Parse M1 cluster assignments ─────────────────────────────────────────

m1_df = pd.read_excel(CLUSTER_PATH, sheet_name=M1_SHEET)
m1_row = m1_df.iloc[0]

def parse_recipe_list(cell_value):
    if pd.isna(cell_value):
        return []
    return [r.strip() for r in str(cell_value).split("\n") if r.strip()]

clusters = []  # list of (label, recipe_ids)
for col, label in CLUSTER_COLUMNS:
    recipes = parse_recipe_list(m1_row[col])
    # Verify all recipe IDs are in pivot
    missing = [r for r in recipes if r not in pivot.index]
    for r in missing:
        print(f"  WARNING: {r!r} ({label}) not found in pivot — skipping")
    recipes = [r for r in recipes if r in pivot.index]
    clusters.append((label, recipes))
    print(f"M1 {label}: {len(recipes)} recipes")

# ── 3. Per-cluster analysis ──────────────────────────────────────────────────

def analyse_cluster(pivot_full, recipe_ids, cluster_label):
    """Returns (stats_df, top20_df, summary_dict)."""
    sub = pivot_full.loc[recipe_ids].copy()
    n_recipes = len(sub)

    # Remove CAS columns that are entirely 0 in this cluster
    active_mask = (sub > 0).any(axis=0)
    sub_active = sub.loc[:, active_mask]
    n_active_cas = sub_active.shape[1]

    # ── Frequency statistics (all active CAS) ────────────────────────────
    freq       = (sub_active > 0).mean(axis=0)
    n_rec      = (sub_active > 0).sum(axis=0)
    mat_pres   = sub_active.replace(0, np.nan)
    avg_pres   = mat_pres.mean(axis=0)
    avg_all    = sub_active.mean(axis=0)

    # ── PCA ──────────────────────────────────────────────────────────────
    X = sub_active.values
    n_comp = min(N_PC, n_recipes - 1, n_active_cas)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X_sc)
    ev     = pca.explained_variance_ratio_ * 100
    loadings = pca.components_  # (n_comp, n_active_cas)

    # Global importance: sum |contrib| across all recipes and PCs
    contrib = X_sc[:, np.newaxis, :] * loadings[np.newaxis, :, :]  # (R, K, I)
    global_imp = np.abs(contrib).sum(axis=(0, 1))  # (n_active_cas,)

    # Build stats DataFrame for ALL active ingredients
    stats = pd.DataFrame({
        "Cluster":              cluster_label,
        "CAS":                  sub_active.columns,
        "Ingredient":           [cas_name.get(c, c) for c in sub_active.columns],
        "Frequency":            freq.values.round(4),
        "Recipes_Count":        n_rec.values.astype(int),
        "Avg_Norm_When_Present": avg_pres.values.round(6),
        "Avg_Norm_All_Recipes": avg_all.values.round(6),
        "Global_Importance":    global_imp.round(4),
    })

    # Add PC loadings
    for k in range(n_comp):
        stats[f"PC{k+1}_Loading"] = loadings[k].round(4)

    # Add rank by Global_Importance within cluster
    stats = stats.sort_values("Global_Importance", ascending=False)
    stats.insert(1, "PCA_Rank", range(1, len(stats) + 1))
    stats = stats.reset_index(drop=True)

    top20 = stats.head(20).copy()

    summary = {
        "Cluster":          cluster_label,
        "N_Recipes":        n_recipes,
        "N_Active_CAS":     n_active_cas,
        "N_PCs_Computed":   n_comp,
        "PC1_Var_Expl_%":   round(float(ev[0]), 2) if len(ev) > 0 else None,
        "PC2_Var_Expl_%":   round(float(ev[1]), 2) if len(ev) > 1 else None,
        "PC3_Var_Expl_%":   round(float(ev[2]), 2) if len(ev) > 2 else None,
        "PC4_Var_Expl_%":   round(float(ev[3]), 2) if len(ev) > 3 else None,
        "Total_Var_Expl_%": round(float(ev.sum()), 2),
    }

    return stats, top20, summary

results = []  # list of (label, stats, top20, summary)
for label, recipes in clusters:
    stats, top20, summary = analyse_cluster(pivot, recipes, label)
    results.append((label, stats, top20, summary))
    print(f"\n{label}: {summary}")

# ── 4. Build output sheets ───────────────────────────────────────────────────

# Sheet 1 – long format all ingredients (clusters in order, sorted by importance within each)
all_ingredients = pd.concat([r[1] for r in results], ignore_index=True)

# Sheet 2 – wide comparison of top 20 per cluster (one block of columns per cluster)
SUB_COLS = [
    ("CAS",            "CAS"),
    ("Ingredient",     "Ingredient"),
    ("Global_Imp",     "Global_Importance"),
    ("Frequency",      "Frequency"),
    ("Recipes_Count",  "Recipes_Count"),
    ("Avg_Norm_Pres",  "Avg_Norm_When_Present"),
    ("PC1_Loading",    "PC1_Loading"),
    ("PC2_Loading",    "PC2_Loading"),
]

wide_data = {"Rank": range(1, 21)}
for label, _stats, top20, _summary in results:
    t = top20.reindex(range(20))  # pad to 20 rows with NaN for small clusters
    for suffix, src_col in SUB_COLS:
        col_name = f"{label}_{suffix}"
        wide_data[col_name] = t[src_col].values if src_col in t.columns else [np.nan] * 20
wide = pd.DataFrame(wide_data)

# Sheet 3 – summary
summary_df = pd.DataFrame([r[3] for r in results])

# ── 5. Export ────────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    all_ingredients.to_excel(writer, sheet_name="All_Ingredients", index=False)
    wide.to_excel(writer, sheet_name="Top20_Comparison", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"\nSaved: {OUT_PATH}")
print(f"  All_Ingredients : {len(all_ingredients)} rows  "
      f"({', '.join(f'{lbl}={len(st)}' for lbl, st, _, _ in results)})")
print(f"  Top20_Comparison: top 20 per cluster side-by-side ({len(results)} clusters)")
print(f"  Summary         : PCA metadata ({len(summary_df)} clusters)")
