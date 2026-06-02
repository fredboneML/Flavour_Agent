"""
PCA within each HDBSCAN cluster + combined frequency/PCA statistics export.

Outputs: outputs/hdbscan_cluster_pca_comparison.xlsx
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
CLUSTER_PATH = BASE / "outputs/cluster_assignments_all_algorithms.xlsx"
OUT_PATH  = BASE / "outputs/hdbscan_cluster_pca_comparison.xlsx"

N_PC = 4   # PCs to use for global importance

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

# ── 2. Parse HDBSCAN cluster assignments ────────────────────────────────────

clusters_df = pd.read_excel(CLUSTER_PATH)
hdbscan_row = clusters_df[clusters_df["Algorithm"] == "HDBSCAN"].iloc[0]

def parse_recipe_list(cell_value):
    if pd.isna(cell_value):
        return []
    return [r.strip() for r in str(cell_value).split("\n") if r.strip()]

c1_recipes = parse_recipe_list(hdbscan_row["Cluster_1"])
c2_recipes = parse_recipe_list(hdbscan_row["Cluster_2"])

print(f"HDBSCAN C1: {len(c1_recipes)} recipes | C2: {len(c2_recipes)} recipes")

# Verify all recipe IDs are in pivot
for r in c1_recipes + c2_recipes:
    if r not in pivot.index:
        print(f"  WARNING: {r!r} not found in pivot — skipping")

c1_recipes = [r for r in c1_recipes if r in pivot.index]
c2_recipes = [r for r in c2_recipes if r in pivot.index]
print(f"After matching: C1={len(c1_recipes)}, C2={len(c2_recipes)}")

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

c1_stats, c1_top20, c1_summary = analyse_cluster(pivot, c1_recipes, "C1")
c2_stats, c2_top20, c2_summary = analyse_cluster(pivot, c2_recipes, "C2")

print(f"\nC1: {c1_summary}")
print(f"C2: {c2_summary}")

# ── 4. Build output sheets ───────────────────────────────────────────────────

# Sheet 1 – long format all ingredients (C1 then C2, sorted by Global_Importance within each)
all_ingredients = pd.concat([c1_stats, c2_stats], ignore_index=True)

# Sheet 2 – wide comparison of top 20 per cluster
def build_top20_comparison(t1, t2):
    prefix_c1 = {col: f"C1_{col}" for col in t1.columns if col not in ("CAS", "Ingredient", "Cluster")}
    prefix_c2 = {col: f"C2_{col}" for col in t2.columns if col not in ("CAS", "Ingredient", "Cluster")}
    t1r = t1.rename(columns=prefix_c1).drop(columns=["Cluster"])
    t2r = t2.rename(columns=prefix_c2).drop(columns=["Cluster"])
    t1r["Comparison_Rank"] = range(1, len(t1r) + 1)
    t2r["Comparison_Rank"] = range(1, len(t2r) + 1)
    return t1r.merge(t2r, on="Comparison_Rank", how="outer", suffixes=("", "_c2"))

wide = pd.DataFrame({
    "Rank":              range(1, 21),
    "C1_CAS":            c1_top20["CAS"].values,
    "C1_Ingredient":     c1_top20["Ingredient"].values,
    "C1_Global_Imp":     c1_top20["Global_Importance"].values,
    "C1_Frequency":      c1_top20["Frequency"].values,
    "C1_Recipes_Count":  c1_top20["Recipes_Count"].values,
    "C1_Avg_Norm_Pres":  c1_top20["Avg_Norm_When_Present"].values,
    "C1_PC1_Loading":    c1_top20["PC1_Loading"].values,
    "C1_PC2_Loading":    c1_top20["PC2_Loading"].values,
    "C2_CAS":            c2_top20["CAS"].values,
    "C2_Ingredient":     c2_top20["Ingredient"].values,
    "C2_Global_Imp":     c2_top20["Global_Importance"].values,
    "C2_Frequency":      c2_top20["Frequency"].values,
    "C2_Recipes_Count":  c2_top20["Recipes_Count"].values,
    "C2_Avg_Norm_Pres":  c2_top20["Avg_Norm_When_Present"].values,
    "C2_PC1_Loading":    c2_top20["PC1_Loading"].values,
    "C2_PC2_Loading":    c2_top20["PC2_Loading"].values,
})

# Sheet 3 – summary
summary_df = pd.DataFrame([c1_summary, c2_summary])

# ── 5. Export ────────────────────────────────────────────────────────────────

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    all_ingredients.to_excel(writer, sheet_name="All_Ingredients", index=False)
    wide.to_excel(writer, sheet_name="Top20_Comparison", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"\nSaved: {OUT_PATH}")
print(f"  All_Ingredients : {len(all_ingredients)} rows  (C1={len(c1_stats)}, C2={len(c2_stats)})")
print(f"  Top20_Comparison: top 20 per cluster side-by-side")
print(f"  Summary         : PCA metadata")
