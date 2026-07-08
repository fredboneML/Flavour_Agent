"""
PCA within each cluster using PANEL/EXPERT-VERIFIED true labels (not model output).

Motivation
----------
The per-cluster PCA in `scripts/pca_per_m1_cluster.py` subsets recipes by the M1
model prediction. Because M1/M2 misclassify some recipes (especially in *green* and
*unpleasant*), wrong points pollute each cluster's PCA and degrade the ingredient
statistics the experts mine for rules. This script keeps only recipes whose cluster
membership is verified, so the exported statistics are clean.

True-label sources (precedence high -> low)
--------------------------------------------
1. panel_green       — `Verkostung Cluster Green ...xlsx` / Ergebnisse, panel agreement > 50%  -> green
2. panel_unpleasant  — `Verkostung Cluster unpleasant ...xlsx` / Ergebnisse, agreement > 50%   -> unpleasant
3. anchor            — Vorgabe `Target Recipes` per cluster (expert-pinned)                     -> that cluster
4. m1m2_correct      — `Panel_Scorecard`: GT not in {green,unpleasant} AND both M1 & M2 = ✓     -> GT cluster

Recipes rejected by the panel (agreement <= 50%) are dropped entirely.

Preprocessing (identical to pca_v2_noOT / pca_per_m1_cluster)
------------------------------------------------------------
Zero out ignore substances, per-recipe normalization of Totalmenge (sums to 1),
pivot Recipe x CAS. No odour-type, no threshold.

Output: outputs/pca_true_cluster_ingredient_statistics.xlsx
  - All_Ingredients   : long format, all clusters, full stats incl. median columns
  - Top20_Comparison  : top-20 per cluster side-by-side (wide)
  - Summary           : PCA metadata per cluster (variance explained, N, PCA status)
  - Recipe_Labels     : recipe -> true_cluster, source (provenance / handover)
  - Dropped_Recipes   : rejected-by-panel + unmatched IDs (auditable)
Plus summary PNGs under outputs/ (green & unpleasant biplots, variance bar, grid, heatmaps).
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE         = Path(".")
DATA_PATH    = BASE / "data/gold/Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv"
IGNORE_PATH  = BASE / "data/gold/ignone_substances.csv"
VORGABE_PATH = BASE / "data/gold/Erdbeer Clustering Sensorik Vorgabe.xlsx"
GREEN_PATH   = BASE / "data/gold/Verkostung Cluster Green KI vom 07_07_2026.xlsx"
UNPL_PATH    = BASE / "data/gold/Verkostung Cluster unpleasant KI vom 03_07_2026.xlsx"
SCORE_PATH   = BASE / "outputs/cluster_assignments_expert_seeded_all_strategies.xlsx"
OUT_PATH     = BASE / "outputs/pca_true_cluster_ingredient_statistics.xlsx"

N_PC        = 4      # PCs to use for global importance
PANEL_CUTOFF = 50.0  # panel agreement %; strictly greater than this counts as a true member
MIN_PCA_RECIPES = 3  # need >=3 recipes for a non-trivial PCA (>=1 PC)

# Cluster display order (green & unpleasant first — the analysis focus)
CLUSTER_ORDER = ["green", "unpleasant", "warm", "dairy", "floral", "fruity", "Walderdbeere"]

# Clusters populated by guarded model consensus (no panel tasting available for these).
# green & unpleasant are DELIBERATELY excluded: we have panel truth there and the models
# are known-unreliable, so we trust the panel only.
CONSENSUS_CLUSTERS = ["warm", "dairy", "floral", "fruity", "Walderdbeere"]


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_de_float(val):
    if pd.isna(val):
        return float("nan")
    s = str(val).strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def match_recipe(rid, recipes):
    """Map a source id ('187.657', '185.237') to a dataset recipe ('187.657P', '185.237H').

    Reuses the canonical resolution from notebooks/_build_v3_expert_seeded.py::_match_verk.
    """
    rid = str(rid).strip()
    if rid in recipes:
        return rid
    pref = [r for r in recipes if r.startswith(rid)]
    if pref:
        return pref[0]
    num = rid.rstrip("PpHhNnXx ")            # strip trailing letter suffixes
    pref = [r for r in recipes if r.startswith(num)]
    return pref[0] if pref else None


def _find_header_col(header_rows, needle):
    """Return the column index whose (multi-row) header contains `needle`."""
    for ci in range(header_rows.shape[1]):
        for ri in range(header_rows.shape[0]):
            v = header_rows.iat[ri, ci]
            if pd.notna(v) and needle.lower() in str(v).lower():
                return ci
    return None


def read_panel(path, cluster, recipes):
    """Return (kept, rejected) lists of (matched_recipe, pct) from a Verkostung Ergebnisse sheet."""
    raw = pd.read_excel(path, sheet_name="Ergebnisse", header=None)
    hdr = raw.iloc[:2]
    c_pct = _find_header_col(hdr, "Prozent")
    if c_pct is None:
        raise ValueError(f"No 'Prozent %' column found in {path}")
    kept, rejected = [], []
    for ri in range(2, raw.shape[0]):
        rid = raw.iat[ri, 0]
        if pd.isna(rid):
            continue                          # skip the trailing Mittelwert/average row
        rid = str(rid).strip()
        try:
            pct = float(raw.iat[ri, c_pct])
        except (TypeError, ValueError):
            continue
        matched = match_recipe(rid, recipes)
        target = matched if matched else rid
        (kept if pct > PANEL_CUTOFF else rejected).append((target, rid, pct, matched is not None))
    return kept, rejected


# ── 1. Load and preprocess data (identical to PCA notebook / pca_per_m1_cluster) ──

df = pd.read_csv(DATA_PATH, dtype=str)
df["Totalmenge"] = df["Totalmenge"].apply(parse_de_float)

if IGNORE_PATH.exists():
    ign = pd.read_csv(IGNORE_PATH)
    ign_idents      = set(ign["Ident"].dropna().astype(str).str.strip())
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
    values="Norm_Totalmenge", aggfunc="sum", fill_value=0,
)
print(f"Full pivot: {pivot.shape}")

recipes  = list(pivot.index)
cas_name = df.groupby("CAS-Nr.")["Name"].first().to_dict()


# ── 2. Build the recipe -> true_cluster label table (precedence enforced) ────────

label_of  = {}   # recipe -> cluster
source_of = {}   # recipe -> source tag
dropped   = []   # (recipe_id, reason, detail)


def assign(recipe, cluster, source):
    if recipe in label_of:
        return False                          # higher-precedence source already claimed it
    label_of[recipe]  = cluster
    source_of[recipe] = source
    return True


# 2a. Green panel  (agreement > 50% -> green)
gk, gr = read_panel(GREEN_PATH, "green", recipes)
for target, rid, pct, ok in gk:
    if not ok:
        dropped.append((rid, "unmatched_id", f"green panel {pct:.1f}%"));  continue
    assign(target, "green", "panel_green")
for target, rid, pct, ok in gr:
    dropped.append((rid, "panel_rejected", f"green {pct:.1f}%"))

# 2b. Unpleasant panel  (agreement > 50% -> unpleasant)
uk, ur = read_panel(UNPL_PATH, "unpleasant", recipes)
for target, rid, pct, ok in uk:
    if not ok:
        dropped.append((rid, "unmatched_id", f"unpleasant panel {pct:.1f}%"));  continue
    assign(target, "unpleasant", "panel_unpleasant")
for target, rid, pct, ok in ur:
    dropped.append((rid, "panel_rejected", f"unpleasant {pct:.1f}%"))

# 2c. Expert anchors (Vorgabe Target Recipes)
vg = pd.read_excel(VORGABE_PATH, sheet_name="Tabelle1")
for _, row in vg.iterrows():
    cluster = row.get("Cluster")
    targets = row.get("Target Recipes")
    if pd.isna(cluster) or pd.isna(targets) or str(cluster).strip() == "Black List":
        continue
    cluster = str(cluster).strip()
    for raw_id in str(targets).split(";"):
        raw_id = raw_id.strip()
        if not raw_id or raw_id.lower() == "nan":
            continue
        matched = match_recipe(raw_id, recipes)
        if matched is None:
            dropped.append((raw_id, "unmatched_id", f"anchor {cluster}"));  continue
        assign(matched, cluster, "anchor")

# 2d. M1&M2 both correct, for clusters other than green/unpleasant
sc = pd.read_excel(SCORE_PATH, sheet_name="Panel_Scorecard")
for _, row in sc.iterrows():
    gt = str(row["GT"]).strip()
    if gt in ("green", "unpleasant"):
        continue
    if "✓" in str(row["M1"]) and "✓" in str(row["M2"]):
        matched = match_recipe(row["Recipe"], recipes)
        if matched is None:
            dropped.append((str(row["Recipe"]), "unmatched_id", f"m1m2_correct {gt}"));  continue
        assign(matched, gt, "m1m2_correct")

# 2e. Guarded model consensus — populate clusters with no panel tasting.
#     A recipe joins cluster C only if ALL of:
#       (i)   M1 == M2 == M2_kw == C                       (independent-method consensus)
#       (ii)  no conflicting panel/expert positive truth   (panel override)
#       (iii) its nearest confirmed-cluster centroid == C  (ingredient-space geometry)
#     Green & unpleasant are excluded (panel-only). Verified recipes already in
#     label_of are never overwritten (precedence).

# Positive ground-truth map (recipe -> cluster) from panel + scorecard, for the override.
panel_truth = {}
for target, rid, pct, ok in gk:
    if ok:
        panel_truth[target] = "green"
for target, rid, pct, ok in uk:
    if ok:
        panel_truth[target] = "unpleasant"
for _, row in sc.iterrows():                      # includes 'corrected' rows (e.g. 187.167P -> unpleasant)
    matched = match_recipe(row["Recipe"], recipes)
    if matched is not None:
        panel_truth[matched] = str(row["GT"]).strip()

# Geometry guard reference: the verified seed recipes (panel + anchor + m1m2_correct).
# Compared in STANDARDIZED feature space (z-score each CAS across all recipes). This
# de-emphasizes the shared fruity/sweet base that dominates raw cosine and lets the
# distinguishing ingredients drive the geometry — the same reasoning behind StandardScaler
# in the PCA and the S6 contrast strategy. We take the single NEAREST verified recipe
# (1-NN, unbiased by cluster size) rather than an averaged centroid (biased by density).
_Z = StandardScaler().fit_transform(pivot.values)
Z = pd.DataFrame(_Z, index=pivot.index, columns=pivot.columns)
seed_recipes = list(label_of.keys())            # verified before consensus pass


def nearest_seed_cluster(recipe):
    v = Z.loc[recipe].values
    dists = {r: float(np.linalg.norm(v - Z.loc[r].values)) for r in seed_recipes}
    nn = min(dists, key=dists.get)
    return label_of[nn], nn


sc_cmp = pd.read_excel(SCORE_PATH, sheet_name="Strategy_Comparison")
for col in ("M1_label_prop", "M2_rule_based", "M2_kw"):
    sc_cmp[col] = sc_cmp[col].astype(str).str.strip()

n_consensus = 0
geom_flag = {}   # recipe -> bool (nearest verified recipe agrees with consensus cluster)
for _, row in sc_cmp.iterrows():
    m1, m2, m2kw = row["M1_label_prop"], row["M2_rule_based"], row["M2_kw"]
    if not (m1 == m2 == m2kw) or m1 not in CONSENSUS_CLUSTERS:
        continue
    cluster = m1
    matched = match_recipe(row["Recipe"], recipes)
    if matched is None:
        dropped.append((str(row["Recipe"]), "unmatched_id", f"consensus {cluster}"));  continue
    if matched in label_of:
        continue                                   # verified label already wins (precedence)
    # (ii) panel override
    truth = panel_truth.get(matched)
    if truth is not None and truth != cluster:
        dropped.append((matched, "consensus_panel_override", f"models={cluster} but panel={truth}"))
        continue
    # (iii) geometry: SOFT corroboration flag (not a gate). Nearest verified recipe in
    #       standardized space; recorded per recipe so experts see chemical confidence.
    near, nn = nearest_seed_cluster(matched)
    if assign(matched, cluster, "consensus"):
        geom_flag[matched] = (near == cluster)
        n_consensus += 1
print(f"Consensus additions: {n_consensus} "
      f"({sum(geom_flag.values())} geometry-corroborated)")

labels_df = (
    pd.DataFrame({"Recipe": list(label_of), "true_cluster": list(label_of.values())})
    .assign(
        source=lambda d: d["Recipe"].map(source_of),
        geom_corroborated=lambda d: d["Recipe"].map(
            lambda r: {True: "yes", False: "no"}.get(geom_flag.get(r), "")  # blank = not a consensus recipe
        ),
    )
    .sort_values(["true_cluster", "Recipe"]).reset_index(drop=True)
)

# Cluster -> recipe list, in the display order
clusters = []
for label in CLUSTER_ORDER:
    recs = sorted(labels_df.loc[labels_df["true_cluster"] == label, "Recipe"])
    recs = [r for r in recs if r in pivot.index]
    clusters.append((label, recs))
# any unexpected clusters not in CLUSTER_ORDER
for label in sorted(set(labels_df["true_cluster"]) - set(CLUSTER_ORDER)):
    recs = sorted(labels_df.loc[labels_df["true_cluster"] == label, "Recipe"])
    clusters.append((label, [r for r in recs if r in pivot.index]))

print("\nTrue-label cluster sizes:")
for label, recs in clusters:
    print(f"  {label:14s}: {len(recs):2d} recipes  {recs}")
print(f"Dropped: {len(dropped)} (panel_rejected / unmatched)")


# ── 3. Per-cluster analysis (frequency + median stats + PCA when feasible) ───────

def analyse_cluster(pivot_full, recipe_ids, cluster_label):
    """Return (stats_df, top20_df, summary_dict, scores_or_None, loadings_or_None, active_cols)."""
    sub = pivot_full.loc[recipe_ids].copy()
    n_recipes = len(sub)

    active_mask = (sub > 0).any(axis=0)
    sub_active  = sub.loc[:, active_mask]
    n_active_cas = sub_active.shape[1]

    # Frequency / concentration statistics (avg AND median, when-present AND all-recipes)
    freq     = (sub_active > 0).mean(axis=0)
    n_rec    = (sub_active > 0).sum(axis=0)
    mat_pres = sub_active.replace(0, np.nan)
    avg_pres = mat_pres.mean(axis=0)
    med_pres = mat_pres.median(axis=0)
    avg_all  = sub_active.mean(axis=0)
    med_all  = sub_active.median(axis=0)

    stats = pd.DataFrame({
        "Cluster":               cluster_label,
        "CAS":                   sub_active.columns,
        "Ingredient":            [cas_name.get(c, c) for c in sub_active.columns],
        "Frequency":             freq.values.round(4),
        "Recipes_Count":         n_rec.values.astype(int),
        "Avg_Norm_When_Present": avg_pres.values.round(6),
        "Med_Norm_When_Present": med_pres.values.round(6),
        "Avg_Norm_All_Recipes":  avg_all.values.round(6),
        "Med_Norm_All_Recipes":  med_all.values.round(6),
    })

    n_comp = min(N_PC, n_recipes - 1, n_active_cas)
    scores = loadings = None
    ev = np.array([])

    if n_comp >= 1 and n_recipes >= MIN_PCA_RECIPES:
        X       = sub_active.values
        X_sc    = StandardScaler().fit_transform(X)
        pca     = PCA(n_components=n_comp, random_state=42)
        scores  = pca.fit_transform(X_sc)
        ev       = pca.explained_variance_ratio_ * 100
        loadings = pca.components_                                   # (n_comp, n_active_cas)
        contrib  = X_sc[:, np.newaxis, :] * loadings[np.newaxis, :, :]
        global_imp = np.abs(contrib).sum(axis=(0, 1))               # (n_active_cas,)
        stats["Global_Importance"] = global_imp.round(4)
        for k in range(n_comp):
            stats[f"PC{k+1}_Loading"] = loadings[k].round(4)
        stats = stats.sort_values("Global_Importance", ascending=False)
        pca_status = "ok"
    else:
        stats["Global_Importance"] = np.nan
        stats = stats.sort_values("Frequency", ascending=False)
        pca_status = f"skipped (n_recipes={n_recipes} < {MIN_PCA_RECIPES})"

    stats.insert(1, "PCA_Rank", range(1, len(stats) + 1))
    stats = stats.reset_index(drop=True)
    top20 = stats.head(20).copy()

    summary = {
        "Cluster":          cluster_label,
        "N_Recipes":        n_recipes,
        "N_Active_CAS":     n_active_cas,
        "N_PCs_Computed":   int(n_comp) if pca_status == "ok" else 0,
        "PCA_Status":       pca_status,
        "PC1_Var_Expl_%":   round(float(ev[0]), 2) if len(ev) > 0 else None,
        "PC2_Var_Expl_%":   round(float(ev[1]), 2) if len(ev) > 1 else None,
        "PC3_Var_Expl_%":   round(float(ev[2]), 2) if len(ev) > 2 else None,
        "PC4_Var_Expl_%":   round(float(ev[3]), 2) if len(ev) > 3 else None,
        "Total_Var_Expl_%": round(float(ev.sum()), 2) if len(ev) > 0 else None,
    }
    return stats, top20, summary, scores, loadings, list(sub_active.columns)


results = []   # (label, stats, top20, summary, scores, loadings, active_cols, recipe_ids)
for label, recs in clusters:
    if not recs:
        print(f"  {label}: 0 recipes — skipped entirely")
        continue
    stats, top20, summary, scores, loadings, active_cols = analyse_cluster(pivot, recs, label)
    results.append((label, stats, top20, summary, scores, loadings, active_cols, recs))
    print(f"  {label:14s}: {summary['PCA_Status']}, PC1={summary['PC1_Var_Expl_%']}%")


# ── 4. Build output sheets ───────────────────────────────────────────────────────

all_ingredients = pd.concat([r[1] for r in results], ignore_index=True)

SUB_COLS = [
    ("CAS",            "CAS"),
    ("Ingredient",     "Ingredient"),
    ("Global_Imp",     "Global_Importance"),
    ("Frequency",      "Frequency"),
    ("Recipes_Count",  "Recipes_Count"),
    ("Avg_Norm_Pres",  "Avg_Norm_When_Present"),
    ("Med_Norm_Pres",  "Med_Norm_When_Present"),
    ("PC1_Loading",    "PC1_Loading"),
    ("PC2_Loading",    "PC2_Loading"),
]
wide_data = {"Rank": range(1, 21)}
for label, _stats, top20, *_ in results:
    t = top20.reindex(range(20))
    for suffix, src_col in SUB_COLS:
        wide_data[f"{label}_{suffix}"] = t[src_col].values if src_col in t.columns else [np.nan] * 20
wide = pd.DataFrame(wide_data)

summary_df = pd.DataFrame([r[3] for r in results])
dropped_df = pd.DataFrame(dropped, columns=["Recipe_ID", "Reason", "Detail"])

with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
    all_ingredients.to_excel(writer, sheet_name="All_Ingredients", index=False)
    wide.to_excel(writer, sheet_name="Top20_Comparison", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    labels_df.to_excel(writer, sheet_name="Recipe_Labels", index=False)
    dropped_df.to_excel(writer, sheet_name="Dropped_Recipes", index=False)

print(f"\nSaved workbook: {OUT_PATH}")


# ── 5. Summary plots ─────────────────────────────────────────────────────────────

OUT_DIR = BASE / "outputs"
pca_results = [r for r in results if r[4] is not None]   # clusters with real PCA


def biplot(label, stats, scores, loadings, active_cols, recipe_ids, n_arrows=10):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(scores[:, 0], scores[:, 1], s=70, c="#2b6cb0", edgecolor="white", zorder=3)
    for i, rid in enumerate(recipe_ids):
        ax.annotate(rid, (scores[i, 0], scores[i, 1]), fontsize=7, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")
    if loadings.shape[0] >= 2:
        imp = np.abs(loadings[:2]).sum(axis=0)
        top = np.argsort(imp)[::-1][:n_arrows]
        scale = 0.9 * max(np.abs(scores[:, :2]).max(), 1e-9) / max(np.abs(loadings[:2]).max(), 1e-9)
        for j in top:
            ax.arrow(0, 0, loadings[0, j] * scale, loadings[1, j] * scale,
                     color="#c05621", alpha=0.7, head_width=scale * 0.02, zorder=2)
            ax.annotate(cas_name.get(active_cols[j], active_cols[j])[:22],
                        (loadings[0, j] * scale, loadings[1, j] * scale),
                        fontsize=7, color="#7b341e")
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{label} — true-label PCA (n={len(recipe_ids)})")
    fig.tight_layout()
    p = OUT_DIR / f"pca_true_cluster_{label}_biplot.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# 5a. Focus biplots (green & unpleasant if they have PCA), plus any other PCA cluster
for label, stats, top20, summary, scores, loadings, active_cols, recs in pca_results:
    biplot(label, stats, scores, loadings, active_cols, recs)

# 5b. Variance-explained bar chart across clusters
if pca_results:
    fig, ax = plt.subplots(figsize=(9, 5))
    labs = [r[0] for r in pca_results]
    pc1  = [r[3]["PC1_Var_Expl_%"] or 0 for r in pca_results]
    pc2  = [r[3]["PC2_Var_Expl_%"] or 0 for r in pca_results]
    x = np.arange(len(labs))
    ax.bar(x - 0.2, pc1, 0.4, label="PC1", color="#2b6cb0")
    ax.bar(x + 0.2, pc2, 0.4, label="PC2", color="#c05621")
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("Per-cluster PCA variance (true labels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_true_cluster_variance.png", dpi=130); plt.close(fig)

# 5c. Summary grid: PC1–PC2 scatter tile per PCA cluster
if pca_results:
    n = len(pca_results); ncol = min(3, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.2 * nrow), squeeze=False)
    for idx, (label, stats, top20, summary, scores, loadings, active_cols, recs) in enumerate(pca_results):
        ax = axes[idx // ncol][idx % ncol]
        ax.scatter(scores[:, 0], scores[:, 1], s=45, c="#2b6cb0", edgecolor="white")
        ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
        ax.set_title(f"{label} (n={len(recs)}, PC1={summary['PC1_Var_Expl_%']}%)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Within-cluster PCA on panel/expert-verified true labels", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_true_cluster_summary_grid.png", dpi=130); plt.close(fig)

# 5d. Top-ingredient importance heatmap for green & unpleasant
for label, stats, top20, summary, scores, loadings, active_cols, recs in pca_results:
    if label not in ("green", "unpleasant"):
        continue
    pc_cols = [c for c in stats.columns if c.startswith("PC") and c.endswith("_Loading")]
    if not pc_cols:
        continue
    top = stats.head(15)
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(top[pc_cols].values, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(top[pc_cols].values).max(), vmax=np.abs(top[pc_cols].values).max())
    ax.set_xticks(range(len(pc_cols))); ax.set_xticklabels(pc_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([str(n)[:26] for n in top["Ingredient"]], fontsize=8)
    ax.set_title(f"{label} — top-15 ingredient PC loadings")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"pca_true_cluster_{label}_loading_heatmap.png", dpi=130); plt.close(fig)

print("Saved plots:")
for p in sorted(OUT_DIR.glob("pca_true_cluster_*.png")):
    print(f"  {p}")
print(f"\nDone. Clusters with PCA: {[r[0] for r in pca_results]}; "
      f"stats-only: {[r[0] for r in results if r[4] is None]}")
