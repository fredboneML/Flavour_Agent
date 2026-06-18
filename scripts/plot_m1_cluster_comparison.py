"""
Visual at-a-glance summary of the M1 (Label Propagation) per-cluster PCA Excel.

Source: outputs/m1_cluster_pca_comparison.xlsx  (7 named clusters)

Outputs (in outputs/):
  m1_clusters_overview_dashboard.png   – 4-panel summary (size, variance, profile scatter)
  m1_clusters_top_ingredients.png      – 7 small-multiple bar charts, top ingredients per cluster
  m1_clusters_ingredient_heatmap.png   – top ingredients x cluster Global-Importance heatmap
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT  = Path("outputs")
XLSX = OUT / "m1_cluster_pca_comparison.xlsx"

BG   = "#F7F7F7"
GRID = "#E0E0E0"

# Cluster order + distinct colour per cluster
CLUSTERS = ["Walderdbeere", "dairy", "floral", "fruity", "green", "unpleasant", "warm"]
COLORS = {
    "Walderdbeere": "#B5179E",   # berry magenta
    "dairy":        "#F2C14E",   # cream yellow
    "floral":       "#E07A9E",   # pink
    "fruity":       "#E63946",   # red
    "green":        "#43AA8B",   # green
    "unpleasant":   "#6C757D",   # grey
    "warm":         "#E76F51",   # warm orange
}


def short_name(name, maxlen=26):
    """Strip cert suffixes and truncate."""
    for suffix in (" Kosher Halal", " Halal Kosher", " Kosher", " Halal",
                   " natürlich", ", natürlich", " BG"):
        name = name.replace(suffix, "")
    return name.strip()[:maxlen]


# ── Load ──────────────────────────────────────────────────────────────────────
df      = pd.read_excel(XLSX, sheet_name="All_Ingredients")
summary = pd.read_excel(XLSX, sheet_name="Summary").set_index("Cluster").reindex(CLUSTERS)

per_cluster = {c: df[df["Cluster"] == c].reset_index(drop=True) for c in CLUSTERS}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Overview dashboard (2 x 2)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor=BG)
gs  = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)
bar_colors = [COLORS[c] for c in CLUSTERS]

# Panel A – cluster sizes (recipes) ------------------------------------------------
axA = fig.add_subplot(gs[0, 0]); axA.set_facecolor(BG)
y = np.arange(len(CLUSTERS))
axA.barh(y, summary["N_Recipes"], color=bar_colors, alpha=0.9)
axA.set_yticks(y); axA.set_yticklabels(CLUSTERS, fontsize=11)
axA.invert_yaxis()
axA.set_xlabel("Number of recipes", fontsize=10)
axA.set_title("Cluster size (recipes)", fontsize=12, fontweight="bold")
axA.grid(axis="x", color=GRID, lw=0.7); axA.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(summary["N_Recipes"]):
    axA.text(v + 0.6, i, str(int(v)), va="center", fontsize=9, color="#333")

# Panel B – active ingredients (CAS) ----------------------------------------------
axB = fig.add_subplot(gs[0, 1]); axB.set_facecolor(BG)
axB.barh(y, summary["N_Active_CAS"], color=bar_colors, alpha=0.9)
axB.set_yticks(y); axB.set_yticklabels(CLUSTERS, fontsize=11)
axB.invert_yaxis()
axB.set_xlabel("Active ingredients (distinct CAS)", fontsize=10)
axB.set_title("Ingredient diversity per cluster", fontsize=12, fontweight="bold")
axB.grid(axis="x", color=GRID, lw=0.7); axB.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(summary["N_Active_CAS"]):
    axB.text(v + 1, i, str(int(v)), va="center", fontsize=9, color="#333")

# Panel C – variance explained, stacked PC1-PC4 -----------------------------------
axC = fig.add_subplot(gs[1, 0]); axC.set_facecolor(BG)
pc_cols = ["PC1_Var_Expl_%", "PC2_Var_Expl_%", "PC3_Var_Expl_%", "PC4_Var_Expl_%"]
pc_shades = ["#264653", "#2A9D8F", "#8AB17D", "#E9C46A"]
x = np.arange(len(CLUSTERS))
bottom = np.zeros(len(CLUSTERS))
for col, shade, lbl in zip(pc_cols, pc_shades, ["PC1", "PC2", "PC3", "PC4"]):
    vals = summary[col].fillna(0).values
    axC.bar(x, vals, bottom=bottom, color=shade, label=lbl, width=0.7)
    bottom += vals
axC.set_xticks(x); axC.set_xticklabels(CLUSTERS, fontsize=10, rotation=20, ha="right")
axC.set_ylabel("Variance explained (%)", fontsize=10)
axC.set_title("PCA variance explained (PC1–PC4 stacked)\nLow total = diffuse / heterogeneous cluster",
              fontsize=12, fontweight="bold")
axC.legend(fontsize=9, ncol=4, loc="upper right")
axC.grid(axis="y", color=GRID, lw=0.7); axC.spines[["top", "right"]].set_visible(False)
for i, v in enumerate(summary["Total_Var_Expl_%"]):
    axC.text(i, bottom[i] + 1.2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#333")

# Panel D – profile scatter: size vs structure ------------------------------------
axD = fig.add_subplot(gs[1, 1]); axD.set_facecolor(BG)
sizes = 80 + (summary["N_Active_CAS"] / summary["N_Active_CAS"].max()) * 1200
axD.scatter(summary["N_Recipes"], summary["Total_Var_Expl_%"],
            s=sizes, c=bar_colors, alpha=0.8, edgecolors="white", linewidths=1.2, zorder=3)
for c in CLUSTERS:
    axD.annotate(c, (summary.loc[c, "N_Recipes"], summary.loc[c, "Total_Var_Expl_%"]),
                 fontsize=9.5, fontweight="bold", color="#333",
                 xytext=(6, 6), textcoords="offset points")
axD.set_xlabel("Number of recipes", fontsize=10)
axD.set_ylabel("Total variance explained by 4 PCs (%)", fontsize=10)
axD.set_title("Cluster profile: size vs PCA compactness\nBubble size = ingredient diversity (CAS)",
              fontsize=12, fontweight="bold")
axD.grid(color=GRID, lw=0.6); axD.spines[["top", "right"]].set_visible(False)

fig.suptitle("M1 (Label Propagation) — 7-Cluster PCA Overview", fontsize=16, fontweight="bold", y=0.98)
out1 = OUT / "m1_clusters_overview_dashboard.png"
fig.savefig(out1, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Top ingredients per cluster (small multiples)
# ─────────────────────────────────────────────────────────────────────────────
N_TOP = 8
fig, axes = plt.subplots(2, 4, figsize=(20, 10), facecolor=BG)
axes = axes.flatten()

for idx, c in enumerate(CLUSTERS):
    ax = axes[idx]; ax.set_facecolor(BG)
    top = per_cluster[c].head(N_TOP).iloc[::-1]   # reverse so #1 is on top
    labels = top["Ingredient"].apply(lambda s: short_name(s, 24))
    yy = np.arange(len(top))
    ax.barh(yy, top["Global_Importance"], color=COLORS[c], alpha=0.9, height=0.7)
    ax.set_yticks(yy); ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(f"{c}\n(n={int(summary.loc[c,'N_Recipes'])} recipes, "
                 f"{summary.loc[c,'Total_Var_Expl_%']:.0f}% var.)",
                 fontsize=11, fontweight="bold", color=COLORS[c])
    ax.grid(axis="x", color=GRID, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(left=False)
    for i, (imp, freq) in enumerate(zip(top["Global_Importance"], top["Frequency"])):
        ax.text(imp + imp * 0.01, i, f" {freq:.0%}", va="center", fontsize=7, color="#666")

# Hide the unused 8th axis
axes[-1].axis("off")
axes[-1].text(0.5, 0.5,
              "Top 8 ingredients per cluster\nby PCA Global Importance\n\n% label = frequency\nwithin cluster",
              ha="center", va="center", fontsize=12, color="#555", transform=axes[-1].transAxes)

fig.suptitle("M1 Clusters — Defining Ingredients (Top 8 by PCA Global Importance)",
             fontsize=16, fontweight="bold", y=1.00)
plt.tight_layout()
out2 = OUT / "m1_clusters_top_ingredients.png"
fig.savefig(out2, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Cross-cluster ingredient heatmap
# Union of top-5 ingredients per cluster; matrix = Global Importance per cluster.
# ─────────────────────────────────────────────────────────────────────────────
TOP_PER = 5
top_cas = []
for c in CLUSTERS:
    top_cas.extend(per_cluster[c].head(TOP_PER)["CAS"].tolist())
# preserve order, dedupe
seen, ordered_cas = set(), []
for cas in top_cas:
    if cas not in seen:
        seen.add(cas); ordered_cas.append(cas)

# Build importance matrix (rows = CAS, cols = clusters)
imp_lookup = {c: per_cluster[c].set_index("CAS")["Global_Importance"].to_dict() for c in CLUSTERS}
name_lookup = df.drop_duplicates("CAS").set_index("CAS")["Ingredient"].to_dict()

mat = np.array([[imp_lookup[c].get(cas, np.nan) for c in CLUSTERS] for cas in ordered_cas])
row_labels = [short_name(name_lookup.get(cas, cas), 30) for cas in ordered_cas]

# Normalise per column (cluster) for colour comparability
col_max = np.nanmax(mat, axis=0)
mat_norm = mat / np.where(col_max == 0, 1, col_max)

fig, ax = plt.subplots(figsize=(11, max(8, 0.42 * len(ordered_cas))), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(np.nan_to_num(mat_norm, nan=0.0), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

for i in range(len(ordered_cas)):
    for j in range(len(CLUSTERS)):
        val = mat[i, j]
        if np.isnan(val):
            ax.text(j, i, "·", ha="center", va="center", fontsize=10, color="#BBB")
        else:
            shade = mat_norm[i, j]
            fg = "white" if shade > 0.6 else "#333"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color=fg)

ax.set_xticks(range(len(CLUSTERS)))
ax.set_xticklabels(CLUSTERS, fontsize=10, rotation=25, ha="right")
for tick, c in zip(ax.get_xticklabels(), CLUSTERS):
    tick.set_color(COLORS[c]); tick.set_fontweight("bold")
ax.set_yticks(range(len(ordered_cas)))
ax.set_yticklabels(row_labels, fontsize=8.5)
ax.set_title("M1 Clusters — Key Ingredient Fingerprint\n"
             "Global PCA Importance per cluster (colour = relative within each cluster column; "
             "· = not active)",
             fontsize=12, fontweight="bold")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Relative importance within cluster", fontsize=9)
plt.tight_layout()
out3 = OUT / "m1_clusters_ingredient_heatmap.png"
fig.savefig(out3, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out3}")

print("\nAll 3 figures saved to outputs/")
