"""
Visualise differences between HDBSCAN clusters C1 vs C2.

Sources: outputs/hdbscan_cluster_pca_comparison.xlsx

Outputs (in outputs/):
  hdbscan_c1_c2_mirror_bars.png        – mirrored horizontal bars (Global_Importance)
  hdbscan_c1_c2_bubble_scatter.png     – Frequency vs PCA importance bubble chart
  hdbscan_c1_c2_loading_biplot.png     – PC1/PC2 loading positions per cluster
  hdbscan_c1_c2_overview_panel.png     – variance + set overlap overview (2-panel)
  hdbscan_c1_c2_shared_heatmap.png     – shared-ingredient stat comparison heatmap
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT   = Path("outputs")
XLSX  = OUT / "hdbscan_cluster_pca_comparison.xlsx"

# ── colour palette ────────────────────────────────────────────────────────────
C1_COL  = "#2D7DD2"   # blue
C2_COL  = "#E05A2B"   # orange
BG      = "#F7F7F7"
GRID    = "#E0E0E0"

def short_name(name, maxlen=28):
    """Strip cert suffixes and truncate."""
    for suffix in (" Kosher Halal", " Halal Kosher", " Kosher", " Halal", " natürlich", ", natürlich"):
        name = name.replace(suffix, "")
    return name.strip()[:maxlen]

# ── Load data ─────────────────────────────────────────────────────────────────
df       = pd.read_excel(XLSX, sheet_name="All_Ingredients")
summary  = pd.read_excel(XLSX, sheet_name="Summary")

c1 = df[df["Cluster"] == "C1"].reset_index(drop=True)
c2 = df[df["Cluster"] == "C2"].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 – Mirrored horizontal bars: Top 15 by Global_Importance per cluster
# ─────────────────────────────────────────────────────────────────────────────
N = 15

t1 = c1.head(N).copy()
t2 = c2.head(N).copy()
t1["label"] = t1["Ingredient"].apply(short_name)
t2["label"] = t2["Ingredient"].apply(short_name)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig.subplots_adjust(wspace=0.0)

y = np.arange(N)

# ── left panel: C1 ───────────────────────────────────────────────────────────
bars = ax_l.barh(y, t1["Global_Importance"], color=C1_COL, alpha=0.88, height=0.65)
ax_l.set_xlim(ax_l.get_xlim()[1], 0)   # invert so bars grow rightward
ax_l.set_yticks(y)
ax_l.set_yticklabels(t1["label"], fontsize=9.5, ha="right")
ax_l.yaxis.set_tick_params(pad=6)
ax_l.set_xlabel("Global PCA Importance", fontsize=10)
ax_l.set_title("Cluster C1\n(n = 84 recipes)", fontsize=12, fontweight="bold", color=C1_COL)
ax_l.tick_params(left=False)
ax_l.set_facecolor(BG)
ax_l.grid(axis="x", color=GRID, lw=0.7)
ax_l.spines[["top", "right", "left"]].set_visible(False)
for i, (imp, freq) in enumerate(zip(t1["Global_Importance"], t1["Frequency"])):
    ax_l.text(imp + 0.15, i, f" {freq:.0%}", va="center", fontsize=7.5, color="#555555")

# ── right panel: C2 ──────────────────────────────────────────────────────────
ax_r.barh(y, t2["Global_Importance"], color=C2_COL, alpha=0.88, height=0.65)
ax_r.set_yticks(y)
ax_r.set_yticklabels(t2["label"], fontsize=9.5, ha="left")
ax_r.yaxis.set_label_position("right")
ax_r.yaxis.tick_right()
ax_r.yaxis.set_tick_params(pad=6)
ax_r.set_xlabel("Global PCA Importance", fontsize=10)
ax_r.set_title("Cluster C2\n(n = 45 recipes)", fontsize=12, fontweight="bold", color=C2_COL)
ax_r.tick_params(right=False)
ax_r.set_facecolor(BG)
ax_r.grid(axis="x", color=GRID, lw=0.7)
ax_r.spines[["top", "left", "right"]].set_visible(False)
for i, (imp, freq) in enumerate(zip(t2["Global_Importance"], t2["Frequency"])):
    ax_r.text(imp + 0.08, i, f" {freq:.0%}", va="center", fontsize=7.5, color="#555555")

fig.suptitle("Top 15 Ingredients by PCA Global Importance – HDBSCAN C1 vs C2\n"
             "(% label = frequency within cluster)", fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out1 = OUT / "hdbscan_c1_c2_mirror_bars.png"
fig.savefig(out1, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out1}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 – Bubble scatter: Frequency vs Global_Importance (top 40 per cluster)
# Bubble size = Avg_Norm_When_Present
# ─────────────────────────────────────────────────────────────────────────────
TOP = 40
s1 = c1.head(TOP).copy()
s2 = c2.head(TOP).copy()

fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
ax.set_facecolor(BG)

def bubble_sizes(avg_norm):
    """Map avg norm values to reasonable bubble areas."""
    a = avg_norm.fillna(0).values
    # min 40, max 800
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.full(len(a), 150)
    return 40 + (a - lo) / (hi - lo) * 760

sc1 = ax.scatter(
    s1["Frequency"], s1["Global_Importance"],
    s=bubble_sizes(s1["Avg_Norm_When_Present"]),
    c=C1_COL, alpha=0.75, linewidths=0.5, edgecolors="white", label="C1 (n=84)", zorder=3
)
sc2 = ax.scatter(
    s2["Frequency"], s2["Global_Importance"],
    s=bubble_sizes(s2["Avg_Norm_When_Present"]),
    c=C2_COL, alpha=0.75, linewidths=0.5, edgecolors="white", label="C2 (n=45)", zorder=3,
    marker="D"
)

# Annotate top-8 per cluster
for _, row in s1.head(8).iterrows():
    ax.annotate(
        short_name(row["Ingredient"], 22),
        (row["Frequency"], row["Global_Importance"]),
        fontsize=7, color=C1_COL,
        xytext=(5, 3), textcoords="offset points"
    )
for _, row in s2.head(8).iterrows():
    ax.annotate(
        short_name(row["Ingredient"], 22),
        (row["Frequency"], row["Global_Importance"]),
        fontsize=7, color=C2_COL,
        xytext=(5, -9), textcoords="offset points"
    )

ax.set_xlabel("Frequency (share of recipes in cluster)", fontsize=11)
ax.set_ylabel("Global PCA Importance", fontsize=11)
ax.set_title("Frequency vs PCA Importance – Top 40 Ingredients per Cluster\n"
             "Bubble size = Avg normalised amount when present",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=11, framealpha=0.8)
ax.grid(color=GRID, lw=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

plt.tight_layout()
out2 = OUT / "hdbscan_c1_c2_bubble_scatter.png"
fig.savefig(out2, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out2}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 – PC1 / PC2 loading biplot (top 15 per cluster)
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

def loading_panel(ax, data, color, title, n=15):
    top = data.head(n)
    sizes = 60 + (top["Global_Importance"] / top["Global_Importance"].max()) * 200

    ax.axhline(0, color=GRID, lw=1)
    ax.axvline(0, color=GRID, lw=1)
    ax.scatter(
        top["PC1_Loading"], top["PC2_Loading"],
        s=sizes, c=color, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3
    )
    for _, row in top.iterrows():
        ax.annotate(
            short_name(row["Ingredient"], 20),
            (row["PC1_Loading"], row["PC2_Loading"]),
            fontsize=7.5, color="#333333",
            xytext=(5, 4), textcoords="offset points"
        )
    ax.set_xlabel("PC1 Loading", fontsize=10)
    ax.set_ylabel("PC2 Loading", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=color)
    ax.set_facecolor(BG)
    ax.grid(color=GRID, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

ev1 = summary.loc[summary["Cluster"] == "C1", ["PC1_Var_Expl_%", "PC2_Var_Expl_%"]].values[0]
ev2 = summary.loc[summary["Cluster"] == "C2", ["PC1_Var_Expl_%", "PC2_Var_Expl_%"]].values[0]

loading_panel(ax1, c1, C1_COL,
              f"C1 – PC1 ({ev1[0]:.1f}%) vs PC2 ({ev1[1]:.1f}%)\nTop 15 ingredients by Global Importance")
loading_panel(ax2, c2, C2_COL,
              f"C2 – PC1 ({ev2[0]:.1f}%) vs PC2 ({ev2[1]:.1f}%)\nTop 15 ingredients by Global Importance")

fig.suptitle("PCA Loading Space: How Top Ingredients Drive Variance per Cluster\n"
             "Dot size = Global Importance", fontsize=13, fontweight="bold")
plt.tight_layout()
out3 = OUT / "hdbscan_c1_c2_loading_biplot.png"
fig.savefig(out3, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out3}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 – Overview panel: variance explained (bar) + set overlap (Venn-style)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6), facecolor=BG)
gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Sub-panel A: variance explained per PC ───────────────────────────────────
ax_v = fig.add_subplot(gs[0, 0])
ax_v.set_facecolor(BG)

pc_labels = ["PC1", "PC2", "PC3", "PC4"]
c1_ev = summary.loc[summary["Cluster"] == "C1",
        ["PC1_Var_Expl_%", "PC2_Var_Expl_%", "PC3_Var_Expl_%", "PC4_Var_Expl_%"]].values.flatten()
c2_ev = summary.loc[summary["Cluster"] == "C2",
        ["PC1_Var_Expl_%", "PC2_Var_Expl_%", "PC3_Var_Expl_%", "PC4_Var_Expl_%"]].values.flatten()

x = np.arange(4)
w = 0.35
b1 = ax_v.bar(x - w/2, c1_ev, w, color=C1_COL, alpha=0.88, label="C1 (n=84)")
b2 = ax_v.bar(x + w/2, c2_ev, w, color=C2_COL, alpha=0.88, label="C2 (n=45)")

for bar in b1:
    ax_v.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color=C1_COL)
for bar in b2:
    ax_v.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
              f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8, color=C2_COL)

ax_v.set_xticks(x)
ax_v.set_xticklabels(pc_labels, fontsize=11)
ax_v.set_ylabel("Variance Explained (%)", fontsize=10)
ax_v.set_title("PCA Variance Explained per Component\n(C2 shows more concentrated structure)",
               fontsize=11, fontweight="bold")
ax_v.legend(fontsize=10)
ax_v.grid(axis="y", color=GRID, lw=0.7)
ax_v.spines[["top", "right"]].set_visible(False)

# ── Sub-panel B: ingredient set overlap ──────────────────────────────────────
ax_s = fig.add_subplot(gs[0, 1])
ax_s.set_aspect("equal")
ax_s.set_facecolor(BG)
ax_s.axis("off")

shared_cas = set(c1["CAS"]) & set(c2["CAS"])
only_c1    = set(c1["CAS"]) - set(c2["CAS"])
only_c2    = set(c2["CAS"]) - set(c1["CAS"])

circ1 = plt.Circle((-0.55, 0), 0.85, color=C1_COL, alpha=0.35)
circ2 = plt.Circle(( 0.55, 0), 0.85, color=C2_COL, alpha=0.35)
ax_s.add_patch(circ1)
ax_s.add_patch(circ2)

ax_s.text(-1.05, 0,  f"{len(only_c1)}\nexcl. to C1",
          ha="center", va="center", fontsize=13, fontweight="bold", color=C1_COL)
ax_s.text( 0,    0,  f"{len(shared_cas)}\nshared",
          ha="center", va="center", fontsize=13, fontweight="bold", color="#333333")
ax_s.text( 1.05, 0,  f"{len(only_c2)}\nexcl. to C2",
          ha="center", va="center", fontsize=13, fontweight="bold", color=C2_COL)

ax_s.text(-0.9, 1.0, "C1", fontsize=16, fontweight="bold", color=C1_COL, ha="center")
ax_s.text( 0.9, 1.0, "C2", fontsize=16, fontweight="bold", color=C2_COL, ha="center")
ax_s.text(0, -1.1, f"Total: {len(c1)} C1 CAS  |  {len(c2)} C2 CAS",
          ha="center", fontsize=9, color="#666666")
ax_s.set_xlim(-2.0, 2.0)
ax_s.set_ylim(-1.4, 1.5)
ax_s.set_title("Active Ingredient Overlap\n(CAS numbers present in each cluster)",
               fontsize=11, fontweight="bold")

fig.suptitle("HDBSCAN Cluster Comparison – Structural Overview", fontsize=14, fontweight="bold")
plt.tight_layout()
out4 = OUT / "hdbscan_c1_c2_overview_panel.png"
fig.savefig(out4, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out4}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 5 – Shared ingredients heatmap: C1 vs C2 for top shared CAS
# ─────────────────────────────────────────────────────────────────────────────
# Take top-25 shared CAS ranked by max(C1_GlobalImp, C2_GlobalImp)
shared_cas = set(c1["CAS"]) & set(c2["CAS"])
shared_c1  = c1[c1["CAS"].isin(shared_cas)].set_index("CAS")
shared_c2  = c2[c2["CAS"].isin(shared_cas)].set_index("CAS")

merged = shared_c1[["Ingredient", "Global_Importance", "Frequency", "Avg_Norm_When_Present"]].join(
    shared_c2[["Global_Importance", "Frequency", "Avg_Norm_When_Present"]],
    rsuffix="_C2"
).rename(columns={
    "Global_Importance":    "C1_GlobalImp",
    "Frequency":            "C1_Freq",
    "Avg_Norm_When_Present": "C1_AvgNorm",
    "Global_Importance_C2": "C2_GlobalImp",
    "Frequency_C2":         "C2_Freq",
    "Avg_Norm_When_Present_C2": "C2_AvgNorm",
})
merged["max_imp"] = merged[["C1_GlobalImp", "C2_GlobalImp"]].max(axis=1)
top_shared = merged.nlargest(25, "max_imp")
top_shared["label"] = top_shared["Ingredient"].apply(lambda x: short_name(x, 26))
top_shared = top_shared.sort_values("max_imp")

# Build heatmap matrix: rows = ingredients, columns = [C1_Imp, C2_Imp, C1_Freq, C2_Freq]
cols_to_plot = ["C1_GlobalImp", "C2_GlobalImp", "C1_Freq", "C2_Freq", "C1_AvgNorm", "C2_AvgNorm"]
col_labels   = ["C1\nGlobal Imp.", "C2\nGlobal Imp.",
                "C1\nFrequency", "C2\nFrequency",
                "C1\nAvg Norm.", "C2\nAvg Norm."]

mat = top_shared[cols_to_plot].values.astype(float)
# Normalise each column 0-1 for colouring
mat_norm = (mat - mat.min(axis=0)) / (mat.max(axis=0) - mat.min(axis=0) + 1e-9)

fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG)
ax.set_facecolor(BG)

im = ax.imshow(mat_norm, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="none")

# Annotate each cell with the raw value
for row_i in range(len(top_shared)):
    for col_j, col in enumerate(cols_to_plot):
        val = mat[row_i, col_j]
        if col.endswith("Imp"):
            txt = f"{val:.1f}"
        elif col.endswith("Freq"):
            txt = f"{val:.0%}"
        else:
            txt = f"{val:.3f}"
        brightness = mat_norm[row_i, col_j]
        fg = "black" if 0.3 < brightness < 0.75 else ("white" if brightness <= 0.3 else "black")
        ax.text(col_j, row_i, txt, ha="center", va="center",
                fontsize=8, color=fg, fontweight="bold")

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, fontsize=10)
ax.set_yticks(range(len(top_shared)))
ax.set_yticklabels(top_shared["label"].values, fontsize=8.5)
ax.set_title("Top 25 Shared Ingredients – C1 vs C2 Comparison\n"
             "(colour = relative value within each metric column, green=high, red=low)",
             fontsize=12, fontweight="bold")

# Add cluster colour bands at top
for j, col in enumerate(cols_to_plot):
    colour = C1_COL if col.startswith("C1") else C2_COL
    ax.add_patch(plt.Rectangle((j - 0.5, len(top_shared) - 0.5),
                                1, 0.4, color=colour, clip_on=False))

plt.tight_layout()
out5 = OUT / "hdbscan_c1_c2_shared_heatmap.png"
fig.savefig(out5, dpi=160, bbox_inches="tight", facecolor=BG)
plt.close()
print(f"Saved: {out5}")

print("\nAll 5 plots saved to outputs/")
