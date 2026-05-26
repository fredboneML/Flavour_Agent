"""Script to create v2 (without-threshold) versions of both notebooks."""
import json, copy, textwrap
from pathlib import Path

NB_DIR = Path(__file__).parent


# ── helpers ───────────────────────────────────────────────────────────────────

def strip_outputs(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "outputs": [], "execution_count": None}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def replace_cell(nb, idx, new_source, cell_type="code"):
    if cell_type == "code":
        nb["cells"][idx] = code_cell(new_source)
    else:
        nb["cells"][idx] = md_cell(new_source)


def insert_after(nb, idx, new_source, cell_type="code"):
    if cell_type == "code":
        new_cell = code_cell(new_source)
    else:
        new_cell = md_cell(new_source)
    nb["cells"].insert(idx + 1, new_cell)
    return idx + 1


# ═══════════════════════════════════════════════════════════════════════════════
# PCA v2
# ═══════════════════════════════════════════════════════════════════════════════

EXPLAIN_RECIPE = textwrap.dedent('''\
def explain_recipe(rez_nr, n_top=8, plot=True):
    # Print and optionally plot top ingredient drivers for a recipe
    if rez_nr not in recipe_idx_map:
        print(f"Recipe {rez_nr!r} not found.")
        print("Available (first 10):", sorted(recipe_idx_map.keys())[:10])
        return

    r_idx    = recipe_idx_map[rez_nr]
    r_scores = scores_oav[r_idx, :N_PC_ATTR]
    r_contrib = contrib[r_idx]
    dom_ot   = dom_ot_labels[r_idx]

    print("=" * 50)
    print(f"  Recipe: {rez_nr}   |   Dominant Odour Type: {dom_ot}")
    print("=" * 50)
    for k in range(N_PC_ATTR):
        print(f"  PC{k+1} score: {r_scores[k]:+.3f}  (explains {ev[k]:.1f}% of total variance)")
    print()

    for k in range(N_PC_ATTR):
        c_vec    = r_contrib[k]
        top_idx_k = np.argsort(np.abs(c_vec))[::-1][:n_top]
        print(f"  -- PC{k+1}: top {n_top} ingredient drivers --")
        for i in top_idx_k:
            cas  = cas_labels[i]
            name = cas_name.get(cas, cas).split(",")[0][:40]
            val  = pivot_oav.iloc[r_idx, i]  # normalized proportion
            sign = "+" if c_vec[i] >= 0 else "-"
            print(f"    {sign}  contrib={c_vec[i]:+.4f}  norm={val:.6f}  "
                  f"{cas:15s}  {name}")
        print()

    dists  = np.linalg.norm(scores_oav - r_scores, axis=1)
    nn_idx = np.argsort(dists)[1:9]
    print("  -- 8 nearest recipes --")
    for ri in nn_idx:
        print(f"    {recipes_oav[ri]:12s}  dist={dists[ri]:.3f}"
              f"  PC1={scores_oav[ri,0]:+.2f}  PC2={scores_oav[ri,1]:+.2f}"
              f"  ({dom_ot_labels[ri]})")

    if plot:
        fig, ax = plt.subplots(figsize=(10, 7))
        for ot in sorted(set(dom_ot_labels)):
            mask = np.array(dom_ot_labels) == ot
            ax.scatter(scores_oav[mask, 0], scores_oav[mask, 1],
                       c=ODOUR_PALETTE.get(ot, "#BBBBBB"),
                       s=45, alpha=0.5, edgecolors="white", linewidths=0.3)
        ax.scatter(scores_oav[r_idx, 0], scores_oav[r_idx, 1],
                   s=220, c="gold", edgecolors="#333", linewidths=1.5, zorder=5,
                   label=f"Query: {rez_nr}")
        ax.scatter(scores_oav[nn_idx, 0], scores_oav[nn_idx, 1],
                   s=100, facecolors="none", edgecolors="#222", linewidths=1.5,
                   zorder=4, label="8 nearest")
        for ri in nn_idx:
            ax.annotate(recipes_oav[ri], (scores_oav[ri, 0], scores_oav[ri, 1]),
                        fontsize=7, xytext=(5, 3), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=11)
        ax.set_title(f"Recipe {rez_nr} - PCA position and nearest neighbours", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

print("explain_recipe() defined.")
''')

FIND_BY_INGREDIENT = textwrap.dedent('''\
def find_by_ingredient(query, n_top=15, plot=True):
    # Find recipes containing a given ingredient (by CAS or name fragment)
    q = str(query).strip()
    if q in cas_labels:
        matched_cas = [q]
    else:
        matched_cas = [c for c in cas_labels if q.lower() in cas_name.get(c, "").lower()]

    if not matched_cas:
        print(f"No ingredient found matching {q!r}.")
        return None

    print(f"Matched {len(matched_cas)} CAS number(s):")
    for cas in matched_cas:
        print(f"  {cas}  -->  {cas_name.get(cas, cas)}")
    print()

    norm_per_recipe = pivot_oav[matched_cas].sum(axis=1)
    nonzero         = norm_per_recipe[norm_per_recipe > 0].sort_values(ascending=False)
    print(f"Present in {len(nonzero)} / {len(recipes_oav)} recipes.")
    print()

    table_rows = []
    for rez, norm_val in nonzero.head(n_top).items():
        r_idx = recipes_oav.index(rez)
        table_rows.append({
            "Rez.-Nr.":   rez,
            "Norm_Total": round(norm_val, 6),
            "PC1":        round(scores_oav[r_idx, 0], 3),
            "PC2":        round(scores_oav[r_idx, 1], 3),
            "PC3":        round(scores_oav[r_idx, 2], 3),
            "Dom. OT":    dom_ot_labels[r_idx],
        })

    result = pd.DataFrame(table_rows).set_index("Rez.-Nr.")
    print(result.to_string())
    print()

    if plot:
        fig, ax = plt.subplots(figsize=(10, 7))
        match_mask = norm_per_recipe.reindex(recipes_oav, fill_value=0).values > 0
        for ot in sorted(set(dom_ot_labels)):
            mask = np.array(dom_ot_labels) == ot
            ax.scatter(scores_oav[mask & ~match_mask, 0],
                       scores_oav[mask & ~match_mask, 1],
                       c=ODOUR_PALETTE.get(ot, "#BBBBBB"),
                       s=40, alpha=0.4, edgecolors="white", linewidths=0.3)
        ax.scatter(scores_oav[match_mask, 0], scores_oav[match_mask, 1],
                   s=100, c="red", zorder=4, edgecolors="#333", linewidths=0.8,
                   label=f"Contains: {q} (n={match_mask.sum()})")
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=11)
        ax.set_title(f"Recipes containing: {q}", fontsize=10)
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()

    return result

print("find_by_ingredient() defined.")
print(\'Usage:  find_by_ingredient("furaneol")  or  find_by_ingredient("3658-77-3")\')
''')

NEAREST_RECIPES = textwrap.dedent('''\
def nearest_recipes(ingredient_dict, n=10, plot=True, label="Virtual Recipe"):
    # Find the nearest real recipes to a virtual recipe.
    # ingredient_dict: {cas_or_name_fragment: norm_proportion}
    # Values are normalised proportions (will be auto-normalised to sum=1).
    resolved = {}
    for key, amt in ingredient_dict.items():
        if key in cas_labels:
            resolved[key] = amt
        else:
            hit = next(
                (c for c in cas_labels if key.lower() in cas_name.get(c, "").lower()),
                None
            )
            if hit:
                resolved[hit] = amt
            else:
                print(f"  WARNING: {key!r} not found in model - skipped.")

    if not resolved:
        print("No valid ingredients after resolution. Aborting.")
        return None

    print(f"  {label}  -  resolved {len(resolved)} ingredient(s):")
    for cas, amt in resolved.items():
        print(f"    {cas:15s}  {cas_name.get(cas, cas).split(\',\')[0][:40]:40s}  norm={amt:.6f}")
    print()

    virt_vec = np.zeros(len(cas_labels))
    for cas, amt in resolved.items():
        virt_vec[cas_labels.index(cas)] = max(0.0, float(amt))

    if virt_vec.sum() > 0:
        virt_vec /= virt_vec.sum()

    virt_scaled = scaler_oav.transform(virt_vec.reshape(1, -1))[0]
    virt_score  = pca_oav.transform(virt_scaled.reshape(1, -1))[0]

    print(f"  Virtual recipe in PCA space: "
          f"PC1={virt_score[0]:+.3f}  PC2={virt_score[1]:+.3f}  PC3={virt_score[2]:+.3f}")
    print()

    dists  = np.linalg.norm(scores_oav - virt_score, axis=1)
    nn_idx = np.argsort(dists)[:n]

    result_rows = []
    print(f"  {n} nearest real recipes:")
    for rank, ri in enumerate(nn_idx, 1):
        result_rows.append({
            "Rank":     rank,
            "Rez.-Nr.": recipes_oav[ri],
            "Distance": round(dists[ri], 3),
            "PC1":      round(scores_oav[ri, 0], 3),
            "PC2":      round(scores_oav[ri, 1], 3),
            "PC3":      round(scores_oav[ri, 2], 3),
            "Dom. OT":  dom_ot_labels[ri],
        })
        print(f"    #{rank:2d}  {recipes_oav[ri]:12s}  dist={dists[ri]:.3f}"
              f"  PC1={scores_oav[ri,0]:+.2f}  PC2={scores_oav[ri,1]:+.2f}"
              f"  ({dom_ot_labels[ri]})")

    result_df = pd.DataFrame(result_rows).set_index("Rank")

    if plot:
        fig, ax = plt.subplots(figsize=(11, 8))
        for ot in sorted(set(dom_ot_labels)):
            mask = np.array(dom_ot_labels) == ot
            ax.scatter(scores_oav[mask, 0], scores_oav[mask, 1],
                       c=ODOUR_PALETTE.get(ot, "#BBBBBB"),
                       s=42, alpha=0.50, edgecolors="white", linewidths=0.3, zorder=2)
        ax.scatter(scores_oav[nn_idx, 0], scores_oav[nn_idx, 1],
                   s=130, facecolors="none", edgecolors="#111", linewidths=1.8,
                   zorder=4, label=f"Top {n} nearest")
        for rank, ri in enumerate(nn_idx, 1):
            ax.annotate(f"#{rank} {recipes_oav[ri]}", (scores_oav[ri, 0], scores_oav[ri, 1]),
                        fontsize=7, xytext=(6, 4), textcoords="offset points")
        ax.scatter(virt_score[0], virt_score[1],
                   marker="*", s=380, c="gold", edgecolors="#333", linewidths=1.2,
                   zorder=5, label=label)
        ax.annotate(label, (virt_score[0], virt_score[1]), fontsize=9, fontweight="bold",
                    xytext=(9, 7), textcoords="offset points")
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)", fontsize=11)
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)", fontsize=11)
        ax.set_title(
            f"Nearest Real Recipes to: {label}\\n"
            f"({len(resolved)} ingredient(s) specified | {n} nearest shown)",
            fontsize=10
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        safe = label[:25].replace(" ", "_").replace("/", "-").replace("+", "x")
        fig.savefig(OUT_DIR / f"pca_v2_nearest_{safe}.png", dpi=150, bbox_inches="tight")
        plt.show()

    return result_df

print("nearest_recipes() defined.")
print(\'Usage:  nearest_recipes({"furaneol": 0.30, "vanillin": 0.10, "linalool": 0.05})\')
''')


def build_pca_v2():
    with open(NB_DIR / "recipe_pca_v1_strawberry_erdbeere_executed.ipynb") as f:
        nb = json.load(f)
    nb = strip_outputs(copy.deepcopy(nb))

    # ── Cell 0: title ─────────────────────────────────────────────────────────
    replace_cell(nb, 0, (
        "# PCA - Erdbeere Gesamt v2 (Without Threshold)\n\n"
        "**Version:** v2 - no OAV / olfactory threshold used.\n"
        "**Pipeline:** Remove ignore-list substances (CAS-based) -> normalise Totalmenge per recipe -> PCA.\n"
        "**Dataset:** `data/gold/Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv`\n"
        "**Outputs:** `outputs/pca_v2_without_threshold_*`\n"
    ), "markdown")

    # ── Cell 5: normalization ─────────────────────────────────────────────────
    replace_cell(nb, 5, textwrap.dedent('''\
        # -- Normalize Totalmenge per recipe (no threshold, no OAV) ------------------
        # After ignore-list zeroing, keep only rows with positive Totalmenge then
        # express each ingredient as a relative proportion within its recipe.

        df_t = df[df["Totalmenge"] > 0].copy()

        per_recipe_sum = df_t.groupby("Rez.-Nr.")["Totalmenge"].transform("sum")
        df_t["Norm_Totalmenge"] = df_t["Totalmenge"] / per_recipe_sum

        print(f"Rows with positive Totalmenge: {len(df_t):,}")
        print(f"Recipes after filtering      : {df_t[\'Rez.-Nr.\'].nunique()}")
        print(f"Unique CAS numbers retained  : {df_t[\'CAS-Nr.\'].nunique()}")
        print()
        print("Norm_Totalmenge distribution:")
        print(df_t["Norm_Totalmenge"].describe().round(5))
        '''))

    # ── Cell 6: normalized pivot, no log ─────────────────────────────────────
    replace_cell(nb, 6, textwrap.dedent('''\
        # -- Build Recipe x CAS Normalized Totalmenge matrix --------------------------
        pivot_oav = df_t.pivot_table(
            index="Rez.-Nr.", columns="CAS-Nr.",
            values="Norm_Totalmenge", aggfunc="sum", fill_value=0
        )
        print(f"Recipe x CAS matrix       : {pivot_oav.shape}")
        print(f"Avg CAS per recipe        : {(pivot_oav > 0).sum(axis=1).mean():.1f}")
        print(f"Matrix value range        : {pivot_oav.values.min():.5f} - {pivot_oav.values.max():.5f}")

        # No log transformation -- normalised proportions are already on a comparable scale
        X_oav = pivot_oav.values.copy()
        '''))

    # ── Cell 7: OdourType profile, no log ────────────────────────────────────
    replace_cell(nb, 7, textwrap.dedent('''\
        # -- Build Recipe x OdourType Normalized Totalmenge profile ------------------
        rows = []
        for _, row in df_t.iterrows():
            for col in ["Odour-Type 1", "Odour-Type 2", "Odour-Type 3"]:
                ot = row[col]
                if pd.notna(ot):
                    rows.append({"Recipe": row["Rez.-Nr."], "OdourType": ot,
                                 "Norm_Totalmenge": row["Norm_Totalmenge"]})

        ot_df    = pd.DataFrame(rows)
        pivot_ot = ot_df.groupby(["Recipe", "OdourType"])["Norm_Totalmenge"].sum().unstack(fill_value=0)
        print(f"Recipe x OdourType matrix : {pivot_ot.shape}")
        print(f"Odour types               : {pivot_ot.columns.tolist()}")

        # No log transformation
        X_ot = pivot_ot.values.copy()

        dom_ot_series = pivot_ot.idxmax(axis=1)
        print("\\nDominant odour type distribution:")
        print(dom_ot_series.value_counts())
        '''))

    # ── NEW cell after 7: ingredient statistics ───────────────────────────────
    insert_after(nb, 7, textwrap.dedent('''\
        # -- Ingredient Statistics: Frequency & Average (after removal + normalisation) --
        cas_name_full = df.groupby("CAS-Nr.")["Name"].first().to_dict()

        # Frequency = fraction of recipes where ingredient is present (proportion > 0)
        freq      = (pivot_oav > 0).mean(axis=0)
        n_recipes = (pivot_oav > 0).sum(axis=0)

        # Average normalised proportion among recipes that contain the ingredient
        avg_present_mat = pivot_oav.copy()
        avg_present_mat[avg_present_mat == 0] = np.nan
        avg_when_present = avg_present_mat.mean(axis=0)

        # Average normalised proportion across ALL recipes (including zeros)
        avg_all = pivot_oav.mean(axis=0)

        ing_stats = pd.DataFrame({
            "CAS":                   pivot_oav.columns,
            "Ingredient":            [cas_name_full.get(c, c) for c in pivot_oav.columns],
            "Frequency":             freq.values.round(4),
            "Recipes_Count":         n_recipes.values,
            "Avg_Norm_When_Present": avg_when_present.values.round(6),
            "Avg_Norm_All_Recipes":  avg_all.values.round(6),
        }).sort_values("Frequency", ascending=False).reset_index(drop=True)

        print(f"Ingredient statistics for {len(ing_stats)} unique CAS numbers "
              f"across {len(pivot_oav)} recipes")
        print()
        print(ing_stats.head(30).to_string(index=False))

        ing_stats.to_csv(OUT_DIR / "pca_v2_ingredient_statistics.csv", index=False)
        ing_stats.to_excel(OUT_DIR / "pca_v2_ingredient_statistics.xlsx", index=False)
        print("\\nExported: pca_v2_ingredient_statistics.csv / .xlsx")
        '''))
    # After this insert, cells 8+ shift by 1.

    # ── Cell 9 (was 8): section heading ──────────────────────────────────────
    replace_cell(nb, 9,
                 "## 2 · PCA on Normalised Totalmenge Fingerprint (recipes x CAS)\n",
                 "markdown")

    # ── Cell 11 (was 10): scree plot ─────────────────────────────────────────
    src = "".join(nb["cells"][11]["source"])
    src = src.replace("OAV Fingerprint PCA", "Normalised Totalmenge PCA")
    src = src.replace("pca_v1_scree_oav.png", "pca_v2_scree_norm.png")
    replace_cell(nb, 11, src)

    # ── Cell 13 (was 12): PC1 vs PC2 scatter ─────────────────────────────────
    src = "".join(nb["cells"][13]["source"])
    src = src.replace("OAV Fingerprint", "Normalised Totalmenge")
    src = src.replace("pca_v1_pc1pc2_oav.png", "pca_v2_pc1pc2_norm.png")
    replace_cell(nb, 13, src)

    # ── Cell 16 (was 15): biplot ──────────────────────────────────────────────
    src = "".join(nb["cells"][16]["source"])
    src = src.replace("(OAV)", "(Norm. Totalmenge)")
    src = src.replace("pca_v1_biplot_oav.png", "pca_v2_biplot_norm.png")
    replace_cell(nb, 16, src)

    # ── Cell 21 (was 20): dashboard ───────────────────────────────────────────
    src = "".join(nb["cells"][21]["source"])
    src = src.replace("OAV Fingerprint", "Normalised Totalmenge")
    src = src.replace("pca_v1_dashboard_pc123.png", "pca_v2_dashboard_pc123.png")
    replace_cell(nb, 21, src)

    # ── Cell 23 (was 22): loading heatmap ────────────────────────────────────
    src = "".join(nb["cells"][23]["source"])
    src = src.replace("pca_v1_loading_heatmap.png", "pca_v2_loading_heatmap.png")
    replace_cell(nb, 23, src)

    # ── Cell 25 (was 24): scores table ───────────────────────────────────────
    src = "".join(nb["cells"][25]["source"])
    src = src.replace("pca_v1_scores.csv", "pca_v2_scores.csv")
    replace_cell(nb, 25, src)

    # ── Cell 29 (was 28): explain_recipe ────────────────────────────────────
    replace_cell(nb, 29, EXPLAIN_RECIPE)

    # ── Cell 32 (was 31): find_by_ingredient ─────────────────────────────────
    replace_cell(nb, 32, FIND_BY_INGREDIENT)

    # ── Cell 36 (was 35): nearest_recipes ────────────────────────────────────
    replace_cell(nb, 36, NEAREST_RECIPES)

    # ── Cell 37 (was 36): demo 1 – remove cas_thresh ─────────────────────────
    replace_cell(nb, 37, textwrap.dedent('''\
        # -- Demo 1: auto-build virtual recipe from top PC1 ingredients --------------
        top3_pc1_idx = np.argsort(np.abs(loadings[0]))[::-1][:3]
        top3_pc1_cas = [cas_labels[i] for i in top3_pc1_idx]

        # Use median normalised proportion across recipes that contain each ingredient
        pc1_mix = {}
        for cas in top3_pc1_cas:
            if cas not in pivot_oav.columns:
                continue
            vals = pivot_oav.loc[pivot_oav[cas] > 0, cas]
            if len(vals) == 0:
                continue
            pc1_mix[cas] = float(vals.median())

        print("Demo virtual recipe (top-3 PC1 ingredients at median normalised proportion):")
        for cas, amt in pc1_mix.items():
            print(f"  {cas}  {cas_name.get(cas, \'?\').split(\',\')[0][:40]}  norm={amt:.6f}")
        print()
        _ = nearest_recipes(pc1_mix, n=10, label="PC1 Dominant Mix")
        '''))

    # ── Cell 38 (was 37): demo 2 ─────────────────────────────────────────────
    replace_cell(nb, 38, textwrap.dedent('''\
        # -- Demo 2: freely-defined mix (edit to explore) ----------------------------
        # Keys can be partial ingredient names (case-insensitive) or exact CAS numbers.
        # Values are normalised proportions (will be auto-normalised to sum=1).
        my_mix = {
            "furaneol":  0.30,    # key strawberry character odorant
            "vanillin":  0.10,    # warm vanilla support
            "linalool":  0.05,    # floral/fresh lift
        }
        _ = nearest_recipes(my_mix, n=10, label="Furaneol + Vanillin + Linalool")
        '''))

    # ── Cell 40 (was 39): PC1 heatmap ────────────────────────────────────────
    src = "".join(nb["cells"][40]["source"])
    src = src.replace("pca_v1_contrib_heatmap_pc1.png", "pca_v2_contrib_heatmap_pc1.png")
    replace_cell(nb, 40, src)

    # ── Cell 41 (was 40): attribution CSV ────────────────────────────────────
    src = "".join(nb["cells"][41]["source"])
    src = src.replace("pca_v1_recipe_attribution.csv", "pca_v2_recipe_attribution.csv")
    replace_cell(nb, 41, src)

    # ── Cell 42 (was 41): section heading for top 20 per recipe ──────────────
    replace_cell(nb, 42, (
        "---\n"
        "## 12 · Top 20 Ingredients per Recipe (PCA Contribution Export)\n\n"
        "For every recipe the 20 ingredients that contribute most to its PCA fingerprint "
        "are ranked.\n"
        "The `Norm_Totalmenge` column shows the **normalised relative proportion** within "
        "the recipe (no log, no OAV).\n"
    ), "markdown")

    # ── Cell 43 (was 42): top 20 per recipe ──────────────────────────────────
    replace_cell(nb, 43, textwrap.dedent('''\
        # -- Top 20 ingredients per recipe by combined PCA contribution ---------------
        total_abs_contrib = np.abs(contrib).sum(axis=1)  # (n_recipes, n_cas)

        N_TOP_PER_RECIPE = 20
        rows = []
        for r_idx, rez in enumerate(recipes_oav):
            top_idx = np.argsort(total_abs_contrib[r_idx])[::-1][:N_TOP_PER_RECIPE]
            for rank, i in enumerate(top_idx, 1):
                cas = cas_labels[i]
                rows.append({
                    "Rez.-Nr.":        rez,
                    "Dom_OT":          dom_ot_labels[r_idx],
                    "Rank":            rank,
                    "CAS":             cas,
                    "Ingredient":      cas_name.get(cas, cas),
                    "Norm_Totalmenge": round(float(pivot_oav.iloc[r_idx, i]), 6),
                    "Total_Contrib":   round(float(total_abs_contrib[r_idx, i]), 5),
                    "PC1_Contrib":     round(float(contrib[r_idx, 0, i]), 5),
                    "PC2_Contrib":     round(float(contrib[r_idx, 1, i]), 5),
                    "PC3_Contrib":     round(float(contrib[r_idx, 2, i]), 5),
                    "PC4_Contrib":     round(float(contrib[r_idx, 3, i]), 5),
                })

        top20_df = pd.DataFrame(rows)
        print(f"Shape: {top20_df.shape}  ({len(recipes_oav)} recipes x {N_TOP_PER_RECIPE} ingredients)")
        print()

        xlsx_path = OUT_DIR / "pca_v2_top20_ingredients_per_recipe.xlsx"
        csv_path  = OUT_DIR / "pca_v2_top20_ingredients_per_recipe.csv"
        top20_df.to_excel(xlsx_path, index=False)
        top20_df.to_csv(csv_path, index=False)
        print(f"Exported: {xlsx_path.name}")
        print(f"Exported: {csv_path.name}")
        print()
        display(top20_df.head(30))
        '''))

    # ── NEW: markdown heading for global top 20 ──────────────────────────────
    insert_after(nb, 43, (
        "---\n"
        "## 13 · Top 20 Ingredients Globally (Across All Recipes)\n\n"
        "These are the ingredients with the highest **summed absolute PCA contribution** "
        "across all recipes and all four principal components - i.e. the most analytically "
        "impactful substances in the entire portfolio regardless of which recipe they come from.\n"
    ), "markdown")

    # ── NEW: top 20 globally code ─────────────────────────────────────────────
    insert_after(nb, 44, textwrap.dedent('''\
        # -- Top 20 Ingredients Globally (summed |contrib| over all recipes x PC1-PC4) -
        global_importance = np.abs(contrib).sum(axis=(0, 1))   # (n_cas,)

        top20_global_idx = np.argsort(global_importance)[::-1][:20]

        global_rows = []
        for rank, i in enumerate(top20_global_idx, 1):
            cas = cas_labels[i]
            col = pivot_oav.iloc[:, i]
            global_rows.append({
                "Rank":              rank,
                "CAS":               cas,
                "Ingredient":        cas_name.get(cas, cas),
                "Global_Importance": round(float(global_importance[i]), 4),
                "Frequency":         round(float((col > 0).mean()), 4),
                "Avg_Norm_Present":  round(float(col[col > 0].mean() if (col > 0).any() else 0), 6),
                "Avg_Norm_All":      round(float(col.mean()), 6),
                "Recipes_Count":     int((col > 0).sum()),
                "PC1_Loading":       round(float(loadings[0, i]), 4),
                "PC2_Loading":       round(float(loadings[1, i]), 4),
                "PC3_Loading":       round(float(loadings[2, i]), 4),
                "PC4_Loading":       round(float(loadings[3, i]), 4),
            })

        top20_global_df = pd.DataFrame(global_rows)
        print("=== Top 20 Ingredients Globally (all recipes combined) ===")
        print()
        print(top20_global_df.to_string(index=False))

        top20_global_df.to_csv(OUT_DIR / "pca_v2_top20_ingredients_global.csv", index=False)
        top20_global_df.to_excel(OUT_DIR / "pca_v2_top20_ingredients_global.xlsx", index=False)
        print("\\nExported: pca_v2_top20_ingredients_global.csv / .xlsx")

        # -- Quick bar chart of global importance ---------------------------------
        names_short = [cas_name.get(cas_labels[i], cas_labels[i]).split(",")[0][:30]
                       for i in top20_global_idx]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.barh(range(20), global_importance[top20_global_idx][::-1],
                color="#4A90D9", alpha=0.85, edgecolor="white")
        ax.set_yticks(range(20))
        ax.set_yticklabels(names_short[::-1], fontsize=9)
        ax.set_xlabel("Global PCA Importance (summed |contrib| over all recipes x PC1-PC4)")
        ax.set_title("Top 20 Ingredients Globally - Normalised Totalmenge PCA", fontsize=11)
        plt.tight_layout()
        fig.savefig(OUT_DIR / "pca_v2_top20_global_importance.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved: pca_v2_top20_global_importance.png")
        '''))

    # ── Update Key Findings cell ──────────────────────────────────────────────
    for idx, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "Key Findings" in src and cell["cell_type"] == "markdown":
            replace_cell(nb, idx, (
                "---\n"
                "## 14 · Key Findings (v2 - No Threshold)\n\n"
                "### Data Processing\n"
                "- Ignore list applied (CAS-based masking) before any analysis.\n"
                "- Totalmenge normalised per recipe to relative proportions (sum = 1).\n"
                "- **No OAV / olfactory threshold** used.\n"
                "- **No log transformation** - proportions are already on a comparable scale.\n\n"
                "### Ingredient Statistics\n"
                "- Frequency and average proportion per ingredient computed across all recipes.\n"
                "- Full statistics exported to `pca_v2_ingredient_statistics.csv`.\n\n"
                "### PCA Results\n"
                "- Top 20 ingredients per recipe: `pca_v2_top20_ingredients_per_recipe.csv`.\n"
                "- Top 20 globally: `pca_v2_top20_ingredients_global.csv`.\n"
            ), "markdown")
            break

    for idx, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "Recipes analysed" in src and cell["cell_type"] == "markdown":
            replace_cell(nb, idx, (
                "## Summary\n\n"
                "| | Value |\n|---|---|\n"
                "| Version | v2 - without threshold |\n"
                "| Pipeline | Remove ignore list -> Normalise per recipe -> PCA |\n"
                "| Features | CAS-level normalised Totalmenge |\n"
                "| Log transform | None |\n"
                "| Top-N per recipe | 20 |\n"
                "| Outputs | `outputs/pca_v2_*` |\n"
            ), "markdown")
            break

    out_path = NB_DIR / "recipe_pca_v2_without_threshold_strawberry_erdbeere.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {out_path.name}  ({len(nb['cells'])} cells)")


# ═══════════════════════════════════════════════════════════════════════════════
# MDS v2
# ═══════════════════════════════════════════════════════════════════════════════

def build_mds_v2():
    with open(NB_DIR / "recipe_clustering_erdbeere_mds_all_algorithms_executed.ipynb") as f:
        nb = json.load(f)
    nb = strip_outputs(copy.deepcopy(nb))

    # ── Cell 0: title ─────────────────────────────────────────────────────────
    replace_cell(nb, 0, (
        "# Erdbeere Recipe Clustering - MDS + All Algorithms (v2 Without Threshold)\n\n"
        "**Version:** v2 - no olfactory threshold used in feature vectors.\n"
        "**Pipeline:** Remove ignore-list substances (CAS-based) -> normalise Totalmenge "
        "per recipe -> MDS clustering.\n"
        "All feature vectors use **`use_threshold=False`** - quantities are compared by "
        "normalised relative proportion only.\n"
    ), "markdown")

    # ── Cell 4: data loading – CAS-based ignore list ──────────────────────────
    replace_cell(nb, 4, textwrap.dedent('''\
        def to_float(v, fallback=0.0):
            # Parse European decimal strings (\'1,3E-5\') and native floats/ints
            if v is None: return fallback
            if isinstance(v, (int, float)): return float(v)
            try: return float(str(v).strip().replace(",", "."))
            except: return fallback

        # -- Load CSV ------------------------------------------------------------------
        df_raw = pd.read_csv(CSV_PATH, dtype=str)
        df_raw[TOTAL_COL]     = df_raw[TOTAL_COL].apply(to_float)
        df_raw[THRESHOLD_COL] = df_raw[THRESHOLD_COL].apply(to_float)
        df = df_raw[df_raw[REZ_COL].notna()].copy()

        # -- Ignore list: CAS-based masking -------------------------------------------
        # Match by Ident or Name first, then zero ALL rows sharing the same CAS number
        # (catches alternate Idents of the same substance, e.g. Triethylcitrat variants)
        if IGNORE_PATH.exists():
            ign             = pd.read_csv(IGNORE_PATH)
            ign_idents      = set(ign[IDENT_COL].dropna().astype(str).str.strip())
            names_to_ignore = {str(n).lower().strip() for n in ign[NAME_COL]}
            mask = (
                df[IDENT_COL].astype(str).str.strip().isin(ign_idents) |
                df[NAME_COL].str.lower().str.strip().isin(names_to_ignore)
            )
            cas_to_ignore = set(df.loc[mask, CAS_COL].dropna().astype(str).str.strip())
            df.loc[df[CAS_COL].astype(str).str.strip().isin(cas_to_ignore), TOTAL_COL] = 0.0
            print(f"Ignored idents: {len(ign_idents)} | Ignored CAS: {len(cas_to_ignore)}")
        else:
            print("No ignore list found -- all ingredients included.")

        # -- Normalize Totalmenge per recipe (relative proportions) -------------------
        per_recipe_total = df.groupby(REZ_COL)[TOTAL_COL].transform("sum")
        df[TOTAL_COL] = np.where(per_recipe_total > 0,
                                  df[TOTAL_COL] / per_recipe_total,
                                  df[TOTAL_COL])

        recipes = df[REZ_COL].unique().tolist()
        print(f"Recipes           : {len(recipes)}")
        print(f"Ingredients (rows) : {len(df)}")
        print(f"Sample recipes    : {recipes[:6]}")
        print(f"Odour types present: OT1={df[OT1].notna().sum()} rows, "
              f"OT2={df[OT2].notna().sum()}, OT3={df[OT3].notna().sum()}")
        n = len(recipes)
        '''))

    # ── Cell 7: auto-detect heading ───────────────────────────────────────────
    replace_cell(nb, 7, (
        "## 3. Auto-detect Optimal Number of Clusters\n\n"
        "Since no panelist free-sorting is available yet, we let the data decide.\n"
        "Four complementary methods are used on the **M1 feature space** "
        "(OT1 weighted by normalised Totalmenge, **no threshold**):\n\n"
        "| Method | Optimum |\n|---|---|\n"
        "| Ward dendrogram | Largest gap between merges |\n"
        "| Silhouette score | Maximise |\n"
        "| Davies-Bouldin index | Minimise |\n"
        "| Calinski-Harabasz score | Maximise |\n"
    ), "markdown")

    # ── Cell 8: optimal k – use_threshold=False ───────────────────────────────
    src = "".join(nb["cells"][8]["source"])
    src = src.replace(
        "build_recipe_vectors(df, recipes, [(OT1, 1.0)], use_threshold=True)",
        "build_recipe_vectors(df, recipes, [(OT1, 1.0)], use_threshold=False)"
    )
    src = src.replace(
        "# Build M2 vectors (OT1 + threshold) -- used as the reference for cluster detection",
        "# Build M1 vectors (OT1, no threshold) -- used as the reference for cluster detection"
    )
    src = src.replace(
        "# Build M2 vectors (OT1 + threshold) — used as the reference for cluster detection",
        "# Build M1 vectors (OT1, no threshold) — used as the reference for cluster detection"
    )
    src = src.replace("MDS stress (M2, Ward)", "MDS stress (M1, Ward)")
    replace_cell(nb, 8, src)

    # ── Cell 10: section heading for models ───────────────────────────────────
    replace_cell(nb, 10, (
        "## 4. All Models - MDS Maps at Optimal k\n\n"
        "Two models are compared - both without threshold:\n\n"
        "| Model | Features | Threshold |\n|-------|----------|-----------|\n"
        "| M1: OT1 | OT1 weighted by normalised Totalmenge | None |\n"
        "| M2: OT1+OT2+OT3 | Three OTs weighted by position | None |\n"
    ), "markdown")

    # ── Cell 11: MODEL_CONFIGS – no-threshold only ────────────────────────────
    replace_cell(nb, 11, textwrap.dedent('''\
        MODEL_CONFIGS = [
            {"name": "M1: OT1\\n(no threshold)",
             "fcw": [(OT1, 1.0)],
             "use_thresh": False},
            {"name": "M2: OT1+OT2+OT3\\n(no threshold)",
             "fcw": [(OT1, pos_weight(1, 4)), (OT2, pos_weight(2, 4)), (OT3, pos_weight(3, 4))],
             "use_thresh": False},
        ]

        all_model_results = []
        for cfg in MODEL_CONFIGS:
            _, vecs = build_recipe_vectors(df, recipes, cfg["fcw"], cfg["use_thresh"])
            diss    = cosine_dissimilarity(vecs)
            coords, Z, labels, stress = run_mds(diss, n_clusters=OPTIMAL_K)
            all_model_results.append({
                "name": cfg["name"], "coords": coords, "labels": labels,
                "stress": stress, "n_clusters": len(np.unique(labels))
            })
            print(f"{cfg[\'name\'].split(chr(10))[0]:40s}  stress={stress:.3f}  "
                  f"{len(np.unique(labels))} clusters")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, res in zip(axes.flatten(), all_model_results):
            mds_plot(ax, res["coords"], recipes, res["labels"],
                     f"{res[\'name\']}\\nk={OPTIMAL_K}, stress={res[\'stress\']:.3f}",
                     show_legend=True)
        fig.suptitle(f"MDS - 2 Models, No Threshold (cosine dissimilarity + Ward, k={OPTIMAL_K})",
                     fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/erdbeere_v2_mds_2models.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Saved.")
        '''))

    # ── Cell 12: multi-algo heading ───────────────────────────────────────────
    replace_cell(nb, 12, (
        "## 5. Multi-Algorithm Comparison at Optimal k\n\n"
        "All 10 algorithms are applied to the **M1 recipe vectors** "
        "(OT1, normalised Totalmenge, no threshold) with `k = OPTIMAL_K`.\n"
        "DBSCAN and HDBSCAN are density-based and may find a different natural k; "
        "their parameters are scanned to best approximate `OPTIMAL_K`, and residual "
        "noise points are resolved to the nearest cluster.\n\n"
        "| Algorithm | Type |\n|---|---|\n"
        "| k-Means | Centroid partition |\n"
        "| k-Medoids (PAM) | Medoid partition on cosine distance |\n"
        "| Ward Linkage | Hierarchical (baseline) |\n"
        "| DBSCAN | Density-based |\n"
        "| HDBSCAN | Hierarchical density |\n"
        "| GMM | Probabilistic (PCA pre-reduction) |\n"
        "| Spectral Clustering | Graph-based (cosine affinity) |\n"
        "| Fuzzy c-Means | Soft partition (PCA pre-reduction) |\n"
        "| SOM | Topological 2D map |\n"
        "| DEC (simplified) | PCA encoding + Student-t refinement |\n"
    ), "markdown")

    # ── Cell 15: run all algos – use_threshold=False ──────────────────────────
    src = "".join(nb["cells"][15]["source"])
    src = src.replace(
        "build_recipe_vectors(df, recipes, [(OT1, 1.0)], use_threshold=True)",
        "build_recipe_vectors(df, recipes, [(OT1, 1.0)], use_threshold=False)"
    )
    src = src.replace(
        "# -- Ward (M2 reference) — already computed via auto-detected k ────────────────",
        "# -- Ward (M1 reference, no threshold) -- already computed via auto-detected k --"
    )
    src = src.replace(
        "# ── Ward (M2 reference) — already computed via auto-detected k ────────────────",
        "# -- Ward (M1 reference, no threshold) -- already computed via auto-detected k --"
    )
    replace_cell(nb, 15, src)

    # ── Cell 16: all-algos plot title ─────────────────────────────────────────
    src = "".join(nb["cells"][16]["source"])
    src = src.replace(
        "All Algorithms — M2 MDS Space",
        "All Algorithms - M1 MDS Space (No Threshold)"
    )
    src = src.replace(
        "erdbeere_all_algorithms_mds.png",
        "erdbeere_v2_all_algorithms_mds.png"
    )
    replace_cell(nb, 16, src)

    # ── Cell 24: cluster profiles – normalized Totalmenge (no OAV) ───────────
    replace_cell(nb, 24, textwrap.dedent('''\
        K_PROF = 5   # Silhouette-optimal

        # -- Normalised Totalmenge x Odour-Type matrix --------------------------------
        # Use the already-normalised Totalmenge (no OAV, no threshold)
        df_ot    = df[df[TOTAL_COL] > 0].copy()
        ot_vocab = sorted({str(x).strip() for x in df_ot[OT1].dropna()
                           if str(x).strip() not in ("", "nan")})

        ot_matrix = pd.DataFrame(0.0, index=recipes, columns=ot_vocab)
        for _, row in df_ot.iterrows():
            rez = row[REZ_COL]
            ot  = str(row[OT1]).strip()
            if rez in ot_matrix.index and ot in ot_vocab:
                ot_matrix.loc[rez, ot] += row[TOTAL_COL]   # already normalised proportion

        row_sum = ot_matrix.sum(axis=1).replace(0, np.nan)
        ot_prof = ot_matrix.div(row_sum, axis=0).fillna(0)

        # -- Re-fit algorithms at K_PROF -----------------------------------------------
        prof_algos = {}

        km_p = KMeans(n_clusters=K_PROF, n_init=20, random_state=42).fit(vecs_m2)
        prof_algos["k-Means"] = km_p.labels_ + 1

        if _has_faiss and K_PROF in faiss_results:
            prof_algos["FAISS k-Means"] = faiss_results[K_PROF]

        prof_algos["Ward"] = fcluster(Z_m2_sq, t=K_PROF, criterion="maxclust")

        gmm_p = GaussianMixture(n_components=K_PROF, n_init=10, random_state=42).fit(vecs_m2)
        prof_algos["GMM"] = gmm_p.predict(vecs_m2) + 1

        prof_algos["k-Medoids"] = kmedoids(diss_m2, K_PROF, random_state=42)

        spec_p = SpectralClustering(n_clusters=K_PROF, affinity="precomputed",
                                     random_state=42, n_init=10).fit(1 - diss_m2)
        prof_algos["Spectral"] = spec_p.labels_ + 1

        prof_algos["Fuzzy c-Means"] = fuzzy_cmeans(vecs_m2, c=K_PROF, random_state=42)
        prof_algos["SOM"] = som_cluster(vecs_m2, k=K_PROF, random_state=42)

        print("Algorithms re-fitted at k=5:")
        for name, lbl in prof_algos.items():
            sil = silhouette_score(diss_m2, np.array(lbl), metric="precomputed")
            print(f"  {name:<18s}  {len(np.unique(lbl))} clusters  sil={sil:.3f}")
        '''))

    # ── Cell 25: key findings ─────────────────────────────────────────────────
    replace_cell(nb, 25, (
        "## 7. Key Findings (v2 - No Threshold)\n\n"
        "### Data Processing\n"
        "- Ignore list applied using CAS-based masking (covers all variants of a substance).\n"
        "- Totalmenge normalised per recipe (sum = 1) - **no OAV / olfactory threshold**.\n"
        "- Feature vectors are normalised proportion-weighted odour-type profiles.\n\n"
        "### Models Compared\n"
        "Only no-threshold models are used:\n"
        "- **M1**: OT1 only\n"
        "- **M2**: OT1 + OT2 + OT3 (position-weighted)\n\n"
        "Without the threshold divisor, ingredients are weighted purely by their relative quantity.\n"
    ), "markdown")

    # ── Cell 29: Section 11 heading ───────────────────────────────────────────
    replace_cell(nb, 29, (
        "---\n"
        "## 11 · All Algorithms at Optimal k - M1 Feature Space (No Threshold)\n\n"
        "Silhouette best k and Davies-Bouldin best k determined above.\n"
        "All 10 algorithms are run at both k values; "
        "recipes are plotted in MDS Dim1 x Dim2 space.\n"
    ), "markdown")

    # ── Cell 31: feature space assignment ────────────────────────────────────
    replace_cell(nb, 31, textwrap.dedent('''\
        K_SIL_MDS = best_k_sil   # Silhouette best k (from auto-detection above)
        K_DB_MDS  = best_k_db    # Davies-Bouldin best k

        # Feature space: M1 recipe vectors (no threshold, already computed above)
        feats_mds   = vecs_m2
        feats_mds_n = normalize(feats_mds)
        diss_mds    = diss_m2
        recipes_mds = list(recipes)
        n_mds       = n
        print(f"Feature space : {feats_mds.shape}")
        print(f"Distance matrix: {diss_mds.shape}")
        print(f"Silhouette best k = {K_SIL_MDS}  |  Davies-Bouldin best k = {K_DB_MDS}")
        '''))

    # ── Cells 33/34: update print and filenames to not hardcode k values ──────
    replace_cell(nb, 33, textwrap.dedent('''\
        print("=" * 60)
        print(f"k = {K_SIL_MDS}  -- Silhouette optimum")
        print("=" * 60)
        algo_k5_mds = run_all_algos_mds(K_SIL_MDS)
        plot_algo_grid_mds(
            algo_k5_mds, K_SIL_MDS,
            subtitle=f"Silhouette best k={K_SIL_MDS}",
            fname=f"erdbeere_v2_mds_all_algos_k{K_SIL_MDS}_silhouette.png"
        )
        '''))

    replace_cell(nb, 34, textwrap.dedent('''\
        print("=" * 60)
        print(f"k = {K_DB_MDS}  -- Davies-Bouldin optimum")
        print("=" * 60)
        algo_k8_mds = run_all_algos_mds(K_DB_MDS)
        plot_algo_grid_mds(
            algo_k8_mds, K_DB_MDS,
            subtitle=f"Davies-Bouldin best k={K_DB_MDS}",
            fname=f"erdbeere_v2_mds_all_algos_k{K_DB_MDS}_davies_bouldin.png"
        )
        '''))

    out_path = NB_DIR / "recipe_clustering_erdbeere_mds_all_algorithms_v2_without_threshold.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {out_path.name}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    build_pca_v2()
    build_mds_v2()
    print("Done.")
