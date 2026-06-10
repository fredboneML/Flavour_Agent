# Design: Four New Clustering Strategies for the Expert-Seeded Erdbeere Notebook

**Date:** 2026-06-10
**Target:** `notebooks/_build_v3_expert_seeded.py` (build script) → regenerates
`notebooks/recipe_clustering_erdbeere_v3_expert_seeded.ipynb`
**Output:** `outputs/cluster_assignments_expert_seeded_all_strategies.xlsx` (expert handover)

## Goal

Extend the v3 expert-seeded clustering notebook with four new strategies for
constructing recipe clusters from the expert-provided example centroids in
`data/gold/Erdbeer Clustering Sensorik Vorgabe.xlsx`, then produce an Excel the
domain experts can review to tell us **which strategy aligns best with the
panelists' free-sorting selection**.

Existing strategies (unchanged): S1–S5 (target-recipe / ingredient seeds ×
mean / median aggregation, hybrid).

## Two Strategy Families

The new methods do not all fit the existing "build centroid → run 10
algorithms" grid, so we split into two families.

### Family A — Centroid strategies (slot into existing `STRATEGIES` grid)

**S6 — Contrast (Rocchio) centroids**
- `centroid[c] = normalize( mean(seed_c) − β · mean(all recipes not in seed_c) )`
- `β = 0.5` (constant, documented).
- Seed source = S1's target recipes (so S1 vs S6 isolates the contrast effect).
- Rationale: the shared "fruity/sweet" background dominates OT1 space and blurs
  every current centroid; subtracting the global background emphasizes what
  *distinguishes* each cluster.
- Integration: add to `STRATEGIES` dict → flows through `run_all_algos`
  (all 10 algorithms) → gets its own MDS grid PNG + Excel detail sheet,
  exactly like S1–S5. **Zero structural change.**

### Family B — Direct methods (one label vector each, new `direct_results` dict)

These produce a single labeling, not a 10-algorithm grid.

**M1 — Label Propagation**
- Build a kNN cosine-similarity graph over all recipes (`k = 10`, RBF/knn kernel
  via sklearn `LabelSpreading`).
- Pin expert target recipes to their cluster; `-1` (unlabeled) for the rest.
- Diffuse labels through the manifold. Respects non-spherical cluster shapes
  that centroid methods assume away.

**M2 — Rule-based pre-assignment (full interpretation)**
- Deterministically assign recipes that fire an expert rule (see Rule Table
  below), then assign the remainder by nearest expert centroid (S1 centroids).
- Operates on **per-recipe Totalmenge-normalized CAS concentrations**
  (`df[TOTAL_COL]`), already available in the notebook — not OT1 space.

**M3 — Consensus**
- Co-association matrix `A[i,j] = fraction of runs where recipe i and j share a
  cluster`, over all Family-A algorithm runs **excluding known-degenerate runs**
  (DEC failed, Fuzzy collapsed in v3) plus M1.
- Cluster `A` via average-linkage agglomerative at k=7, Hungarian-align cluster
  ids to expert names (reuse existing `hungarian_align` on cluster centroids).
- Free: reuses runs already computed; answers "which assignment is most stable".

## M2 Rule Table (full interpretation)

Parsed from the `Regeln/ Notizen` column. Concentration `conc(recipe, CAS)` =
per-recipe normalized Totalmenge of that CAS in that recipe.

| Cluster | Marker ingredient(s) (CAS) | Rule text | Encoded condition |
|---|---|---|---|
| warm | maltol 118-71-8, Furaneol 3658-77-3, Vanillin 121-33-5 | "alle drei Rohstoffe müssen vorhanden sein" | all three present (conc > 0) |
| warm | trans,trans-2,4-Decadienal 25152-84-5 | "wenn Rohstoff vorhanden" | present (conc > 0) |
| floral | Jasmin-Absolue 8022-96-6 | "wenn Rohstoff vorhanden" | present (conc > 0) |
| Walderdbeere | Methylanthranilat 134-20-3, Dimethylanthranilat 85-91-6 | "beide Rohstoffe müssen vorhanden sein" | both present (conc > 0) |
| dairy | diacetyl 431-03-8, Acetoin 513-86-0 | "beide Rohstoffe müssen vorhanden sein" | both present (conc > 0) |
| unpleasant | Dimethylsulfid 75-18-3 | ">0,0009 eindeutig" | conc > 0.0009 |
| unpleasant | S-(Methylthio)butyrat 2432-51-1, Methyl-3-(methylthio)propionat 13532-18-8 | "wenn Rohstoff vorhanden" (Black List) | present (conc > 0) |
| fruity | Ethyl-trans,cis-2,4-decadienoat 3025-30-7 | "> 0,004 eindeutig" | conc > 0.004 |
| fruity | Isoamylacetat 123-92-2 | "> 0,1 eindeutig" | conc > 0.1 |
| green | cis-3-Hexenol, trans-2-Hexenol, trans-2-Hexenal | "keine eindeutige Regel gefunden" | **no rule** — centroid fallback only |
| unpleasant | Hexansäure kräftig 142-62-1 | "keine eindeutige Regel gefunden" | **no rule** — centroid fallback only |

### Proposed precedence (when a recipe fires multiple rules) — **flag for review**

1. **`eindeutig` threshold rules** (Dimethylsulfid, decadienoat, Isoamylacetat) —
   highest priority; the expert wording "eindeutig" = definitive.
2. **multi-ingredient AND rules** ("alle drei", "beide") — strong, specific.
3. **single-presence rules** ("wenn Rohstoff vorhanden") — weakest.
4. Within the same tier, break ties by cosine similarity to the S1 centroid of
   the candidate clusters.
5. Recipes firing no rule → nearest S1 expert centroid.

The notebook will print a per-recipe audit (which rule fired, which tier) so the
experts can sanity-check the precedence.

## Excel for Expert Handover

- **Keep** existing per-strategy detail sheets; add sheets for S6, M1, M2, M3
  (same format: Algorithm rows × Cluster columns of newline-joined recipe ids;
  direct methods M1–M3 are single-row).
- **NEW master sheet `Strategy_Comparison`** — the expert review view:
  - One row per recipe.
  - One column per strategy/method.
  - Centroid strategies (S1–S6) represented by their **NearestCentroid**
    assignment (purest "closest expert centroid" — chosen over majority-vote /
    single-algorithm).
  - A `Target_Recipe?` flag column + `Expert_Intended_Cluster` column so experts
    immediately see where target recipes landed under each strategy.
- **Extend** the `Agreement` (ARI/NMI) sheet to include the new methods.

## `Next strategies to try` (markdown section, not implemented)

#1 direct ingredient-profile centroid · #3 medoid centroid · #4 dose-weighted
mean · #6 constrained k-means (COP/PCKMeans) · #7 few-shot classifier ·
#8 supervised metric learning / LDA space · #9 seeded soft GMM ·
#12 active-disagreement report.

## Verification

- Build script runs without error; notebook executes top-to-bottom.
- All 4 new strategies produce 7 expert clusters (or documented partials).
- M2 audit print shows each rule firing on at least the expert's own target
  recipes where applicable.
- Excel opens with: S1–S6 + M1–M3 detail sheets, `Strategy_Comparison` master
  sheet, `Agreement` sheet.

## Out of Scope

- No change to the OT1 feature space or the existing S1–S5 logic.
- No new clustering algorithms added to the 10-algorithm suite.
- "Next strategies" remain documentation only.
