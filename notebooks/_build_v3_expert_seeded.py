"""Build the v3 expert-seeded clustering notebook.

This is a notebook-generation script. It writes
recipe_clustering_erdbeere_mds_all_algorithms_v3_expert_seeded.ipynb
based on the v2 notebook structure but with:
  - K fixed at 7 (from the expert Vorgabe.xlsx)
  - 6 centroid-seeding strategies + 3 direct methods x all algorithms

Run from project root:
    python notebooks/_build_v3_expert_seeded.py
"""
import json
from pathlib import Path

NB_PATH = Path("notebooks/recipe_clustering_erdbeere_mds_all_algorithms_v3_expert_seeded.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []

# ---------------------------------------------------------------------------
# 0. Title / overview
# ---------------------------------------------------------------------------
CELLS.append(md(
    """# Erdbeere Recipe Clustering - Expert-Seeded Centroids (v3)

This notebook mirrors the v2 *all algorithms / no threshold* notebook with one
key change: the **number of clusters and their centroids are seeded from
expert knowledge** in `data/gold/Erdbeer Clustering Sensorik Vorgabe.xlsx`.

The aim is to evaluate six different ways of building seed centroids and to
compare how each clustering algorithm responds to that seeding.

## Six centroid strategies

| Strategy | Seed source | Aggregation | Floral fallback |
|---|---|---|---|
| **S1 - Target Recipes (mean)** | OT1 vectors of `Target Recipes` per cluster | mean | top recipes where OT1=`floral` dominates |
| **S2 - Ingredients (mean)** | OT1 vectors of recipes containing each cluster's characteristic CAS-Nr. | mean | same logic - recipes containing Jasmin-Absolue |
| **S3 - Hybrid** | Target Recipes where available; ingredient-based for floral | mean | ingredient (Jasmin-Absolue) |
| **S4 - Target Recipes (median)** | same source as S1 | element-wise median | OT1-dominant fallback, median |
| **S5 - Ingredients (median)** | same source as S2 | element-wise median | ingredient (Jasmin-Absolue), median |
| **S6 - Contrast (Rocchio)** | same source as S1 | mean(seed) − 0.5·mean(rest) | OT1-dominant fallback, then contrast |

S4 and S5 mirror S1 and S2 exactly except for the aggregation. The median is
more robust to outlier recipes (e.g. recipes dominated by a single
descriptor) and tends to produce sparser centroids on this 13-d OT1 space.

S6 keeps S1's seed source but subtracts the global background profile
(β=0.5·mean of all non-seed recipes), so each centroid emphasizes what
*distinguishes* its cluster rather than the shared fruity/sweet character that
dominates every recipe. Negative components are clipped to keep the centroid a
valid non-negative OT1 profile.

## Expert clusters (k=7)
`warm, floral, Walderdbeere, green, dairy, unpleasant, fruity`

Black-List rows from the expert file are merged back into their parent
clusters as extra characteristic ingredients. The `186.521` Target Recipe is
**missing from the dataset** (likely typo) and is dropped.
"""))

# ---------------------------------------------------------------------------
# 0b. Strategy overview table (all strategies in the Excel)
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## Strategies in the Excel file - overview

The companion file `cluster_assignments_expert_seeded_all_strategies.xlsx`
contains **9 strategies** for turning the expert centroids into recipe clusters.
Six are *centroid* strategies (S1-S6, each run through all 10 algorithms), three
are *direct* methods (M1-M3, one labeling each). In the `Strategy_Comparison`
master sheet the centroid strategies are represented by their **NearestCentroid**
assignment.

| Strategy | Type | Short description |
|---|---|---|
| **S1 - Target Recipes (mean)** | centroid | Mean OT1 vector of the expert's Target Recipes per cluster. |
| **S2 - Ingredients (mean)** | centroid | Mean OT1 vector of all recipes that contain the cluster's characteristic ingredient (CAS-Nr.). |
| **S3 - Hybrid** | centroid | Target Recipes where available, otherwise ingredient-based. |
| **S4 - Target Recipes (median)** | centroid | Like S1 but element-wise **median** (more robust to outlier recipes). |
| **S5 - Ingredients (median)** | centroid | Like S2 but element-wise **median**. |
| **S6 - Contrast (Rocchio)** | centroid | Like S1 but subtracts the global background (β=0.5·mean of all other recipes) to emphasise what **distinguishes** each cluster from the shared fruity/sweet base. |
| **M1 - Label Propagation** | direct | Expert target recipes are pinned to their cluster; labels diffuse to the rest through a **cosine k-NN graph** (k=10) over all recipes. No round-cluster assumption. |
| **M2 - Rule-based** | direct | The expert `Regeln/Notizen` rules (e.g. `> 0,004 eindeutig`, `alle drei Rohstoffe müssen vorhanden sein`) applied on per-recipe normalized ingredient concentrations; recipes that fire no rule fall back to the nearest S1 centroid. |
| **M3 - Consensus** | direct | Co-association across all non-degenerate algorithm runs + M1, re-clustered at k=7 — the most stable "majority" partition. |

**Similarity / distance throughout:** recipes are vectors over the OT1
descriptor vocabulary, **L2-normalized**, so cosine similarity = dot product.
M1's k-NN graph and the S1-centroid assignments all use this cosine measure;
M2's rules instead operate directly on ingredient (CAS) concentrations.

Excel sheets: `Strategy_Comparison` (master, one row per recipe), one detail
sheet per strategy (`S1_target_recipes` ... `M3_consensus`), and two agreement
sheets (`Agreement_Strategies`, `Agreement_byAlgo`).
"""))

# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------
CELLS.append(md("## 1. Setup & Imports\n"))

CELLS.append(code(
    """%matplotlib inline
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.cluster import HDBSCAN as _HDBSCAN
    _has_hdbscan = True
except ImportError:
    _has_hdbscan = False

OUTPUT_DIR = '../outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH    = Path('../data/gold/Third_Trial_Set_PDM Erdbeere Gesamt 8-5-2026.csv')
IGNORE_PATH = Path('../data/gold/ignone_substances.csv')
EXPERT_XLSX = Path('../data/gold/Erdbeer Clustering Sensorik Vorgabe.xlsx')

OT1, OT2, OT3 = 'Odour-Type 1', 'Odour-Type 2', 'Odour-Type 3'
THRESHOLD_COL = 'Threshold'
REZ_COL, IDENT_COL, CAS_COL, NAME_COL, TOTAL_COL = 'Rez.-Nr.', 'Ident', 'CAS-Nr.', 'Name', 'Totalmenge'

# Expert cluster names → stable color used across every plot in the notebook.
EXPERT_CLUSTERS = ['warm', 'floral', 'Walderdbeere', 'green', 'dairy', 'unpleasant', 'fruity']
EXPERT_COLORS = {
    'warm':         '#E63946',
    'floral':       '#F4A261',
    'Walderdbeere': '#9B59B6',
    'green':        '#2A9D8F',
    'dairy':        '#E9C46A',
    'unpleasant':   '#264653',
    'fruity':       '#E76F51',
}

K_EXPERT = len(EXPERT_CLUSTERS)
print(f'Libraries loaded. HDBSCAN available: {_has_hdbscan}')
print(f'Expert clusters (k={K_EXPERT}): {EXPERT_CLUSTERS}')
"""))

# ---------------------------------------------------------------------------
# 2. Load data (same as v2)
# ---------------------------------------------------------------------------
CELLS.append(md("## 2. Load Erdbeere Recipe Data\n"))

CELLS.append(code(
    """def to_float(v, fallback=0.0):
    if v is None: return fallback
    if isinstance(v, (int, float)): return float(v)
    try: return float(str(v).strip().replace(',', '.'))
    except: return fallback

df_raw = pd.read_csv(CSV_PATH, dtype=str)
df_raw[TOTAL_COL]     = df_raw[TOTAL_COL].apply(to_float)
df_raw[THRESHOLD_COL] = df_raw[THRESHOLD_COL].apply(to_float)
df = df_raw[df_raw[REZ_COL].notna()].copy()

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
    print(f'Ignored idents: {len(ign_idents)} | Ignored CAS: {len(cas_to_ignore)}')

per_recipe_total = df.groupby(REZ_COL)[TOTAL_COL].transform('sum')
df[TOTAL_COL] = np.where(per_recipe_total > 0, df[TOTAL_COL] / per_recipe_total, df[TOTAL_COL])

recipes = df[REZ_COL].unique().tolist()
n = len(recipes)
print(f'Recipes: {n}  |  Ingredient rows: {len(df)}')
"""))

# ---------------------------------------------------------------------------
# 3. Build OT1 vectors + MDS coords
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 3. Build OT1 Feature Vectors

Same feature space as the v2 notebook: each recipe → normalized vector over
the OT1 (Odour-Type 1) vocabulary, weighted by `Totalmenge`. All
centroid construction below happens in **this** space.
"""))

CELLS.append(code(
    """def norm_term(term):
    if pd.isna(term) or not isinstance(term, str): return None
    t = term.lower().strip().replace('"', '').replace("'", '').rstrip('.,;:')
    return t if len(t) >= 2 else None

def build_vocabulary(df, feature_cols):
    vocab = set()
    for col in feature_cols:
        if col in df.columns:
            for t in df[col].dropna().map(norm_term):
                if t: vocab.add(t)
    return sorted(vocab)

def build_recipe_vectors(df, recipes, feature_cols_weighted):
    feature_cols = [col for col, _ in feature_cols_weighted]
    vocab        = build_vocabulary(df, feature_cols)
    vocab_to_idx = {t: i for i, t in enumerate(vocab)}
    vectors      = np.zeros((len(recipes), len(vocab)), dtype=np.float64)
    for r_idx, recipe in enumerate(recipes):
        for _, row in df[df[REZ_COL] == recipe].iterrows():
            qty = float(row[TOTAL_COL])
            if qty <= 0: continue
            for col, col_weight in feature_cols_weighted:
                term = norm_term(row.get(col))
                if term and term in vocab_to_idx:
                    vectors[r_idx, vocab_to_idx[term]] += col_weight * qty
    return vocab, vocab_to_idx, normalize(vectors)

vocab, vocab_to_idx, vecs = build_recipe_vectors(df, recipes, [(OT1, 1.0)])
print(f'OT1 vocabulary ({len(vocab)} terms): {vocab}')
print(f'Recipe vectors shape: {vecs.shape}')

# Precompute cosine dissimilarity matrix (used by every algo + MDS)
sim  = np.clip(vecs @ vecs.T, -1.0, 1.0)
diss = 1.0 - sim
np.fill_diagonal(diss, 0.0)

# Run MDS once for visualization
mds_coords = MDS(n_components=2, dissimilarity='precomputed', metric=True,
                 n_init=10, max_iter=1000, random_state=42,
                 normalized_stress='auto').fit_transform(diss)
print(f'MDS coords shape: {mds_coords.shape}')
"""))

# ---------------------------------------------------------------------------
# 4. Parse expert Excel
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 4. Parse Expert Centroids from Vorgabe.xlsx

The Excel uses a forward-fill convention: the `Cluster` column only carries
the name on the first row of each cluster. The `Black List` section adds
**extra characteristic ingredients** to existing clusters (unpleasant,
fruity, warm). We merge those back.
"""))

CELLS.append(code(
    """xlsx_raw = pd.read_excel(EXPERT_XLSX)

# Forward-fill cluster name, BUT reset to None when we hit the 'Black List' separator
# (Black List is a section marker, not a cluster — rows below it carry their own Cluster value)
expert_df = xlsx_raw.copy()
bl_idx = expert_df.index[expert_df['Cluster'] == 'Black List']
if len(bl_idx) > 0:
    bl_pos = bl_idx[0]
    # Forward-fill only up to (and not including) the Black List row
    expert_df.loc[:bl_pos - 1, 'Cluster'] = expert_df.loc[:bl_pos - 1, 'Cluster'].ffill()
    # After Black List, forward-fill within each cluster the user named
    expert_df.loc[bl_pos + 1:, 'Cluster'] = expert_df.loc[bl_pos + 1:, 'Cluster'].ffill()
else:
    expert_df['Cluster'] = expert_df['Cluster'].ffill()

# Drop the Black List separator row and rows without ingredient info
expert_df = expert_df[(expert_df['Cluster'] != 'Black List') & expert_df['Cluster'].notna()].copy()

# Build the per-cluster spec
expert_spec = {}
for cluster_name, sub in expert_df.groupby('Cluster'):
    cas_list = sub['CAS-Nr.'].dropna().astype(str).str.strip().tolist()
    rohstoffe = sub['Rohstoff'].dropna().astype(str).str.strip().tolist()
    target_recipes_raw = sub['Target Recipes'].dropna().tolist()
    target_recipes = [f'{t:.3f}' for t in target_recipes_raw]
    descriptors = sub['Deskriptoren allgemein'].dropna().astype(str).str.strip().tolist()
    expert_spec[cluster_name] = {
        'cas': cas_list,
        'rohstoffe': rohstoffe,
        'target_recipes': target_recipes,
        'descriptors': descriptors,
    }

# Sanity print
for name in EXPERT_CLUSTERS:
    s = expert_spec.get(name, {})
    print(f'{name:14s}  ingredients={len(s.get("cas", [])):2d}  targets={len(s.get("target_recipes", [])):2d}  '
          f'descriptors={s.get("descriptors", [])}')
"""))

CELLS.append(code(
    """# Map every Target Recipe ID (e.g. '185.237') to a dataset recipe (e.g. '185.237H')
def match_recipe_prefix(target_id, recipes):
    matches = [r for r in recipes if r.startswith(target_id)]
    return matches[0] if matches else None

print('Target Recipe → dataset match resolution:')
unmatched = []
for cname, s in expert_spec.items():
    resolved = []
    for t in s['target_recipes']:
        m = match_recipe_prefix(t, recipes)
        if m is None: unmatched.append((cname, t))
        else: resolved.append(m)
    s['target_recipes_resolved'] = resolved
    print(f'  {cname:14s}  {len(resolved)}/{len(s["target_recipes"])} matched  →  {resolved}')

if unmatched:
    print(f'\\n⚠ Dropped unmatched targets: {unmatched}')
"""))

# ---------------------------------------------------------------------------
# 5. Six centroid strategies
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 5. Build Six Centroid Strategies

Each strategy returns a dict `{cluster_name → centroid_vector}` in the same
OT1 vocabulary space as the recipe vectors. Centroids are L2-normalized so
cosine similarity (= dot product) is the natural distance.
"""))

CELLS.append(code(
    """def _normalize_vec(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def _mean_of_indices(idxs, vectors):
    if not idxs: return None
    c = vectors[idxs].mean(axis=0)
    return _normalize_vec(c)

def _median_of_indices(idxs, vectors):
    if not idxs: return None
    c = np.median(vectors[idxs], axis=0)
    # If element-wise median collapses to all-zero (can happen on sparse
    # normalized vectors when fewer than half the recipes activate a dim),
    # fall back to the mean so the centroid remains well-defined.
    if np.linalg.norm(c) < 1e-12:
        c = vectors[idxs].mean(axis=0)
    return _normalize_vec(c)

def _recipes_containing_cas(cas_list, df, recipes):
    \"\"\"Return indices of recipes whose ingredient list contains any of the given CAS numbers.\"\"\"
    if not cas_list: return []
    cas_set = {str(c).strip() for c in cas_list}
    rec_ids = df[df[CAS_COL].astype(str).str.strip().isin(cas_set)][REZ_COL].unique().tolist()
    return [i for i, r in enumerate(recipes) if r in rec_ids]

def _ot1_dominant_recipes(ot1_term, vectors, vocab_to_idx, top_n=5):
    \"\"\"Return indices of the top_n recipes with the highest weight on ``ot1_term``.\"\"\"
    if ot1_term not in vocab_to_idx: return []
    col = vocab_to_idx[ot1_term]
    return list(np.argsort(vectors[:, col])[::-1][:top_n])

# ── Strategy 1: Target Recipes (with OT1-dominant fallback) ──────────────────
def build_strategy_target_recipes(expert_spec, vectors, recipes, vocab_to_idx):
    centroids = {}
    for cname in EXPERT_CLUSTERS:
        s = expert_spec.get(cname, {})
        resolved = s.get('target_recipes_resolved', [])
        idxs = [recipes.index(r) for r in resolved if r in recipes]
        if idxs:
            centroids[cname] = _mean_of_indices(idxs, vectors)
        else:
            # OT1-only fallback: if the cluster name itself is an OT1 term, use it
            fallback_term = cname.lower()
            idxs = _ot1_dominant_recipes(fallback_term, vectors, vocab_to_idx, top_n=5)
            centroids[cname] = _mean_of_indices(idxs, vectors)
            print(f'  [S1] {cname}: no targets, OT1-dominant fallback on "{fallback_term}" → {len(idxs)} recipes')
    return centroids

# ── Strategy 2: Characteristic Ingredients ───────────────────────────────────
def build_strategy_ingredients(expert_spec, vectors, recipes, df):
    centroids = {}
    for cname in EXPERT_CLUSTERS:
        s = expert_spec.get(cname, {})
        idxs = _recipes_containing_cas(s.get('cas', []), df, recipes)
        if not idxs:
            print(f'  [S2] {cname}: no recipes contain any expert CAS — leaving as None')
            centroids[cname] = None
            continue
        centroids[cname] = _mean_of_indices(idxs, vectors)
    return centroids

# ── Strategy 3: Hybrid (target recipes + ingredient fallback) ────────────────
def build_strategy_hybrid(expert_spec, vectors, recipes, df):
    centroids = {}
    for cname in EXPERT_CLUSTERS:
        s = expert_spec.get(cname, {})
        resolved = s.get('target_recipes_resolved', [])
        idxs = [recipes.index(r) for r in resolved if r in recipes]
        if not idxs:
            idxs = _recipes_containing_cas(s.get('cas', []), df, recipes)
            print(f'  [S3] {cname}: no targets, ingredient fallback → {len(idxs)} recipes')
        centroids[cname] = _mean_of_indices(idxs, vectors)
    return centroids

# ── Strategy 4: Target Recipes (median aggregation) ──────────────────────────
def build_strategy_target_recipes_median(expert_spec, vectors, recipes, vocab_to_idx):
    centroids = {}
    for cname in EXPERT_CLUSTERS:
        s = expert_spec.get(cname, {})
        resolved = s.get('target_recipes_resolved', [])
        idxs = [recipes.index(r) for r in resolved if r in recipes]
        if idxs:
            centroids[cname] = _median_of_indices(idxs, vectors)
        else:
            fallback_term = cname.lower()
            idxs = _ot1_dominant_recipes(fallback_term, vectors, vocab_to_idx, top_n=5)
            centroids[cname] = _median_of_indices(idxs, vectors)
            print(f'  [S4] {cname}: no targets, OT1-dominant fallback on "{fallback_term}" → {len(idxs)} recipes (median)')
    return centroids

# ── Strategy 5: Characteristic Ingredients (median aggregation) ──────────────
def build_strategy_ingredients_median(expert_spec, vectors, recipes, df):
    centroids = {}
    for cname in EXPERT_CLUSTERS:
        s = expert_spec.get(cname, {})
        idxs = _recipes_containing_cas(s.get('cas', []), df, recipes)
        if not idxs:
            print(f'  [S5] {cname}: no recipes contain any expert CAS — leaving as None')
            centroids[cname] = None
            continue
        centroids[cname] = _median_of_indices(idxs, vectors)
    return centroids

# ── Strategy 6: Contrast (Rocchio) centroids ─────────────────────────────────
# centroid = normalize( mean(seed) - BETA * mean(all recipes NOT in seed) ).
# Subtracting the global "fruity/sweet" background that dominates OT1 space
# sharpens what *distinguishes* each cluster. Seed source = S1 target recipes
# (so S1 vs S6 isolates the contrast effect).
CONTRAST_BETA = 0.5

def build_strategy_contrast(expert_spec, vectors, recipes, vocab_to_idx, beta=CONTRAST_BETA):
    centroids = {}
    n_all = vectors.shape[0]
    for cname in EXPERT_CLUSTERS:
        s = expert_spec.get(cname, {})
        resolved = s.get('target_recipes_resolved', [])
        idxs = [recipes.index(r) for r in resolved if r in recipes]
        if not idxs:
            fallback_term = cname.lower()
            idxs = _ot1_dominant_recipes(fallback_term, vectors, vocab_to_idx, top_n=5)
            print(f'  [S6] {cname}: no targets, OT1-dominant fallback on "{fallback_term}" → {len(idxs)} recipes (contrast)')
        if not idxs:
            centroids[cname] = None
            continue
        in_mean  = vectors[idxs].mean(axis=0)
        out_idxs = [i for i in range(n_all) if i not in set(idxs)]
        out_mean = vectors[out_idxs].mean(axis=0) if out_idxs else np.zeros_like(in_mean)
        c = in_mean - beta * out_mean
        c = np.clip(c, 0.0, None)  # keep non-negative so it stays a valid profile in OT1 space
        if np.linalg.norm(c) < 1e-12:
            c = in_mean  # contrast wiped everything out → fall back to the plain mean
        centroids[cname] = _normalize_vec(c)
    return centroids

STRATEGIES = {
    'S1_target_recipes':        build_strategy_target_recipes(expert_spec, vecs, recipes, vocab_to_idx),
    'S2_ingredients':           build_strategy_ingredients(expert_spec, vecs, recipes, df),
    'S3_hybrid':                build_strategy_hybrid(expert_spec, vecs, recipes, df),
    'S4_target_recipes_median': build_strategy_target_recipes_median(expert_spec, vecs, recipes, vocab_to_idx),
    'S5_ingredients_median':    build_strategy_ingredients_median(expert_spec, vecs, recipes, df),
    'S6_contrast':              build_strategy_contrast(expert_spec, vecs, recipes, vocab_to_idx),
}

# Validate: every strategy must have all 7 centroids
for s_name, c_dict in STRATEGIES.items():
    missing = [c for c, v in c_dict.items() if v is None]
    if missing:
        print(f'⚠ {s_name} missing centroids for: {missing}')
    else:
        print(f'✅ {s_name}: 7 centroids built')
"""))

CELLS.append(code(
    """# Quick look at the centroids — which OT1 terms dominate each cluster, per strategy
for s_name, c_dict in STRATEGIES.items():
    print(f'\\n=== {s_name} ===')
    for cname in EXPERT_CLUSTERS:
        v = c_dict.get(cname)
        if v is None:
            print(f'  {cname:14s}  (missing)')
            continue
        top3 = sorted(enumerate(v), key=lambda kv: kv[1], reverse=True)[:3]
        top3_named = [(vocab[i], round(w, 3)) for i, w in top3]
        print(f'  {cname:14s}  top OT1: {top3_named}')
"""))

# ---------------------------------------------------------------------------
# 6. Algorithm implementations + Hungarian alignment
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 6. Clustering Algorithms (seeded + Hungarian-aligned variants)

Algorithms that natively accept seed centroids (K-Means, GMM, K-Medoids,
Fuzzy c-Means, DEC, NearestCentroid) consume the seeds directly. The rest
(Ward, DBSCAN, HDBSCAN, Spectral, SOM) run unseeded at k=7 and have their
cluster IDs **re-labeled to expert cluster names** via Hungarian assignment
on cluster-centroid cosine distance.

All algorithms return labels as **expert cluster name strings**, so plots and
exports stay on the same color and naming scheme.
"""))

CELLS.append(code(
    """def centroids_to_matrix(centroids_dict, cluster_order=None):
    \"\"\"Stack a {name: vec} dict into (matrix, name_list) in a stable order.\"\"\"
    names = cluster_order or list(centroids_dict.keys())
    mat = np.array([centroids_dict[n] for n in names])
    return mat, names

def nearest_centroid_labels(vectors, centroid_mat, name_list):
    \"\"\"Argmin cosine distance to each centroid; returns array of expert-cluster name strings.\"\"\"
    # vectors and centroid_mat are L2-normalized → cosine sim = dot product
    sims = vectors @ centroid_mat.T
    idx  = np.argmax(sims, axis=1)
    return np.array([name_list[i] for i in idx])

def hungarian_align(algo_labels_int, vectors, centroid_mat, name_list):
    \"\"\"Map integer algorithm labels to expert cluster name strings via Hungarian assignment.

    Extra algorithm clusters (when an algo finds >7 clusters) get the
    'extra_<id>' label and are kept visible but uncolored.
    \"\"\"
    uniq = sorted(set(algo_labels_int))
    n_e  = len(name_list)
    # cluster centers in OT1 space (normalized so cosine = dot)
    centers = np.array([vectors[algo_labels_int == c].mean(axis=0) for c in uniq])
    norms   = np.linalg.norm(centers, axis=1, keepdims=True)
    centers = centers / np.maximum(norms, 1e-12)

    cost = 1.0 - centers @ centroid_mat.T        # (n_uniq, n_e)
    n_u  = cost.shape[0]
    K    = max(n_u, n_e)
    padded = np.full((K, K), 10.0)
    padded[:n_u, :n_e] = cost
    row_ind, col_ind = linear_sum_assignment(padded)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if r < n_u and c < n_e:
            mapping[uniq[r]] = name_list[c]
    return np.array([mapping.get(l, f'extra_{l}') for l in algo_labels_int])

print('Alignment helpers defined.')
"""))

CELLS.append(code(
    """# ── K-Medoids with seeded medoids ────────────────────────────────────────────
def kmedoids_seeded(dist_matrix, centroid_mat, name_list, vectors, max_iter=500):
    # Initial medoids = the recipe nearest to each seed centroid
    sims = vectors @ centroid_mat.T
    medoids = [int(np.argmax(sims[:, j])) for j in range(centroid_mat.shape[0])]
    seen = set(); medoids_uniq = []
    for m in medoids:
        if m not in seen: medoids_uniq.append(m); seen.add(m)
    # If duplicates collapsed, fill with random unused
    while len(medoids_uniq) < centroid_mat.shape[0]:
        for i in range(dist_matrix.shape[0]):
            if i not in seen: medoids_uniq.append(i); seen.add(i); break
    medoids = medoids_uniq

    for _ in range(max_iter):
        labels = np.argmin(dist_matrix[:, medoids], axis=1)
        new_medoids = []
        for c in range(len(medoids)):
            pts = np.where(labels == c)[0]
            if len(pts) == 0:
                new_medoids.append(medoids[c]); continue
            sub = dist_matrix[np.ix_(pts, pts)]
            new_medoids.append(int(pts[np.argmin(sub.sum(axis=1))]))
        if new_medoids == medoids: break
        medoids = new_medoids
    labels = np.argmin(dist_matrix[:, medoids], axis=1)
    return np.array([name_list[i] for i in labels])

# ── Fuzzy c-Means with seeded centroids ──────────────────────────────────────
def fuzzy_cmeans_seeded(X, centroid_mat, name_list, m=2.0, max_iter=500, tol=1e-7):
    n  = X.shape[0]
    centroids = centroid_mat.copy()
    c = centroids.shape[0]
    # Init U from distances
    dist = np.array([[np.linalg.norm(X[j] - centroids[i]) for j in range(n)] for i in range(c)])
    dist = np.maximum(dist, 1e-12)
    ratio = dist[None, :, :] / dist[:, None, :]
    U = 1.0 / (ratio ** (2.0 / (m - 1))).sum(axis=1)
    U /= U.sum(axis=0, keepdims=True)
    for _ in range(max_iter):
        Um = U ** m
        centroids = (Um @ X) / Um.sum(axis=1, keepdims=True)
        dist = np.array([[np.linalg.norm(X[j] - centroids[i]) for j in range(n)] for i in range(c)])
        dist = np.maximum(dist, 1e-12)
        ratio = dist[None, :, :] / dist[:, None, :]
        U_new = 1.0 / (ratio ** (2.0 / (m - 1))).sum(axis=1)
        U_new /= U_new.sum(axis=0, keepdims=True)
        if np.max(np.abs(U_new - U)) < tol: break
        U = U_new
    labels = np.argmax(U, axis=0)
    return np.array([name_list[i] for i in labels])

# ── DEC simplified with seeded initial PCA-kmeans centers ────────────────────
def dec_simplified_seeded(X, centroid_mat, name_list, enc_dim=8, n_iter=300):
    d = min(enc_dim, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=d, random_state=42).fit(X)
    Z   = pca.transform(X)
    Z_seeds = pca.transform(centroid_mat)  # project seeds to PCA space
    centers = Z_seeds.copy()
    for _ in range(n_iter):
        dist2 = np.sum((Z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        q     = 1.0 / (1.0 + dist2)
        q    /= q.sum(axis=1, keepdims=True)
        f     = q.sum(axis=0)
        p     = (q ** 2) / np.maximum(f, 1e-12)
        p    /= p.sum(axis=1, keepdims=True)
        for j in range(centers.shape[0]):
            centers[j] = (p[:, j:j+1] * Z).sum(axis=0) / max(p[:, j].sum(), 1e-12)
    labels = np.argmax(q, axis=1)
    return np.array([name_list[i] for i in labels])

# ── SOM with k-Means on nodes seeded by centroids ────────────────────────────
def som_cluster_seeded(X, centroid_mat, name_list,
                       lr=0.5, sigma=1.5, n_iter=8000, random_state=42):
    rng     = np.random.RandomState(random_state)
    k       = centroid_mat.shape[0]
    g       = int(np.ceil(np.sqrt(k)))
    n_nodes = g * g
    weights = normalize(rng.randn(n_nodes, X.shape[1]))
    node_pos = np.array([(r, c) for r in range(g) for c in range(g)], dtype=float)
    for t in range(n_iter):
        frac  = 1.0 - t / n_iter
        lr_t  = lr * frac
        sig_t = max(sigma * frac, 0.01)
        xi    = X[rng.randint(0, X.shape[0])]
        bmu   = np.argmin(np.linalg.norm(weights - xi, axis=1))
        d2    = ((node_pos - node_pos[bmu]) ** 2).sum(axis=1)
        h     = np.exp(-d2 / (2 * sig_t ** 2))
        weights += lr_t * h[:, None] * (xi - weights)
    # Cluster node weights with k-Means *seeded by expert centroids*
    node_labels = KMeans(n_clusters=k, init=centroid_mat, n_init=1,
                         random_state=random_state).fit_predict(weights)
    recipe_labels = np.array([node_labels[np.argmin(np.linalg.norm(weights - xi, axis=1))] for xi in X])
    return np.array([name_list[i] for i in recipe_labels])

# ── DBSCAN / HDBSCAN: scan params for k≈7, then Hungarian align ──────────────
def dbscan_target_k(diss_m, K, eps_grid=np.arange(0.01, 1.00, 0.01)):
    best_eps, best_raw = None, None
    best_diff = float('inf')
    for eps in eps_grid:
        raw = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit_predict(diss_m)
        n_clust = len(set(raw) - {-1})
        diff = abs(n_clust - K)
        if diff < best_diff:
            best_diff, best_eps, best_raw = diff, eps, raw
        if n_clust == K and (raw == -1).sum() == 0: break
    # Resolve noise to nearest non-noise
    labels = best_raw.copy()
    noise  = np.where(labels == -1)[0]
    non_n  = np.where(labels != -1)[0]
    if len(non_n) == 0:
        return np.zeros(len(labels), dtype=int)
    for i in noise:
        labels[i] = labels[non_n[np.argmin(diss_m[i, non_n])]]
    return labels

def hdbscan_target_k(diss_m, K, n_recipes):
    if not _has_hdbscan: return None
    best_mcs, best_raw = None, None
    best_diff = float('inf')
    for mcs in range(2, min(n_recipes, 20)):
        raw = _HDBSCAN(min_cluster_size=mcs, metric='precomputed').fit_predict(diss_m)
        n_clust = len(set(raw) - {-1})
        diff = abs(n_clust - K)
        if diff < best_diff:
            best_diff, best_mcs, best_raw = diff, mcs, raw
    if best_raw is None: return None
    labels = best_raw.copy()
    noise  = np.where(labels == -1)[0]
    non_n  = np.where(labels != -1)[0]
    if len(non_n) == 0:
        return np.zeros(len(labels), dtype=int)
    for i in noise:
        labels[i] = labels[non_n[np.argmin(diss_m[i, non_n])]]
    return labels

print('Algorithm implementations ready.')
"""))

# ---------------------------------------------------------------------------
# 7. Run all combinations
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 7. Run All Strategies × All Algorithms

10 algorithms × 3 strategies = 30 partitions. Stored in
`results[strategy][algo] = labels` where labels are expert cluster name
strings.
"""))

CELLS.append(code(
    """def run_all_algos(centroids_dict, vectors, diss_m):
    centroid_mat, name_list = centroids_to_matrix(centroids_dict, cluster_order=EXPERT_CLUSTERS)
    K = len(name_list)
    out = {}

    # ── Seeded algorithms (return expert names directly) ─────────────────────
    out['NearestCentroid'] = nearest_centroid_labels(vectors, centroid_mat, name_list)

    km_int = KMeans(n_clusters=K, init=centroid_mat, n_init=1,
                    random_state=42).fit_predict(vectors)
    out['k-Means'] = np.array([name_list[i] for i in km_int])

    out['k-Medoids'] = kmedoids_seeded(diss_m, centroid_mat, name_list, vectors)

    enc_dim_gmm = min(K, vectors.shape[0] - 1, vectors.shape[1])
    Z_gmm = PCA(n_components=enc_dim_gmm, random_state=42).fit_transform(vectors)
    means_init_gmm = PCA(n_components=enc_dim_gmm, random_state=42).fit(vectors).transform(centroid_mat)
    gmm_int = GaussianMixture(n_components=K, n_init=1, means_init=means_init_gmm,
                              covariance_type='full', random_state=42).fit_predict(Z_gmm)
    out['GMM'] = np.array([name_list[i] for i in gmm_int])

    enc_dim_fcm = min(max(K, 4), vectors.shape[0] - 1, vectors.shape[1])
    pca_fcm = PCA(n_components=enc_dim_fcm, random_state=42).fit(vectors)
    Z_fcm = pca_fcm.transform(vectors)
    seeds_fcm = pca_fcm.transform(centroid_mat)
    out['Fuzzy c-Means'] = fuzzy_cmeans_seeded(Z_fcm, seeds_fcm, name_list)

    out['SOM'] = som_cluster_seeded(vectors, centroid_mat, name_list)

    out['DEC (simpl.)'] = dec_simplified_seeded(vectors, centroid_mat, name_list)

    # ── Unseedable algorithms — Hungarian align after the fact ───────────────
    Z_ward = linkage(squareform(diss_m, checks=False), method='ward')
    ward_int = fcluster(Z_ward, t=K, criterion='maxclust')
    out['Ward'] = hungarian_align(ward_int, vectors, centroid_mat, name_list)

    db_int = dbscan_target_k(diss_m, K)
    out['DBSCAN'] = hungarian_align(db_int, vectors, centroid_mat, name_list)

    if _has_hdbscan:
        hdb_int = hdbscan_target_k(diss_m, K, len(vectors))
        out['HDBSCAN'] = hungarian_align(hdb_int, vectors, centroid_mat, name_list)
    else:
        out['HDBSCAN'] = None

    affinity = np.clip(1.0 - diss_m, 0.0, 1.0); np.fill_diagonal(affinity, 1.0)
    spec_int = SpectralClustering(n_clusters=K, affinity='precomputed',
                                  n_init=30, random_state=42).fit_predict(affinity)
    out['Spectral'] = hungarian_align(spec_int, vectors, centroid_mat, name_list)
    return out

results = {}
for s_name, c_dict in STRATEGIES.items():
    if any(v is None for v in c_dict.values()):
        print(f'⚠ Skipping {s_name} (missing centroids)')
        continue
    print(f'\\nRunning {s_name}...')
    results[s_name] = run_all_algos(c_dict, vecs, diss)
    # Brief per-algo summary
    for algo, lbl in results[s_name].items():
        if lbl is None: continue
        uniq, counts = np.unique(lbl, return_counts=True)
        print(f'  {algo:18s}  {dict(zip(uniq, counts))}')
"""))

# ---------------------------------------------------------------------------
# 7b. Direct methods (M1 label-prop, M2 rule-based, M3 consensus)
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 7b. Direct Methods (Family B)

Three methods that do **not** fit the "build centroid -> run 10 algorithms"
grid. Each produces a single labeling, collected in `direct_results`:

- **M1 - Label Propagation**: kNN cosine graph over all recipes; expert target
  recipes are pinned to their cluster and labels diffuse through the manifold.
  Clusters with no Target Recipe are seeded from a few ingredient recipes.
- **M2 - Rule-based pre-assignment**: the expert `Regeln/ Notizen` rules,
  transcribed in full and applied on per-recipe Totalmenge-normalized CAS
  concentrations. Recipes that fire no rule fall back to the nearest S1 centroid.
- **M3 - Consensus**: co-association across all Family-A algorithm runs
  (excluding the degenerate DEC / Fuzzy c-Means) plus M1, clustered at k=7.
"""))

# ── Keyword signal (from Verkostung Übersicht 'fertige Aromen') ──────────────
CELLS.append(md(
    """### 7b.0  Keyword signal (soft, confidence-gated)

The `Key-Words Übersicht fertige Aromen` column of
`Verkostung Cluster KI vom 11_06_2026.xlsx` carries recipe-level sensory
descriptors. We map the **discriminative** German words to expert clusters
(`KW_LEXICON`); generic words that appear across every cluster
(`fruchtig`, `bonbonartig`, `reif`, `saftig`, `süß`, `erdbeere`) are
deliberately excluded. Per recipe we count cluster hits and keep a **confident
vote** only when the top cluster leads the runner-up by ≥1.

This vote is a *soft* signal used by the `M1_kw` / `M2_kw` variants below — the
hard chemistry rules from the Vorgabe stay authoritative. Keywords exist for
only the tasted subset, so recipes without a confident vote keep their original
(rule / centroid / propagation) assignment. Note: these descriptors come from
the same finished aromas the panel later tasted, so treat panel-agreement gains
as suggestive, not independent proof."""))

CELLS.append(code(
    """import re
VERK_XLSX = '../data/gold/Verkostung Cluster KI vom 11_06_2026.xlsx'

# Discriminative German descriptor -> expert cluster. Generic tokens omitted.
KW_LEXICON = {
    'karamell': 'warm', 'marmelade': 'warm',
    'grün': 'green', 'frisch': 'green',
    'blumig': 'floral',
    'sahnig': 'dairy', 'milchig': 'dairy', 'buttrig': 'dairy',
    'überreif': 'unpleasant', 'bier': 'unpleasant', 'holzig': 'unpleasant',
    'mango': 'fruity', 'tutti-frutti': 'fruity',
    'walderdbeere': 'Walderdbeere',
}
KW_MARGIN = 2   # top cluster must lead the runner-up by >= this to count as a vote
                # (>=2 = at least two agreeing keywords; filters noisy single-word votes)

_ub = pd.read_excel(VERK_XLSX, sheet_name='Übersicht')
_kwcol = [c for c in _ub.columns if 'Key-Words' in str(c)][0]
_ridcol = _ub.columns[0]

def _match_kw(rid, recipes):
    rid = str(rid).strip()
    if rid in recipes:
        return rid
    pref = [r for r in recipes if r.startswith(rid)]
    if pref:
        return pref[0]
    num = rid.rstrip('PpHhNnXxKk ')
    pref = [r for r in recipes if r.startswith(num)]
    return pref[0] if pref else None

kw_scores = {}   # recipe -> {cluster: hit count}
for _, row in _ub.iterrows():
    rid, kws = row[_ridcol], row[_kwcol]
    if pd.isna(rid) or pd.isna(kws):
        continue
    ds = _match_kw(rid, recipes)
    if ds is None:
        continue
    sc = kw_scores.setdefault(ds, {})
    for tok in re.split(r'[,/]', str(kws)):
        cl = KW_LEXICON.get(tok.strip().lower())
        if cl:
            sc[cl] = sc.get(cl, 0) + 1

kw_vote = {}     # recipe -> cluster, only when confident
for recipe, sc in kw_scores.items():
    if not sc:
        continue
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    top_c, top_n = ranked[0]
    run_n = ranked[1][1] if len(ranked) > 1 else 0
    if top_n - run_n >= KW_MARGIN:
        kw_vote[recipe] = top_c

print(f'Keyword signal: {len(kw_scores)} recipes have mapped keywords, '
      f'{len(kw_vote)} give a confident vote (margin >= {KW_MARGIN}).')
for recipe in sorted(kw_vote):
    print(f'  {recipe:12s} -> {kw_vote[recipe]:14s}  (scores {kw_scores[recipe]})')
"""))

# ── M1: Label Propagation ────────────────────────────────────────────────────
CELLS.append(code(
    """from sklearn.semi_supervised import LabelSpreading

direct_results = {}

cluster_to_int = {c: i for i, c in enumerate(EXPERT_CLUSTERS)}
int_to_cluster = {i: c for c, i in cluster_to_int.items()}

# Seed labels: target recipes pinned; clusters with no target seeded from a few
# ingredient recipes (hybrid-style) so every expert cluster can be propagated.
seed_labels = np.full(len(recipes), -1, dtype=int)
for cname in EXPERT_CLUSTERS:
    s = expert_spec.get(cname, {})
    resolved = [r for r in s.get('target_recipes_resolved', []) if r in recipes]
    if not resolved:
        idxs = _recipes_containing_cas(s.get('cas', []), df, recipes)
        resolved = [recipes[i] for i in idxs[:3]]
        if resolved:
            print(f'  [M1] {cname}: no targets, seeding from {len(resolved)} ingredient recipes')
    for r in resolved:
        seed_labels[recipes.index(r)] = cluster_to_int[cname]

n_seeded = int((seed_labels >= 0).sum())
seeded_clusters = sorted({int_to_cluster[i] for i in seed_labels if i >= 0})
missing = [c for c in EXPERT_CLUSTERS if c not in seeded_clusters]
print(f'M1 Label Propagation: {n_seeded} seeded recipes across {len(seeded_clusters)} clusters')
if missing:
    print(f'  ⚠ no seed for: {missing} — these clusters cannot be propagated')

# vecs are L2-normalized, so kNN ordering matches cosine similarity.
ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2, max_iter=1000)
ls.fit(vecs, seed_labels)
direct_results['M1_label_prop'] = np.array([int_to_cluster[i] for i in ls.transduction_])
print('  sizes:', {c: int((direct_results['M1_label_prop'] == c).sum()) for c in EXPERT_CLUSTERS})
"""))

# ── M2: Rule-based pre-assignment (full interpretation) ──────────────────────
CELLS.append(code(
    """# Expert rules are PARSED from the 'Regeln/ Notizen' column of
# Erdbeer Clustering Sensorik Vorgabe.xlsx (no longer hardcoded — edits to the
# Excel propagate automatically). conc = per-recipe Totalmenge-normalized amount
# of the marker CAS in that recipe.
#   mode 'threshold' : marker conc > thr        (rule text '> X eindeutig')
#   mode 'all'       : ALL markers present > 0   ('alle drei' / 'beide ... müssen vorhanden')
#   mode 'any'       : ANY marker present > 0    ('wenn Rohstoff vorhanden')
# tier = precedence (1 strongest): threshold=1, all=2, any=3.
# 'keine eindeutige Regel gefunden' / blank -> no rule, centroid fallback only.
import re

def _parse_thr(text):
    \"\"\"Parse a German-decimal threshold like '> 0,0009' / '>0.004' -> float.\"\"\"
    m = re.search(r'>\\s*([0-9]+[.,][0-9]+)', text)
    return float(m.group(1).replace(',', '.')) if m else None

def build_rules(edf):
    \"\"\"Build M2 rule dicts from the ordered expert rows.

    A rule-bearing 'all'/'any' row absorbs the following same-cluster rows whose
    Regeln cell is blank into its marker set (so 'alle drei'/'beide' reconstruct
    the full AND group). Threshold rules apply to their own CAS only.\"\"\"
    rows = edf.reset_index(drop=True)
    n = len(rows)
    rules, i = [], 0
    while i < n:
        row = rows.iloc[i]
        cluster = row['Cluster']
        cas = str(row['CAS-Nr.']).strip() if pd.notna(row['CAS-Nr.']) else None
        regeln = row['Regeln/ Notizen']
        text = '' if pd.isna(regeln) else str(regeln).strip()
        low = text.lower()
        if not text or 'keine eindeutige' in low or cas is None:
            i += 1
            continue
        thr = _parse_thr(text)
        if thr is not None:
            rules.append({'cluster': cluster, 'mode': 'threshold', 'cas': [cas],
                          'thr': thr, 'tier': 1, 'note': f'{cluster}: {text}'})
            i += 1
            continue
        if 'müssen vorhanden' in low or 'alle ' in low or 'beide' in low:
            mode, tier = 'all', 2
        elif 'vorhanden' in low:                       # 'wenn Rohstoff vorhanden'
            mode, tier = 'any', 3
        else:
            i += 1                                     # unrecognized note -> ignore
            continue
        group_cas, j = [cas], i + 1
        while j < n:                                   # absorb blank-rule same-cluster rows
            nrow = rows.iloc[j]
            if nrow['Cluster'] != cluster:
                break
            nreg = nrow['Regeln/ Notizen']
            if pd.notna(nreg) and str(nreg).strip():
                break
            if pd.notna(nrow['CAS-Nr.']):
                group_cas.append(str(nrow['CAS-Nr.']).strip())
            j += 1
        rules.append({'cluster': cluster, 'mode': mode, 'cas': group_cas,
                      'tier': tier, 'note': f'{cluster}: {text}'})
        i = j
    return rules

RULES = build_rules(expert_df)
print(f'Parsed {len(RULES)} M2 rules from Vorgabe Excel:')
for r in sorted(RULES, key=lambda r: (r['tier'], r['cluster'])):
    extra = f" thr={r['thr']}" if 'thr' in r else ''
    print(f"  tier{r['tier']} {r['mode']:9s} {r['cluster']:14s} cas={r['cas']}{extra}")

# Precompute per-recipe normalized concentration for every marker CAS.
_marker_cas = {c for rule in RULES for c in rule['cas']}
_conc = {}
for cas in _marker_cas:
    sub = df[df[CAS_COL].astype(str).str.strip() == cas]
    _conc[cas] = sub.groupby(REZ_COL)[TOTAL_COL].sum().to_dict()

def C(recipe, cas):
    return _conc.get(cas, {}).get(recipe, 0.0)

def fired_rules(recipe):
    out = []
    for rule in RULES:
        cas = rule['cas']
        if rule['mode'] == 'threshold':
            ok = C(recipe, cas[0]) > rule['thr']
        elif rule['mode'] == 'all':
            ok = all(C(recipe, c) > 0 for c in cas)
        else:  # 'any'
            ok = any(C(recipe, c) > 0 for c in cas)
        if ok:
            out.append(rule)
    return out

s1_centroids = STRATEGIES['S1_target_recipes']
m2_labels, audit = [], []
for ri, recipe in enumerate(recipes):
    fired = fired_rules(recipe)
    if fired:
        best_tier = min(r['tier'] for r in fired)
        cands = [r for r in fired if r['tier'] == best_tier]
        cand_clusters = sorted({r['cluster'] for r in cands})
        if len(cand_clusters) == 1:
            chosen = cand_clusters[0]
            reason = cands[0]['note']
        else:
            v = vecs[ri]
            sims = {c: (float(v @ s1_centroids[c]) if s1_centroids.get(c) is not None else -1.0)
                    for c in cand_clusters}
            chosen = max(sims, key=sims.get)
            reason = f"tie@tier{best_tier} {cand_clusters} -> {chosen} (cosine)"
        m2_labels.append(chosen)
        audit.append((recipe, f'tier{best_tier}', chosen, reason))
    else:
        v = vecs[ri]
        sims = {c: float(v @ cv) for c, cv in s1_centroids.items() if cv is not None}
        chosen = max(sims, key=sims.get)
        m2_labels.append(chosen)
        audit.append((recipe, 'centroid', chosen, 'no rule fired'))

direct_results['M2_rule_based'] = np.array(m2_labels)
n_rule = sum(1 for a in audit if a[1] != 'centroid')
print(f'M2 Rule-based: {n_rule}/{len(recipes)} assigned by rule, {len(recipes)-n_rule} by nearest centroid')
print('  sizes:', {c: int((direct_results['M2_rule_based'] == c).sum()) for c in EXPERT_CLUSTERS})
print('\\n  Audit — recipes assigned by an expert rule:')
for recipe, src, cl, reason in audit:
    if src != 'centroid':
        print(f'    {recipe:12s}  [{src}]  -> {cl:14s}  ({reason})')
"""))

# ── M1_kw / M2_kw: keyword-enhanced variants ─────────────────────────────────
CELLS.append(md(
    """### 7b.1  Keyword-enhanced variant (`M2_kw`)

Same machinery as M2, plus the confident keyword vote — added **alongside** the
original so we can measure whether keywords help (see the `Verkostung_Compare`
sheet). Rules stay authoritative:

- **`M2_kw`** = M2 where the keyword vote replaces nearest-centroid for the
  *no-rule fallback* and breaks *same-tier rule ties*; rule-fired assignments and
  expert target/anchor recipes are untouched.

An earlier `M1_kw` (label-propagation with keyword seeds) was dropped: even 8
seeds swung 27 of 130 labels and collapsed real clusters (Walderdbeere, unpleasant),
so the propagation channel proved too noisy. With the gate raised to
`KW_MARGIN = 2` (two agreeing keywords) only a handful of high-confidence votes
remain — most of which the chemistry rules already cover, so M2_kw stays close
to M2 by design."""))

CELLS.append(code(
    """# M2_kw — keyword vote for no-rule fallback and same-tier ties; rules authoritative.
# Expert target/anchor recipes are never overridden by a keyword (same guard as M1_kw).
_target_set = {r for s in expert_spec.values() for r in s.get('target_recipes_resolved', [])}
m2kw_labels, audit_kw = [], []
for ri, recipe in enumerate(recipes):
    fired = fired_rules(recipe)
    vote = None if recipe in _target_set else kw_vote.get(recipe)
    if fired:
        best_tier = min(r['tier'] for r in fired)
        cand_clusters = sorted({r['cluster'] for r in fired if r['tier'] == best_tier})
        if len(cand_clusters) == 1:
            chosen, reason = cand_clusters[0], 'rule'
        elif vote in cand_clusters:
            chosen, reason = vote, f'tie@tier{best_tier} {cand_clusters} -> keyword'
        else:
            v = vecs[ri]
            sims = {c: (float(v @ s1_centroids[c]) if s1_centroids.get(c) is not None else -1.0)
                    for c in cand_clusters}
            chosen, reason = max(sims, key=sims.get), f'tie@tier{best_tier} -> cosine'
    elif vote is not None:
        chosen, reason = vote, 'no rule -> keyword'
    else:
        v = vecs[ri]
        sims = {c: float(v @ cv) for c, cv in s1_centroids.items() if cv is not None}
        chosen, reason = max(sims, key=sims.get), 'no rule -> nearest centroid'
    m2kw_labels.append(chosen)
    audit_kw.append((recipe, chosen, reason))

direct_results['M2_kw'] = np.array(m2kw_labels)
diff2 = int((direct_results['M2_kw'] != direct_results['M2_rule_based']).sum())
print(f'M2_kw: differs from M2 on {diff2} recipes')
print('  sizes:', {c: int((direct_results['M2_kw'] == c).sum()) for c in EXPERT_CLUSTERS})
print('  keyword-driven assignments:')
for recipe, cl, reason in audit_kw:
    if 'keyword' in reason:
        print(f'    {recipe:12s} -> {cl:14s} ({reason})')
"""))

# ── M3: Consensus clustering ─────────────────────────────────────────────────
CELLS.append(code(
    """from sklearn.cluster import AgglomerativeClustering

DEGENERATE = {'DEC (simpl.)', 'Fuzzy c-Means'}  # collapsed in v3 — excluded from consensus

label_vectors = []
for s_name, algo_dict in results.items():
    for algo, lbl in algo_dict.items():
        if lbl is None or algo in DEGENERATE:
            continue
        label_vectors.append(np.asarray(lbl))
label_vectors.append(direct_results['M1_label_prop'])

M = len(label_vectors)
N = len(recipes)
co = np.zeros((N, N), dtype=np.float64)
for lbl in label_vectors:
    same = (lbl[:, None] == lbl[None, :]).astype(np.float64)
    co += same
co /= max(M, 1)

d = 1.0 - co
np.fill_diagonal(d, 0.0)
agg = AgglomerativeClustering(n_clusters=K_EXPERT, metric='precomputed', linkage='average')
m3_int = agg.fit_predict(d)

# Re-label integer consensus clusters to expert names via Hungarian on S1 centroids.
cmat, cnames = centroids_to_matrix(STRATEGIES['S1_target_recipes'], EXPERT_CLUSTERS)
direct_results['M3_consensus'] = hungarian_align(m3_int, vecs, cmat, cnames)
print(f'M3 Consensus over {M} label vectors (excluded: {sorted(DEGENERATE)})')
print('  sizes:', {c: int((direct_results['M3_consensus'] == c).sum())
                   for c in sorted(set(direct_results['M3_consensus']))})
"""))

# ---------------------------------------------------------------------------
# 8-10. Plot each strategy
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 8. MDS Visualization Helper
"""))

CELLS.append(code(
    """def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 2:
        ax.scatter(x, y, s=80, color=kwargs.get('facecolor', 'gray'), zorder=4)
        return
    cov = np.cov(x, y)
    ev, evec = np.linalg.eigh(cov)
    order = ev.argsort()[::-1]
    ev, evec = ev[order], evec[:, order]
    angle  = np.degrees(np.arctan2(*evec[:, 0][::-1]))
    width  = max(2 * n_std * np.sqrt(abs(ev[0])), 0.001)
    height = max(2 * n_std * np.sqrt(abs(ev[1])), 0.001)
    ax.add_patch(Ellipse(xy=(np.mean(x), np.mean(y)),
                         width=width, height=height, angle=angle, **kwargs))

def mds_plot_named(ax, coords, names, str_labels, title, show_legend=False):
    \"\"\"Plot MDS coords colored by expert cluster name (string label).\"\"\"
    unique = sorted(set(str_labels))
    ax.set_facecolor('#FAFAFA')
    ax.axhline(0, color='#CCCCCC', lw=0.7, zorder=1)
    ax.axvline(0, color='#CCCCCC', lw=0.7, zorder=1)
    for lbl in unique:
        mask = np.array(str_labels) == lbl
        cx, cy = coords[mask, 0], coords[mask, 1]
        col = EXPERT_COLORS.get(lbl, '#888888')  # extra_* → gray
        confidence_ellipse(cx, cy, ax, n_std=1.5,
                           facecolor=col, alpha=0.15, edgecolor=col,
                           linewidth=1.2, linestyle='--', zorder=2)
        ax.scatter(cx, cy, color=col, s=55, zorder=4, edgecolors='white', lw=0.7)
        for i, name in enumerate(np.array(names)[mask]):
            ax.annotate(name, (cx[i], cy[i]), fontsize=6, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', color=col)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('MDS Dim 1', fontsize=8)
    ax.set_ylabel('MDS Dim 2', fontsize=8)
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.tick_params(labelsize=7)
    if show_legend:
        present = [c for c in EXPERT_CLUSTERS if c in unique]
        patches = [mpatches.Patch(color=EXPERT_COLORS[c], label=c) for c in present]
        extras = [c for c in unique if c.startswith('extra_')]
        for e in extras:
            patches.append(mpatches.Patch(color='#888888', label=e))
        ax.legend(handles=patches, fontsize=7, loc='best', framealpha=0.85)

def plot_strategy_grid(strategy_name, results_for_strategy, save_path):
    algos = list(results_for_strategy.keys())
    n     = len(algos)
    cols  = 4
    rows  = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows))
    axes = axes.flatten()
    for i, algo in enumerate(algos):
        lbl = results_for_strategy[algo]
        ax = axes[i]
        if lbl is None:
            ax.text(0.5, 0.5, f'{algo}\\n(not available)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(algo, fontsize=9, fontweight='bold')
            continue
        title = f'{algo}  ({len(set(lbl))} clusters)'
        mds_plot_named(ax, mds_coords, recipes, lbl, title, show_legend=True)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'{strategy_name} - All Algorithms (Erdbeere, n={len(recipes)}, k={K_EXPERT}, expert-seeded)',
                 fontsize=13, fontweight='bold', y=1.005)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Saved: {save_path}')

print('Plotting helpers defined.')
"""))

CELLS.append(md("## 9. Strategy 1 - Target Recipes as Centroids\n"))
CELLS.append(code(
    """plot_strategy_grid(
    'S1: Target Recipes as Centroids',
    results['S1_target_recipes'],
    f'{OUTPUT_DIR}/erdbeere_v3_S1_target_recipes_mds.png',
)
"""))

CELLS.append(md("## 10. Strategy 2 - Characteristic Ingredients as Centroids\n"))
CELLS.append(code(
    """plot_strategy_grid(
    'S2: Characteristic Ingredients as Centroids',
    results['S2_ingredients'],
    f'{OUTPUT_DIR}/erdbeere_v3_S2_ingredients_mds.png',
)
"""))

CELLS.append(md("## 11. Strategy 3 - Hybrid (Target Recipes + Ingredient Fallback)\n"))
CELLS.append(code(
    """plot_strategy_grid(
    'S3: Hybrid (Target Recipes + Ingredient Fallback)',
    results['S3_hybrid'],
    f'{OUTPUT_DIR}/erdbeere_v3_S3_hybrid_mds.png',
)
"""))

CELLS.append(md("## 11b. Strategy 4 - Target Recipes (median aggregation)\n"))
CELLS.append(code(
    """plot_strategy_grid(
    'S4: Target Recipes (median)',
    results['S4_target_recipes_median'],
    f'{OUTPUT_DIR}/erdbeere_v3_S4_target_recipes_median_mds.png',
)
"""))

CELLS.append(md("## 11c. Strategy 5 - Characteristic Ingredients (median aggregation)\n"))
CELLS.append(code(
    """plot_strategy_grid(
    'S5: Characteristic Ingredients (median)',
    results['S5_ingredients_median'],
    f'{OUTPUT_DIR}/erdbeere_v3_S5_ingredients_median_mds.png',
)
"""))

CELLS.append(md("## 11d. Strategy 6 - Contrast (Rocchio) centroids\n"))
CELLS.append(code(
    """plot_strategy_grid(
    'S6: Contrast (Rocchio)',
    results['S6_contrast'],
    f'{OUTPUT_DIR}/erdbeere_v3_S6_contrast_mds.png',
)
"""))

CELLS.append(md("## 11e. Direct Methods - MDS comparison (M1 / M2 / M3)\n"))
CELLS.append(code(
    """_methods = [('M1_label_prop', 'M1: Label Propagation'),
            ('M2_rule_based', 'M2: Rule-based'),
            ('M3_consensus',  'M3: Consensus'),
            ('M2_kw',         'M2_kw: + keyword vote')]

_nm = len(_methods)
_cols = 3
_rows = -(-_nm // _cols)
fig, axes = plt.subplots(_rows, _cols, figsize=(6.5 * _cols, 5.5 * _rows))
axes = np.atleast_1d(axes).flatten()
for ax, (key, title) in zip(axes, _methods):
    lbl = direct_results[key]
    mds_plot_named(ax, mds_coords, recipes, lbl,
                   f'{title}  ({len(set(lbl))} clusters)', show_legend=True)
for j in range(_nm, len(axes)):
    axes[j].set_visible(False)
fig.suptitle(f'Direct Methods (Erdbeere, n={len(recipes)}, k={K_EXPERT}, expert-seeded)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/erdbeere_v3_direct_methods_mds.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {OUTPUT_DIR}/erdbeere_v3_direct_methods_mds.png')
"""))

# ---------------------------------------------------------------------------
# 11. Cross-strategy agreement
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 12. Cross-Strategy Cluster Agreement

For each algorithm, how stable is the partition across the six strategies?
Higher ARI / NMI means the choice of seeding strategy matters less for that
algorithm. Lower values mean the algorithm is sensitive to seed centroids.

The pairs of greatest interest are:

- **S1 vs S4**: mean vs median on the same Target-Recipe source — pure
  aggregation-function effect.
- **S2 vs S5**: mean vs median on the same ingredient-recipe source — same
  question, larger sample per cluster.
"""))

CELLS.append(code(
    """from itertools import combinations

algos = list(next(iter(results.values())).keys())
strategy_names = list(results.keys())

rows = []
for algo in algos:
    row = {'Algorithm': algo}
    for s_a, s_b in combinations(strategy_names, 2):
        lbl_a = results[s_a][algo]
        lbl_b = results[s_b][algo]
        if lbl_a is None or lbl_b is None:
            row[f'ARI {s_a} vs {s_b}'] = None
            row[f'NMI {s_a} vs {s_b}'] = None
            continue
        row[f'ARI {s_a} vs {s_b}'] = round(adjusted_rand_score(lbl_a, lbl_b), 3)
        row[f'NMI {s_a} vs {s_b}'] = round(normalized_mutual_info_score(lbl_a, lbl_b), 3)
    rows.append(row)

agreement_df = pd.DataFrame(rows)
print('Cross-strategy agreement per algorithm:')
print(agreement_df.to_string(index=False))

# Strategy representatives: Family-A strategies via NearestCentroid (purest
# 'closest expert centroid' assignment), direct methods M1-M3 as-is. These are
# the columns the experts compare in the Strategy_Comparison sheet.
representative = {s_name: algo_dict['NearestCentroid']
                  for s_name, algo_dict in results.items()
                  if algo_dict.get('NearestCentroid') is not None}
for m_name, lbl in direct_results.items():
    representative[m_name] = lbl

rep_names = list(representative.keys())
rep_rows = []
for s_a, s_b in combinations(rep_names, 2):
    la, lb = representative[s_a], representative[s_b]
    rep_rows.append({
        'Strategy A': s_a, 'Strategy B': s_b,
        'ARI': round(adjusted_rand_score(la, lb), 3),
        'NMI': round(normalized_mutual_info_score(la, lb), 3),
    })
rep_agreement_df = pd.DataFrame(rep_rows)
print('\\nPairwise agreement across strategy representatives:')
print(rep_agreement_df.to_string(index=False))
"""))

# ---------------------------------------------------------------------------
# 12. Export Excel
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 13. Export Cluster Assignments to Excel

The first sheet, **`Strategy_Comparison`**, is the expert hand-over view: one
row per recipe, one column per strategy/method. Family-A strategies (S1–S6) are
represented by their **NearestCentroid** assignment; the direct methods
(M1–M3) appear as their own columns. `Target_Recipe?` and
`Expert_Intended_Cluster` flag the expert's seed recipes so you can see where
each one landed under every strategy.

Per-strategy detail sheets follow (algorithm rows × cluster columns of recipe
ids), then one sheet per direct method, then two agreement sheets
(`Agreement_Strategies` = pairwise ARI/NMI across strategy representatives,
`Agreement_byAlgo` = per-algorithm cross-strategy stability).
"""))

CELLS.append(code(
    """out_xlsx = f'{OUTPUT_DIR}/cluster_assignments_expert_seeded_all_strategies.xlsx'

# Map each resolved Target Recipe to the cluster the expert intended for it.
target_to_cluster = {}
for cname, sp in expert_spec.items():
    for r in sp.get('target_recipes_resolved', []):
        target_to_cluster[r] = cname

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    # Master expert-review sheet: one row per recipe, one column per strategy.
    # Family-A strategies represented by NearestCentroid; direct methods as-is.
    master_rows = []
    for ri, recipe in enumerate(recipes):
        row = {
            'Recipe': recipe,
            'Target_Recipe?': 'yes' if recipe in target_to_cluster else '',
            'Expert_Intended_Cluster': target_to_cluster.get(recipe, ''),
        }
        for col_name, lbl in representative.items():
            row[col_name] = lbl[ri]
        master_rows.append(row)
    pd.DataFrame(master_rows).to_excel(writer, sheet_name='Strategy_Comparison', index=False)

    # Per-strategy detail sheets (Family A: algorithm rows x cluster columns).
    for s_name, algo_dict in results.items():
        rows = []
        for algo, lbl in algo_dict.items():
            if lbl is None:
                rows.append({'Algorithm': algo})
                continue
            row = {'Algorithm': algo}
            for cname in sorted(set(lbl)):
                mask = lbl == cname
                row[f'Cluster_{cname}'] = '\\n'.join(sorted(np.array(recipes)[mask].tolist()))
            rows.append(row)
        pd.DataFrame(rows).fillna('').to_excel(writer, sheet_name=s_name[:31], index=False)

    # Direct-method detail sheets (single row each).
    for m_name, lbl in direct_results.items():
        row = {'Method': m_name}
        for cname in sorted(set(lbl)):
            mask = lbl == cname
            row[f'Cluster_{cname}'] = '\\n'.join(sorted(np.array(recipes)[mask].tolist()))
        pd.DataFrame([row]).fillna('').to_excel(writer, sheet_name=m_name[:31], index=False)

    # Agreement sheets.
    rep_agreement_df.to_excel(writer, sheet_name='Agreement_Strategies', index=False)
    agreement_df.to_excel(writer, sheet_name='Agreement_byAlgo', index=False)

print(f'Exported: {out_xlsx}')
print(f'  Sheets: Strategy_Comparison + {len(results)} strategy + {len(direct_results)} direct + 2 agreement')

# Per-cluster count summary across strategy representatives.
print('\\nCluster sizes per strategy representative:')
for col_name, lbl in representative.items():
    sizes = {c: int((lbl == c).sum()) for c in sorted(set(lbl))}
    print(f'  {col_name:26s}  {sizes}')
"""))

# ---------------------------------------------------------------------------
# 13a. Re-score vs the sensory panel (Verkostung) — old vs new M1/M2
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 13a. Re-score against the Verkostung panel — old vs new M1/M2

The updated `Erdbeer Clustering Sensorik Vorgabe.xlsx` changed the rules (notably
**Dimethylsulfid 75-18-3 now maps to `warm`**, previously `unpleasant`). This cell
compares the **new** M1/M2 assignments against the **previous run** (the
`Fred M1/M2` columns) recorded in `Verkostung Cluster KI vom 11_06_2026.xlsx`.

**Important caveat:** the panel agreement % (`Prozent %`) was judged against the
*old* placements — "is the aroma right in the cluster it was given". It therefore
**cannot be recomputed** for the new labels without re-tasting. We use it only to
read intent: a *changed* assignment where the panel **disagreed** with the old
placement (low %) is a **potential fix**; a change where the panel **agreed** with
the old placement (high %) is a **risk** worth a human look. `Jan Free Sorting` and
`Cluster vorgegeben` are free-text descriptors (not cluster labels), so they are
shown for reference only — mapping them to clusters is the deferred keyword step."""))

CELLS.append(code(
    """VERK_XLSX = '../data/gold/Verkostung Cluster KI vom 11_06_2026.xlsx'

# Ergebnisse sheet has the cleanest old labels + panel %. Positional columns:
# 0 Rezept Nr | 2 old M1 | 3 old M2 | 4 Cluster vorgegeben | 5 Jan Free Sorting | 7 Prozent %
_erg = pd.read_excel(VERK_XLSX, sheet_name='Ergebnisse', header=None)
_erg = _erg.iloc[2:].reset_index(drop=True)          # drop the 2 header rows

def _norm_cluster(x):
    \"\"\"Pull the canonical EXPERT_CLUSTER out of a possibly-annotated label.\"\"\"
    if pd.isna(x):
        return ''
    s = str(x).strip().lower()
    for c in EXPERT_CLUSTERS:
        if c.lower() in s:
            return c
    return str(x).strip()

def _match_verk(rid, recipes):
    \"\"\"Map a Verkostung id ('187.796P') to a dataset recipe.\"\"\"
    rid = str(rid).strip()
    if rid in recipes:
        return rid
    pref = [r for r in recipes if r.startswith(rid)]
    if pref:
        return pref[0]
    num = rid.rstrip('PpHhNnXx ')                     # strip trailing letter suffixes
    pref = [r for r in recipes if r.startswith(num)]
    return pref[0] if pref else None

new_m1 = {recipes[i]: direct_results['M1_label_prop'][i] for i in range(len(recipes))}
new_m2 = {recipes[i]: direct_results['M2_rule_based'][i] for i in range(len(recipes))}
new_m2kw = {recipes[i]: direct_results['M2_kw'][i] for i in range(len(recipes))}

cmp_rows = []
for _, r in _erg.iterrows():
    rid = r[0]
    if pd.isna(rid):
        continue
    ds = _match_verk(rid, recipes)
    if ds is None:
        cmp_rows.append({'Recipe': str(rid).strip(), 'matched': '(no dataset match)'})
        continue
    old1, old2 = _norm_cluster(r[2]), _norm_cluster(r[3])
    n1, n2 = new_m1[ds], new_m2[ds]
    try:
        pct = float(r[7])
    except (TypeError, ValueError):
        pct = np.nan
    def _flag(old, new):
        if old == new or not old:
            return ''
        if not np.isnan(pct) and pct < 50:
            return 'potential fix'
        if not np.isnan(pct) and pct >= 75:
            return 'RISK (panel liked old)'
        return 'changed'
    cmp_rows.append({
        'Recipe': str(rid).strip(), 'matched': ds,
        'old_M1': old1, 'new_M1': n1, 'M1_change': _flag(old1, n1),
        'old_M2': old2, 'new_M2': n2, 'M2_change': _flag(old2, n2),
        'M2_kw': new_m2kw[ds], 'M2kw_vs_M2': '*' if new_m2kw[ds] != n2 else '',
        'kw_vote': kw_vote.get(ds, ''),
        'panel_%': pct,
        'Cluster_vorgegeben': '' if pd.isna(r[4]) else str(r[4]).strip(),
        'Jan_Free_Sorting': '' if pd.isna(r[5]) else str(r[5]).strip(),
    })

compare_df = pd.DataFrame(cmp_rows)
pd.set_option('display.max_rows', 100)
print('Old vs New M1/M2 on the', len(compare_df), 'tasted recipes:')
print(compare_df.to_string(index=False))

m1_changed = compare_df['M1_change'].fillna('').ne('').sum() if 'M1_change' in compare_df else 0
m2_changed = compare_df['M2_change'].fillna('').ne('').sum() if 'M2_change' in compare_df else 0
print(f'\\nM1 changed assignment on {m1_changed} recipes; M2 changed on {m2_changed} recipes.')
for col, lbl in [('M1_change', 'M1'), ('M2_change', 'M2')]:
    if col in compare_df:
        fixes = int((compare_df[col] == 'potential fix').sum())
        risks = int((compare_df[col] == 'RISK (panel liked old)').sum())
        print(f'  {lbl}: {fixes} potential fixes (panel disliked old), {risks} risks (panel liked old)')

m2kw_d = int(compare_df['M2kw_vs_M2'].eq('*').sum()) if 'M2kw_vs_M2' in compare_df else 0
print(f'\\nKeyword variant on the tasted recipes: M2_kw differs from M2 on {m2kw_d} '
      f'(marked * in the M2kw_vs_M2 column).')

# Append the comparison as a sheet to the strategy export.
with pd.ExcelWriter(out_xlsx, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    compare_df.fillna('').to_excel(writer, sheet_name='Verkostung_Compare', index=False)
print(f'\\nAppended sheet Verkostung_Compare to {out_xlsx}')
"""))

# ---------------------------------------------------------------------------
# 13b. Strategy_Comparison overview plots
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 13b. Strategy_Comparison Overview (at a glance)

Four panels summarising the `Strategy_Comparison` sheet so the whole picture
is visible without scrolling 130 rows:

1. **Pairwise agreement (ARI)** - which strategies produce similar partitions.
2. **Cluster-size composition** - which strategies keep balanced clusters vs
   collapse everything into a few.
3. **Target-recipe recovery** - for each of the expert's seed recipes, where it
   lands under every strategy (✓ = matches the expert-intended cluster).
4. **Per-recipe consensus** - on the MDS map, how many strategies agree on each
   recipe (bright = unanimous, dark = contested); expert targets ringed in red.
"""))

CELLS.append(code(
    """from matplotlib.colors import ListedColormap

strat_cols  = list(representative.keys())
short       = lambda nm: nm.split('_')[0]
short_names = [short(s) for s in strat_cols]
labels_mat  = {k: np.asarray(v) for k, v in representative.items()}
nS, N = len(strat_cols), len(recipes)
ci    = {c: i for i, c in enumerate(EXPERT_CLUSTERS)}
cmap7 = ListedColormap([EXPERT_COLORS[c] for c in EXPERT_CLUSTERS])
ridx  = {r: i for i, r in enumerate(recipes)}

fig, axes = plt.subplots(2, 2, figsize=(20, 17))

# ── Panel 1: pairwise ARI agreement heatmap ──
ax = axes[0, 0]
ari = np.eye(nS)
for a in range(nS):
    for b in range(a + 1, nS):
        v = adjusted_rand_score(labels_mat[strat_cols[a]], labels_mat[strat_cols[b]])
        ari[a, b] = ari[b, a] = v
im = ax.imshow(ari, cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(nS)); ax.set_xticklabels(short_names, rotation=45, ha='right')
ax.set_yticks(range(nS)); ax.set_yticklabels(short_names)
for a in range(nS):
    for b in range(nS):
        ax.text(b, a, f'{ari[a, b]:.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Pairwise strategy agreement (Adjusted Rand Index)\\nhigher = more similar partitions',
             fontweight='bold')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Panel 2: cluster-size composition (stacked bars) ──
ax = axes[0, 1]
bottom = np.zeros(nS)
for c in EXPERT_CLUSTERS:
    vals = np.array([int((labels_mat[k] == c).sum()) for k in strat_cols])
    ax.bar(range(nS), vals, bottom=bottom, color=EXPERT_COLORS[c],
           label=c, edgecolor='white', linewidth=0.5)
    bottom += vals
ax.set_xticks(range(nS)); ax.set_xticklabels(short_names, rotation=45, ha='right')
ax.set_ylabel('recipes')
ax.set_title(f'Cluster-size composition per strategy (n={N})\\nbalanced vs collapsed clusters at a glance',
             fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='upper right')

# ── Panel 3: target-recipe recovery grid ──
ax = axes[1, 0]
targets = sorted(target_to_cluster, key=lambda r: (ci[target_to_cluster[r]], r))
T = len(targets)
mat = np.array([[ci.get(labels_mat[k][ridx[r]], 0) for k in strat_cols] for r in targets])
ax.imshow(mat, cmap=cmap7, vmin=0, vmax=len(EXPERT_CLUSTERS) - 1, aspect='auto')
ax.set_xticks(range(nS)); ax.set_xticklabels(short_names, rotation=45, ha='right')
ax.set_yticks(range(T))
ax.set_yticklabels([f'{r}  [{target_to_cluster[r]}]' for r in targets], fontsize=8)
for ti, r in enumerate(targets):
    for si, k in enumerate(strat_cols):
        if labels_mat[k][ridx[r]] == target_to_cluster[r]:
            ax.text(si, ti, '✓', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')
ax.set_title('Target-recipe recovery: where each expert seed recipe lands\\n'
             '✓ = matches expert-intended cluster; cell colour = assigned cluster',
             fontweight='bold')

# ── Panel 4: per-recipe consensus on the MDS map ──
ax = axes[1, 1]
alllab = np.array([labels_mat[k] for k in strat_cols])
agreement = np.array([np.unique(alllab[:, j], return_counts=True)[1].max() / nS for j in range(N)])
sc = ax.scatter(mds_coords[:, 0], mds_coords[:, 1], c=agreement, cmap='viridis',
                vmin=1.0 / nS, vmax=1.0, s=35, edgecolor='white', linewidth=0.3)
tmask = np.array([recipes[i] in target_to_cluster for i in range(N)])
ax.scatter(mds_coords[tmask, 0], mds_coords[tmask, 1], s=140, facecolors='none',
           edgecolors='red', linewidths=1.6, label='target recipes')
ax.set_title('Cross-strategy consensus per recipe (MDS)\\nbright = strategies agree, dark = contested',
             fontweight='bold')
ax.legend(fontsize=8, loc='best')
fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='fraction of strategies agreeing')

fig.suptitle('Strategy_Comparison overview - Erdbeere expert-seeded clustering',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
_ov = f'{OUTPUT_DIR}/erdbeere_v3_strategy_comparison_overview.png'
plt.savefig(_ov, dpi=150, bbox_inches='tight')
plt.show()
print('Saved overview:', _ov)

unanimous = int((agreement == 1.0).sum())
print(f'Recipes with unanimous assignment across all {nS} strategies: {unanimous}/{N}')
print('Target-recipe recovery rate per strategy (matches / {} targets):'.format(T))
for k in strat_cols:
    hit = sum(1 for r in targets if labels_mat[k][ridx[r]] == target_to_cluster[r])
    print(f'  {short(k):4s}  {hit}/{T}')
"""))

# ---------------------------------------------------------------------------
# 13. Quick takeaways
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 14. Quick Takeaways

- All 10 algorithms can be made to honour the expert-supplied k=7 partition,
  either by direct centroid seeding (K-Means, GMM, K-Medoids, Fuzzy c-Means,
  SOM, DEC, NearestCentroid) or by Hungarian re-labeling (Ward, DBSCAN,
  HDBSCAN, Spectral).
- The cross-strategy ARI/NMI table ranks algorithms by how much they care
  about the choice of seed centroid - higher = less sensitive.
- **For expert hand-over**, the `Strategy_Comparison` sheet is the place to
  start: scan each recipe across all strategy columns and tell us which column
  best matches the panelists' free-sorting groups.

Outputs written:

- `outputs/erdbeere_v3_S1_target_recipes_mds.png`
- `outputs/erdbeere_v3_S2_ingredients_mds.png`
- `outputs/erdbeere_v3_S3_hybrid_mds.png`
- `outputs/erdbeere_v3_S4_target_recipes_median_mds.png`
- `outputs/erdbeere_v3_S5_ingredients_median_mds.png`
- `outputs/erdbeere_v3_S6_contrast_mds.png`
- `outputs/erdbeere_v3_direct_methods_mds.png`
- `outputs/erdbeere_v3_strategy_comparison_overview.png`
- `outputs/cluster_assignments_expert_seeded_all_strategies.xlsx`
  (Strategy_Comparison + 6 strategy + 3 direct-method + 2 agreement sheets)
"""))

# ---------------------------------------------------------------------------
# 15. Next strategies to try
# ---------------------------------------------------------------------------
CELLS.append(md(
    """## 15. Next Strategies to Try

Not yet implemented — candidate directions for the next iteration, ordered
roughly by expected payoff / effort.

| # | Idea | Why it might help |
|---|------|-------------------|
| 1 | **Direct ingredient-profile centroid** | Average the OT1 vectors of the expert *ingredients themselves* (not the recipes containing them), giving a pure prototype uncontaminated by everything else in those recipes. |
| 2 | **Medoid centroid** | Use the real target recipe closest to the group mean as the centroid — stays on the data manifold; robust when a cluster has only 2–3 targets and one is an outlier. |
| 3 | **Dose-weighted mean** | Weight each seed recipe by how much of the signature ingredient it contains, so a recipe at 8 % counts more than a trace. |
| 4 | **Constrained k-means (COP/PCKMeans)** | Must-link among same-cluster targets, cannot-link across clusters — cluster freely but never violate expert judgments. |
| 5 | **Few-shot classifier** | With fixed named clusters + labeled targets this is really classification: kNN / nearest-shrunken-centroid / logistic regression on the labeled targets. |
| 6 | **Supervised metric learning / LDA** | Learn a transform of OT1 space (LDA, NCA, LMNN) that pulls same-cluster targets together, then run the 10 algorithms in the learned space. Fixes the root cause if expert clusters aren't separable under plain cosine. |
| 7 | **Seeded soft GMM** | Freeze component means at expert centroids and return soft memberships ("60 % Grün, 40 % Frucht-Süß") — closer to how panelists perceive borderline recipes. |
| 8 | **Active-disagreement report** | Rank recipes by how much the strategies disagree and hand that shortlist back to the experts — cheapest way to get more labels exactly where they matter. |
"""))


# ---------------------------------------------------------------------------
# Assemble JSON
# ---------------------------------------------------------------------------
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, indent=1))
print(f'Wrote {NB_PATH}  ({len(CELLS)} cells)')
