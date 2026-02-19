"""
Recipe Clustering V2 – Multi-Feature Weighted Profiles
======================================================

Improvements over V1:
- Strips defined solvents/carriers (ignore_substances.csv) before clustering
- Re-normalises remaining ingredients to aliquots (Totalmenge sums to 1 per recipe)
- Feature vector now combines:
    * Sensorik_1..4         positional weight (pos 1 = 1.0, pos 4 = 0.25)
    * Odour Type 1..3       positional weight (pos 1 = 1.0, pos 3 = 0.33)
    * Totalmenge            scales each ingredient's contribution by its share
    * Threshold ppm         scales by odour potency (1 / threshold)
  → combined term weight = positional_weight × totalmenge × (1 / threshold)

  The positional_weight is the same base multiplier for both Sensorik and
  Odour Type columns — what differs is the n_cols used in the formula:
      pos_weight(position, n_cols) = (n_cols + 1 - position) / n_cols
  Sensorik (n_cols=4): pos 1→1.00, pos 2→0.75, pos 3→0.50, pos 4→0.25
  Odour Type (n_cols=3): pos 1→1.00, pos 2→0.67, pos 3→0.33

  Sensorik and Odour Type terms share the same vocabulary (feature space).
  If the same term appears in both a Sensorik column and an Odour Type column
  for the same ingredient, both positional weights are added to the same
  vector dimension — so overlap reinforces that term's importance.
- Rows with Totalmenge = 0 are excluded from vector building
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# Weight helpers
# ──────────────────────────────────────────────────────────────────────────────

def positional_weight(position: int, n_cols: int) -> float:
    """
    Descending importance weight.
    position 1 → 1.0,  position n_cols → 1/n_cols
    Formula: (n_cols + 1 - position) / n_cols
    """
    return (n_cols + 1 - position) / n_cols


def threshold_factor(threshold_ppm, fallback: float = 1.0) -> float:
    """
    Odour-potency factor: 1 / threshold_ppm.
    Lower threshold → substance is detectable at tiny concentrations → higher impact.
    NaN / non-positive values → fallback (neutral, no adjustment).
    """
    try:
        t = float(threshold_ppm)
        if np.isnan(t) or t <= 0:
            return fallback
        return 1.0 / t
    except (TypeError, ValueError):
        return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(data_path: str, ignore_path: str) -> pd.DataFrame:
    """
    1. Load Versuchsdaten_2.csv.
    2. Set Totalmenge = 0 for all substances listed in ignore_substances.csv.
    3. Re-normalise Totalmenge per recipe to aliquots (sum → 1).

    Returns the preprocessed DataFrame.
    """
    df = pd.read_csv(data_path)
    ign = pd.read_csv(ignore_path)

    # Clean up ignore list (match on Name column)
    ignore_names = set(ign['Name'].dropna().str.strip())

    mask = df['Name'].str.strip().isin(ignore_names)
    n_zeroed = mask.sum()
    df.loc[mask, 'Totalmenge'] = 0.0

    # Per-recipe aliquot normalisation (vectorised)
    per_total = df.groupby('Rez.-Nr.')['Totalmenge'].transform('sum')
    df['Totalmenge'] = np.where(per_total > 0, df['Totalmenge'] / per_total, df['Totalmenge'])

    print(f"[preprocess] {len(df)} rows loaded, "
          f"{n_zeroed} ingredient rows zeroed out (ignored substances), "
          f"{(df['Totalmenge'] > 0).sum()} active rows remaining")
    print(f"[preprocess] {df['Rez.-Nr.'].nunique()} unique recipes")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary & vector extraction
# ──────────────────────────────────────────────────────────────────────────────

SENSORIK_COLS = ['Sensorik_1', 'Sensorik_2', 'Sensorik_3', 'Sensorik_4']
ODOUR_COLS = [
    'Odour Type 1 FlavourWheel',
    'Odour Type 2 Flavour Wheel',
    'Odour Type 3 Flavour Wheel',
]


def _norm_term(term) -> Optional[str]:
    if pd.isna(term) or not isinstance(term, str):
        return None
    t = term.lower().strip().replace('"', '').replace("'", '').rstrip('.,;:')
    return t if len(t) >= 2 else None


def build_vocabulary(df: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """Build combined vocabulary from Sensorik_1..4 + Odour Type 1..3."""
    all_terms: set = set()
    for col in SENSORIK_COLS + ODOUR_COLS:
        if col in df.columns:
            for t in df[col].dropna().map(_norm_term):
                if t:
                    all_terms.add(t)
    vocabulary = sorted(all_terms)
    vocab_to_idx = {t: i for i, t in enumerate(vocabulary)}
    print(f"[vocabulary] {len(vocabulary)} unique terms "
          f"(sensorik + odour-type combined)")
    return vocabulary, vocab_to_idx


def extract_recipe_vectors(
    df: pd.DataFrame,
    vocabulary: List[str],
    vocab_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build one L2-normalised feature vector per recipe.

    Term contribution = positional_weight × totalmenge × (1 / threshold)

    Only rows with Totalmenge > 0 contribute (ignored substances are skipped).
    """
    recipes = df['Rez.-Nr.'].unique().tolist()
    n_recipes = len(recipes)
    n_feat = len(vocabulary)

    # Pre-compute per-row threshold factors (cache for speed)
    thresh_factors = (
        df['Threshold ppm (Datenbank)']
        .apply(threshold_factor)
        .values
    )

    vectors = np.zeros((n_recipes, n_feat), dtype=np.float64)

    for r_idx, recipe in enumerate(recipes):
        rows = df[df['Rez.-Nr.'] == recipe]
        row_indices = rows.index.tolist()

        for i, row_i in enumerate(row_indices):
            row = df.loc[row_i]
            qty = float(row['Totalmenge'])
            if qty <= 0:
                continue  # ignored substance

            t_factor = thresh_factors[df.index.get_loc(row_i)]
            ingredient_weight = qty * t_factor

            # ── Sensorik columns (4 columns) ──────────────────────────────
            n_s = len(SENSORIK_COLS)
            for pos, col in enumerate(SENSORIK_COLS, start=1):
                if col not in df.columns:
                    continue
                term = _norm_term(row.get(col))
                if term and term in vocab_to_idx:
                    pw = positional_weight(pos, n_s)
                    vectors[r_idx, vocab_to_idx[term]] += pw * ingredient_weight

            # ── Odour type columns (3 columns) ────────────────────────────
            n_o = len(ODOUR_COLS)
            for pos, col in enumerate(ODOUR_COLS, start=1):
                if col not in df.columns:
                    continue
                term = _norm_term(row.get(col))
                if term and term in vocab_to_idx:
                    pw = positional_weight(pos, n_o)
                    vectors[r_idx, vocab_to_idx[term]] += pw * ingredient_weight

    # L2 normalise
    recipe_vectors = normalize(vectors)

    print(f"[vectors] {n_recipes} recipes × {n_feat} features extracted")
    return recipe_vectors, recipes


# ──────────────────────────────────────────────────────────────────────────────
# Cluster naming / details
# ──────────────────────────────────────────────────────────────────────────────

def generate_cluster_names(
    cluster_labels: np.ndarray,
    recipe_vectors: np.ndarray,
    vocabulary: List[str],
    top_n: int = 3,
) -> Dict[int, str]:
    """Generate descriptive names per cluster based on distinctive terms."""
    unique_labels = sorted(set(cluster_labels))
    global_centroid = recipe_vectors.mean(axis=0)

    centroids: Dict[int, np.ndarray] = {}
    for label in unique_labels:
        if label == -1:
            continue
        mask = cluster_labels == label
        centroids[label] = recipe_vectors[mask].mean(axis=0)

    names: Dict[int, str] = {}
    for label in unique_labels:
        if label == -1:
            names[label] = "Outliers"
            continue
        centroid = centroids[label]
        distinctiveness = centroid - global_centroid * 0.8
        top_idx = np.argsort(distinctiveness)[-6:][::-1]
        terms = []
        for idx in top_idx:
            if distinctiveness[idx] > 0 and centroid[idx] > 0.05:
                terms.append(vocabulary[idx].capitalize())
            if len(terms) >= top_n:
                break
        if len(terms) < 2:
            top_idx = np.argsort(centroid)[-top_n:][::-1]
            terms = [vocabulary[i].capitalize() for i in top_idx]
        names[label] = "-".join(terms[:top_n])

    return names


def get_cluster_details(
    cluster_labels: np.ndarray,
    recipe_vectors: np.ndarray,
    vocabulary: List[str],
    recipes: List[str],
    cluster_names: Dict[int, str],
) -> Dict[int, Dict]:
    details = {}
    for label in sorted(set(cluster_labels)):
        mask = cluster_labels == label
        cluster_recipes = [recipes[i] for i, m in enumerate(mask) if m]
        cluster_vecs = recipe_vectors[mask]
        centroid = cluster_vecs.mean(axis=0)
        top_idx = np.argsort(centroid)[-10:][::-1]
        top_terms = [(vocabulary[i], float(centroid[i])) for i in top_idx]
        details[label] = {
            'name': cluster_names.get(label, f"Cluster {label}"),
            'recipes': cluster_recipes,
            'centroid': centroid,
            'top_terms': top_terms,
            'size': len(cluster_recipes),
        }
    return details


def print_cluster_summary(details: Dict[int, Dict]) -> None:
    for label in sorted(details):
        info = details[label]
        print(f"\n{'─'*50}")
        print(f"CLUSTER {label}: {info['name']}")
        print(f"{'─'*50}")
        print(f"Recipes ({info['size']}):")
        for r in info['recipes']:
            print(f"  • {r}")
        print("Top profile:")
        for term, w in info['top_terms'][:7]:
            bar = "█" * int(w * 40)
            print(f"  {term:18} {bar} ({w:.3f})")


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tsne_coords(recipe_vectors: np.ndarray) -> np.ndarray:
    """Reduce to 2D via PCA → t-SNE."""
    n_samples = recipe_vectors.shape[0]
    data = recipe_vectors
    if data.shape[1] > 50:
        pca = PCA(n_components=min(30, n_samples - 1))
        data = pca.fit_transform(data)
    perplexity = min(5, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, max_iter=1000)
    return tsne.fit_transform(data)


def visualize_clusters(
    recipe_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str],
    recipes: List[str],
    details: Dict[int, Dict],
    title: str = "Recipe Clusters",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 8),
) -> plt.Figure:
    coords = _tsne_coords(recipe_vectors)
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_labels), 8)))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Left: t-SNE scatter ──────────────────────────────────────────────────
    ax1 = axes[0]
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        pts = coords[mask]
        color = 'gray' if label == -1 else colors[i % len(colors)]
        marker = 'x' if label == -1 else 'o'
        alpha = 0.5 if label == -1 else 0.85
        ax1.scatter(pts[:, 0], pts[:, 1], c=[color], marker=marker,
                    s=160, alpha=alpha,
                    label=cluster_names.get(label, f"C{label}"),
                    edgecolors='black', linewidths=0.5)

    for i, recipe in enumerate(recipes):
        ax1.annotate(recipe[:12], (coords[i, 0], coords[i, 1]),
                     fontsize=6.5, alpha=0.7, ha='center', va='bottom')

    ax1.set_xlabel("t-SNE Dim 1", fontsize=10)
    ax1.set_ylabel("t-SNE Dim 2", fontsize=10)
    ax1.set_title(f"t-SNE Cluster Map\n{title}", fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=7.5, framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    # ── Right: top terms per cluster (bar chart) ─────────────────────────────
    ax2 = axes[1]
    y_pos, y_lab, bar_vals, bar_cols = [], [], [], []
    y = 0
    for label in unique_labels:
        if label == -1:
            continue
        info = details[label]
        color = colors[label % len(colors)]
        for term, w in info['top_terms'][:5]:
            y_pos.append(y)
            y_lab.append(term)
            bar_vals.append(w)
            bar_cols.append(color)
            y += 1
        y += 0.6

    ax2.barh(y_pos, bar_vals, color=bar_cols, alpha=0.82)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(y_lab, fontsize=8)
    ax2.set_xlabel("Weighted Importance", fontsize=10)
    ax2.set_title("Top Terms per Cluster", fontsize=11, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.25)

    # Cluster name annotations
    y = 0
    for label in unique_labels:
        if label == -1:
            continue
        info = details[label]
        ax2.annotate(
            f"C{label}: {info['name']} ({info['size']}r)",
            xy=(ax2.get_xlim()[1] * 0.55, y + 2),
            fontsize=7.5, fontweight='bold',
            color=colors[label % len(colors)],
        )
        y += 5 + 0.6

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


def visualize_radar(
    details: Dict[int, Dict],
    title: str = "Cluster Sensorik Profiles",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Optional[plt.Figure]:
    all_terms: set = set()
    for label, info in details.items():
        if label == -1:
            continue
        all_terms.update([t[0] for t in info['top_terms'][:8]])
    all_terms_list = sorted(all_terms)[:12]
    n_terms = len(all_terms_list)
    n_clusters = sum(1 for l in details if l != -1)
    if n_clusters == 0 or n_terms == 0:
        return None

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n_terms, endpoint=False).tolist()
    angles += angles[:1]
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 8)))

    for i, (label, info) in enumerate(details.items()):
        if label == -1:
            continue
        tw = {t: w for t, w in info['top_terms']}
        vals = [tw.get(t, 0) for t in all_terms_list] + [tw.get(all_terms_list[0], 0)]
        ax.plot(angles, vals, 'o-', linewidth=2,
                label=info['name'], color=colors[i % len(colors)])
        ax.fill(angles, vals, alpha=0.12, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_terms_list, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


def visualize_faiss_similarity(
    recipe_vectors: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str],
    faiss_index,
    cluster_centroids: np.ndarray,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (17, 6),
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    n = len(recipe_vectors)
    vectors = recipe_vectors.astype('float32')

    # 1. Similarity heatmap
    distances, _ = faiss_index.search(vectors, n)
    similarity = 1 / (1 + distances)
    im1 = axes[0].imshow(similarity, cmap='YlOrRd', aspect='auto')
    axes[0].set_title("Recipe Similarity Matrix\n(FAISS L2)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Recipe index")
    axes[0].set_ylabel("Recipe index")
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label="Similarity")

    # 2. Cluster sizes
    unique_labels = sorted(set(cluster_labels))
    sizes = [list(cluster_labels).count(l) for l in unique_labels]
    names = [cluster_names.get(l, f"C{l}")[:22] for l in unique_labels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    bars = axes[1].barh(range(len(unique_labels)), sizes, color=colors)
    axes[1].set_yticks(range(len(unique_labels)))
    axes[1].set_yticklabels(names, fontsize=8)
    axes[1].set_xlabel("Number of Recipes")
    axes[1].set_title("Cluster Sizes\n(FAISS k-means)", fontsize=11, fontweight='bold')
    axes[1].invert_yaxis()
    for bar, sz in zip(bars, sizes):
        axes[1].text(bar.get_width() + 0.1,
                     bar.get_y() + bar.get_height() / 2,
                     str(sz), va='center', fontsize=9)

    # 3. Centroid distances
    if cluster_centroids is not None:
        k = len(cluster_centroids)
        dist_mat = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                dist_mat[i, j] = np.linalg.norm(
                    cluster_centroids[i] - cluster_centroids[j])
        im3 = axes[2].imshow(dist_mat, cmap='Blues', aspect='auto')
        axes[2].set_title("Inter-Cluster Distances\n(Centroid L2)", fontsize=11, fontweight='bold')
        axes[2].set_xlabel("Cluster")
        axes[2].set_ylabel("Cluster")
        axes[2].set_xticks(range(k))
        axes[2].set_yticks(range(k))
        plt.colorbar(im3, ax=axes[2], shrink=0.8, label="Distance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {save_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main clusterer class
# ──────────────────────────────────────────────────────────────────────────────

class RecipeClustererV2:
    """
    Full clustering pipeline for Versuchsdaten_2.csv with:
    - Solvent/carrier removal
    - Aliquot normalisation
    - Combined sensorik + odour-type feature vectors weighted by Totalmenge and Threshold
    """

    DATA_PATH = 'data/gold/Versuchsdaten_2.csv'
    IGNORE_PATH = 'data/gold/ignone_substances.csv'

    def __init__(self, data_path: str = DATA_PATH, ignore_path: str = IGNORE_PATH):
        self.data_path = data_path
        self.ignore_path = ignore_path
        self.df: Optional[pd.DataFrame] = None
        self.recipe_vectors: Optional[np.ndarray] = None
        self.vocabulary: Optional[List[str]] = None
        self.vocab_to_idx: Optional[Dict[str, int]] = None
        self.recipes: Optional[List[str]] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_names: Optional[Dict[int, str]] = None
        self.faiss_index = None
        self.cluster_centroids: Optional[np.ndarray] = None

    # ── data prep ────────────────────────────────────────────────────────────

    def load_and_preprocess(self) -> pd.DataFrame:
        self.df = preprocess(self.data_path, self.ignore_path)
        return self.df

    def build_vocabulary(self) -> List[str]:
        self.vocabulary, self.vocab_to_idx = build_vocabulary(self.df)
        return self.vocabulary

    def extract_vectors(self) -> Tuple[np.ndarray, List[str]]:
        self.recipe_vectors, self.recipes = extract_recipe_vectors(
            self.df, self.vocabulary, self.vocab_to_idx)
        return self.recipe_vectors, self.recipes

    # ── clustering ───────────────────────────────────────────────────────────

    def cluster_hdbscan(self, min_cluster_size: int = 2, min_samples: int = 1) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        self.cluster_labels = clusterer.fit_predict(self.recipe_vectors)
        n_c = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        print(f"HDBSCAN → {n_c} clusters, {n_noise} outliers")
        return self.cluster_labels

    def cluster_agglomerative(self, k_range: tuple = (3, 12)) -> np.ndarray:
        best_k, best_score, scores = k_range[0], -1, []
        for k in range(k_range[0], min(k_range[1] + 1, len(self.recipes))):
            ac = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
            labels = ac.fit_predict(self.recipe_vectors)
            score = silhouette_score(self.recipe_vectors, labels)
            scores.append((k, score))
            if score > best_score:
                best_score = score
                best_k = k
        print(f"Agglomerative silhouette scores: {[(k, f'{s:.3f}') for k, s in scores]}")
        print(f"Optimal k={best_k} (score={best_score:.3f})")
        ac = AgglomerativeClustering(n_clusters=best_k, metric='euclidean', linkage='ward')
        self.cluster_labels = ac.fit_predict(self.recipe_vectors)
        return self.cluster_labels

    def cluster_faiss(self, k_range: tuple = (3, 12), niter: int = 50) -> np.ndarray:
        if not FAISS_AVAILABLE:
            raise ImportError("Run: pip install faiss-cpu")
        vectors = np.ascontiguousarray(self.recipe_vectors.astype('float32'))
        n, d = vectors.shape
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(vectors)

        best_k, best_score = k_range[0], -1
        best_labels, best_centroids = None, None
        scores = []

        for k in range(k_range[0], min(k_range[1] + 1, n)):
            km = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=42)
            km.train(vectors)
            _, labels = km.index.search(vectors, 1)
            labels = labels.flatten()
            if len(set(labels)) > 1:
                score = silhouette_score(vectors, labels)
                scores.append((k, score))
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels.copy()
                    best_centroids = km.centroids.copy()

        print(f"FAISS silhouette scores: {[(k, f'{s:.3f}') for k, s in scores]}")
        print(f"Optimal k={best_k} (score={best_score:.3f})")
        self.cluster_labels = best_labels
        self.cluster_centroids = best_centroids
        self.centroid_index = faiss.IndexFlatL2(d)
        self.centroid_index.add(best_centroids)
        return self.cluster_labels

    # ── naming & details ─────────────────────────────────────────────────────

    def make_names(self) -> Dict[int, str]:
        self.cluster_names = generate_cluster_names(
            self.cluster_labels, self.recipe_vectors, self.vocabulary)
        print("\nCluster names:")
        for label, name in sorted(self.cluster_names.items()):
            count = list(self.cluster_labels).count(label)
            print(f"  {label:3d}  {name}  ({count} recipes)")
        return self.cluster_names

    def details(self) -> Dict[int, Dict]:
        return get_cluster_details(
            self.cluster_labels, self.recipe_vectors,
            self.vocabulary, self.recipes, self.cluster_names)

    # ── similarity search (FAISS only) ───────────────────────────────────────

    def find_similar(self, recipe_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
        query = self.recipe_vectors[recipe_idx:recipe_idx + 1].astype('float32')
        dists, idxs = self.faiss_index.search(query, top_k + 1)
        return [(self.recipes[i], float(d))
                for d, i in zip(dists[0], idxs[0]) if i != recipe_idx][:top_k]

    # ── full pipelines ────────────────────────────────────────────────────────

    def _common_prep(self):
        print("\n[1/5] Loading & preprocessing data...")
        self.load_and_preprocess()
        print("\n[2/5] Building vocabulary...")
        self.build_vocabulary()
        print("\n[3/5] Extracting weighted recipe vectors...")
        self.extract_vectors()

    def run_hdbscan_pipeline(self, min_cluster_size: int = 2,
                             save_plots: bool = True,
                             output_dir: str = ".") -> Dict:
        print("=" * 60)
        print("PIPELINE: HDBSCAN")
        print("=" * 60)
        self._common_prep()
        print("\n[4/5] Clustering (HDBSCAN)...")
        self.cluster_hdbscan(min_cluster_size=min_cluster_size)
        print("\n[5/5] Naming clusters...")
        self.make_names()
        d = self.details()
        print_cluster_summary(d)

        if save_plots:
            visualize_clusters(
                self.recipe_vectors, self.cluster_labels,
                self.cluster_names, self.recipes, d,
                title="HDBSCAN",
                save_path=f"{output_dir}/v2_cluster_hdbscan.png")
            visualize_radar(d, title="HDBSCAN Cluster Profiles",
                            save_path=f"{output_dir}/v2_cluster_hdbscan_profiles.png")
        return d

    def run_agglomerative_pipeline(self, k_range: tuple = (3, 12),
                                   save_plots: bool = True,
                                   output_dir: str = ".") -> Dict:
        print("=" * 60)
        print("PIPELINE: AGGLOMERATIVE")
        print("=" * 60)
        self._common_prep()
        print("\n[4/5] Clustering (Agglomerative, auto-k)...")
        self.cluster_agglomerative(k_range=k_range)
        print("\n[5/5] Naming clusters...")
        self.make_names()
        d = self.details()
        print_cluster_summary(d)

        if save_plots:
            visualize_clusters(
                self.recipe_vectors, self.cluster_labels,
                self.cluster_names, self.recipes, d,
                title="Agglomerative",
                save_path=f"{output_dir}/v2_cluster_agglomerative.png")
            visualize_radar(d, title="Agglomerative Cluster Profiles",
                            save_path=f"{output_dir}/v2_cluster_agglomerative_profiles.png")
        return d

    def run_faiss_pipeline(self, k_range: tuple = (3, 12),
                           save_plots: bool = True,
                           output_dir: str = ".") -> Dict:
        print("=" * 60)
        print("PIPELINE: FAISS")
        print("=" * 60)
        self._common_prep()
        print("\n[4/5] Clustering (FAISS k-means)...")
        self.cluster_faiss(k_range=k_range)
        print("\n[5/5] Naming clusters...")
        self.make_names()
        d = self.details()
        print_cluster_summary(d)

        if save_plots:
            visualize_clusters(
                self.recipe_vectors, self.cluster_labels,
                self.cluster_names, self.recipes, d,
                title="FAISS",
                save_path=f"{output_dir}/v2_cluster_faiss.png")
            visualize_radar(d, title="FAISS Cluster Profiles",
                            save_path=f"{output_dir}/v2_cluster_faiss_profiles.png")
            visualize_faiss_similarity(
                self.recipe_vectors, self.cluster_labels,
                self.cluster_names, self.faiss_index, self.cluster_centroids,
                save_path=f"{output_dir}/v2_cluster_faiss_similarity.png")
        return d


if __name__ == "__main__":
    c = RecipeClustererV2(
        data_path="data/gold/Versuchsdaten_2.csv",
        ignore_path="data/gold/ignone_substances.csv",
    )
    c.run_hdbscan_pipeline(save_plots=True, output_dir="outputs")
    plt.show()
