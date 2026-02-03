"""
Recipe Clustering based on Weighted Sensorik Profiles

This module clusters recipes using their sensorik descriptors with positional
importance weighting (Sensorik_1 is most important, Sensorik_16 is least).
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
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RecipeClusterer:
    """Cluster recipes based on weighted sensorik profiles."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.recipe_vectors = None
        self.vocabulary = None
        self.vocab_to_idx = None
        self.recipes = None
        self.cluster_labels = None
        self.cluster_names = None
        self.clusterer = None
        self.faiss_index = None
        self.cluster_centroids = None

    def load_data(self) -> pd.DataFrame:
        """Load the CSV data."""
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} rows")
        return self.df

    def _calculate_weight(self, position: int) -> float:
        """
        Calculate importance weight based on sensorik position.
        Position 1 (Sensorik_1) = highest importance = 1.0
        Position 16 (Sensorik_16) = lowest importance = 0.0625
        """
        return (17 - position) / 16

    def _normalize_term(self, term: str) -> str:
        """Normalize a sensorik term for consistency."""
        if pd.isna(term) or not isinstance(term, str):
            return None
        term = term.lower().strip()
        # Remove quotes and special characters
        term = term.replace('"', '').replace("'", "")
        # Remove trailing punctuation
        term = term.rstrip('.,;:')
        if len(term) < 2:
            return None
        return term

    def build_vocabulary(self) -> List[str]:
        """Build vocabulary from all sensorik columns."""
        sensorik_cols = [f'Sensorik_{i}' for i in range(1, 17)]
        all_terms = set()

        for col in sensorik_cols:
            if col in self.df.columns:
                terms = self.df[col].dropna().apply(self._normalize_term)
                all_terms.update([t for t in terms if t])

        self.vocabulary = sorted(all_terms)
        self.vocab_to_idx = {term: idx for idx, term in enumerate(self.vocabulary)}
        print(f"Built vocabulary with {len(self.vocabulary)} unique terms")
        return self.vocabulary

    def extract_recipe_vectors(self) -> Tuple[np.ndarray, List[str]]:
        """
        Extract weighted feature vectors for each recipe.

        Each recipe is represented by a vector where each dimension corresponds
        to a sensorik term, and the value is the sum of position-weighted occurrences
        across all ingredients in that recipe.
        """
        recipe_col = 'Rez.-Nr.'
        sensorik_cols = [f'Sensorik_{i}' for i in range(1, 17)]

        # Group by recipe
        self.recipes = self.df[recipe_col].unique().tolist()
        n_recipes = len(self.recipes)
        n_features = len(self.vocabulary)

        # Initialize recipe vectors
        vectors = np.zeros((n_recipes, n_features))

        for recipe_idx, recipe in enumerate(self.recipes):
            recipe_data = self.df[self.df[recipe_col] == recipe]

            # Aggregate weighted sensorik terms
            for _, row in recipe_data.iterrows():
                for position, col in enumerate(sensorik_cols, start=1):
                    if col in self.df.columns:
                        term = self._normalize_term(row.get(col))
                        if term and term in self.vocab_to_idx:
                            weight = self._calculate_weight(position)
                            vectors[recipe_idx, self.vocab_to_idx[term]] += weight

        # L2 normalize vectors for better clustering
        self.recipe_vectors = normalize(vectors)

        print(f"Extracted vectors for {n_recipes} recipes")
        print(f"Vector dimensionality: {n_features}")

        return self.recipe_vectors, self.recipes

    def cluster_recipes(self, min_cluster_size: int = 2, min_samples: int = 1) -> np.ndarray:
        """
        Cluster recipes using HDBSCAN for automatic cluster detection.

        Args:
            min_cluster_size: Minimum number of recipes to form a cluster
            min_samples: How conservative clustering should be

        Returns:
            Array of cluster labels
        """
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )

        self.cluster_labels = self.clusterer.fit_predict(self.recipe_vectors)

        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)

        print(f"\nClustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Noise points (outliers): {n_noise}")

        return self.cluster_labels

    def cluster_recipes_agglomerative(
        self,
        k_range: tuple = (3, 10)
    ) -> np.ndarray:
        """
        Cluster recipes using Agglomerative Clustering.
        Automatically finds optimal k using silhouette score.
        GUARANTEES all recipes get assigned to a cluster (no outliers).

        Args:
            k_range: Range of k values to try (min, max)

        Returns:
            Array of cluster labels
        """
        print("\nFinding optimal number of clusters...")

        best_k = k_range[0]
        best_score = -1
        scores = []

        # Try different values of k
        for k in range(k_range[0], min(k_range[1] + 1, len(self.recipes))):
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric='euclidean',
                linkage='ward'
            )
            labels = clusterer.fit_predict(self.recipe_vectors)
            score = silhouette_score(self.recipe_vectors, labels)
            scores.append((k, score))

            if score > best_score:
                best_score = score
                best_k = k

        print(f"  Silhouette scores: {[(k, f'{s:.3f}') for k, s in scores]}")
        print(f"  Optimal k: {best_k} (silhouette: {best_score:.3f})")

        # Fit with optimal k
        self.clusterer = AgglomerativeClustering(
            n_clusters=best_k,
            metric='euclidean',
            linkage='ward'
        )
        self.cluster_labels = self.clusterer.fit_predict(self.recipe_vectors)

        n_clusters = len(set(self.cluster_labels))
        print(f"\nClustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  All {len(self.recipes)} recipes assigned (no outliers)")

        return self.cluster_labels

    def cluster_recipes_faiss(
        self,
        k_range: tuple = (3, 12),
        niter: int = 50
    ) -> np.ndarray:
        """
        Cluster recipes using FAISS vector database with k-means.

        FAISS provides:
        - Efficient similarity search
        - GPU acceleration (if available)
        - Production-ready vector indexing

        The weighted importance is preserved in the vectors themselves.

        Args:
            k_range: Range of k values to try (min, max)
            niter: Number of k-means iterations

        Returns:
            Array of cluster labels
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        print("\nBuilding FAISS index and finding optimal clusters...")

        # Prepare vectors for FAISS (needs float32, contiguous)
        vectors = np.ascontiguousarray(self.recipe_vectors.astype('float32'))
        n_samples, d = vectors.shape

        # Build FAISS index for similarity search
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(vectors)
        print(f"  FAISS index built: {self.faiss_index.ntotal} vectors, {d} dimensions")

        # Find optimal k using silhouette score
        best_k = k_range[0]
        best_score = -1
        best_centroids = None
        best_labels = None
        scores = []

        for k in range(k_range[0], min(k_range[1] + 1, n_samples)):
            # FAISS k-means clustering
            kmeans = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=42)
            kmeans.train(vectors)

            # Get cluster assignments
            _, labels = kmeans.index.search(vectors, 1)
            labels = labels.flatten()

            # Calculate silhouette score
            if len(set(labels)) > 1:
                score = silhouette_score(vectors, labels)
                scores.append((k, score))

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_centroids = kmeans.centroids.copy()
                    best_labels = labels.copy()

        print(f"  Silhouette scores: {[(k, f'{s:.3f}') for k, s in scores]}")
        print(f"  Optimal k: {best_k} (silhouette: {best_score:.3f})")

        self.cluster_labels = best_labels
        self.cluster_centroids = best_centroids

        # Build index with centroids for similarity search
        self.centroid_index = faiss.IndexFlatL2(d)
        self.centroid_index.add(best_centroids)

        n_clusters = len(set(self.cluster_labels))
        print(f"\nClustering Results:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  All {len(self.recipes)} recipes assigned (no outliers)")

        return self.cluster_labels

    def find_similar_recipes(self, recipe_idx: int, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find most similar recipes using FAISS index.

        Args:
            recipe_idx: Index of the query recipe
            top_k: Number of similar recipes to return

        Returns:
            List of (recipe_name, distance) tuples
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Run cluster_recipes_faiss first.")

        query = self.recipe_vectors[recipe_idx:recipe_idx+1].astype('float32')
        distances, indices = self.faiss_index.search(query, top_k + 1)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != recipe_idx:  # Exclude the query itself
                results.append((self.recipes[idx], float(dist)))

        return results[:top_k]

    def find_recipes_by_profile(
        self,
        target_profile: Dict[str, float],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find recipes most similar to a target sensorik profile.

        Args:
            target_profile: Dict of {sensorik_term: weight}
            top_k: Number of recipes to return

        Returns:
            List of (recipe_name, distance) tuples
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Run cluster_recipes_faiss first.")

        # Build query vector from profile
        query = np.zeros((1, len(self.vocabulary)), dtype='float32')
        for term, weight in target_profile.items():
            term_normalized = term.lower().strip()
            if term_normalized in self.vocab_to_idx:
                query[0, self.vocab_to_idx[term_normalized]] = weight

        # Normalize query
        query = normalize(query).astype('float32')

        # Search
        distances, indices = self.faiss_index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append((self.recipes[idx], float(dist)))

        return results

    def create_new_recipe_profile_faiss(
        self,
        cluster_weights: Dict[int, float]
    ) -> Tuple[np.ndarray, List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Create a new recipe profile by weighted averaging of FAISS cluster centroids.
        Also returns similar existing recipes.

        Args:
            cluster_weights: Dict mapping cluster label to weight

        Returns:
            Tuple of (profile vector, top sensorik terms, similar existing recipes)
        """
        if self.cluster_centroids is None:
            raise ValueError("FAISS centroids not available. Run cluster_recipes_faiss first.")

        # Normalize weights
        total_weight = sum(cluster_weights.values())
        normalized_weights = {k: v/total_weight for k, v in cluster_weights.items()}

        # Calculate weighted average of centroids
        new_profile = np.zeros(len(self.vocabulary), dtype='float32')

        for label, weight in normalized_weights.items():
            new_profile += weight * self.cluster_centroids[label]

        # Normalize
        new_profile = normalize(new_profile.reshape(1, -1))[0]

        # Get top terms
        top_indices = np.argsort(new_profile)[-15:][::-1]
        top_terms = [(self.vocabulary[i], float(new_profile[i])) for i in top_indices]

        # Find similar existing recipes
        query = new_profile.reshape(1, -1).astype('float32')
        distances, indices = self.faiss_index.search(query, 5)

        similar_recipes = [
            (self.recipes[idx], float(dist))
            for dist, idx in zip(distances[0], indices[0])
        ]

        return new_profile, top_terms, similar_recipes

    def generate_cluster_names(self, top_n: int = 3) -> Dict[int, str]:
        """
        Generate human-readable names for each cluster based on
        DISTINCTIVE sensorik terms (not just top terms, which may be common to all).
        """
        self.cluster_names = {}
        unique_labels = sorted(set(self.cluster_labels))

        # First, calculate global average to find common terms
        global_centroid = self.recipe_vectors.mean(axis=0)

        # Calculate centroids for each cluster
        cluster_centroids = {}
        for label in unique_labels:
            if label == -1:
                continue
            cluster_mask = self.cluster_labels == label
            cluster_vectors = self.recipe_vectors[cluster_mask]
            cluster_centroids[label] = cluster_vectors.mean(axis=0)

        # For each cluster, find terms that are distinctive (higher than average)
        for label in unique_labels:
            if label == -1:
                self.cluster_names[label] = "Outliers"
                continue

            centroid = cluster_centroids[label]

            # Calculate distinctiveness: how much above global average
            distinctiveness = centroid - global_centroid * 0.8  # Allow some baseline

            # Get top distinctive terms
            top_indices = np.argsort(distinctiveness)[-6:][::-1]

            # Filter to only terms with positive distinctiveness and decent weight
            distinctive_terms = []
            for idx in top_indices:
                if distinctiveness[idx] > 0 and centroid[idx] > 0.05:
                    distinctive_terms.append(self.vocabulary[idx].capitalize())
                if len(distinctive_terms) >= top_n:
                    break

            # Fallback to top terms if no distinctive ones found
            if len(distinctive_terms) < 2:
                top_indices = np.argsort(centroid)[-top_n:][::-1]
                distinctive_terms = [self.vocabulary[i].capitalize() for i in top_indices]

            # Create name
            name = "-".join(distinctive_terms[:top_n])
            self.cluster_names[label] = name

        print("\nCluster Names:")
        for label, name in sorted(self.cluster_names.items()):
            count = list(self.cluster_labels).count(label)
            print(f"  Cluster {label}: {name} ({count} recipes)")

        return self.cluster_names

    def get_cluster_details(self) -> Dict[int, Dict]:
        """Get detailed information about each cluster."""
        details = {}

        for label in sorted(set(self.cluster_labels)):
            cluster_mask = self.cluster_labels == label
            cluster_recipes = [self.recipes[i] for i, m in enumerate(cluster_mask) if m]
            cluster_vectors = self.recipe_vectors[cluster_mask]

            # Calculate centroid
            centroid = cluster_vectors.mean(axis=0)

            # Get top 10 terms
            top_indices = np.argsort(centroid)[-10:][::-1]
            top_terms = [(self.vocabulary[i], centroid[i]) for i in top_indices]

            details[label] = {
                'name': self.cluster_names.get(label, f"Cluster {label}"),
                'recipes': cluster_recipes,
                'centroid': centroid,
                'top_terms': top_terms,
                'size': len(cluster_recipes)
            }

        return details

    def create_new_recipe_profile(
        self,
        cluster_weights: Dict[int, float]
    ) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        """
        Create a new recipe profile by weighted averaging of cluster centroids.

        Args:
            cluster_weights: Dict mapping cluster label to weight (should sum to 1)

        Returns:
            Tuple of (profile vector, top sensorik terms with weights)
        """
        # Normalize weights
        total_weight = sum(cluster_weights.values())
        normalized_weights = {k: v/total_weight for k, v in cluster_weights.items()}

        # Calculate weighted average
        new_profile = np.zeros(len(self.vocabulary))

        for label, weight in normalized_weights.items():
            if label == -1:
                continue
            cluster_mask = self.cluster_labels == label
            cluster_vectors = self.recipe_vectors[cluster_mask]
            centroid = cluster_vectors.mean(axis=0)
            new_profile += weight * centroid

        # Get top terms
        top_indices = np.argsort(new_profile)[-15:][::-1]
        top_terms = [(self.vocabulary[i], new_profile[i]) for i in top_indices]

        return new_profile, top_terms

    def visualize_clusters(
        self,
        method: str = 'tsne',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10)
    ) -> plt.Figure:
        """
        Visualize clusters using dimensionality reduction.

        Args:
            method: 'tsne' or 'pca'
            save_path: Path to save the figure
            figsize: Figure size
        """
        # Reduce dimensions
        if method == 'tsne':
            # Use PCA first if many dimensions
            if self.recipe_vectors.shape[1] > 50:
                pca = PCA(n_components=min(30, self.recipe_vectors.shape[0]-1))
                reduced = pca.fit_transform(self.recipe_vectors)
            else:
                reduced = self.recipe_vectors

            perplexity = min(5, len(self.recipes) - 1)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
            coords = reducer.fit_transform(reduced)
            method_name = "t-SNE"
        else:
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(self.recipe_vectors)
            method_name = "PCA"

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Color palette
        unique_labels = sorted(set(self.cluster_labels))
        n_clusters = len(unique_labels)
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 8)))

        # Plot 1: Cluster visualization
        ax1 = axes[0]
        for i, label in enumerate(unique_labels):
            mask = self.cluster_labels == label
            cluster_coords = coords[mask]

            if label == -1:
                color = 'gray'
                marker = 'x'
                alpha = 0.5
            else:
                color = colors[i % len(colors)]
                marker = 'o'
                alpha = 0.8

            ax1.scatter(
                cluster_coords[:, 0],
                cluster_coords[:, 1],
                c=[color],
                marker=marker,
                s=150,
                alpha=alpha,
                label=self.cluster_names.get(label, f"Cluster {label}"),
                edgecolors='black',
                linewidths=0.5
            )

        # Add recipe labels
        for i, recipe in enumerate(self.recipes):
            ax1.annotate(
                recipe[:12],  # Truncate long names
                (coords[i, 0], coords[i, 1]),
                fontsize=7,
                alpha=0.7,
                ha='center',
                va='bottom'
            )

        ax1.set_xlabel(f'{method_name} Dimension 1', fontsize=10)
        ax1.set_ylabel(f'{method_name} Dimension 2', fontsize=10)
        ax1.set_title('Recipe Clusters by Sensorik Profile', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Cluster composition (top terms per cluster)
        ax2 = axes[1]
        details = self.get_cluster_details()

        # Prepare data for bar chart
        cluster_data = []
        for label in unique_labels:
            if label == -1:
                continue
            info = details[label]
            terms = [t[0] for t in info['top_terms'][:5]]
            weights = [t[1] for t in info['top_terms'][:5]]
            cluster_data.append({
                'label': label,
                'name': info['name'],
                'terms': terms,
                'weights': weights,
                'size': info['size']
            })

        # Create horizontal bar chart
        y_positions = []
        y_labels = []
        bar_data = []

        y = 0
        for data in cluster_data:
            for i, (term, weight) in enumerate(zip(data['terms'], data['weights'])):
                y_positions.append(y)
                y_labels.append(f"{term}")
                bar_data.append((data['label'], weight))
                y += 1
            y += 0.5  # Gap between clusters

        # Color bars by cluster
        bar_colors = [colors[d[0] % len(colors)] for d in bar_data]
        bars = ax2.barh(y_positions, [d[1] for d in bar_data], color=bar_colors, alpha=0.8)

        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(y_labels, fontsize=8)
        ax2.set_xlabel('Weighted Importance', fontsize=10)
        ax2.set_title('Top Sensorik Terms per Cluster', fontsize=12, fontweight='bold')

        # Add cluster name annotations
        y = 0
        for data in cluster_data:
            ax2.annotate(
                f"Cluster {data['label']}: {data['name']}\n({data['size']} recipes)",
                xy=(ax2.get_xlim()[1] * 0.6, y + 2),
                fontsize=8,
                fontweight='bold',
                color=colors[data['label'] % len(colors)]
            )
            y += len(data['terms']) + 0.5

        ax2.invert_yaxis()
        ax2.grid(True, axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig

    def visualize_cluster_profiles(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """Create a radar chart showing cluster profiles."""

        details = self.get_cluster_details()

        # Get all top terms across clusters (union)
        all_top_terms = set()
        for label, info in details.items():
            if label == -1:
                continue
            all_top_terms.update([t[0] for t in info['top_terms'][:8]])

        all_top_terms = sorted(all_top_terms)[:12]  # Limit for readability

        # Prepare data
        n_clusters = len([l for l in details.keys() if l != -1])
        n_terms = len(all_top_terms)

        if n_clusters == 0 or n_terms == 0:
            print("Not enough data for radar chart")
            return None

        # Create radar chart
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, n_terms, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop

        colors = plt.cm.Set2(np.linspace(0, 1, max(n_clusters, 8)))

        for i, (label, info) in enumerate(details.items()):
            if label == -1:
                continue

            # Get values for each term
            term_to_weight = {t[0]: t[1] for t in info['top_terms']}
            values = [term_to_weight.get(term, 0) for term in all_top_terms]
            values += values[:1]  # Complete the loop

            ax.plot(angles, values, 'o-', linewidth=2, label=info['name'],
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_top_terms, fontsize=9)
        ax.set_title('Cluster Sensorik Profiles', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig

    def run_full_pipeline(
        self,
        min_cluster_size: int = 2,
        save_plots: bool = True,
        output_dir: str = "."
    ) -> Dict:
        """Run the complete clustering pipeline."""

        print("=" * 60)
        print("RECIPE CLUSTERING PIPELINE")
        print("=" * 60)

        # Step 1: Load data
        print("\n[1/5] Loading data...")
        self.load_data()

        # Step 2: Build vocabulary
        print("\n[2/5] Building vocabulary...")
        self.build_vocabulary()

        # Step 3: Extract vectors
        print("\n[3/5] Extracting weighted recipe vectors...")
        self.extract_recipe_vectors()

        # Step 4: Cluster
        print("\n[4/5] Clustering recipes...")
        self.cluster_recipes(min_cluster_size=min_cluster_size)

        # Step 5: Generate names
        print("\n[5/5] Generating cluster names...")
        self.generate_cluster_names()

        # Get details
        details = self.get_cluster_details()

        # Print summary
        print("\n" + "=" * 60)
        print("CLUSTERING SUMMARY")
        print("=" * 60)

        for label in sorted(details.keys()):
            info = details[label]
            print(f"\n{'─' * 50}")
            print(f"CLUSTER {label}: {info['name']}")
            print(f"{'─' * 50}")
            print(f"Recipes ({info['size']}):")
            for recipe in info['recipes']:
                print(f"  • {recipe}")
            print(f"\nTop Sensorik Profile:")
            for term, weight in info['top_terms'][:7]:
                bar = "█" * int(weight * 50)
                print(f"  {term:15} {bar} ({weight:.3f})")

        # Save results
        if save_plots:
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATIONS")
            print("=" * 60)

            # Main cluster plot
            fig1 = self.visualize_clusters(
                method='tsne',
                save_path=f"{output_dir}/cluster_visualization.png"
            )

            # Radar chart
            fig2 = self.visualize_cluster_profiles(
                save_path=f"{output_dir}/cluster_profiles.png"
            )

        return {
            'cluster_labels': self.cluster_labels,
            'cluster_names': self.cluster_names,
            'details': details,
            'recipes': self.recipes,
            'vectors': self.recipe_vectors,
            'vocabulary': self.vocabulary
        }

    def run_agglomerative_pipeline(
        self,
        k_range: tuple = (3, 10),
        save_plots: bool = True,
        output_dir: str = "."
    ) -> Dict:
        """
        Run clustering with Agglomerative algorithm (no outliers).
        All recipes guaranteed to be assigned to a cluster.
        """

        print("=" * 60)
        print("RECIPE CLUSTERING (AGGLOMERATIVE - NO OUTLIERS)")
        print("=" * 60)

        # Step 1: Load data
        print("\n[1/5] Loading data...")
        self.load_data()

        # Step 2: Build vocabulary
        print("\n[2/5] Building vocabulary...")
        self.build_vocabulary()

        # Step 3: Extract vectors
        print("\n[3/5] Extracting weighted recipe vectors...")
        self.extract_recipe_vectors()

        # Step 4: Cluster with Agglomerative
        print("\n[4/5] Clustering recipes (Agglomerative with auto-k)...")
        self.cluster_recipes_agglomerative(k_range=k_range)

        # Step 5: Generate names
        print("\n[5/5] Generating cluster names...")
        self.generate_cluster_names()

        # Get details
        details = self.get_cluster_details()

        # Print summary
        print("\n" + "=" * 60)
        print("CLUSTERING SUMMARY")
        print("=" * 60)

        for label in sorted(details.keys()):
            info = details[label]
            print(f"\n{'─' * 50}")
            print(f"CLUSTER {label}: {info['name']}")
            print(f"{'─' * 50}")
            print(f"Recipes ({info['size']}):")
            for recipe in info['recipes']:
                print(f"  • {recipe}")
            print(f"\nTop Sensorik Profile:")
            for term, weight in info['top_terms'][:7]:
                bar = "█" * int(weight * 50)
                print(f"  {term:15} {bar} ({weight:.3f})")

        # Save results
        if save_plots:
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATIONS")
            print("=" * 60)

            # Main cluster plot
            fig1 = self.visualize_clusters(
                method='tsne',
                save_path=f"{output_dir}/cluster_agglomerative.png"
            )

            # Radar chart
            fig2 = self.visualize_cluster_profiles(
                save_path=f"{output_dir}/cluster_agglomerative_profiles.png"
            )

        return {
            'cluster_labels': self.cluster_labels,
            'cluster_names': self.cluster_names,
            'details': details,
            'recipes': self.recipes,
            'vectors': self.recipe_vectors,
            'vocabulary': self.vocabulary
        }

    def run_faiss_pipeline(
        self,
        k_range: tuple = (3, 12),
        save_plots: bool = True,
        output_dir: str = "."
    ) -> Dict:
        """
        Run clustering with FAISS vector database.
        Provides efficient similarity search and k-means clustering.
        """

        print("=" * 60)
        print("RECIPE CLUSTERING (FAISS VECTOR DATABASE)")
        print("=" * 60)

        # Step 1: Load data
        print("\n[1/5] Loading data...")
        self.load_data()

        # Step 2: Build vocabulary
        print("\n[2/5] Building vocabulary...")
        self.build_vocabulary()

        # Step 3: Extract vectors
        print("\n[3/5] Extracting weighted recipe vectors...")
        self.extract_recipe_vectors()

        # Step 4: Cluster with FAISS
        print("\n[4/5] Clustering with FAISS k-means...")
        self.cluster_recipes_faiss(k_range=k_range)

        # Step 5: Generate names
        print("\n[5/5] Generating cluster names...")
        self.generate_cluster_names()

        # Get details
        details = self.get_cluster_details()

        # Print summary
        print("\n" + "=" * 60)
        print("CLUSTERING SUMMARY")
        print("=" * 60)

        for label in sorted(details.keys()):
            info = details[label]
            print(f"\n{'─' * 50}")
            print(f"CLUSTER {label}: {info['name']}")
            print(f"{'─' * 50}")
            print(f"Recipes ({info['size']}):")
            for recipe in info['recipes']:
                print(f"  • {recipe}")
            print(f"\nTop Sensorik Profile:")
            for term, weight in info['top_terms'][:7]:
                bar = "█" * int(weight * 50)
                print(f"  {term:15} {bar} ({weight:.3f})")

        # Demo similarity search
        print("\n" + "=" * 60)
        print("FAISS SIMILARITY SEARCH DEMO")
        print("=" * 60)

        # Show similar recipes for a few examples
        sample_indices = [0, len(self.recipes)//2, len(self.recipes)-1]
        for i in sample_indices:
            recipe = self.recipes[i]
            similar = self.find_similar_recipes(i, top_k=3)
            print(f"\nRecipes similar to '{recipe}':")
            for sim_recipe, dist in similar:
                print(f"  • {sim_recipe} (distance: {dist:.4f})")

        # Save results
        if save_plots:
            print("\n" + "=" * 60)
            print("GENERATING VISUALIZATIONS")
            print("=" * 60)

            # Main cluster plot
            fig1 = self.visualize_clusters(
                method='tsne',
                save_path=f"{output_dir}/cluster_faiss.png"
            )

            # Radar chart
            fig2 = self.visualize_cluster_profiles(
                save_path=f"{output_dir}/cluster_faiss_profiles.png"
            )

            # FAISS-specific visualization
            fig3 = self.visualize_faiss_similarity(
                save_path=f"{output_dir}/cluster_faiss_similarity.png"
            )

        return {
            'cluster_labels': self.cluster_labels,
            'cluster_names': self.cluster_names,
            'details': details,
            'recipes': self.recipes,
            'vectors': self.recipe_vectors,
            'vocabulary': self.vocabulary,
            'faiss_index': self.faiss_index,
            'centroids': self.cluster_centroids
        }

    def visualize_faiss_similarity(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10)
    ) -> plt.Figure:
        """
        Visualize FAISS similarity matrix and cluster structure.
        """
        if self.faiss_index is None:
            print("FAISS index not available")
            return None

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 1. Similarity heatmap
        ax1 = axes[0]
        n = len(self.recipes)
        vectors = self.recipe_vectors.astype('float32')

        # Compute pairwise distances
        distances, _ = self.faiss_index.search(vectors, n)

        # Convert to similarity (inverse of distance)
        similarity = 1 / (1 + distances)

        im = ax1.imshow(similarity, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Recipe Similarity Matrix\n(FAISS L2 Distance)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Recipe Index')
        ax1.set_ylabel('Recipe Index')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Similarity', fontsize=9)

        # 2. Cluster sizes bar chart
        ax2 = axes[1]
        unique_labels = sorted(set(self.cluster_labels))
        sizes = [list(self.cluster_labels).count(l) for l in unique_labels]
        names = [self.cluster_names.get(l, f"C{l}")[:20] for l in unique_labels]
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        bars = ax2.barh(range(len(unique_labels)), sizes, color=colors)
        ax2.set_yticks(range(len(unique_labels)))
        ax2.set_yticklabels(names, fontsize=8)
        ax2.set_xlabel('Number of Recipes')
        ax2.set_title('Cluster Sizes\n(FAISS k-means)', fontsize=11, fontweight='bold')
        ax2.invert_yaxis()

        # Add count labels
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(size), va='center', fontsize=9)

        # 3. Centroid distances
        ax3 = axes[2]
        if self.cluster_centroids is not None:
            n_centroids = len(self.cluster_centroids)
            centroid_dist = np.zeros((n_centroids, n_centroids))

            for i in range(n_centroids):
                for j in range(n_centroids):
                    centroid_dist[i, j] = np.linalg.norm(
                        self.cluster_centroids[i] - self.cluster_centroids[j]
                    )

            im3 = ax3.imshow(centroid_dist, cmap='Blues', aspect='auto')
            ax3.set_title('Inter-Cluster Distances\n(Centroid L2)', fontsize=11, fontweight='bold')
            ax3.set_xlabel('Cluster')
            ax3.set_ylabel('Cluster')
            ax3.set_xticks(range(n_centroids))
            ax3.set_yticks(range(n_centroids))

            # Add colorbar
            cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
            cbar3.set_label('Distance', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")

        return fig


def demo_new_recipe_creation(clusterer: RecipeClusterer):
    """Demonstrate creating a new recipe profile by mixing clusters."""

    print("\n" + "=" * 60)
    print("DEMO: Creating New Recipe Profile")
    print("=" * 60)

    # Get available clusters (excluding outliers)
    available_clusters = [l for l in set(clusterer.cluster_labels) if l != -1]

    if len(available_clusters) < 2:
        print("Need at least 2 clusters to demonstrate mixing")
        return

    # Mix first two clusters 60/40
    cluster_a, cluster_b = available_clusters[0], available_clusters[1]

    print(f"\nMixing:")
    print(f"  60% Cluster {cluster_a} ({clusterer.cluster_names[cluster_a]})")
    print(f"  40% Cluster {cluster_b} ({clusterer.cluster_names[cluster_b]})")

    profile, top_terms = clusterer.create_new_recipe_profile({
        cluster_a: 0.6,
        cluster_b: 0.4
    })

    print(f"\nNew Recipe Target Sensorik Profile:")
    for term, weight in top_terms[:10]:
        bar = "█" * int(weight * 50)
        print(f"  {term:15} {bar} ({weight:.3f})")


def demo_faiss_recipe_creation(clusterer: RecipeClusterer):
    """Demonstrate FAISS-specific features for recipe creation."""

    print("\n" + "=" * 60)
    print("DEMO: FAISS New Recipe Creation with Similarity Search")
    print("=" * 60)

    if clusterer.cluster_centroids is None:
        print("FAISS centroids not available")
        return

    # Get available clusters
    available_clusters = list(range(len(clusterer.cluster_centroids)))

    if len(available_clusters) < 2:
        print("Need at least 2 clusters to demonstrate mixing")
        return

    # Mix first two clusters 70/30
    cluster_a, cluster_b = available_clusters[0], available_clusters[1]

    print(f"\nMixing clusters:")
    print(f"  70% Cluster {cluster_a} ({clusterer.cluster_names.get(cluster_a, 'Unknown')})")
    print(f"  30% Cluster {cluster_b} ({clusterer.cluster_names.get(cluster_b, 'Unknown')})")

    profile, top_terms, similar_recipes = clusterer.create_new_recipe_profile_faiss({
        cluster_a: 0.7,
        cluster_b: 0.3
    })

    print(f"\nNew Recipe Target Sensorik Profile:")
    for term, weight in top_terms[:10]:
        bar = "█" * int(weight * 50)
        print(f"  {term:15} {bar} ({weight:.3f})")

    print(f"\nMost Similar Existing Recipes (via FAISS search):")
    for recipe, dist in similar_recipes:
        print(f"  • {recipe} (distance: {dist:.4f})")

    # Demo: Search by custom profile
    print("\n" + "-" * 50)
    print("CUSTOM PROFILE SEARCH")
    print("-" * 50)

    custom_profile = {
        'fruity': 0.8,
        'sweet': 0.6,
        'tropical': 0.4,
        'pineapple': 0.3
    }

    print(f"\nSearching for recipes matching:")
    for term, weight in custom_profile.items():
        print(f"  {term}: {weight}")

    similar = clusterer.find_recipes_by_profile(custom_profile, top_k=5)

    print(f"\nBest matching recipes:")
    for recipe, dist in similar:
        print(f"  • {recipe} (distance: {dist:.4f})")


if __name__ == "__main__":
    # Run the clustering pipeline
    clusterer = RecipeClusterer("data/gold/Versuchsdaten.csv")
    results = clusterer.run_full_pipeline(
        min_cluster_size=2,
        save_plots=True,
        output_dir="."
    )

    # Demo new recipe creation
    demo_new_recipe_creation(clusterer)

    # Show plots
    plt.show()
