"""
Strategy Clustering Diagnostics
Tracks strategy clustering and behavioral diversity across generations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class StrategyClusteringLogger:
    """Logger for strategy clustering diagnostics"""

    cluster_centers: List[np.ndarray] = []
    cluster_labels: List[List[int]] = []
    silhouette_scores: List[float] = []
    strategy_diversities: List[float] = []

    @staticmethod
    def log_generation_strategies(strategy_vectors: List[np.ndarray], n_clusters: int = 3):
        """Log strategy clustering for current generation"""
        if not strategy_vectors:
            return

        # Convert to numpy array
        X = np.array(strategy_vectors)

        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(strategy_vectors)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = 0.0

        # Calculate strategy diversity (average distance to cluster centers)
        diversity = np.mean([np.linalg.norm(vec - centers[label]) for vec, label in zip(X, labels)])

        # Store results
        StrategyClusteringLogger.cluster_centers.append(centers)
        StrategyClusteringLogger.cluster_labels.append(labels.tolist())
        StrategyClusteringLogger.silhouette_scores.append(float(silhouette))
        StrategyClusteringLogger.strategy_diversities.append(float(diversity))

    @staticmethod
    def plot_strategy_clustering(filename="diagnostics/strategy_clustering.png"):
        """Plot strategy clustering metrics vs generation"""
        if not StrategyClusteringLogger.silhouette_scores:
            print("No strategy clustering data recorded for plotting")
            return

        generations = list(range(len(StrategyClusteringLogger.silhouette_scores)))

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Silhouette scores
        ax1.plot(generations, StrategyClusteringLogger.silhouette_scores, 'bo-', linewidth=2)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Silhouette Score")
        ax1.set_title("Strategy Clustering Quality")
        ax1.grid(True, alpha=0.3)

        # Strategy diversity
        ax2.plot(generations, StrategyClusteringLogger.strategy_diversities, 'ro-', linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Strategy Diversity")
        ax2.set_title("Strategy Diversity")
        ax2.grid(True, alpha=0.3)

        # Number of clusters over time
        n_clusters = [len(centers) for centers in StrategyClusteringLogger.cluster_centers]
        ax3.plot(generations, n_clusters, 'go-', linewidth=2)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Number of Clusters")
        ax3.set_title("Number of Strategy Clusters")
        ax3.grid(True, alpha=0.3)

        # Cluster sizes distribution (most recent)
        if StrategyClusteringLogger.cluster_labels:
            recent_labels = StrategyClusteringLogger.cluster_labels[-1]
            unique_labels, counts = np.unique(recent_labels, return_counts=True)
            ax4.bar(unique_labels, counts, alpha=0.7)
            ax4.set_xlabel("Cluster ID")
            ax4.set_ylabel("Number of Strategies")
            ax4.set_title(f"Cluster Sizes (Generation {len(generations)-1})")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def get_clustering_data() -> Dict[str, Any]:
        """Get all logged clustering data"""
        return {
            'cluster_centers': StrategyClusteringLogger.cluster_centers.copy(),
            'cluster_labels': StrategyClusteringLogger.cluster_labels.copy(),
            'silhouette_scores': StrategyClusteringLogger.silhouette_scores.copy(),
            'strategy_diversities': StrategyClusteringLogger.strategy_diversities.copy()
        }
