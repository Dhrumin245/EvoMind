"""
Meta-Gene Entropy Diagnostics
Tracks entropy and selection pressure on meta-learning genes across generations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


class MetaGeneEntropyLogger:
    """Logger for meta-gene entropy diagnostics"""

    meta_gene_entropies: List[float] = []
    plasticity_weight_variances: List[float] = []
    learning_rate_entropies: List[float] = []
    plastic_neuron_fractions: List[float] = []

    # META-3.2: New metrics for adaptability selection pressure
    adaptability_pressures: List[float] = []
    meta_effectiveness_scores: List[float] = []

    @staticmethod
    def log_generation_meta_entropy(
        meta_gene_entropy: float,
        plasticity_weight_variance: float,
        learning_rate_entropy: float,
        plastic_neuron_fraction: float
    ):
        """Log meta-gene diagnostics for current generation"""
        MetaGeneEntropyLogger.meta_gene_entropies.append(meta_gene_entropy)
        MetaGeneEntropyLogger.plasticity_weight_variances.append(plasticity_weight_variance)
        MetaGeneEntropyLogger.learning_rate_entropies.append(learning_rate_entropy)
        MetaGeneEntropyLogger.plastic_neuron_fractions.append(plastic_neuron_fraction)

    @staticmethod
    def plot_meta_gene_entropy(filename="diagnostics/meta_gene_entropy.png"):
        """Plot meta-gene entropy metrics vs generation"""
        if not MetaGeneEntropyLogger.meta_gene_entropies:
            print("No meta-gene entropy data recorded for plotting")
            return

        generations = list(range(len(MetaGeneEntropyLogger.meta_gene_entropies)))

        # Create subplot grid - expand to 3x2 for new metrics
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

        # Meta-gene entropy
        ax1.plot(generations, MetaGeneEntropyLogger.meta_gene_entropies, 'ro-', linewidth=2)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Meta-gene Entropy")
        ax1.set_title("Meta-Gene Selection Pressure")
        ax1.grid(True, alpha=0.3)

        # Plasticity weight variance
        ax2.plot(generations, MetaGeneEntropyLogger.plasticity_weight_variances, 'bo-', linewidth=2)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Plasticity Weight Variance")
        ax2.set_title("Plasticity Weight Variance")
        ax2.grid(True, alpha=0.3)

        # Learning rate entropy
        ax3.plot(generations, MetaGeneEntropyLogger.learning_rate_entropies, 'go-', linewidth=2)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Learning Rate Entropy")
        ax3.set_title("Learning Rate Gene Entropy")
        ax3.grid(True, alpha=0.3)

        # Plastic neuron fraction
        ax4.plot(generations, MetaGeneEntropyLogger.plastic_neuron_fractions, 'mo-', linewidth=2)
        ax4.set_xlabel("Generation")
        ax4.set_ylabel("Plastic Neuron Fraction")
        ax4.set_title("Fraction of Plastic Neurons")
        ax4.grid(True, alpha=0.3)

        # META-3.2: Adaptability selection pressure
        if hasattr(MetaGeneEntropyLogger, 'adaptability_pressures') and MetaGeneEntropyLogger.adaptability_pressures:
            ax5.plot(generations, MetaGeneEntropyLogger.adaptability_pressures, 'co-', linewidth=2)
            ax5.set_xlabel("Generation")
            ax5.set_ylabel("Adaptability Selection Pressure")
            ax5.set_title("Evolution Pressure on Adaptability")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No adaptability pressure data', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title("Evolution Pressure on Adaptability")

        # META-3.2: Meta-parameter effectiveness
        if hasattr(MetaGeneEntropyLogger, 'meta_effectiveness_scores') and MetaGeneEntropyLogger.meta_effectiveness_scores:
            ax6.plot(generations, MetaGeneEntropyLogger.meta_effectiveness_scores, 'yo-', linewidth=2)
            ax6.set_xlabel("Generation")
            ax6.set_ylabel("Meta-Parameter Effectiveness")
            ax6.set_title("Meta-Parameter Effectiveness Score")
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No effectiveness data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title("Meta-Parameter Effectiveness Score")

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_meta_entropy_data() -> Dict[str, Any]:
        """Get all logged meta-gene entropy data"""
        return {
            'meta_gene_entropies': MetaGeneEntropyLogger.meta_gene_entropies.copy(),
            'plasticity_weight_variances': MetaGeneEntropyLogger.plasticity_weight_variances.copy(),
            'learning_rate_entropies': MetaGeneEntropyLogger.learning_rate_entropies.copy(),
            'plastic_neuron_fractions': MetaGeneEntropyLogger.plastic_neuron_fractions.copy()
        }
