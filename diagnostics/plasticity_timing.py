"""
DIAGNOSTIC 2: Plastic Weight Activation Timing
Tracks ||ΔW|| per timestep across plastic layers.
"""

import matplotlib.pyplot as plt
from typing import List, Dict


class PlasticityTimingLogger:
    """Logger for plastic weight activation timing diagnostics"""

    delta_logs: List[List[List[float]]] = []  # Per-generation per-layer delta logs

    @staticmethod
    def log_generation_delta_logs(layer_delta_logs: List[List[float]]):
        """Log delta logs for all plastic layers in current generation"""
        PlasticityTimingLogger.delta_logs.append(layer_delta_logs)

    @staticmethod
    def plot_activation_timing(filename="diagnostics/plastic_weight_activation_timing.png"):
        """Plot ||ΔW|| vs timestep per layer"""
        if not PlasticityTimingLogger.delta_logs:
            print("No delta logs recorded for plotting")
            return

        # Use the most recent generation's data
        recent_logs = PlasticityTimingLogger.delta_logs[-1]

        if not recent_logs:
            print("No plastic layers found for plotting")
            return

        plt.figure(figsize=(12, 8))

        for i, delta_logs in enumerate(recent_logs):
            if delta_logs:
                timesteps = list(range(len(delta_logs)))
                plt.plot(timesteps, delta_logs, label=f"Layer {i} ||ΔW||", linewidth=2)

        plt.xlabel("Timestep")
        plt.ylabel("||ΔW|| (Weight Change Norm)")
        plt.title("DIAGNOSTIC 2: Plastic Weight Activation Timing")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def get_delta_logs() -> List[List[List[float]]]:
        """Get all logged delta logs"""
        return PlasticityTimingLogger.delta_logs.copy()
