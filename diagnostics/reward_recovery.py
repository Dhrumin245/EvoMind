"""
DIAGNOSTIC 3: Learning Speed Compression
Tracks recovery times after perturbations across generations.
"""

import matplotlib.pyplot as plt
from typing import List, Optional


class RewardRecoveryLogger:
    """Logger for reward recovery diagnostics"""

    # List of (generation, recovery_time_or_None) tuples
    recovery_records: List[tuple] = []

    @staticmethod
    def calculate_recovery_time(pre_shock_reward: float, post_shock_rewards: List[float]) -> Optional[int]:
        """Calculate recovery time: min t where reward[t] returns to within 80% of pre-shock baseline.

        Works correctly for both positive and negative reward baselines:
        - Positive baseline: threshold = 0.8 * pre_shock (must reach 80% of positive baseline)
        - Negative baseline: threshold = pre_shock / 0.8 (allows up to 25% more negative than baseline)
        """
        if pre_shock_reward >= 0:
            threshold = 0.8 * pre_shock_reward
        else:
            # For negative baseline, recovery means not falling more than 20% further below baseline
            threshold = pre_shock_reward / 0.8  # e.g. -0.1 / 0.8 = -0.125
        for t, reward in enumerate(post_shock_rewards):
            if reward >= threshold:
                return t
        return None  # Did not recover

    @staticmethod
    def log_recovery_time(recovery_time: Optional[int], generation: int = -1):
        """Log a recovery time for a specific generation"""
        RewardRecoveryLogger.recovery_records.append((generation, recovery_time))

    @staticmethod
    def plot_recovery_times(filename="diagnostics/learning_speed_compression.png"):
        """Plot recovery time vs generation"""
        if not RewardRecoveryLogger.recovery_records:
            print("No recovery times recorded for plotting")
            return

        # Filter out None (did not recover) entries and use real generation numbers
        gen_indices = [gen for gen, t in RewardRecoveryLogger.recovery_records if t is not None]
        recovery_times_filtered = [t for _, t in RewardRecoveryLogger.recovery_records if t is not None]

        plt.figure(figsize=(10, 6))
        if gen_indices:
            plt.plot(gen_indices, recovery_times_filtered, 'bo-', linewidth=2, markersize=6)
            # Annotate did-not-recover generations at the top of the chart
            no_recovery_gens = [gen for gen, t in RewardRecoveryLogger.recovery_records if t is None]
            if no_recovery_gens:
                max_y = max(recovery_times_filtered) if recovery_times_filtered else 1
                plt.scatter(no_recovery_gens, [max_y * 1.05] * len(no_recovery_gens),
                            marker='x', color='red', s=60, label='No recovery', zorder=5)
                plt.legend()
        else:
            # All generations failed to recover — show them as crosses on y=0
            no_recovery_gens = [gen for gen, t in RewardRecoveryLogger.recovery_records if t is None]
            plt.scatter(no_recovery_gens, [0] * len(no_recovery_gens),
                        marker='x', color='red', s=80, label='No recovery')
            plt.legend()
        plt.xlabel("Generation")
        plt.ylabel("Recovery Time (timesteps)")
        plt.title("DIAGNOSTIC 3: Learning Speed Compression")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_recovery_times() -> List[Optional[int]]:
        """Get all logged recovery times (backward-compat: returns just the times)"""
        return [t for _, t in RewardRecoveryLogger.recovery_records]
