"""
DIAGNOSTIC 3: Learning Speed Compression
Tracks recovery times after perturbations across generations.
"""

import matplotlib.pyplot as plt
from typing import List, Optional


class RewardRecoveryLogger:
    """Logger for reward recovery diagnostics"""

    recovery_times: List[Optional[int]] = []

    @staticmethod
    def calculate_recovery_time(pre_shock_reward: float, post_shock_rewards: List[float]) -> Optional[int]:
        """Calculate recovery time: min t where reward[t] >= 0.8 * reward_pre_shock"""
        threshold = 0.8 * pre_shock_reward
        for t, reward in enumerate(post_shock_rewards):
            if reward >= threshold:
                return t
        return None  # Did not recover

    @staticmethod
    def log_recovery_time(recovery_time: Optional[int]):
        """Log a recovery time for the current generation"""
        RewardRecoveryLogger.recovery_times.append(recovery_time)

    @staticmethod
    def plot_recovery_times(filename="diagnostics/learning_speed_compression.png"):
        """Plot recovery time vs generation"""
        if not RewardRecoveryLogger.recovery_times:
            print("No recovery times recorded for plotting")
            return

        generations = list(range(len(RewardRecoveryLogger.recovery_times)))
        recovery_times_filtered = [t for t in RewardRecoveryLogger.recovery_times if t is not None]

        plt.figure(figsize=(10, 6))
        plt.plot(generations[:len(recovery_times_filtered)], recovery_times_filtered, 'bo-', linewidth=2, markersize=6)
        plt.xlabel("Generation")
        plt.ylabel("Recovery Time (timesteps)")
        plt.title("DIAGNOSTIC 3: Learning Speed Compression")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def get_recovery_times() -> List[Optional[int]]:
        """Get all logged recovery times"""
        return RewardRecoveryLogger.recovery_times.copy()
