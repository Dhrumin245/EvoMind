"""
Plotting functions for diagnostics and visualization.
Moved from main.py to separate diagnostics folder.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List, cast, Optional

from genomes.genome_prey import PreyGenome
from genomes.genome_predator import PredatorGenome
from core.genome import Genome as EvolvableGenome
from diagnostics.reward_recovery import RewardRecoveryLogger
from diagnostics.strategy_clustering import StrategyClusteringLogger


def compute_architecture_clustering_stats(population: List[EvolvableGenome], n_clusters: int = 3) -> Dict[str, Any]:
    """Compute architecture clustering stats using genome structural features."""
    vectors = [g.get_architecture_vector() for g in population if hasattr(g, "get_architecture_vector")]
    if len(vectors) < 2:
        return {
            'num_clusters': 0,
            'silhouette': 0.0,
            'diversity': 0.0,
            'cluster_sizes': []
        }

    X = np.array(vectors)
    k = min(n_clusters, len(vectors))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    if len(np.unique(labels)) > 1:
        silhouette = float(silhouette_score(X, labels))
    else:
        silhouette = 0.0

    diversity = float(np.mean([np.linalg.norm(vec - centers[label]) for vec, label in zip(X, labels)]))
    unique_labels, counts = np.unique(labels, return_counts=True)

    return {
        'num_clusters': int(len(unique_labels)),
        'silhouette': silhouette,
        'diversity': diversity,
        'cluster_sizes': counts.tolist()
    }


def plot_meta_gene_histograms(stats: Dict[str, Any]) -> None:
    """Plot histograms for META gene distribution (reward_gain and reward_bias)"""
    if 'meta_gain' not in stats or 'meta_bias' not in stats:
        return

    generation = stats['generation']
    meta_gain = stats['meta_gain']
    meta_bias = stats['meta_bias']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot reward_gain histogram
    ax1.hist(meta_gain, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title(f'META Gene Distribution - Generation {generation}')
    ax1.set_xlabel('Reward Gain')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Plot reward_bias histogram
    ax2.hist(meta_bias, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title(f'META Gene Distribution - Generation {generation}')
    ax2.set_xlabel('Reward Bias')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'output_logs/meta_gene_distribution_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   📊 Saved: META gene distribution analysis")
    print(f"      └─ output_logs/meta_gene_distribution_gen_{generation:04d}.png")


def plot_plastic_norm_evolution(generation_stats: List[Dict[str, Any]]) -> None:
    """Plot plastic weight norm evolution over generations"""
    if not generation_stats:
        return

    generations = [stats['generation'] for stats in generation_stats if 'mean_plastic_norm' in stats]
    mean_norms = [stats['mean_plastic_norm'] for stats in generation_stats if 'mean_plastic_norm' in stats]
    max_norms = [stats['max_plastic_norm'] for stats in generation_stats if 'max_plastic_norm' in stats]

    if not generations:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean plastic norm
    ax.plot(generations, mean_norms, 'b-', linewidth=2, label='Mean Plastic Norm', alpha=0.8)

    # Plot max plastic norm
    ax.plot(generations, max_norms, 'r--', linewidth=1.5, label='Max Plastic Norm', alpha=0.7)

    ax.set_title('Plastic Weight Norm Evolution')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Plastic Weight Norm')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('output_logs/plastic_norm_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("   📊 Saved: Plasticity (learning mechanisms) evolution over generations")
    print("      └─ output_logs/plastic_norm_evolution.png")


def plot_learning_rule_stats(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]) -> None:
    """Plot learning rule parameter distributions per generation"""
    population = prey_population + predator_population
    rules = ["mean_delta_w", "std_delta_w", "max_delta_w", "min_delta_w"]
    rule_labels = ["Mean ΔW", "Std ΔW", "Max ΔW", "Min ΔW"]

    pop_with_net = [g for g in population if getattr(g, 'learning_rule_net', None) is not None]
    if not pop_with_net:
        return

    fig, axes = plt.subplots(1, len(rules), figsize=(20, 4))
    for i, (k, label) in enumerate(zip(rules, rule_labels)):
        vals = [g.learning_rule_net.get_parameters_as_dict()[k] for g in pop_with_net]
        axes[i].hist(vals, bins=30, alpha=0.7, edgecolor='black')
        if vals:
            axes[i].axvline(float(np.mean(vals)), color='r', linewidth=2, label='Mean')
        axes[i].set_title(f"{label} — Gen {generation}")
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f'output_logs/learning_rule_stats_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_rule_vs_fitness(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]) -> None:
    """Plot scatter plots of learning rule parameters vs fitness for each gene"""
    population = prey_population + predator_population
    rules = ["mean_delta_w", "std_delta_w", "max_delta_w", "min_delta_w"]
    rule_labels = ["Mean ΔW", "Std ΔW", "Max ΔW", "Min ΔW"]

    pop_with_net = [g for g in population if getattr(g, 'learning_rule_net', None) is not None]
    if not pop_with_net:
        return

    fig, axes = plt.subplots(1, len(rules), figsize=(20, 4))

    for i, (rule, label) in enumerate(zip(rules, rule_labels)):
        x = [g.learning_rule_net.get_parameters_as_dict()[rule] for g in pop_with_net]
        y = [g.fitness for g in pop_with_net]
        axes[i].scatter(x, y, alpha=0.6)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel("Fitness")
        axes[i].set_title(f"{label} vs Fitness - Gen {generation}")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'output_logs/learning_rule_vs_fitness_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   📊 Saved: Learning mechanisms vs performance analysis")
    print(f"      └─ output_logs/learning_rule_vs_fitness_gen_{generation:04d}.png")


def plot_strategy_clustering(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]) -> None:
    """Plot fitness per cluster after clustering genomes by learning rule strategies"""
    population = prey_population + predator_population
    rules = ["mean_delta_w", "std_delta_w", "max_delta_w", "min_delta_w"]

    # Filter population to only include genomes with learning_rule_net defined
    population_with_rules: List[PreyGenome | PredatorGenome] = [g for g in population if getattr(g, 'learning_rule_net', None) is not None]
    if not population_with_rules:
        return

    # Create feature matrix X from learning rule net summary statistics
    X = np.array([[g.learning_rule_net.get_parameters_as_dict()[k] for k in rules] for g in population_with_rules])

    # Feed into StrategyClusteringLogger for longitudinal tracking
    StrategyClusteringLogger.log_generation_strategies([x for x in X], n_clusters=3)

    # Perform K-means clustering with 3 clusters
    labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)

    # Collect fitness per cluster
    fitness_per_cluster = {0: [], 1: [], 2: []}
    for i, genome in enumerate(population_with_rules):
        cluster = labels[i]
        fitness_per_cluster[cluster].append(genome.fitness)

    # Plot boxplot of fitness per cluster
    fig, ax = plt.subplots(figsize=(10, 6))

    cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']
    fitness_data = [fitness_per_cluster[0], fitness_per_cluster[1], fitness_per_cluster[2]]

    ax.boxplot(fitness_data, tick_labels=cluster_names, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red'),
               whiskerprops=dict(color='blue'),
               capprops=dict(color='blue'))

    ax.set_title(f'Strategy Clustering - Fitness per Cluster - Generation {generation}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'output_logs/strategy_clustering_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   📊 Saved: Strategy diversity/clustering analysis")
    print(f"      └─ output_logs/strategy_clustering_gen_{generation:04d}.png")


def plot_architecture_clustering(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]) -> None:
    """Plot fitness per cluster after clustering genomes by architecture features."""
    population: List[EvolvableGenome] = list(prey_population + predator_population)
    vectors = [g.get_architecture_vector() for g in population if hasattr(g, "get_architecture_vector")]
    if len(vectors) < 2:
        return

    X = np.array(vectors)
    k = min(3, len(vectors))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

    fitness_per_cluster: Dict[int, List[float]] = {i: [] for i in range(k)}
    for idx, genome in enumerate(population[:len(labels)]):
        fitness_per_cluster[int(labels[idx])].append(genome.fitness)

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_names = [f'Cluster {i}' for i in range(k)]
    fitness_data = [fitness_per_cluster[i] for i in range(k)]

    ax.boxplot(fitness_data, tick_labels=cluster_names, patch_artist=True,
               boxprops=dict(facecolor='lightgreen', color='green'),
               medianprops=dict(color='red'),
               whiskerprops=dict(color='green'),
               capprops=dict(color='green'))

    ax.set_title(f'Architecture Clustering - Fitness per Cluster - Generation {generation}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'output_logs/architecture_clustering_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   📊 Saved: Network architecture pattern analysis")
    print(f"      └─ output_logs/architecture_clustering_gen_{generation:04d}.png")


def plot_in_lifetime_learning_curve(generation: int, episode_data: Dict[str, List[float]]) -> None:
    """Plot in-lifetime learning curve: Reward vs time and Plastic change vs time"""
    steps = episode_data["steps"]
    rewards = episode_data["rewards"]
    delta_norms = episode_data["delta_norms"]
    plastic_changes = episode_data["plastic_changes"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Reward vs time
    ax1.plot(steps, rewards, 'b-', linewidth=2, label='Reward')
    ax1.set_title(f'In-Lifetime Learning Curve - Generation {generation}')
    ax1.set_xlabel('Episode Step')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot Plastic change vs time
    if delta_norms:
        ax2.plot(range(len(delta_norms)), delta_norms, 'r-', linewidth=2, label='Plastic Change (Δw norm)')
        ax2.set_title(f'Plastic Change vs Time - Generation {generation}')
        ax2.set_xlabel('Episode Step')
        ax2.set_ylabel('Plastic Change')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No Plastic Layers', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Plastic Change vs Time - Generation {generation}')

    plt.tight_layout()
    plt.savefig(f'output_logs/in_lifetime_learning_curve_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   📊 Saved: In-episode learning improvement curves")
    print(f"      └─ output_logs/in_lifetime_learning_curve_gen_{generation:04d}.png")


def evaluate_single_episode_with_logging(genome, seed: int, max_steps: int = 50) -> Dict[str, List[float]]:
    """Evaluate a single episode for a genome and log episode data for plotting"""
    from environments.deterministic_env import DeterministicVectorizedArena

    # Reset episode tracking
    if hasattr(genome, "brain"):
        genome.brain.reset_episode_tracking()

    env = DeterministicVectorizedArena(
        num_envs=1,  # Single environment
        max_steps=max_steps,
        seed=seed
    )

    state = env.reset()
    rewards = []
    steps = []

    for step in range(max_steps):
        # Get action
        action = genome.act(state[0])  # Single state
        actions = np.array([action])

        # Step environment
        next_state, step_reward, done = env.step(actions)

        # Update plasticity (gated)
        if hasattr(genome, "brain"):
            r = float(step_reward[0])
            if abs(r) > 0.05:
                genome.brain.update_plasticity(r, bool(done[0]))

        rewards.append(float(step_reward[0]))
        steps.append(step)

        state = next_state

        if done[0]:
            break

    env.close()

    # Get episode data
    if hasattr(genome, "brain"):
        episode_data = genome.brain.get_episode_data()
        return {
            "steps": steps,
            "rewards": rewards,
            "delta_norms": episode_data["delta_norms"],
            "plastic_changes": episode_data["rewards"]  # This is the modulated reward r
        }
    else:
        return {
            "steps": steps,
            "rewards": rewards,
            "delta_norms": [],
            "plastic_changes": []
        }


def evaluate_recovery_after_perturbation(
    genome,
    seed: int,
    max_steps: int = 50,
    shock_noise_std: float = 0.1,
    baseline_steps: int = 10,
    generation: int = -1,
) -> None:
    """Measure reward recovery after a weight perturbation shock.

    Runs `baseline_steps` steps to establish a pre-shock baseline reward,
    applies Gaussian noise to all plastic weights (simulating a perturbation),
    then runs the remaining steps and computes how quickly reward returns to
    80 % of the baseline. The result is logged via RewardRecoveryLogger.
    """
    import copy
    from environments.deterministic_env import DeterministicVectorizedArena
    from core.torch_brain import TorchBrain, PlasticLinear

    env = DeterministicVectorizedArena(num_envs=1, max_steps=max_steps, seed=seed)
    state = env.reset()

    pre_shock_rewards: List[float] = []
    post_shock_rewards: List[float] = []

    brain = genome.get_brain() if hasattr(genome, "get_brain") else getattr(genome, "brain", None)

    for step in range(max_steps):
        action = genome.act(state[0])
        next_state, step_reward, done = env.step(np.array([action]))
        r = float(step_reward[0])

        if step < baseline_steps:
            pre_shock_rewards.append(r)
        else:
            if step == baseline_steps and brain is not None:
                # Apply weight-noise shock to all plastic layers
                for layer in getattr(brain, "layers", []):
                    if isinstance(layer, PlasticLinear):
                        with __import__("torch").no_grad():
                            layer.plastic_weight.add_(
                                __import__("torch").randn_like(layer.plastic_weight) * shock_noise_std
                            )
            post_shock_rewards.append(r)

        if brain is not None and abs(r) > 0.05:
            brain.update_plasticity(r, bool(done[0]))

        state = next_state
        if done[0]:
            break

    env.close()

    pre_shock_mean = float(np.mean(pre_shock_rewards)) if pre_shock_rewards else 0.0
    recovery_time = RewardRecoveryLogger.calculate_recovery_time(pre_shock_mean, post_shock_rewards)
    RewardRecoveryLogger.log_recovery_time(recovery_time, generation=generation)

