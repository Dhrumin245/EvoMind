
import os

# Configure threadpools BEFORE importing NumPy/Torch.
# This prevents CPU oversubscription and huge slowdowns on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import warnings
import numpy as np
import asyncio
import json
import time
import concurrent.futures
import threading
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from typing import Dict, Any, Optional, List, cast
from dataclasses import dataclass, field
from enum import Enum

try:
    import torch

    # In serial evaluation, allowing a few Torch threads can be faster.
    # (We still cap MKL/OMP above to keep note-worthy slowdowns away.)
    cpu_count = os.cpu_count() or 4
    torch_threads = max(1, min(4, cpu_count))
    torch.set_num_threads(torch_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)
except Exception:
    # Torch might not be installed or may fail to import in some setups.
    pass

print("Event loop:", asyncio.get_event_loop_policy())

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Import multi-agent modules
from arena_multi import MultiAgentArena
from self_play import evaluate_self_play, evolve_population

# Import other modules
from curriculum import CurriculumStage, get_stage_config
from evolution import EvolutionEngine
from async_evaluator import AsyncDeterministicEvaluator
from ppo_trainer import PPOTrainer, PPOConfig
# Import prey and predator genomes
from genome_prey import PreyGenome
from genome_predator import PredatorGenome, PredatorPackBrain

# Import multi-task generalization harness
from multi_task_harness import (
    get_multi_task_evaluator, TaskSuite, GeneralizationReport,
    MultiTaskEvaluator, get_default_task_suite
)

@dataclass
class EvolutionConfig:
    """Configuration dataclass to avoid global mutable state"""
    population_size: int = 100
    predator_population_size: int = 80
    generations: int = 1000
    tournament_size: int = 5
    elite_count: int = 5
    mutation_rate: float = 0.001
    mutation_strength: float = 0.1
    architecture_mutation_rate: float = 0.01
    base_seed: int = 42
    envs_per_genome: int = 8
    batch_size: int = 2
    num_workers: int = 8
    max_steps: int = 80
    # Evaluation performance controls
    # NOTE: Threaded evaluation is often slower on Windows (oversubscription/GIL)
    # and is not safe unless opponents are deep-copied (brains are mutated per rollout).
    use_threaded_eval: bool = False
    # Fraction of evaluations that also run the non-plastic rollout.
    # Setting this to 0.0 makes evaluation ~2x faster.
    nonplastic_check_fraction: float = 0.0
    # Plotting/diagnostics are expensive; do them every N generations.
    plot_every: int = 20
    # Multi-agent specific
    num_prey_per_arena: int = 10
    num_predators_per_arena: int = 3
    # Co-evolution parameters
    num_opponents_per_eval: int = 1
    hall_of_fame_size: int = 10
    # PPO inner-loop training
    ppo_training_steps: int = 100  # Number of PPO training steps per genome
    enable_ppo_inner_loop: bool = True  # Enable/disable PPO training before evaluation
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.population_size > 0, "Population size must be positive"
        assert self.predator_population_size > 0, "Predator population size must be positive"
        assert self.mutation_rate >= 0 and self.mutation_rate <= 1, "Mutation rate must be between 0 and 1"

class EvolutionMode(Enum):
    SINGLE_AGENT = "single"
    CO_EVOLUTION = "coevolution"
    SELF_PLAY = "selfplay"

@dataclass
class TrainingState:
    """Encapsulates the entire training state"""
    config: EvolutionConfig
    generation: int = 0
    prey_population: List[PreyGenome] = field(default_factory=list)
    predator_population: List[PredatorGenome] = field(default_factory=list)
    prey_hall_of_fame: List[PreyGenome] = field(default_factory=list)
    predator_hall_of_fame: List[PredatorGenome] = field(default_factory=list)
    best_prey_fitness_history: List[float] = field(default_factory=list)
    best_predator_fitness_history: List[float] = field(default_factory=list)
    generation_stats: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_hall_of_fame(self):
        """Update hall of fame with best individuals"""
        if self.prey_population:
            best_prey = max(self.prey_population, key=lambda g: g.fitness)
            self.prey_hall_of_fame.append(cast(PreyGenome, best_prey.copy()))
            self.prey_hall_of_fame = sorted(
                self.prey_hall_of_fame,
                key=lambda g: g.fitness, 
                reverse=True
            )[:self.config.hall_of_fame_size]

        if self.predator_population:
            best_predator = max(self.predator_population, key=lambda g: g.fitness)
            self.predator_hall_of_fame.append(cast(PredatorGenome, best_predator.copy()))
            self.predator_hall_of_fame = sorted(
                self.predator_hall_of_fame,
                key=lambda g: g.fitness,
                reverse=True
            )[:self.config.hall_of_fame_size]

def select_multi_agent_stage(generation: int, config: EvolutionConfig) -> CurriculumStage:
    """Curriculum stage selection for multi-agent co-evolution"""
    if generation < 200:
        return CurriculumStage.FORAGING  # Basic movement
    elif generation < 500:
        return CurriculumStage.PRECISION  # Refined control
    elif generation < 800:
        return CurriculumStage.SCARCITY  # Multi-agent coordination
    elif generation < 1200:
        return CurriculumStage.THREAT  # Direct competition
    else:
        return CurriculumStage.ADVERSARIAL  # Advanced pack dynamics

def evaluate_population_parallel(
    population,
    opponents,
    stage_config,
    max_steps,
    num_opponents,
    is_prey_evaluation,
    batch_size,
    num_prey_per_env,
    num_predators_per_env,
    num_workers,
    nonplastic_check_fraction: float,
):
    """Parallel evaluation per-genome using a thread pool.
    Each worker owns its own arena to avoid shared-state contention.
    """
    fitnesses: List[float] = []

    # One arena per worker thread (created lazily) to avoid per-genome arena init cost.
    thread_local = threading.local()

    def _eval_one(genome):
        arena = getattr(thread_local, "arena", None)
        if arena is None:
            thread_local.arena = MultiAgentArena(
                batch_size=batch_size,
                num_prey_per_env=num_prey_per_env,
                num_predators_per_env=num_predators_per_env,
            )
            arena = thread_local.arena

        opp_indices = np.random.choice(len(opponents), size=num_opponents, replace=False)
        genome_fitnesses = []
        for opp_idx in opp_indices:
            opponent = opponents[opp_idx]
            if is_prey_evaluation:
                fitness, _ = evaluate_multi_agent_pair(
                    genome,
                    opponent,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=(np.random.random() < nonplastic_check_fraction),
                )
            else:
                _, fitness = evaluate_multi_agent_pair(
                    opponent,
                    genome,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=(np.random.random() < nonplastic_check_fraction),
                )
            genome_fitnesses.append(fitness)

        avg_fitness = float(np.mean(genome_fitnesses)) if genome_fitnesses else 0.0
        genome.fitness = avg_fitness
        return avg_fitness

    # Small populations fall back to serial evaluation to avoid thread overhead
    if len(population) == 0:
        return fitnesses

    # Heuristic: keep worker count modest to avoid fighting BLAS/OpenMP threadpools.
    # User can still raise EvolutionConfig.num_workers if their setup benefits.
    cpu_count = os.cpu_count() or 4
    worker_count = max(1, min(int(num_workers), max(1, cpu_count // 2)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        for fit in executor.map(_eval_one, population):
            fitnesses.append(fit)

    return fitnesses


def evaluate_population_serial(
    population,
    opponents,
    arena,
    stage_config,
    max_steps,
    num_opponents,
    is_prey_evaluation,
    nonplastic_check_fraction: float,
    enable_ppo_inner_loop: bool = True,
    ppo_training_steps: int = 100,
):
    """Serial evaluation that reuses a single arena instance.

    This is typically fastest and most stable on Windows because:
    - NumPy/PyTorch already use native threads internally
    - Opponent brains are mutated during rollouts (thread sharing is unsafe)
    """
    fitnesses: List[float] = []
    if len(population) == 0 or len(opponents) == 0 or num_opponents <= 0:
        return fitnesses

    # Initialize PPO trainer if needed
    ppo_trainer = None
    if enable_ppo_inner_loop:
        ppo_config = PPOConfig(num_steps=ppo_training_steps)
        ppo_trainer = PPOTrainer(ppo_config)

    for genome in population:
        # PPO inner-loop training before evaluation
        if enable_ppo_inner_loop and ppo_trainer is not None:
            print(f"  Training genome {genome.genome_id} with PPO for {ppo_training_steps} steps...")

            # Create environment function for PPO training
            def env_fn():
                return arena  # Use the same arena instance

            # Train with PPO
            training_stats = ppo_trainer.train(genome.brain, env_fn, num_steps=ppo_training_steps)
            print(f"    PPO training completed: final reward {training_stats['final_reward']:.2f}")

        opp_indices = np.random.choice(len(opponents), size=num_opponents, replace=False)
        genome_fitnesses = []
        for opp_idx in opp_indices:
            opponent = opponents[opp_idx]
            if is_prey_evaluation:
                fitness, _ = evaluate_multi_agent_pair(
                    genome,
                    opponent,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=(np.random.random() < nonplastic_check_fraction),
                )
            else:
                _, fitness = evaluate_multi_agent_pair(
                    opponent,
                    genome,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=(np.random.random() < nonplastic_check_fraction),
                )
            genome_fitnesses.append(fitness)

        avg_fitness = float(np.mean(genome_fitnesses)) if genome_fitnesses else 0.0
        genome.fitness = avg_fitness
        fitnesses.append(avg_fitness)

    return fitnesses

async def train_coevolution_async(
    generation: int,
    training_state: TrainingState,
    evaluator: AsyncDeterministicEvaluator,
    stage: CurriculumStage
) -> Dict[str, Any]:
    """
    Async training step for co-evolution
    """
    print(f"\n Generation {generation} - Stage: {stage.name}")
    print(f" Prey Population: {len(training_state.prey_population)}")
    print(f" Predator Population: {len(training_state.predator_population)}")
    
    # Get stage configuration
    stage_config = get_stage_config(stage)
    
    # Evaluate co-evolution
    start_time = time.time()
    
    # Combine hall of fame with current population for more challenging opponents
    all_prey = training_state.prey_population + training_state.prey_hall_of_fame
    all_predators = training_state.predator_population + training_state.predator_hall_of_fame
    
    # Sample opponents for evaluation
    num_prey_opponents = min(training_state.config.num_opponents_per_eval, len(all_predators))
    num_pred_opponents = min(training_state.config.num_opponents_per_eval, len(all_prey))

    # Reuse a single arena for serial evaluation (batch_size=1 for single prey-predator pair).
    arena = MultiAgentArena(
        batch_size=1,
        num_prey_per_env=1,
        num_predators_per_env=1,
    )
    
    # Evaluate prey population
    print("  Evaluating prey population...")
    if training_state.config.use_threaded_eval:
        prey_fitnesses = evaluate_population_parallel(
            training_state.prey_population,
            all_predators,
            stage_config,
            training_state.config.max_steps,
            num_prey_opponents,
            is_prey_evaluation=True,
            batch_size=training_state.config.batch_size,
            num_prey_per_env=training_state.config.num_prey_per_arena,
            num_predators_per_env=training_state.config.num_predators_per_arena,
            num_workers=training_state.config.num_workers,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
        )
    else:
        prey_fitnesses = evaluate_population_serial(
            training_state.prey_population,
            all_predators,
            arena,
            stage_config,
            training_state.config.max_steps,
            num_prey_opponents,
            is_prey_evaluation=True,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
        )

    # Evaluate predator population
    print("  Evaluating predator population...")
    if training_state.config.use_threaded_eval:
        predator_fitnesses = evaluate_population_parallel(
            training_state.predator_population,
            all_prey,
            stage_config,
            training_state.config.max_steps,
            num_pred_opponents,
            is_prey_evaluation=False,
            batch_size=training_state.config.batch_size,
            num_prey_per_env=training_state.config.num_prey_per_arena,
            num_predators_per_env=training_state.config.num_predators_per_arena,
            num_workers=training_state.config.num_workers,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
        )
    else:
        predator_fitnesses = evaluate_population_serial(
            training_state.predator_population,
            all_prey,
            arena,
            stage_config,
            training_state.config.max_steps,
            num_pred_opponents,
            is_prey_evaluation=False,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
        )
    
    eval_time = time.time() - start_time
    
    # Update best fitness history
    best_prey_fitness = max(prey_fitnesses) if prey_fitnesses else 0.0
    best_predator_fitness = max(predator_fitnesses) if predator_fitnesses else 0.0
    
    training_state.best_prey_fitness_history.append(best_prey_fitness)
    training_state.best_predator_fitness_history.append(best_predator_fitness)

    # Log META gene distribution
    meta_gain = [g.meta["reward_gain"] for g in training_state.prey_population + training_state.predator_population]
    meta_bias = [g.meta["reward_bias"] for g in training_state.prey_population + training_state.predator_population]

    # Log plastic weight norms per generation (sample once, not per update)
    from torch_brain import PlasticLinear
    # Only collect from the last evaluation to avoid O(steps × layers × population) growth
    plastic_norms = PlasticLinear.plastic_norms.copy()
    PlasticLinear.plastic_norms.clear()  # Clear for next generation

    if plastic_norms:
        mean_plastic_norm = float(np.mean(plastic_norms))
        max_plastic_norm = float(np.max(plastic_norms))
        p95_plastic_norm = float(np.percentile(plastic_norms, 95))
        print(f"Plastic Weight Norms - Mean: {mean_plastic_norm:.4f} | Max: {max_plastic_norm:.4f} | 95th: {p95_plastic_norm:.4f}")
    else:
        mean_plastic_norm = 0.0
        max_plastic_norm = 0.0
        p95_plastic_norm = 0.0

    # Calculate META-3.2 metrics: adaptability scores and meta-parameter effectiveness
    adaptability_scores = []
    meta_effectiveness_scores = []

    for genome in training_state.prey_population + training_state.predator_population:
        if hasattr(genome, 'plastic_diagnostics') and genome.plastic_diagnostics:
            plastic_usage = genome.plastic_diagnostics.get('mean_final_plastic_delta', 0.0)

            # Calculate adaptability score (same logic as in evolution.py)
            if plastic_usage > 0:
                plasticity_efficiency = 1.0 - abs(plastic_usage - 0.3) / 0.3
                plasticity_efficiency = max(0.0, plasticity_efficiency)
            else:
                plasticity_efficiency = 0.0

            local_meta_gain = genome.meta.get('reward_gain', 1.0)
            local_meta_bias = genome.meta.get('reward_bias', 0.0)
            plastic_lr = genome.meta.get('plastic_lr', 1.0)

            meta_coherence = 0.0
            if plastic_usage > 0.1:
                lr_effectiveness = min(plastic_lr / 10.0, 1.0) * (plastic_usage / 0.5)
                gain_effectiveness = min(abs(local_meta_gain) / 5.0, 1.0)
                meta_coherence = (lr_effectiveness + gain_effectiveness) / 2.0

            stability = 1.0 - abs(genome.plastic_diagnostics.get('mean_final_plastic_delta', 0.0))
            stability_bonus = stability * 0.3

            adaptability_score = (
                plasticity_efficiency * 0.4 +
                meta_coherence * 0.4 +
                stability_bonus * 0.2
            )
            adaptability_scores.append(float(max(0.0, min(1.0, adaptability_score))))

            # Meta-parameter effectiveness (how well meta-params correlate with performance)
            meta_effectiveness = min(abs(local_meta_gain) / 5.0, 1.0) * min(plastic_lr / 10.0, 1.0)
            meta_effectiveness_scores.append(float(meta_effectiveness))

    avg_adaptability = float(np.mean(adaptability_scores)) if adaptability_scores else 0.0
    avg_meta_effectiveness = float(np.mean(meta_effectiveness_scores)) if meta_effectiveness_scores else 0.0

    # Calculate statistics
    stats = {
        'generation': generation,
        'stage': stage.name,
        'best_prey_fitness': best_prey_fitness,
        'best_predator_fitness': best_predator_fitness,
        'mean_prey_fitness': float(np.mean(prey_fitnesses)) if prey_fitnesses else 0.0,
        'mean_predator_fitness': float(np.mean(predator_fitnesses)) if predator_fitnesses else 0.0,
        'eval_time': eval_time,
        'prey_population_size': len(training_state.prey_population),
        'predator_population_size': len(training_state.predator_population),
        'meta_gain': meta_gain,
        'meta_bias': meta_bias,
        'mean_plastic_norm': mean_plastic_norm,
        'max_plastic_norm': max_plastic_norm,
        'p95_plastic_norm': p95_plastic_norm,
        'avg_adaptability_score': avg_adaptability,
        'avg_meta_effectiveness': avg_meta_effectiveness,
        'config': stage_config
    }
    
    # Log generation
    log_coevolution_generation(stats)
    
    return stats



def evaluate_multi_agent_pair(
    prey_genome,
    predator_genome,
    arena,
    stage_config,
    max_steps,
    do_nonplastic_compare: bool = True,
):
    """
    Evaluate a single prey-predator pair in multi-agent arena with META-4 PRESSURE INJECTION
    Enforces Condition A: Plastic agents outperform non-plastic agents within single lifetime
    """
    # Ensure TorchBrain instances exist for plasticity (using cached get_brain())
    prey_genome.get_brain()
    predator_genome.get_brain()

    # Generate arena seed for consistent experimental control
    arena_seed = np.random.randint(0, int(1e9))

    # CONDITION A: Compare plastic vs non-plastic performance within same episode (identical arena state)
    plastic_prey_reward, plastic_pred_reward = evaluate_with_plasticity(
        prey_genome, predator_genome, arena, max_steps, seed=arena_seed
    )

    if not do_nonplastic_compare:
        return plastic_prey_reward, plastic_pred_reward

    # Create non-plastic versions (disable plasticity updates) with same arena seed
    nonplastic_prey_reward, nonplastic_pred_reward = evaluate_without_plasticity(
        prey_genome, predator_genome, arena, max_steps, seed=arena_seed
    )

    # Plastic agents must outperform non-plastic agents within the episode
    plastic_advantage_prey = plastic_prey_reward - nonplastic_prey_reward
    plastic_advantage_pred = plastic_pred_reward - nonplastic_pred_reward

    plasticity_bonus_prey = np.clip(plastic_advantage_prey, -5, 5)
    plasticity_bonus_pred = np.clip(plastic_advantage_pred, -5, 5)

    final_prey_reward = plastic_prey_reward + plasticity_bonus_prey
    final_pred_reward = plastic_pred_reward + plasticity_bonus_pred

    return final_prey_reward, final_pred_reward


def _to_numpy(x):
    """Convert torch tensors or other array-likes to numpy arrays."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def evaluate_with_plasticity(prey_genome, predator_genome, arena, max_steps, seed=None):
    """Evaluate episode with plasticity enabled"""
    # Reset plasticity and episode tracking before rollout
    prey_genome.brain.reset_plasticity()
    predator_genome.brain.reset_plasticity()
    prey_genome.brain.reset_episode_tracking()
    predator_genome.brain.reset_episode_tracking()

    prey_state, pred_state = arena.reset(seed=seed)

    prey_total_reward = 0.0
    predator_total_reward = 0.0

    predator_brain = PredatorPackBrain(predator_genome)

    for step in range(max_steps):
        # Get actions from genomes
        prey_actions = prey_genome.act_batch(prey_state)

        # Use predator pack brain for coordinated actions
        pred_actions = predator_brain.act(pred_state)

        # Step the arena
        (prey_state, pred_state), r_prey, r_pred, info = arena.step(
            prey_actions, pred_actions
        )

        # Use rewards directly (no inverted multiplier that destroys signal)
        prey_reward = _to_numpy(r_prey)
        pred_reward = _to_numpy(r_pred)

        # Gate: only update plasticity for informative reward signals.
        # (Also reduces overhead by skipping near-zero step penalties.)
        prey_r = float(np.mean(prey_reward))
        pred_r = float(np.mean(pred_reward))
        if abs(prey_r) > 0.05:
            prey_genome.brain.update_plasticity(prey_r, done=False)
        if abs(pred_r) > 0.05:
            predator_genome.brain.update_plasticity(pred_r, done=False)

        prey_total_reward += float(np.sum(prey_reward))
        predator_total_reward += float(np.sum(pred_reward))

        if np.any(info['env_done']):
            break

    # Finalize episode plasticity logging (log once per episode, not per step)
    if hasattr(prey_genome.brain, 'finalize_episode_plastic_norms'):
        prey_genome.brain.finalize_episode_plastic_norms()
    if hasattr(predator_genome.brain, 'finalize_episode_plastic_norms'):
        predator_genome.brain.finalize_episode_plastic_norms()

    return prey_total_reward, predator_total_reward

def evaluate_without_plasticity(prey_genome, predator_genome, arena, max_steps, seed=None):
    """Evaluate episode with plasticity disabled (no updates)"""
    # Reset plasticity and episode tracking but don't update plasticity during episode
    prey_genome.brain.reset_plasticity()
    predator_genome.brain.reset_plasticity()
    prey_genome.brain.reset_episode_tracking()
    predator_genome.brain.reset_episode_tracking()

    prey_state, pred_state = arena.reset(seed=seed)

    prey_total_reward = 0.0
    predator_total_reward = 0.0

    # Create predator brain once, outside the loop
    predator_brain = PredatorPackBrain(predator_genome)

    for step in range(max_steps):
        # Get actions from genomes (plasticity not updated)
        prey_actions = prey_genome.act_batch(prey_state)

        # Use predator pack brain for coordinated actions
        pred_actions = predator_brain.act(pred_state)

        # Step the arena
        (prey_state, pred_state), r_prey, r_pred, info = arena.step(
            prey_actions, pred_actions
        )

        # Use rewards directly (no pressure injection multiplier)
        prey_reward = _to_numpy(r_prey)
        pred_reward = _to_numpy(r_pred)

        # NO PLASTICITY UPDATES - this is the key difference

        prey_total_reward += float(np.sum(prey_reward))
        predator_total_reward += float(np.sum(pred_reward))

        if np.any(info['env_done']):
            break

    # Finalize episode plasticity logging (log once per episode, not per step)
    if hasattr(prey_genome.brain, 'finalize_episode_plastic_norms'):
        prey_genome.brain.finalize_episode_plastic_norms()
    if hasattr(predator_genome.brain, 'finalize_episode_plastic_norms'):
        predator_genome.brain.finalize_episode_plastic_norms()

    return prey_total_reward, predator_total_reward

def log_coevolution_generation(stats: Dict[str, Any]):
    """Log co-evolution generation statistics"""
    print(f"{'='*80}")
    print(f"Generation {stats['generation']:04d} - {stats['stage']}")
    print(f"{'-'*80}")
    print(f"Prey Fitness:    Best: {stats['best_prey_fitness']:8.2f} | Mean: {stats['mean_prey_fitness']:8.2f}")
    print(f"Predator Fitness: Best: {stats['best_predator_fitness']:8.2f} | Mean: {stats['mean_predator_fitness']:8.2f}")
    print(f"Evaluation Time: {stats['eval_time']:6.2f}s")
    print(f"Population: {stats['prey_population_size']} prey, {stats['predator_population_size']} predators")
    print(f"{'='*80}")

def plot_meta_gene_histograms(stats: Dict[str, Any]):
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
    plt.savefig(f'meta_gene_distribution_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"META gene histograms saved: meta_gene_distribution_gen_{generation:04d}.png")

def plot_plastic_norm_evolution(generation_stats: List[Dict[str, Any]]):
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

    # Plot max plastic norm (optional)
    ax.plot(generations, max_norms, 'r--', linewidth=1.5, label='Max Plastic Norm', alpha=0.7)

    ax.set_title('Plastic Weight Norm Evolution')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Plastic Weight Norm')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('plastic_norm_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plastic norm evolution plot saved: plastic_norm_evolution.png")

def plot_learning_rule_stats(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]):
    """Plot learning rule parameter distributions per generation"""
    population = prey_population + predator_population
    rules = ["A", "B", "C", "D", "E"]

    for k in rules:
        vals = [g.learning_rule[k] for g in population if g.learning_rule is not None]
        plt.hist(vals, bins=30)
        plt.axvline(float(np.mean(vals).item()), color='r')
        plt.title(f"Learning Rule {k} — Gen {generation}")
        plt.show()

def plot_learning_rule_vs_fitness(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]):
    """Plot scatter plots of learning rule parameters vs fitness for each gene"""
    population = prey_population + predator_population
    rules = ["A", "B", "C", "D", "E"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, rule in enumerate(rules):
        x = [g.learning_rule[rule] for g in population if g.learning_rule is not None]
        y = [g.fitness for g in population if g.learning_rule is not None]
        axes[i].scatter(x, y, alpha=0.6)
        axes[i].set_xlabel(rule)
        axes[i].set_ylabel("Fitness")
        axes[i].set_title(f"Learning Rule {rule} vs Fitness - Gen {generation}")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'learning_rule_vs_fitness_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Learning rule vs fitness scatter plots saved: learning_rule_vs_fitness_gen_{generation:04d}.png")

def plot_strategy_clustering(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]):
    """Plot fitness per cluster after clustering genomes by learning rule strategies (A, B, C, D, E)"""
    population = prey_population + predator_population
    rules = ["A", "B", "C", "D", "E"]

    # Filter population to only include genomes with learning_rule defined
    population_with_rules: List[PreyGenome | PredatorGenome] = [g for g in population if g.learning_rule is not None]
    if not population_with_rules:
        return
    
    # Create feature matrix X from learning rules
    X = np.array([[cast(Dict[str, float], g.learning_rule)[k] for k in rules] for g in population_with_rules])

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

    ax.boxplot(fitness_data, label=cluster_names, patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='blue'),
               medianprops=dict(color='red'),
               whiskerprops=dict(color='blue'),
               capprops=dict(color='blue'))

    ax.set_title(f'Strategy Clustering - Fitness per Cluster - Generation {generation}')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Fitness')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'strategy_clustering_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Strategy clustering plot saved: strategy_clustering_gen_{generation:04d}.png")

def evaluate_single_episode_with_logging(genome, seed: int, max_steps: int = 50) -> Dict[str, List[float]]:
    """Evaluate a single episode for a genome and log episode data for plotting"""
    from deterministic_env import DeterministicVectorizedArena

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

def plot_in_lifetime_learning_curve(generation: int, episode_data: Dict[str, List[float]]):
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
    plt.savefig(f'in_lifetime_learning_curve_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"In-lifetime learning curve plot saved: in_lifetime_learning_curve_gen_{generation:04d}.png")

def save_coevolution_state(training_state: TrainingState, filename: str = "coevolution_state.json"):
    """Save complete co-evolution training state"""
    state = {
        'generation': training_state.generation,
        'config': training_state.config.__dict__,
        'best_prey_fitness_history': training_state.best_prey_fitness_history,
        'best_predator_fitness_history': training_state.best_predator_fitness_history,
        'generation_stats': training_state.generation_stats,
    }
    
    # Save prey population
    state['prey_population'] = []
    for i, genome in enumerate(training_state.prey_population):
        genome_data = genome.to_dict()
        genome_data['id'] = i
        state['prey_population'].append(genome_data)
    
    # Save predator population
    state['predator_population'] = []
    for i, genome in enumerate(training_state.predator_population):
        genome_data = genome.to_dict()
        genome_data['id'] = i
        state['predator_population'].append(genome_data)
    
    # Save hall of fame
    state['prey_hall_of_fame'] = [g.to_dict() for g in training_state.prey_hall_of_fame]
    state['predator_hall_of_fame'] = [g.to_dict() for g in training_state.predator_hall_of_fame]
    
    with open(filename, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print(f"Co-evolution state saved: {filename}")

def load_coevolution_state(filename: str = "coevolution_state.json") -> TrainingState:
    """Load co-evolution training state"""
    with open(filename, 'r') as f:
        state = json.load(f)
    
    # Create config
    config_dict = state['config']
    config = EvolutionConfig(**config_dict)
    
    # Create training state
    training_state = TrainingState(config=config)
    training_state.generation = state['generation']
    
    # Load prey population
    for genome_data in state['prey_population']:
        genome = PreyGenome.from_dict(genome_data)
        training_state.prey_population.append(genome)
    
    # Load predator population
    for genome_data in state['predator_population']:
        genome = PredatorGenome.from_dict(genome_data)
        training_state.predator_population.append(genome)
    
    # Load hall of fame
    for genome_data in state['prey_hall_of_fame']:
        genome = PreyGenome.from_dict(genome_data)
        training_state.prey_hall_of_fame.append(genome)
    
    for genome_data in state['predator_hall_of_fame']:
        genome = PredatorGenome.from_dict(genome_data)
        training_state.predator_hall_of_fame.append(genome)
    
    # Load history
    training_state.best_prey_fitness_history = state['best_prey_fitness_history']
    training_state.best_predator_fitness_history = state['best_predator_fitness_history']
    training_state.generation_stats = state['generation_stats']
    
    print(f"Co-evolution state loaded: {filename}")
    print(f"Generation: {training_state.generation}")
    print(f"Prey: {len(training_state.prey_population)}, Predators: {len(training_state.predator_population)}")
    
    return training_state

async def main_coevolution_async():
    """Main async co-evolution training loop"""
    print("Starting Co-Evolution Training")
    print("=" * 60)
    
    # Initialize configuration
    config = EvolutionConfig()

    # Optional runtime overrides for quick perf testing / CI runs.
    # Example (PowerShell):
    #   $env:MAX_GENERATIONS=1; $env:AUTO_LOAD_COEVOLUTION_STATE='n'; python main.py
    max_gens_env = os.getenv("MAX_GENERATIONS")
    if max_gens_env:
        try:
            config.generations = int(max_gens_env)
        except ValueError:
            pass
    
    # Initialize training state
    training_state = TrainingState(config=config)
    
    # Initialize populations
    print("Initializing populations...")
    training_state.prey_population = [
        PreyGenome.random_initialization() 
        for _ in range(config.population_size)
    ]
    training_state.predator_population = [
        PredatorGenome.random_initialization()
        for _ in range(config.predator_population_size)
    ]
    
    # Initialize evaluator
    evaluator = AsyncDeterministicEvaluator(
        base_seed=config.base_seed,
        num_workers=config.num_workers,
        use_gpu=False,
        envs_per_genome=config.envs_per_genome,
        max_steps=config.max_steps
    )
    
    # Load checkpoint if exists
    if os.path.exists("coevolution_state.json"):
        response = os.getenv("AUTO_LOAD_COEVOLUTION_STATE")
        if response is None:
            response = input("Co-evolution state found. Load? (y/n): ")
        if response.lower() == 'y':
            training_state = load_coevolution_state()
            evaluator.load_seeds("seed_registry.json")
    
    # Initialize evolution engines
    prey_engine = EvolutionEngine(
        population_size=config.population_size,
        tournament_size=config.tournament_size,
        elite_count=config.elite_count,
        mutation_rate=config.mutation_rate,
        mutation_strength=config.mutation_strength,
        architecture_mutation_rate=config.architecture_mutation_rate,
        genome_cls=PreyGenome,
    )
    predator_engine = EvolutionEngine(
        population_size=config.predator_population_size,
        tournament_size=config.tournament_size,
        elite_count=config.elite_count,
        mutation_rate=config.mutation_rate,
        mutation_strength=config.mutation_strength,
        architecture_mutation_rate=config.architecture_mutation_rate,
        genome_cls=PredatorGenome
    )
    
    # Training loop
    for generation in range(training_state.generation, config.generations):
        stage = select_multi_agent_stage(generation, config)
        
        # Control diagnostics - only enable every N generations
        if config.plot_every > 0 and generation % config.plot_every == 0:
            evaluator.enable_diagnostics = True
        else:
            evaluator.enable_diagnostics = False
        
        # Co-evolution training step
        stats = await train_coevolution_async(
            generation, training_state, evaluator, stage
        )
        training_state.generation_stats.append(stats)
        
        # Update hall of fame
        training_state.update_hall_of_fame()
        
        # Stagnation detection
        if len(training_state.best_prey_fitness_history) > 50:
            recent_prey = training_state.best_prey_fitness_history[-50:]
            recent_predator = training_state.best_predator_fitness_history[-50:]
            
            if max(recent_prey) - min(recent_prey) < 0.1:
                print("Prey stagnation detected - adjusting mutation...")
                config.mutation_rate = min(config.mutation_rate * 1.5, 0.05)
                config.architecture_mutation_rate = min(config.architecture_mutation_rate * 2, 0.1)
            
            if max(recent_predator) - min(recent_predator) < 0.1:
                print("Predator stagnation detected - adjusting mutation...")
                config.mutation_strength = min(config.mutation_strength * 1.2, 0.5)
        
        # Evolve populations separately
        print("Evolving populations...")

        if prey_engine:
            prey_population = prey_engine.create_next_generation(
                training_state.prey_population, generation
            )
            training_state.prey_population = prey_population.genomes

        if predator_engine:
            predator_population = predator_engine.create_next_generation(
                training_state.predator_population , generation
            )
            training_state.predator_population = predator_population.genomes

        # Save checkpoint and run diagnostics
        if config.plot_every > 0 and generation % config.plot_every == 0:
            save_coevolution_state(training_state)
            evaluator.save_seeds()
            plot_meta_gene_histograms(stats)
            plot_plastic_norm_evolution(training_state.generation_stats)
            plot_learning_rule_stats(generation, training_state.prey_population, training_state.predator_population)
            plot_learning_rule_vs_fitness(generation, training_state.prey_population, training_state.predator_population)
            plot_strategy_clustering(generation, training_state.prey_population, training_state.predator_population)

            # In-Lifetime Learning Curve: Pick top genome and log during single episode
            top_prey_genome = max(training_state.prey_population, key=lambda g: g.fitness)
            episode_data = evaluate_single_episode_with_logging(top_prey_genome, seed=generation, max_steps=config.max_steps)
            plot_in_lifetime_learning_curve(generation, episode_data)

        training_state.generation += 1
    
    # Final save
    save_coevolution_state(training_state, "final_coevolution_state.json")
    evaluator.save_seeds("final_seed_registry.json")
    
    # Close evaluator
    evaluator.close()
    
    print("\nCo-evolution training completed!")
    print(f"Best prey fitness: {max(training_state.best_prey_fitness_history):.2f}")
    print(f"Best predator fitness: {max(training_state.best_predator_fitness_history):.2f}")

async def main_async():
    """Placeholder for single-agent async training"""
    print("Single-agent async training not implemented in this version")
    print("Use --evolution-type multi for co-evolution")

def main():
    """Placeholder for single-agent sync training"""
    print("Single-agent sync training not implemented in this version")
    print("Use --evolution-type multi for co-evolution")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evolution Arena Training")
    parser.add_argument("--mode", choices=["async", "sync", "coevolution"], default="coevolution",
                       help="Training mode (async, sync, or coevolution)")
    parser.add_argument("--evolution-type", choices=["single", "multi"], default="multi",
                       help="Evolution type (single-agent or multi-agent)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for determinism")
    parser.add_argument("--population", type=int, default=100,
                       help="Prey population size")
    parser.add_argument("--predator-population", type=int, default=80,
                       help="Predator population size")
    parser.add_argument("--generations", type=int, default=1000,
                       help="Number of generations")
    
    args = parser.parse_args()
    
    # Update configuration using dataclass
    config = EvolutionConfig(
        base_seed=args.seed,
        population_size=args.population,
        predator_population_size=args.predator_population,
        generations=args.generations
    )
    
    if args.evolution_type == "multi":
        # Run co-evolution
        if args.mode == "async":
            asyncio.run(main_coevolution_async())
        else:
            print("Co-evolution currently supports async mode only")
            asyncio.run(main_coevolution_async())
    else:
        # Run single-agent evolution (original code)
        if args.mode == "async":
            asyncio.run(main_async())
        else:
            main()
