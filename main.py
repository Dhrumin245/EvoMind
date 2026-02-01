
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
from curriculum_controller import CurriculumController
from evolution import EvolutionEngine
from async_evaluator import AsyncDeterministicEvaluator
from ppo_trainer import PPOTrainer, PPOConfig
# Import prey and predator genomes
from genome_prey import PreyGenome
from genome_predator import PredatorGenome, PredatorPackBrain
from genome import Genome as EvolvableGenome

# Import multi-task generalization harness
from multi_task_harness import (
    get_multi_task_evaluator, TaskSuite, GeneralizationReport,
    MultiTaskEvaluator, get_default_task_suite
)

# Import meta-scientist system
from meta_scientist import MetaScientist
from typing import Sequence

# Import meta-evolution populations
from evolution import ArchitectPopulation, MutatorPopulation

@dataclass
class EpisodeMetrics:
    """Structured decomposition of evaluation metrics for evolution and curriculum reasoning"""
    task_success: bool  # Whether the agent achieved the primary objective (e.g., survived, captured prey, collected food)
    episode_return: float  # Total accumulated reward over the episode
    learning_speed: float  # Rate of adaptation (plasticity effectiveness over time)
    stability: float  # Consistency of performance (variance in rewards/actions)
    energy_cost: float  # Total energy expended during episode
    complexity_penalty: float  # Penalty for overly complex behaviors (optional)
    novelty: float  # Novelty score (exploration of new strategies/behaviors)
    seed: int  # Random seed used for this evaluation
    stage: str  # Curriculum stage name
    opponent_id: Optional[str] = None  # ID of opponent genome (for co-evolution)
    saturation_penalty: Optional[float] = None  # Penalty for high saturation in networks
    dead_unit_penalty: Optional[float] = None  # Penalty for dead units in networks

@dataclass
class ExperimentReport:
    """Report for a self-directed ablation experiment"""
    generation: int
    experiment_name: str
    hypothesis: str
    baseline_fitness: float
    ablated_fitness: float
    fitness_delta: float
    genome_id: str
    genome_type: str  # 'prey' or 'predator'
    metrics: Dict[str, Any]  # Additional metrics from the experiment

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

    # Milestone 4: Speciation + novelty archive knobs
    speciation_enabled: bool = True
    speciation_compatibility_threshold: float = 3.0
    speciation_architecture_weight: float = 0.3
    speciation_behavior_weight: float = 0.4
    speciation_param_weight: float = 0.3
    speciation_min_species_size: int = 5
    speciation_max_stagnation: int = 15

    novelty_archive_enabled: bool = True
    novelty_threshold: float = 0.1
    novelty_max_archive_size: int = 100
    novelty_immigration_rate: float = 0.1
    novelty_archive_add_top_k: int = 5

    # Weight used by fitness shaping (currently EpisodeMetrics.novelty is a stub)
    novelty_weight: float = 0.2
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
    generalization_reports: List[GeneralizationReport] = field(default_factory=list)
    experiment_reports: List[ExperimentReport] = field(default_factory=list)
    
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
    stage: CurriculumStage,
    prey_engine: EvolutionEngine,
    predator_engine: EvolutionEngine,
    architect_population,
    mutator_population,
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
    combined_population = cast(List[Any], training_state.prey_population + training_state.predator_population)
    meta_gain = [g.meta["reward_gain"] for g in combined_population]
    meta_bias = [g.meta["reward_bias"] for g in combined_population]

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

    # Speciation + novelty logging (Milestone 4)
    try:
        prey_species = prey_engine.compute_species_stats(cast(Any, training_state.prey_population), generation)
        predator_species = predator_engine.compute_species_stats(cast(Any, training_state.predator_population), generation)
        stats['prey_species'] = prey_species
        stats['predator_species'] = predator_species
    except Exception:
        stats['prey_species'] = {'num_species': 0, 'avg_species_size': 0.0, 'total_members': 0, 'species_sizes': []}
        stats['predator_species'] = {'num_species': 0, 'avg_species_size': 0.0, 'total_members': 0, 'species_sizes': []}

    try:
        prey_novelty = prey_engine.compute_novelty_stats(
            cast(Any, training_state.prey_population),
            generation,
            add_top_k_to_archive=training_state.config.novelty_archive_add_top_k,
        )
        predator_novelty = predator_engine.compute_novelty_stats(
            cast(Any, training_state.predator_population),
            generation,
            add_top_k_to_archive=training_state.config.novelty_archive_add_top_k,
        )
        stats['prey_novelty'] = prey_novelty
        stats['predator_novelty'] = predator_novelty
    except Exception:
        stats['prey_novelty'] = {'mean': 0.0, 'max': 0.0, 'p95': 0.0, 'archive': {'size': 0, 'avg_fitness': 0.0, 'generations_covered': 0}}
        stats['predator_novelty'] = {'mean': 0.0, 'max': 0.0, 'p95': 0.0, 'archive': {'size': 0, 'avg_fitness': 0.0, 'generations_covered': 0}}
    
    # Log generation
    log_coevolution_generation(stats)

    # Integrate behavioral probes for comprehensive evaluation
    from behavioral_probes import BehavioralProbe
    # Create properly typed list for behavioral probes
    evolvable_genomes: List[EvolvableGenome] = list(combined_population)
    probe_integration_results = BehavioralProbe.integrate_with_evaluation_pipeline(
        evolvable_genomes,
        generation=generation,
        save_reports=True
    )

    # Add probe results to stats
    stats['behavioral_probes'] = probe_integration_results

    # META-EVOLUTION: Evolve architect and mutator populations
    # Prepare performance data for meta-evolution
    performance_data = {
        'avg_fitness': stats.get('mean_prey_fitness', 0.0) + stats.get('mean_predator_fitness', 0.0),
        'architecture_diversity': stats.get('prey_species', {}).get('num_species', 1) + stats.get('predator_species', {}).get('num_species', 1),
        'motif_effectiveness': stats.get('avg_adaptability_score', 0.0),
        'successful_architectures': training_state.prey_population[:5] + training_state.predator_population[:5],  # Top performers
        'mutation_success_rates': {
            'weight': 0.5,  # Placeholder - could be tracked from mutation logs
            'arch': 0.4,
            'layer': 0.3
        },
        'avg_fitness_improvement': 0.1,  # Placeholder
        'diversity_preservation': 0.8,  # Placeholder
        'exploration_success': 0.6  # Placeholder
    }

    # Evolve architect population
    architect_population.evolve_architectures(performance_data)

    # Evolve mutator population
    mutator_population.evolve_mutators(performance_data)

    # Use evolved mutation strategies to adapt main evolution engines
    adaptive_rates = mutator_population.get_adaptive_rates()
    if prey_engine and adaptive_rates:
        prey_engine.mutation_rate = adaptive_rates.get('weight_rate', prey_engine.mutation_rate)
        prey_engine.architecture_mutation_rate = adaptive_rates.get('arch_rate', prey_engine.architecture_mutation_rate)

    if predator_engine and adaptive_rates:
        predator_engine.mutation_rate = adaptive_rates.get('weight_rate', predator_engine.mutation_rate)
        predator_engine.architecture_mutation_rate = adaptive_rates.get('arch_rate', predator_engine.architecture_mutation_rate)

    # Log meta-evolution progress
    best_architect = architect_population.get_best_template()
    best_mutator = mutator_population.get_best_strategy()

    if best_architect:
        print(f"Meta-Evolution: Best architect fitness: {best_architect.get('meta_fitness', 0.0):.3f}")
    if best_mutator:
        print(f"Meta-Evolution: Best mutator effectiveness: {best_mutator.get('current_effectiveness', 0.0):.3f}")

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
    Returns: (prey_fitness, prey_metrics), (predator_fitness, predator_metrics)
    Enforces Condition A: Plastic agents outperform non-plastic agents within single lifetime
    """
    # Ensure TorchBrain instances exist for plasticity (using cached get_brain())
    prey_genome.get_brain()
    predator_genome.get_brain()

    # Generate arena seed for consistent experimental control
    arena_seed = np.random.randint(0, int(1e9))
    stage_name = stage_config.get('name', 'unknown') if stage_config else 'unknown'

    # CONDITION A: Compare plastic vs non-plastic performance within same episode (identical arena state)
    (plastic_prey_reward, plastic_prey_metrics), (plastic_pred_reward, plastic_pred_metrics) = evaluate_with_plasticity(
        prey_genome, predator_genome, arena, max_steps, seed=arena_seed, stage_name=stage_name
    )

    if not do_nonplastic_compare:
        # Compute fitness from metrics (weighted combination)
        prey_fitness = compute_fitness_from_metrics(plastic_prey_metrics)
        predator_fitness = compute_fitness_from_metrics(plastic_pred_metrics)
        return (prey_fitness, plastic_prey_metrics), (predator_fitness, plastic_pred_metrics)

    # Create non-plastic versions (disable plasticity updates) with same arena seed
    (nonplastic_prey_reward, nonplastic_prey_metrics), (nonplastic_pred_reward, nonplastic_pred_metrics) = evaluate_without_plasticity(
        prey_genome, predator_genome, arena, max_steps, seed=arena_seed, stage_name=stage_name
    )

    # Plastic agents must outperform non-plastic agents within the episode
    plastic_advantage_prey = plastic_prey_reward - nonplastic_prey_reward
    plastic_advantage_pred = plastic_pred_reward - nonplastic_pred_reward

    plasticity_bonus_prey = np.clip(plastic_advantage_prey, -5, 5)
    plasticity_bonus_pred = np.clip(plastic_advantage_pred, -5, 5)

    final_prey_reward = plastic_prey_reward + plasticity_bonus_prey
    final_pred_reward = plastic_pred_reward + plasticity_bonus_pred

    # Update metrics with final rewards and compute fitness
    plastic_prey_metrics.episode_return = final_prey_reward
    plastic_pred_metrics.episode_return = final_pred_reward

    prey_fitness = compute_fitness_from_metrics(plastic_prey_metrics)
    predator_fitness = compute_fitness_from_metrics(plastic_pred_metrics)

    return (prey_fitness, plastic_prey_metrics), (predator_fitness, plastic_pred_metrics)


def _to_numpy(x):
    """Convert torch tensors or other array-likes to numpy arrays."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def compute_fitness_from_metrics(metrics: EpisodeMetrics) -> float:
    """Compute scalar fitness from metrics decomposition"""
    # Weighted combination of metrics for fitness
    # Primary: episode return (main objective)
    # Secondary: success bonus, learning speed, stability penalty, energy efficiency
    fitness = metrics.episode_return

    # Success bonus
    if metrics.task_success:
        fitness += 1.0

    # Learning speed bonus (plasticity effectiveness)
    fitness += metrics.learning_speed * 0.1

    # Stability penalty (prefer consistent performance)
    fitness -= metrics.stability * 0.05

    # Energy efficiency bonus (lower cost is better)
    fitness += (1.0 / (1.0 + metrics.energy_cost)) * 0.5

    # Complexity penalty
    fitness -= metrics.complexity_penalty

    # Novelty bonus
    fitness += metrics.novelty * 0.2

    # Milestone 6: Add stability penalties to fitness shaping
    # Penalize networks with high saturation or dead units
    if metrics.saturation_penalty is not None:
        fitness -= float(metrics.saturation_penalty)
    if metrics.dead_unit_penalty is not None:
        fitness -= float(metrics.dead_unit_penalty)

    return float(fitness)

def evaluate_with_plasticity(prey_genome, predator_genome, arena, max_steps, seed=None, stage_name="unknown"):
    """Evaluate episode with plasticity enabled and return metrics"""
    # Reset plasticity and episode tracking before rollout
    prey_genome.brain.reset_plasticity()
    predator_genome.brain.reset_plasticity()
    prey_genome.brain.reset_episode_tracking()
    predator_genome.brain.reset_episode_tracking()

    prey_state, pred_state = arena.reset(seed=seed)

    prey_total_reward = 0.0
    predator_total_reward = 0.0
    prey_rewards = []
    predator_rewards = []
    prey_energy_cost = 0.0
    predator_energy_cost = 0.0
    steps_survived = 0

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
        prey_rewards.append(prey_r)
        predator_rewards.append(pred_r)

        # Track energy costs from info
        if 'prey_energy' in info:
            prey_energy_cost += np.mean(info['prey_energy'])
        if 'predator_energy' in info:
            predator_energy_cost += np.mean(info['predator_energy'])

        steps_survived += 1

        if np.any(info['env_done']):
            break

    # Finalize episode plasticity logging (log once per episode, not per step)
    if hasattr(prey_genome.brain, 'finalize_episode_plastic_norms'):
        prey_genome.brain.finalize_episode_plastic_norms()
    if hasattr(predator_genome.brain, 'finalize_episode_plastic_norms'):
        predator_genome.brain.finalize_episode_plastic_norms()

    # Compute learning speed from plasticity diagnostics
    prey_learning_speed = 0.0
    if hasattr(prey_genome.brain, 'get_plastic_diagnostics'):
        plastic_diag = prey_genome.brain.get_plastic_diagnostics()
        if 'total_plastic_delta' in plastic_diag:
            prey_learning_speed = float(plastic_diag['total_plastic_delta'])

    # Compute metrics
    prey_metrics = EpisodeMetrics(
        task_success=prey_total_reward > 0,  # Basic success: positive reward
        episode_return=prey_total_reward,
        learning_speed=prey_learning_speed,
        stability=float(np.std(prey_rewards)) if prey_rewards else 0.0,
        energy_cost=prey_energy_cost,
        complexity_penalty=0.0,  # TODO: implement complexity measure
        novelty=0.0,  # TODO: compute novelty score
        seed=seed or 0,
        stage=stage_name,
        opponent_id=predator_genome.genome_id if hasattr(predator_genome, 'genome_id') else None
    )

    # Compute learning speed from plasticity diagnostics
    predator_learning_speed = 0.0
    if hasattr(predator_genome.brain, 'get_plastic_diagnostics'):
        plastic_diag = predator_genome.brain.get_plastic_diagnostics()
        if 'total_plastic_delta' in plastic_diag:
            predator_learning_speed = float(plastic_diag['total_plastic_delta'])

    predator_metrics = EpisodeMetrics(
        task_success=predator_total_reward > 0,
        episode_return=predator_total_reward,
        learning_speed=predator_learning_speed,
        stability=float(np.std(predator_rewards)) if predator_rewards else 0.0,
        energy_cost=predator_energy_cost,
        complexity_penalty=0.0,
        novelty=0.0,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=prey_genome.genome_id if hasattr(prey_genome, 'genome_id') else None
    )

    return (prey_total_reward, prey_metrics), (predator_total_reward, predator_metrics)

def evaluate_without_plasticity(prey_genome, predator_genome, arena, max_steps, seed=None, stage_name="unknown"):
    """Evaluate episode with plasticity disabled and return metrics"""
    # Reset plasticity and episode tracking but don't update plasticity during episode
    prey_genome.brain.reset_plasticity()
    predator_genome.brain.reset_plasticity()
    prey_genome.brain.reset_episode_tracking()
    predator_genome.brain.reset_episode_tracking()

    prey_state, pred_state = arena.reset(seed=seed)

    prey_total_reward = 0.0
    predator_total_reward = 0.0
    prey_rewards = []
    predator_rewards = []
    prey_energy_cost = 0.0
    predator_energy_cost = 0.0
    steps_survived = 0

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
        prey_rewards.append(float(np.mean(prey_reward)))
        predator_rewards.append(float(np.mean(pred_reward)))

        # Track energy costs from info
        if 'prey_energy' in info:
            prey_energy_cost += np.mean(info['prey_energy'])
        if 'predator_energy' in info:
            predator_energy_cost += np.mean(info['predator_energy'])

        steps_survived += 1

        if np.any(info['env_done']):
            break

    # Finalize episode plasticity logging (log once per episode, not per step)
    if hasattr(prey_genome.brain, 'finalize_episode_plastic_norms'):
        prey_genome.brain.finalize_episode_plastic_norms()
    if hasattr(predator_genome.brain, 'finalize_episode_plastic_norms'):
        predator_genome.brain.finalize_episode_plastic_norms()

    # Compute metrics (no learning speed for non-plastic)
    prey_metrics = EpisodeMetrics(
        task_success=prey_total_reward > 0,
        episode_return=prey_total_reward,
        learning_speed=0.0,  # No plasticity updates
        stability=float(np.std(prey_rewards)) if prey_rewards else 0.0,
        energy_cost=prey_energy_cost,
        complexity_penalty=0.0,
        novelty=0.0,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=predator_genome.genome_id if hasattr(predator_genome, 'genome_id') else None
    )

    predator_metrics = EpisodeMetrics(
        task_success=predator_total_reward > 0,
        episode_return=predator_total_reward,
        learning_speed=0.0,
        stability=float(np.std(predator_rewards)) if predator_rewards else 0.0,
        energy_cost=predator_energy_cost,
        complexity_penalty=0.0,
        novelty=0.0,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=prey_genome.genome_id if hasattr(prey_genome, 'genome_id') else None
    )

    return (prey_total_reward, prey_metrics), (predator_total_reward, predator_metrics)

def log_coevolution_generation(stats: Dict[str, Any]):
    """Log co-evolution generation statistics with metrics decomposition"""
    print(f"{'='*80}")
    print(f"Generation {stats['generation']:04d} - {stats['stage']}")
    print(f"{'-'*80}")
    print(f"Prey Fitness:    Best: {stats['best_prey_fitness']:8.2f} | Mean: {stats['mean_prey_fitness']:8.2f}")
    print(f"Predator Fitness: Best: {stats['best_predator_fitness']:8.2f} | Mean: {stats['mean_predator_fitness']:8.2f}")
    print(f"Evaluation Time: {stats['eval_time']:6.2f}s")
    print(f"Population: {stats['prey_population_size']} prey, {stats['predator_population_size']} predators")

    # Log metrics decomposition summary
    if 'avg_adaptability_score' in stats:
        print(f"Adaptability: {stats['avg_adaptability_score']:.3f} | Meta Effectiveness: {stats['avg_meta_effectiveness']:.3f}")
    if 'mean_plastic_norm' in stats:
        print(f"Plastic Norms: Mean {stats['mean_plastic_norm']:.4f} | Max {stats['max_plastic_norm']:.4f} | 95th {stats['p95_plastic_norm']:.4f}")

    # Milestone 4: speciation + novelty summary
    prey_species = stats.get('prey_species')
    predator_species = stats.get('predator_species')
    if isinstance(prey_species, dict) and isinstance(predator_species, dict):
        print(
            "Speciation: "
            f"prey {prey_species.get('num_species', 0)} species (avg {prey_species.get('avg_species_size', 0.0):.1f}), "
            f"pred {predator_species.get('num_species', 0)} species (avg {predator_species.get('avg_species_size', 0.0):.1f})"
        )

    prey_novelty = stats.get('prey_novelty')
    predator_novelty = stats.get('predator_novelty')
    if isinstance(prey_novelty, dict) and isinstance(predator_novelty, dict):
        prey_arch = prey_novelty.get('archive', {}) if isinstance(prey_novelty.get('archive', {}), dict) else {}
        pred_arch = predator_novelty.get('archive', {}) if isinstance(predator_novelty.get('archive', {}), dict) else {}
        print(
            "Novelty: "
            f"prey mean {prey_novelty.get('mean', 0.0):.3f} (p95 {prey_novelty.get('p95', 0.0):.3f}) archive {prey_arch.get('size', 0)}, "
            f"pred mean {predator_novelty.get('mean', 0.0):.3f} (p95 {predator_novelty.get('p95', 0.0):.3f}) archive {pred_arch.get('size', 0)}"
        )

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

# Milestone 7: Meta-scientist experiment runners
def run_ablation_frozen_learning_rule(genome, generation: int, stage_config, max_steps: int) -> ExperimentReport:
    """Run ablation experiment with frozen learning rule network"""
    # Create a copy of the genome with frozen learning rule
    ablated_genome = genome.copy()
    if hasattr(ablated_genome.brain, 'freeze_learning_rule'):
        ablated_genome.brain.freeze_learning_rule()

    # Evaluate baseline (normal genome)
    baseline_fitness, _ = evaluate_multi_agent_pair(
        genome, genome,  # Use same genome as opponent for simplicity
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(genome, 'genome_type') and genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    # Evaluate ablated genome
    ablated_fitness, _ = evaluate_multi_agent_pair(
        ablated_genome, ablated_genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(ablated_genome, 'genome_type') and ablated_genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        ablated_genome, ablated_genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    return ExperimentReport(
        generation=generation,
        experiment_name="frozen_learning_rule",
        hypothesis="Freezing the learning rule network will reduce adaptability",
        baseline_fitness=float(baseline_fitness),
        ablated_fitness=float(ablated_fitness),
        fitness_delta=float(ablated_fitness - baseline_fitness),
        genome_id=genome.genome_id,
        genome_type=getattr(genome, 'genome_type', 'unknown'),
        metrics={"learning_rule_frozen": True}
    )

def run_ablation_frozen_architecture(genome, generation: int, stage_config, max_steps: int) -> ExperimentReport:
    """Run ablation experiment with frozen architecture mutations"""
    # Create a copy of the genome with frozen architecture
    ablated_genome = genome.copy()
    if hasattr(ablated_genome, 'freeze_architecture'):
        ablated_genome.freeze_architecture()

    # Evaluate baseline
    baseline_fitness, _ = evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(genome, 'genome_type') and genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    # Evaluate ablated genome
    ablated_fitness, _ = evaluate_multi_agent_pair(
        ablated_genome, ablated_genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(ablated_genome, 'genome_type') and ablated_genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        ablated_genome, ablated_genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    return ExperimentReport(
        generation=generation,
        experiment_name="frozen_architecture",
        hypothesis="Freezing architecture mutations will limit exploration",
        baseline_fitness=float(baseline_fitness),
        ablated_fitness=float(ablated_fitness),
        fitness_delta=float(ablated_fitness - baseline_fitness),
        genome_id=genome.genome_id,
        genome_type=getattr(genome, 'genome_type', 'unknown'),
        metrics={"architecture_frozen": True}
    )

def run_ablation_disabled_plasticity(genome, generation: int, stage_config, max_steps: int) -> ExperimentReport:
    """Run ablation experiment with plasticity disabled"""
    # Evaluate baseline (with plasticity)
    baseline_fitness, _ = evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(genome, 'genome_type') and genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    # Evaluate without plasticity
    ablated_fitness, _ = evaluate_without_plasticity(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        max_steps, seed=generation, stage_name=stage_config.get('name', 'unknown')
    )[0] if hasattr(genome, 'genome_type') and genome.genome_type == 'prey' else evaluate_without_plasticity(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        max_steps, seed=generation, stage_name=stage_config.get('name', 'unknown')
    )[1]

    return ExperimentReport(
        generation=generation,
        experiment_name="disabled_plasticity",
        hypothesis="Disabling plasticity will reduce in-lifetime adaptation",
        baseline_fitness=float(baseline_fitness),
        ablated_fitness=float(ablated_fitness),
        fitness_delta=float(ablated_fitness - baseline_fitness),
        genome_id=genome.genome_id,
        genome_type=getattr(genome, 'genome_type', 'unknown'),
        metrics={"plasticity_disabled": True}
    )

def run_ablation_reward_weights(genome, generation: int, stage_config, max_steps: int) -> ExperimentReport:
    """Run ablation experiment with modified reward weights"""
    # Create modified reward weights (e.g., inverted success bonus)
    original_weights = getattr(genome, 'reward_weights', {'success': 1.0, 'energy': 0.5, 'stability': 0.05})
    modified_weights = {k: -v for k, v in original_weights.items()}  # Invert all weights

    # Create ablated genome with modified weights
    ablated_genome = genome.copy()
    if hasattr(ablated_genome, 'set_reward_weights'):
        ablated_genome.set_reward_weights(modified_weights)

    # Evaluate baseline
    baseline_fitness, _ = evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(genome, 'genome_type') and genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        genome, genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    # Evaluate ablated genome
    ablated_fitness, _ = evaluate_multi_agent_pair(
        ablated_genome, ablated_genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[0] if hasattr(ablated_genome, 'genome_type') and ablated_genome.genome_type == 'prey' else evaluate_multi_agent_pair(
        ablated_genome, ablated_genome,
        MultiAgentArena(batch_size=1, num_prey_per_env=1, num_predators_per_env=1),
        stage_config, max_steps, do_nonplastic_compare=False
    )[1]

    return ExperimentReport(
        generation=generation,
        experiment_name="inverted_reward_weights",
        hypothesis="Inverting reward weights will degrade performance",
        baseline_fitness=float(baseline_fitness),
        ablated_fitness=float(ablated_fitness),
        fitness_delta=float(ablated_fitness - baseline_fitness),
        genome_id=genome.genome_id,
        genome_type=getattr(genome, 'genome_type', 'unknown'),
        metrics={"original_weights": original_weights, "modified_weights": modified_weights}
    )

def save_coevolution_state(training_state: TrainingState, filename: str = "coevolution_state.json"):
    """Save complete co-evolution training state"""
    state = {
        'generation': training_state.generation,
        'config': training_state.config.__dict__,
        'best_prey_fitness_history': training_state.best_prey_fitness_history,
        'best_predator_fitness_history': training_state.best_predator_fitness_history,
        'generation_stats': training_state.generation_stats,
        'experiment_reports': [exp.__dict__ for exp in training_state.experiment_reports],
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
    print(f"  Preserved genome metadata: parent_ids, birth_generation, mutation_history")

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
    training_state.generalization_reports = [
        r if isinstance(r, GeneralizationReport) else GeneralizationReport(**r)
        for r in state.get('generalization_reports', [])
    ]
    
    print(f"Co-evolution state loaded: {filename}")
    print(f"Generation: {training_state.generation}")
    print(f"Prey: {len(training_state.prey_population)}, Predators: {len(training_state.predator_population)}")
    
    return training_state

async def main_coevolution_async():
    """Main async co-evolution training loop with adaptive curriculum"""
    print("Starting Co-Evolution Training with Adaptive Curriculum")
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

    # Initialize curriculum controller
    curriculum_controller = CurriculumController()

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
        speciation_enabled=config.speciation_enabled,
        novelty_archive_enabled=config.novelty_archive_enabled,
        compatibility_threshold=config.speciation_compatibility_threshold,
        speciation_architecture_weight=config.speciation_architecture_weight,
        speciation_behavior_weight=config.speciation_behavior_weight,
        speciation_param_weight=config.speciation_param_weight,
        min_species_size=config.speciation_min_species_size,
        max_species_stagnation=config.speciation_max_stagnation,
        novelty_threshold=config.novelty_threshold,
        max_archive_size=config.novelty_max_archive_size,
        immigration_rate=config.novelty_immigration_rate,
    )
    predator_engine = EvolutionEngine(
        population_size=config.predator_population_size,
        tournament_size=config.tournament_size,
        elite_count=config.elite_count,
        mutation_rate=config.mutation_rate,
        mutation_strength=config.mutation_strength,
        architecture_mutation_rate=config.architecture_mutation_rate,
        genome_cls=PredatorGenome
        ,
        speciation_enabled=config.speciation_enabled,
        novelty_archive_enabled=config.novelty_archive_enabled,
        compatibility_threshold=config.speciation_compatibility_threshold,
        speciation_architecture_weight=config.speciation_architecture_weight,
        speciation_behavior_weight=config.speciation_behavior_weight,
        speciation_param_weight=config.speciation_param_weight,
        min_species_size=config.speciation_min_species_size,
        max_species_stagnation=config.speciation_max_stagnation,
        novelty_threshold=config.novelty_threshold,
        max_archive_size=config.novelty_max_archive_size,
        immigration_rate=config.novelty_immigration_rate,
    )

    # Initialize meta-evolution populations
    architect_population = ArchitectPopulation(population_size=20)
    mutator_population = MutatorPopulation(population_size=15)

    # Training loop
    for generation in range(training_state.generation, config.generations):
        # Get current stage from curriculum controller
        current_stage = curriculum_controller.get_current_config()
        stage_name = current_stage['name']
        stage_config = get_stage_config(CurriculumStage[stage_name.upper()])

        # Control diagnostics - only enable every N generations
        if config.plot_every > 0 and generation % config.plot_every == 0:
            evaluator.enable_diagnostics = True
        else:
            evaluator.enable_diagnostics = False

        # Co-evolution training step
        stats = await train_coevolution_async(
            generation,
            training_state,
            evaluator,
            CurriculumStage[stage_name.upper()],
            prey_engine,
            predator_engine,
            architect_population,
            mutator_population,
        )
        training_state.generation_stats.append(stats)

        # Update hall of fame
        training_state.update_hall_of_fame()

        # Compute diversity score from population
        from population import Population
        combined_population = Population(size=0)
        combined_population.genomes = training_state.prey_population + training_state.predator_population
        diversity_score = combined_population.get_diversity_score()

        # Compute success rate (fraction of positive fitness scores)
        all_fitnesses = [g.fitness for g in training_state.prey_population + training_state.predator_population]
        success_rate = float(np.mean([1.0 if f > 0 else 0.0 for f in all_fitnesses]))

        # Update curriculum controller with performance metrics
        population_stats = {
            'mean': float(np.mean(all_fitnesses)),
            'max': float(max(all_fitnesses)),
            'min': float(min(all_fitnesses)),
            'std': float(np.std(all_fitnesses))
        }

        new_stage = curriculum_controller.update(population_stats, diversity_score, success_rate)

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

            # Milestone 7: Run integrated meta-scientist experiments
            print("Running integrated meta-scientist experiments...")

            # Initialize meta-scientist system
            meta_scientist = MetaScientist()

            # Analyze population failures and generate hypotheses
            combined_population = cast(List[EvolvableGenome], training_state.prey_population + training_state.predator_population)
            task_info = {'name': stage_name, 'generation': generation}
            
            analysis_results = meta_scientist.analyze_population_failures(
                combined_population,
                task_info
            )

            # Run automated experiments based on hypotheses
            experiment_results = meta_scientist.run_automated_experiments(
                analysis_results['hypotheses'],
                combined_population,
                task_info,
                generation
            )

            # Store experiment reports and learn from results
            for exp in experiment_results:
                # Convert experiment record to ExperimentReport format
                exp_report = ExperimentReport(
                    generation=generation,
                    experiment_name=exp.get('experiment_design', {}).get('experiment_type', 'unknown'),
                    hypothesis=exp.get('hypothesis', 'unknown'),
                    baseline_fitness=0.0,  # Simulated experiments don't have real fitness
                    ablated_fitness=0.0,
                    fitness_delta=0.0,
                    genome_id="N/A",
                    genome_type="simulated",
                    metrics=exp.get('result', {})
                )
                training_state.experiment_reports.append(exp_report)

            # Log experiment results
            print(f"Meta-scientist experiments completed for generation {generation}:")
            for exp in experiment_results:
                hypothesis = exp.get('hypothesis', 'unknown')[:50]
                result = exp.get('result', {}).get('hypothesis_supported', False)
                print(f"  {hypothesis}... -> {'SUPPORTED' if result else 'NOT SUPPORTED'}")

        training_state.generation += 1

    # Final save
    save_coevolution_state(training_state, "final_coevolution_state.json")
    evaluator.save_seeds("final_seed_registry.json")

    # Close evaluator
    evaluator.close()

    print("\nCo-evolution training completed!")
    print(f"Best prey fitness: {max(training_state.best_prey_fitness_history):.2f}")
    print(f"Best predator fitness: {max(training_state.best_predator_fitness_history):.2f}")
    print(f"Final curriculum stage: {curriculum_controller.get_current_config()['name']}")

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
