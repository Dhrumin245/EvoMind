
import os
import signal

# Configure threadpools BEFORE importing NumPy/Torch.
# This prevents CPU oversubscription and huge slowdowns on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1") 
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Python 3.13 propagates KeyboardInterrupt through exec() in dataclasses.
# VS Code terminals can send stray SIGINT on attach, crashing torch import.
# Temporarily ignore SIGINT while torch loads its dataclass-heavy internals.
_prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
import torch
signal.signal(signal.SIGINT, _prev_sigint)
import random
import warnings
import logging
import numpy as np
import asyncio
import json
import time
import concurrent.futures
import threading
from collections import OrderedDict
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, Any, Optional, List, cast
from dataclasses import dataclass, field
from enum import Enum

# Fitness cache to avoid re-evaluating identical genome pairs.
_FITNESS_CACHE: "OrderedDict[tuple, Dict[str, float]]" = OrderedDict()
_FITNESS_CACHE_LOCK = threading.Lock()

_ACTIVE_CONFIG: Optional["EvolutionConfig"] = None
_ACTIVE_GENERATION: int = 0


def _make_fitness_cache_key(prey_genome, predator_genome, stage_config, max_steps: int, do_nonplastic_compare: bool) -> tuple:
    prey_sig = getattr(prey_genome, "signature", None) or getattr(prey_genome, "genome_id", "unknown_prey")
    pred_sig = getattr(predator_genome, "signature", None) or getattr(predator_genome, "genome_id", "unknown_pred")
    stage_key = json.dumps(stage_config, sort_keys=True, default=str) if stage_config else "none"
    return (prey_sig, pred_sig, stage_key, int(max_steps), bool(do_nonplastic_compare))

def _get_cached_fitness(cache_key: tuple) -> Optional[Dict[str, float]]:
    with _FITNESS_CACHE_LOCK:
        cached = _FITNESS_CACHE.get(cache_key)
        if cached is not None:
            _FITNESS_CACHE.move_to_end(cache_key)
        return cached

def _set_cached_fitness(cache_key: tuple, value: Dict[str, float], max_size: int) -> None:
    with _FITNESS_CACHE_LOCK:
        _FITNESS_CACHE[cache_key] = value
        _FITNESS_CACHE.move_to_end(cache_key)
        while len(_FITNESS_CACHE) > max_size:
            _FITNESS_CACHE.popitem(last=False)

def _get_effective_max_steps(config: "EvolutionConfig", stage_config: Optional[Dict[str, Any]], generation: int) -> int:
    stage_max_steps = int(stage_config.get("max_steps", config.max_steps)) if stage_config else config.max_steps
    max_steps = min(int(config.max_steps), stage_max_steps)
    if config.reduce_episode_length_early and generation < config.early_curriculum_generations:
        max_steps = max(1, int(max_steps * config.early_curriculum_reduction_factor))
    return max_steps

def _get_opponent_sample_size(num_opponents: int, partial_enabled: bool, partial_fraction: float) -> int:
    if not partial_enabled:
        return num_opponents
    return max(1, int(round(num_opponents * partial_fraction)))

try:
    import torch

    # In serial evaluation, allowing a few Torch threads can be faster.
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
from environments.arena_multi import MultiAgentArena

# Import other modules
from curriculum.curriculum import CurriculumStage, get_stage_config
from curriculum.curriculum_controller import CurriculumController
from evolution.evolution import EvolutionEngine
from core.async_evaluator import AsyncDeterministicEvaluator
# Import prey and predator genomes
from genomes.genome_prey import PreyGenome
from genomes.genome_predator import PredatorGenome, PredatorPackBrain
from core.genome import Genome as EvolvableGenome, NeuralGene
from core.torch_brain import TorchBrain

# Import multi-task generalization harness
from meta.multi_task_harness import (
    get_multi_task_evaluator, TaskSuite, GeneralizationReport,
    MultiTaskEvaluator, get_default_task_suite, TaskType, BenchmarkResult
)

# Import meta-scientist system
from meta.meta_scientist import MetaScientist
from curriculum.task_generator import DiagnosticTaskGenerator
from meta.meta_optimizer import EvolutionModifier

# Import meta-evolution populations
from evolution.evolution import ArchitectPopulation, MutatorPopulation

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
    adaptability: float = 0.0  # Reward delta from lifetime learning (plastic vs non-plastic)
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
    architecture_mutation_rate: float = 0.05
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
    # Higher values improve adaptability measurement but slow evaluation.
    nonplastic_check_fraction: float = 0.25
    # Plotting/diagnostics are expensive; do them every N generations.
    plot_every: int = 50  # Reduced from 20 to 50 for performance

    # Multi-agent specific
    num_prey_per_arena: int = 10
    num_predators_per_arena: int = 3
    # Co-evolution parameters
    num_opponents_per_eval: int = 1
    hall_of_fame_size: int = 10

    # Milestone 4: Speciation + novelty archive knobs
    speciation_enabled: bool = True
    speciation_compatibility_threshold: float = 0.85
    speciation_compatibility_decay_rate: float = 400.0
    speciation_architecture_weight: float = 0.6
    speciation_behavior_weight: float = 0.2
    speciation_param_weight: float = 0.2
    speciation_min_species_size: int = 5
    speciation_max_stagnation: int = 15
    speciation_min_offspring_per_species: int = 5
    speciation_target_species_min: int = 4
    speciation_target_species_max: int = 7
    speciation_adjust_rate: float = 0.05
    speciation_threshold_min: float = 0.1
    speciation_threshold_max: float = 5.0

    # Prey-specific diversity controls
    prey_speciation_compatibility_threshold: float = 0.75
    prey_novelty_weight: float = 0.6
    prey_min_species_enforcement: int = 2
    prey_min_species_adjust_rate: float = 0.2

    # Predator-specific diversity controls (Issue 3: Fix predator monoculture)
    predator_speciation_compatibility_threshold: float = 0.75
    predator_novelty_weight: float = 0.6
    predator_min_species_enforcement: int = 2
    predator_min_species_adjust_rate: float = 0.2

    # Cross-species reproduction rate (5-10% recommended to maintain speciation)
    cross_species_reproduction_rate: float = 0.05

    novelty_archive_enabled: bool = True

    novelty_threshold: float = 0.1
    novelty_max_archive_size: int = 100
    novelty_immigration_rate: float = 0.1
    novelty_archive_add_top_k: int = 5

    # Weight used by fitness shaping (currently EpisodeMetrics.novelty is a stub)
    novelty_weight: float = 0.5
    novelty_fitness_beta: float = 0.2

    # Adaptability pressure schedule (favor learning early, then taper)
    adaptability_weight_base: float = 0.3
    adaptability_weight_boost: float = 1.0
    adaptability_boost_generations: int = 15
    adaptability_taper_generations: int = 10
    # PPO inner-loop training - DISABLED: Contradicts NeuroGenesis philosophy
    ppo_training_steps: int = 100  # Number of PPO training steps per genome
    enable_ppo_inner_loop: bool = False  # PERMANENTLY DISABLED: Evolution discovers learning rules, not gradients

    # === Performance Optimization Parameters ===
    # Early stopping: terminate episode if score is catastrophically low
    early_stopping_enabled: bool = True  # Enable early stopping
    early_stopping_threshold: float = -30.0  # Catastrophic low score threshold
    early_stopping_patience: int = 5  # Steps to wait before terminating

    # Curriculum-aware episode length: reduce steps in early generations
    reduce_episode_length_early: bool = True  # Enable reduced episode length
    early_curriculum_reduction_factor: float = 0.5  # Multiply max_steps by this in early generations
    early_curriculum_generations: int = 50  # Number of generations to apply reduction

    # Partial evaluation: evaluate on subset of seeds
    partial_evaluation_enabled: bool = True  # Enable partial evaluation
    partial_eval_fraction: float = 0.5  # Fraction of seeds to evaluate (0.0-1.0)

    # Fitness caching: cache fitness for identical genomes
    fitness_cache_enabled: bool = True  # Enable fitness caching
    fitness_cache_max_size: int = 1000  # Maximum cache size

    def __post_init__(self):
        """Validate configuration"""
        assert self.population_size > 0, "Population size must be positive"
        assert self.predator_population_size > 0, "Predator population size must be positive"
        assert self.mutation_rate >= 0 and self.mutation_rate <= 1, "Mutation rate must be between 0 and 1"
        # Validate optimization parameters
        assert -100 <= self.early_stopping_threshold <= 0, "Early stopping threshold must be between -100 and 0"
        assert 0 < self.early_curriculum_reduction_factor <= 1, "Reduction factor must be between 0 and 1"
        assert 0 < self.partial_eval_fraction <= 1, "Partial eval fraction must be between 0 and 1"
        assert self.speciation_target_species_min >= 1, "Speciation target min must be >= 1"
        assert self.speciation_target_species_max >= self.speciation_target_species_min, "Speciation target max must be >= min"
        assert 0 < self.speciation_adjust_rate <= 1, "Speciation adjust rate must be in (0, 1]"
        assert 0 < self.speciation_threshold_min <= self.speciation_threshold_max, "Speciation threshold bounds invalid"
        assert self.prey_speciation_compatibility_threshold > 0, "Prey speciation threshold must be > 0"
        assert 0 <= self.prey_novelty_weight <= 1, "Prey novelty weight must be between 0 and 1"
        assert self.prey_min_species_enforcement >= 1, "Prey min species enforcement must be >= 1"
        assert 0 < self.prey_min_species_adjust_rate <= 1, "Prey min species adjust rate must be in (0, 1]"
        assert self.predator_speciation_compatibility_threshold > 0, "Predator speciation threshold must be > 0"
        assert 0 <= self.predator_novelty_weight <= 1, "Predator novelty weight must be between 0 and 1"
        assert self.predator_min_species_enforcement >= 1, "Predator min species enforcement must be >= 1"
        assert 0 < self.predator_min_species_adjust_rate <= 1, "Predator min species adjust rate must be in (0, 1]"
        assert 0.0 <= self.cross_species_reproduction_rate <= 1.0, "Cross-species reproduction rate must be between 0.0 and 1.0"

        assert self.adaptability_weight_base >= 0, "Adaptability base weight must be >= 0"

        assert self.adaptability_weight_boost >= 0, "Adaptability boost weight must be >= 0"
        assert self.adaptability_boost_generations >= 0, "Adaptability boost generations must be >= 0"
        assert self.adaptability_taper_generations >= 0, "Adaptability taper generations must be >= 0"

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

def _select_primary_failure_diagnosis(failure_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not failure_data:
        return None

    return max(
        failure_data,
        key=lambda f: f.get('diagnosis', {}).get('total_severity', 0.0)
    ).get('diagnosis')

def _map_target_capability_to_task_types(target_capability: Optional[str]) -> List[TaskType]:
    mapping = {
        'architectural_capacity': [TaskType.ARENA_CONFIG, TaskType.CURRICULUM_STAGE],
        'learning_plasticity': [TaskType.CURRICULUM_STAGE],
        'exploration_balance': [TaskType.FOOD_DISTRIBUTION],
        'temporal_credit_assignment': [TaskType.PREDATOR_PREY_RATIO],
        'learning_stability': [TaskType.SENSOR_NOISE, TaskType.ENERGY_DYNAMICS],
        'generalization_ability': [TaskType.ARENA_CONFIG, TaskType.SENSOR_NOISE],
    }
    return mapping.get(target_capability or '', [TaskType.CURRICULUM_STAGE])

def _build_targeted_task_suite(target_capability: Optional[str], generation: int, max_tasks: int = 6) -> TaskSuite:
    base_suite = get_default_task_suite()
    task_types = _map_target_capability_to_task_types(target_capability)

    tasks = [task for task in base_suite.tasks if task.task_type in task_types]
    if not tasks:
        tasks = base_suite.sample_tasks(max_tasks, seed=generation)

    if len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]

    return TaskSuite(tasks=tasks, base_seed=base_suite.base_seed)

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
    partial_evaluation_enabled: bool,
    partial_eval_fraction: float,
    fitness_cache_enabled: bool,
    fitness_cache_max_size: int,
    early_stopping_enabled: bool,
    early_stopping_threshold: float,
    early_stopping_patience: int,
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

        effective_opponents = _get_opponent_sample_size(
            num_opponents,
            partial_evaluation_enabled,
            partial_eval_fraction,
        )
        effective_opponents = min(effective_opponents, len(opponents))
        opp_indices = np.random.choice(len(opponents), size=effective_opponents, replace=False)
        genome_fitnesses = []
        for opp_idx in opp_indices:
            opponent = opponents[opp_idx]
            do_nonplastic = np.random.random() < nonplastic_check_fraction
            cache_key = None
            if fitness_cache_enabled:
                prey = genome if is_prey_evaluation else opponent
                predator = opponent if is_prey_evaluation else genome
                cache_key = _make_fitness_cache_key(prey, predator, stage_config, max_steps, do_nonplastic)
                cached = _get_cached_fitness(cache_key)
                if cached is not None:
                    fitness = cached["prey_fitness"] if is_prey_evaluation else cached["predator_fitness"]
                    genome_fitnesses.append(float(fitness))
                    continue

            if is_prey_evaluation:
                (prey_result, pred_result) = evaluate_multi_agent_pair(
                    genome,
                    opponent,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=do_nonplastic,
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_threshold=early_stopping_threshold,
                    early_stopping_patience=early_stopping_patience,
                )
                fitness = prey_result[0]
            else:
                (prey_result, pred_result) = evaluate_multi_agent_pair(
                    opponent,
                    genome,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=do_nonplastic,
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_threshold=early_stopping_threshold,
                    early_stopping_patience=early_stopping_patience,
                )
                fitness = pred_result[0]
            genome_fitnesses.append(fitness)

            if fitness_cache_enabled and cache_key is not None:
                _set_cached_fitness(
                    cache_key,
                    {
                        "prey_fitness": float(prey_result[0]),
                        "predator_fitness": float(pred_result[0]),
                    },
                    max_size=fitness_cache_max_size,
                )

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
    partial_evaluation_enabled: bool,
    partial_eval_fraction: float,
    fitness_cache_enabled: bool,
    fitness_cache_max_size: int,
    early_stopping_enabled: bool,
    early_stopping_threshold: float,
    early_stopping_patience: int,
):
    """Serial evaluation that reuses a single arena instance.

    This is typically fastest and most stable on Windows because:
    - NumPy/PyTorch already use native threads internally
    - Opponent brains are mutated during rollouts (thread sharing is unsafe)

    NOTE: PPO is PERMANENTLY DISABLED - contradicts NeuroGenesis philosophy.
    Lifetime learning happens via evolved plasticity rules, not gradients.
    """
    fitnesses: List[float] = []
    if len(population) == 0 or len(opponents) == 0 or num_opponents <= 0:
        return fitnesses

    for genome in population:
        # Ensure TorchBrain is built for plasticity
        brain = genome.get_brain()
        if hasattr(brain, "layers") and len(brain.layers) == 0:
            brain.build_from_genome(genome)

        effective_opponents = _get_opponent_sample_size(
            num_opponents,
            partial_evaluation_enabled,
            partial_eval_fraction,
        )
        effective_opponents = min(effective_opponents, len(opponents))
        opp_indices = np.random.choice(len(opponents), size=effective_opponents, replace=False)
        genome_fitnesses = []
        for opp_idx in opp_indices:
            opponent = opponents[opp_idx]
            do_nonplastic = np.random.random() < nonplastic_check_fraction
            cache_key = None
            if fitness_cache_enabled:
                prey = genome if is_prey_evaluation else opponent
                predator = opponent if is_prey_evaluation else genome
                cache_key = _make_fitness_cache_key(prey, predator, stage_config, max_steps, do_nonplastic)
                cached = _get_cached_fitness(cache_key)
                if cached is not None:
                    fitness = cached["prey_fitness"] if is_prey_evaluation else cached["predator_fitness"]
                    genome_fitnesses.append(float(fitness))
                    continue

            if is_prey_evaluation:
                (prey_result, pred_result) = evaluate_multi_agent_pair(
                    genome,
                    opponent,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=do_nonplastic,
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_threshold=early_stopping_threshold,
                    early_stopping_patience=early_stopping_patience,
                )
                fitness = prey_result[0]
            else:
                (prey_result, pred_result) = evaluate_multi_agent_pair(
                    opponent,
                    genome,
                    arena,
                    stage_config,
                    max_steps,
                    do_nonplastic_compare=do_nonplastic,
                    early_stopping_enabled=early_stopping_enabled,
                    early_stopping_threshold=early_stopping_threshold,
                    early_stopping_patience=early_stopping_patience,
                )
                fitness = pred_result[0]
            genome_fitnesses.append(fitness)

            if fitness_cache_enabled and cache_key is not None:
                _set_cached_fitness(
                    cache_key,
                    {
                        "prey_fitness": float(prey_result[0]),
                        "predator_fitness": float(pred_result[0]),
                    },
                    max_size=fitness_cache_max_size,
                )

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
    global _ACTIVE_CONFIG, _ACTIVE_GENERATION
    _ACTIVE_CONFIG = training_state.config
    _ACTIVE_GENERATION = generation

    print(f"\n Generation {generation} - Stage: {stage.name}", flush=True)
    print(f" Prey Population: {len(training_state.prey_population)}", flush=True)
    print(f" Predator Population: {len(training_state.predator_population)}", flush=True)
    
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

    effective_max_steps = _get_effective_max_steps(training_state.config, stage_config, generation)
    
    # Evaluate prey population
    print("  Evaluating prey population...")
    if training_state.config.use_threaded_eval:
        prey_fitnesses = evaluate_population_parallel(
            training_state.prey_population,
            all_predators,
            stage_config,
            effective_max_steps,
            num_prey_opponents,
            is_prey_evaluation=True,
            batch_size=training_state.config.batch_size,
            num_prey_per_env=training_state.config.num_prey_per_arena,
            num_predators_per_env=training_state.config.num_predators_per_arena,
            num_workers=training_state.config.num_workers,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=training_state.config.partial_eval_fraction,
            fitness_cache_enabled=training_state.config.fitness_cache_enabled,
            fitness_cache_max_size=training_state.config.fitness_cache_max_size,
            early_stopping_enabled=training_state.config.early_stopping_enabled,
            early_stopping_threshold=training_state.config.early_stopping_threshold,
            early_stopping_patience=training_state.config.early_stopping_patience,
        )
    else:
        prey_fitnesses = evaluate_population_serial(
            training_state.prey_population,
            all_predators,
            arena,
            stage_config,
            effective_max_steps,
            num_prey_opponents,
            is_prey_evaluation=True,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=training_state.config.partial_eval_fraction,
            fitness_cache_enabled=training_state.config.fitness_cache_enabled,
            fitness_cache_max_size=training_state.config.fitness_cache_max_size,
            early_stopping_enabled=training_state.config.early_stopping_enabled,
            early_stopping_threshold=training_state.config.early_stopping_threshold,
            early_stopping_patience=training_state.config.early_stopping_patience,
        )

    # Evaluate predator population
    print("  Evaluating predator population...")
    if training_state.config.use_threaded_eval:
        predator_fitnesses = evaluate_population_parallel(
            training_state.predator_population,
            all_prey,
            stage_config,
            effective_max_steps,
            num_pred_opponents,
            is_prey_evaluation=False,
            batch_size=training_state.config.batch_size,
            num_prey_per_env=training_state.config.num_prey_per_arena,
            num_predators_per_env=training_state.config.num_predators_per_arena,
            num_workers=training_state.config.num_workers,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=training_state.config.partial_eval_fraction,
            fitness_cache_enabled=training_state.config.fitness_cache_enabled,
            fitness_cache_max_size=training_state.config.fitness_cache_max_size,
            early_stopping_enabled=training_state.config.early_stopping_enabled,
            early_stopping_threshold=training_state.config.early_stopping_threshold,
            early_stopping_patience=training_state.config.early_stopping_patience,
        )
    else:
        predator_fitnesses = evaluate_population_serial(
            training_state.predator_population,
            all_prey,
            arena,
            stage_config,
            effective_max_steps,
            num_pred_opponents,
            is_prey_evaluation=False,
            nonplastic_check_fraction=training_state.config.nonplastic_check_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=training_state.config.partial_eval_fraction,
            fitness_cache_enabled=training_state.config.fitness_cache_enabled,
            fitness_cache_max_size=training_state.config.fitness_cache_max_size,
            early_stopping_enabled=training_state.config.early_stopping_enabled,
            early_stopping_threshold=training_state.config.early_stopping_threshold,
            early_stopping_patience=training_state.config.early_stopping_patience,
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
    from core.torch_brain import PlasticLinear
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

    adaptability_scores = []
    meta_effectiveness_scores = []
    reward_before_learning = []
    reward_after_learning = []
    reward_delta_learning = []
    energy_costs = []
    learning_speeds = []
    stabilities = []
    novelties = []
    success_rates = []
    instability_scores = []  # CRITICAL FIX 2: Track instability
    metrics_missing = 0

    for genome in training_state.prey_population + training_state.predator_population:
        if hasattr(genome, 'plastic_diagnostics') and genome.plastic_diagnostics:
            # Use actual measured plastic advantage from evaluation
            plastic_advantage = genome.plastic_diagnostics.get('plastic_advantage', 0.0)
            reward_before = genome.plastic_diagnostics.get('reward_before_learning', 0.0)
            reward_after = genome.plastic_diagnostics.get('reward_after_learning', 0.0)

            # CRITICAL FIX 2: Get instability from plastic diagnostics
            instability = genome.plastic_diagnostics.get('instability', 0.0)
            instability_scores.append(float(instability))

            # Calculate adaptability as signed improvement from plasticity
            denom = max(1.0, abs(float(reward_before)))
            adaptability_score = float(plastic_advantage) / denom
            adaptability_scores.append(float(np.clip(adaptability_score, -1.0, 1.0)))
            reward_before_learning.append(float(reward_before))
            reward_after_learning.append(float(reward_after))
            reward_delta_learning.append(float(plastic_advantage))

            # Meta-parameter effectiveness based on how well they enable plasticity
            local_meta_gain = genome.meta.get('reward_gain', 1.0)
            local_meta_bias = genome.meta.get('reward_bias', 0.0)
            plastic_lr = genome.meta.get('plastic_lr', 1.0)

            # Meta-params are effective if they correlate with actual plastic improvement
            meta_effectiveness = 0.0
            if plastic_advantage > 0:
                # Reward meta-params that enable positive plasticity
                gain_effectiveness = min(abs(local_meta_gain) / 5.0, 1.0)
                lr_effectiveness = min(plastic_lr / 10.0, 1.0)
                meta_effectiveness = (gain_effectiveness + lr_effectiveness) / 2.0
            else:
                # Penalize meta-params that don't enable plasticity
                meta_effectiveness = 0.1  # Small baseline

            meta_effectiveness_scores.append(float(meta_effectiveness))
        else:
            # No plastic diagnostics - assume no adaptability
            adaptability_scores.append(0.0)
            meta_effectiveness_scores.append(0.0)
            instability_scores.append(0.0)  # CRITICAL FIX 2: Default instability


        last_metrics = getattr(genome, 'last_eval_metrics', None)
        if isinstance(last_metrics, dict):
            energy_costs.append(float(last_metrics.get('energy_cost', 0.0)))
            learning_speeds.append(float(last_metrics.get('learning_speed', 0.0)))
            stabilities.append(float(last_metrics.get('stability', 0.0)))
            novelties.append(float(last_metrics.get('novelty', 0.0)))
            success_rates.append(1.0 if last_metrics.get('task_success', False) else 0.0)
        else:
            metrics_missing += 1

    if metrics_missing == len(training_state.prey_population) + len(training_state.predator_population):
        print("[WARN] Evaluator metrics missing for all genomes; check last_eval_metrics wiring.")
    elif energy_costs and learning_speeds and stabilities and novelties and success_rates:
        print(
            "[EvalRaw] "
            f"Energy {np.min(energy_costs):.3f}/{np.mean(energy_costs):.3f}/{np.max(energy_costs):.3f} | "
            f"Learn {np.min(learning_speeds):.3f}/{np.mean(learning_speeds):.3f}/{np.max(learning_speeds):.3f} | "
            f"Stability {np.min(stabilities):.3f}/{np.mean(stabilities):.3f}/{np.max(stabilities):.3f} | "
            f"Novelty {np.min(novelties):.3f}/{np.mean(novelties):.3f}/{np.max(novelties):.3f} | "
            f"Success {np.mean(success_rates):.3f}"
        )

    avg_adaptability = float(np.mean(adaptability_scores)) if adaptability_scores else 0.0
    avg_meta_effectiveness = float(np.mean(meta_effectiveness_scores)) if meta_effectiveness_scores else 0.0
    avg_reward_before = float(np.mean(reward_before_learning)) if reward_before_learning else 0.0
    avg_reward_after = float(np.mean(reward_after_learning)) if reward_after_learning else 0.0
    avg_reward_delta = float(np.mean(reward_delta_learning)) if reward_delta_learning else 0.0
    avg_energy_cost = float(np.mean(energy_costs)) if energy_costs else 0.0
    avg_learning_speed = float(np.mean(learning_speeds)) if learning_speeds else 0.0
    avg_stability = float(np.mean(stabilities)) if stabilities else 0.0
    avg_novelty = float(np.mean(novelties)) if novelties else 0.0
    avg_success_rate = float(np.mean(success_rates)) if success_rates else 0.0
    avg_instability = float(np.mean(instability_scores)) if instability_scores else 0.0  # CRITICAL FIX 2


    # TEMPORARY FIX: Update learning speed normalization stats for population-level normalization
    _update_learning_speed_stats(learning_speeds)

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
        'avg_reward_before_learning': avg_reward_before,
        'avg_reward_after_learning': avg_reward_after,
        'avg_reward_delta': avg_reward_delta,
        'avg_energy_cost': avg_energy_cost,
        'avg_learning_speed': avg_learning_speed,
        'avg_stability': avg_stability,
        'avg_novelty': avg_novelty,
        'avg_success_rate': avg_success_rate,
        'avg_instability': avg_instability,  # CRITICAL FIX 2
        'config': stage_config

    }

    # Speciation + novelty logging
    try:
        prey_species = prey_engine.compute_species_stats(cast(Any, training_state.prey_population), generation)
        predator_species = predator_engine.compute_species_stats(cast(Any, training_state.predator_population), generation)
        stats['prey_species'] = prey_species
        stats['predator_species'] = predator_species

        prey_species_count = int(prey_species.get('num_species', 0)) if isinstance(prey_species, dict) else 0
        if prey_engine and prey_species_count > 0:
            prey_engine.adjust_speciation_threshold(
                num_species=prey_species_count,
                target_min=training_state.config.speciation_target_species_min,
                target_max=training_state.config.speciation_target_species_max,
                adjust_rate=training_state.config.speciation_adjust_rate,
                min_threshold=training_state.config.speciation_threshold_min,
                max_threshold=training_state.config.speciation_threshold_max,
            )

            if prey_species_count < training_state.config.prey_min_species_enforcement:
                prey_engine.adjust_speciation_threshold(
                    num_species=prey_species_count,
                    target_min=training_state.config.prey_min_species_enforcement,
                    target_max=max(
                        training_state.config.prey_min_species_enforcement,
                        training_state.config.speciation_target_species_max,
                    ),
                    adjust_rate=training_state.config.prey_min_species_adjust_rate,
                    min_threshold=training_state.config.speciation_threshold_min,
                    max_threshold=training_state.config.speciation_threshold_max,
                )

        # Adjust predator speciation threshold (Issue 3: Fix predator monoculture)
        predator_species_count = int(predator_species.get('num_species', 0)) if isinstance(predator_species, dict) else 0
        if predator_engine and predator_species_count > 0:
            predator_engine.adjust_speciation_threshold(
                num_species=predator_species_count,
                target_min=training_state.config.speciation_target_species_min,
                target_max=training_state.config.speciation_target_species_max,
                adjust_rate=training_state.config.speciation_adjust_rate,
                min_threshold=training_state.config.speciation_threshold_min,
                max_threshold=training_state.config.speciation_threshold_max,
            )

            if predator_species_count < training_state.config.predator_min_species_enforcement:
                predator_engine.adjust_speciation_threshold(
                    num_species=predator_species_count,
                    target_min=training_state.config.predator_min_species_enforcement,
                    target_max=max(
                        training_state.config.predator_min_species_enforcement,
                        training_state.config.speciation_target_species_max,
                    ),
                    adjust_rate=training_state.config.predator_min_species_adjust_rate,
                    min_threshold=training_state.config.speciation_threshold_min,
                    max_threshold=training_state.config.speciation_threshold_max,
                )
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

    # Cluster every 5 generations to reduce overhead, skip after gen 500
    if generation % 5 == 0 and generation <= 500:
        try:
            arch_cluster_stats = compute_architecture_clustering_stats(cast(List[EvolvableGenome], combined_population))
            stats['architecture_clusters'] = arch_cluster_stats
        except Exception:
            stats['architecture_clusters'] = {'num_clusters': 0, 'silhouette': 0.0, 'diversity': 0.0, 'cluster_sizes': []}
    else:
        # Skip clustering this generation
        stats['architecture_clusters'] = {'num_clusters': 0, 'silhouette': 0.0, 'diversity': 0.0, 'cluster_sizes': [], 'skipped': True}

    
    # Calculate neural health metrics across population
    total_dead_layers = 0
    total_saturated_layers = 0
    genomes_with_issues = 0
    for genome in combined_population:
        if hasattr(genome, 'brain') and hasattr(genome.brain, 'activation_stats'):
            for stat in genome.brain.activation_stats:
                if stat.get('dead_ratio', 0) > 0.5:
                    total_dead_layers += 1
                if stat.get('saturated_ratio', 0) > 0.5:
                    total_saturated_layers += 1
            if genome.brain.activation_stats:
                genomes_with_issues += 1
    
    stats['neural_health'] = {
        'dead_layers': total_dead_layers,
        'saturated_layers': total_saturated_layers,
        'genomes_analyzed': genomes_with_issues
    }
    
    # Log generation
    log_coevolution_generation(stats)
    if evaluator is not None:
        evaluator.log_seed_coverage(generation)

    # Integrate behavioral probes for comprehensive evaluation
    from evaluation.behavioral_probes import BehavioralProbe
    # Start async logging if not already started
    BehavioralProbe.start_async_logging()
    # Create properly typed list for behavioral probes
    evolvable_genomes: List[EvolvableGenome] = list(combined_population)
    # Only save probe reports at checkpoint intervals to avoid file system spam
    save_probe_reports = (generation % training_state.config.plot_every == 0) if training_state.config.plot_every > 0 else False
    probe_integration_results = BehavioralProbe.integrate_with_evaluation_pipeline(
        evolvable_genomes,
        generation=generation,
        save_reports=save_probe_reports
    )

    # Add probe results to stats
    stats['behavioral_probes'] = probe_integration_results

    # META-EVOLUTION: Evolve architect and mutator populations
    # Prepare performance data for meta-evolution

    # Calculate actual fitness improvement
    avg_fitness_improvement = 0.0
    if len(training_state.best_prey_fitness_history) > 1 and len(training_state.best_predator_fitness_history) > 1:
        prev_prey_best = training_state.best_prey_fitness_history[-2]
        prev_pred_best = training_state.best_predator_fitness_history[-2]
        curr_prey_best = training_state.best_prey_fitness_history[-1]
        curr_pred_best = training_state.best_predator_fitness_history[-1]
        avg_fitness_improvement = ((curr_prey_best - prev_prey_best) + (curr_pred_best - prev_pred_best)) / 2.0

    # Calculate diversity preservation (maintain high species count)
    architecture_diversity = stats.get('prey_species', {}).get('num_species', 1) + stats.get('predator_species', {}).get('num_species', 1)
    diversity_preservation = min(architecture_diversity / 10.0, 1.0)  # Normalize to 0-1

    # Calculate exploration success (novelty scores)
    prey_novelty = stats.get('prey_novelty', {}).get('mean', 0.0)
    pred_novelty = stats.get('predator_novelty', {}).get('mean', 0.0)
    exploration_success = (prey_novelty + pred_novelty) / 2.0

    # Calculate mutation success rates (based on fitness variance improvement)
    mutation_success_rates = {}
    if len(training_state.generation_stats) > 1:
        prev_stats = training_state.generation_stats[-2]
        curr_stats = training_state.generation_stats[-1]

        # Weight mutation success: if fitness improved after mutations
        if avg_fitness_improvement > 0:
            mutation_success_rates['weight'] = min(avg_fitness_improvement * 10.0, 1.0)
            mutation_success_rates['arch'] = min(architecture_diversity / 5.0, 1.0)
            mutation_success_rates['layer'] = min(exploration_success * 2.0, 1.0)
        else:
            mutation_success_rates['weight'] = 0.1
            mutation_success_rates['arch'] = 0.1
            mutation_success_rates['layer'] = 0.1
    else:
        mutation_success_rates = {'weight': 0.5, 'arch': 0.4, 'layer': 0.3}

    # Compute motif_effectiveness from the architect population's own improvement trend.
    # Using the delta between the last two meta-fitness scores captures whether the
    # templates are actually getting better, rather than relying on a proxy metric.
    if len(architect_population.meta_fitness_history) >= 2:
        _arch_delta = architect_population.meta_fitness_history[-1] - architect_population.meta_fitness_history[-2]
        motif_effectiveness = float(max(0.0, min(_arch_delta, 1.0)))
    else:
        motif_effectiveness = float(max(0.0, stats.get('avg_adaptability_score', 0.0)))

    # Combine top prey and predator genomes so the architect sees both roles.
    _top_prey = sorted(training_state.prey_population, key=lambda g: g.fitness, reverse=True)[:5]
    _top_pred = sorted(training_state.predator_population, key=lambda g: g.fitness, reverse=True)[:5]

    performance_data = {
        'avg_fitness': stats.get('mean_prey_fitness', 0.0) + stats.get('mean_predator_fitness', 0.0),
        'architecture_diversity': architecture_diversity,
        'motif_effectiveness': motif_effectiveness,
        'successful_architectures': _top_prey + _top_pred,
        'mutation_success_rates': mutation_success_rates,
        'avg_fitness_improvement': avg_fitness_improvement,
        'diversity_preservation': diversity_preservation,
        'exploration_success': exploration_success
    }

    # Evolve architect population (called exactly once per generation here;
    # EvolutionEngine.create_next_generation no longer drives it to avoid triple-calling).
    architect_population.evolve_architectures(performance_data)

    # Evolve mutator population
    mutator_population.evolve_mutators(performance_data)

    # Use evolved mutation strategies to adapt main evolution engines
    adaptive_rates = mutator_population.get_adaptive_rates()
    used_strategy = mutator_population.get_best_strategy()  # Track which strategy was used for effectiveness update
    if prey_engine and adaptive_rates:
        prey_engine.mutation_rate = adaptive_rates.get('weight_rate', prey_engine.mutation_rate)
        prey_engine.architecture_mutation_rate = adaptive_rates.get('arch_rate', prey_engine.architecture_mutation_rate)

    if predator_engine and adaptive_rates:
        predator_engine.mutation_rate = adaptive_rates.get('weight_rate', predator_engine.mutation_rate)
        predator_engine.architecture_mutation_rate = adaptive_rates.get('arch_rate', predator_engine.architecture_mutation_rate)

    def _inject_template_into_population(
        population: list,
        genome_cls,
        template: Dict[str, Any],
        role_tag: str,
    ) -> None:
        """Build a genome from *template* and replace a random slot in *population*."""
        pattern = template.get('pattern', {})
        if not pattern or not pattern.get('layer_pattern'):
            return
        new_genome = genome_cls.random_initialization()
        new_genome.genes = []
        prev_dim = new_genome.input_size
        layer_pattern = pattern['layer_pattern']
        activation_pattern = pattern.get('activation_pattern', ['tanh'] * len(layer_pattern))
        for i, out_dim in enumerate(layer_pattern):
            activation = activation_pattern[i] if i < len(activation_pattern) else 'tanh'
            gene = NeuralGene(
                gene_id=f"evolved_{role_tag}_layer_{i}",
                input_dim=prev_dim,
                output_dim=out_dim,
                activation=activation,
                use_bias=True,
                plasticity=np.random.uniform(-0.1, 0.1, (out_dim, prev_dim)).astype(np.float32),
            )
            gene.initialize_weights(method="he_normal", scale=0.1)
            new_genome.genes.append(gene)
            prev_dim = out_dim
        # Ensure output dimension matches expected network output
        if prev_dim != new_genome.output_size:
            gene = NeuralGene(
                gene_id=f"evolved_{role_tag}_output",
                input_dim=prev_dim,
                output_dim=new_genome.output_size,
                activation='linear',
                use_bias=True,
                plasticity=None,
            )
            gene.initialize_weights(method="he_normal", scale=0.1)
            new_genome.genes.append(gene)
        if population:
            idx = random.randint(0, len(population) - 1)
            population[idx] = new_genome

    # Inject evolved architectures into both prey and predator populations.
    # Use shared_templates (cached by _share_templates) so prey and predator each
    # get a distinct template from the sorted top-5 rather than both using index 0.
    if random.random() < 0.05:  # 5% chance per generation
        shared = architect_population.shared_templates
        best_template = shared[0] if shared else architect_population.get_best_template()
        if best_template:
            _inject_template_into_population(
                training_state.prey_population, PreyGenome, best_template, 'prey'
            )
        # Predator gets the second-best template (or same if only one exists)
        pred_template = shared[1] if len(shared) > 1 else best_template
        if pred_template:
            _inject_template_into_population(
                training_state.predator_population, PredatorGenome, pred_template, 'predator'
            )

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
    early_stopping_enabled: bool = False,
    early_stopping_threshold: float = -30.0,
    early_stopping_patience: int = 5,
):
    """
    Evaluate a single prey-predator pair in multi-agent arena with PRESSURE INJECTION
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
        prey_genome,
        predator_genome,
        arena,
        max_steps,
        seed=arena_seed,
        stage_name=stage_name,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience,
    )

    if not do_nonplastic_compare:
        # Compute fitness from metrics (weighted combination)
        prey_fitness = compute_fitness_from_metrics(plastic_prey_metrics, brain=prey_genome.brain)
        predator_fitness = compute_fitness_from_metrics(plastic_pred_metrics, brain=predator_genome.brain)
        _store_last_eval_metrics(prey_genome, plastic_prey_metrics, prey_fitness)
        _store_last_eval_metrics(predator_genome, plastic_pred_metrics, predator_fitness)
        return (prey_fitness, plastic_prey_metrics), (predator_fitness, plastic_pred_metrics)

    # Create non-plastic versions (disable plasticity updates) with same arena seed
    (nonplastic_prey_reward, nonplastic_prey_metrics), (nonplastic_pred_reward, nonplastic_pred_metrics) = evaluate_without_plasticity(
        prey_genome,
        predator_genome,
        arena,
        max_steps,
        seed=arena_seed,
        stage_name=stage_name,
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_threshold=early_stopping_threshold,
        early_stopping_patience=early_stopping_patience,
    )

    # Plastic agents must outperform non-plastic agents within the episode
    plastic_advantage_prey = plastic_prey_reward - nonplastic_prey_reward
    plastic_advantage_pred = plastic_pred_reward - nonplastic_pred_reward

    plasticity_bonus_prey = np.clip(plastic_advantage_prey, -5, 5)
    plasticity_bonus_pred = np.clip(plastic_advantage_pred, -5, 5)

    final_prey_reward = plastic_prey_reward + plasticity_bonus_prey
    final_pred_reward = plastic_pred_reward + plasticity_bonus_pred

    # Store plastic advantage in diagnostics for adaptability calculation
    if prey_genome.plastic_diagnostics is None:
        prey_genome.plastic_diagnostics = {}
    prey_genome.plastic_diagnostics['plastic_advantage'] = float(plastic_advantage_prey)
    prey_genome.plastic_diagnostics['reward_before_learning'] = float(nonplastic_prey_reward)
    prey_genome.plastic_diagnostics['reward_after_learning'] = float(plastic_prey_reward)

    if predator_genome.plastic_diagnostics is None:
        predator_genome.plastic_diagnostics = {}
    predator_genome.plastic_diagnostics['plastic_advantage'] = float(plastic_advantage_pred)
    predator_genome.plastic_diagnostics['reward_before_learning'] = float(nonplastic_pred_reward)
    predator_genome.plastic_diagnostics['reward_after_learning'] = float(plastic_pred_reward)

    # Update metrics with final rewards and compute fitness
    plastic_prey_metrics.episode_return = final_prey_reward
    plastic_pred_metrics.episode_return = final_pred_reward
    plastic_prey_metrics.adaptability = float(plastic_advantage_prey)
    plastic_pred_metrics.adaptability = float(plastic_advantage_pred)

    prey_fitness = compute_fitness_from_metrics(plastic_prey_metrics, brain=prey_genome.brain)
    predator_fitness = compute_fitness_from_metrics(plastic_pred_metrics, brain=predator_genome.brain)

    _store_last_eval_metrics(prey_genome, plastic_prey_metrics, prey_fitness)
    _store_last_eval_metrics(predator_genome, plastic_pred_metrics, predator_fitness)

    return (prey_fitness, plastic_prey_metrics), (predator_fitness, plastic_pred_metrics)


def _to_numpy(x):
    """Convert torch tensors or other array-likes to numpy arrays."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _action_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts.astype(np.float32) / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
    max_entropy = float(np.log(max(len(counts), 1)))
    if max_entropy <= 0.0:
        return 0.0
    return float(entropy / max_entropy)

def _store_last_eval_metrics(genome, metrics: EpisodeMetrics, fitness: float) -> None:
    genome.last_eval_metrics = {
        'fitness': float(fitness),
        'energy_cost': float(metrics.energy_cost),
        'learning_speed': float(metrics.learning_speed),
        'stability': float(metrics.stability),
        'task_success': bool(metrics.task_success),
        'episode_return': float(metrics.episode_return),
        'complexity_penalty': float(metrics.complexity_penalty),
        'novelty': float(metrics.novelty),
        'saturation_penalty': float(metrics.saturation_penalty or 0.0),
        'dead_unit_penalty': float(metrics.dead_unit_penalty or 0.0),
        'seed': int(metrics.seed),
        'stage': str(metrics.stage),
        'opponent_id': metrics.opponent_id,
        'adaptability': float(metrics.adaptability),
    }

def _get_adaptability_weight() -> float:
    config = _ACTIVE_CONFIG
    if config is None:
        return 0.3

    boost_gens = int(config.adaptability_boost_generations)
    taper_gens = int(config.adaptability_taper_generations)
    base = float(config.adaptability_weight_base)
    boost = float(config.adaptability_weight_boost)
    gen = int(_ACTIVE_GENERATION)

    if boost_gens <= 0:
        return base
    if gen < boost_gens:
        return boost
    if taper_gens <= 0:
        return base

    t = (gen - boost_gens + 1) / float(taper_gens)
    t = max(0.0, min(1.0, t))
    return float(boost + (base - boost) * t)


# Global state for learning speed normalization (population-level statistics)
_LEARNING_SPEED_STATS = {
    'min': 0.0,
    'max': 1.0,
    'mean': 0.0,
    'std': 1.0,
    'initialized': False
}

# TEMPORARY FIX: Force evolution to prioritize learning
# This addresses the issue where adaptability collapsed to 0.002 while learn_speed is 1.493
# Without this, plasticity remains decorative - learning has no evolutionary consequence
_TEMPORARY_LEARN_SPEED_WEIGHT = 2.0  # Force: fitness = reward + 2.0 * normalized_learn_speed


def _update_learning_speed_stats(learning_speeds: List[float]) -> None:
    """Update global learning speed statistics for normalization"""
    global _LEARNING_SPEED_STATS
    if not learning_speeds:
        return
    
    speeds = np.array(learning_speeds, dtype=np.float32)
    _LEARNING_SPEED_STATS['min'] = float(np.min(speeds))
    _LEARNING_SPEED_STATS['max'] = float(np.max(speeds))
    _LEARNING_SPEED_STATS['mean'] = float(np.mean(speeds))
    _LEARNING_SPEED_STATS['std'] = float(np.std(speeds)) if len(speeds) > 1 else 1.0
    _LEARNING_SPEED_STATS['initialized'] = True


def _normalize_learning_speed(learning_speed: float) -> float:
    """Normalize learning speed using z-score normalization"""
    global _LEARNING_SPEED_STATS
    
    if not _LEARNING_SPEED_STATS['initialized']:
        # Fallback: if not initialized, use raw value capped at reasonable range
        return float(np.clip(learning_speed / 10.0, 0.0, 2.0))
    
    # Z-score normalization: (x - mean) / std
    std = max(_LEARNING_SPEED_STATS['std'], 1e-6)
    normalized = (learning_speed - _LEARNING_SPEED_STATS['mean']) / std
    
    # Scale to 0-2 range for reasonable fitness contribution
    # Typical z-scores range from -3 to 3, so we scale accordingly
    normalized = float(np.clip(normalized * 0.5 + 1.0, 0.0, 2.0))
    
    return normalized


def compute_fitness_from_metrics(metrics: EpisodeMetrics, brain: "Optional[TorchBrain]" = None) -> float:
    """Compute scalar fitness from metrics decomposition with neural health penalties"""
    # Weighted combination of metrics for fitness
    # Primary: episode return (main objective)
    # Secondary: success bonus, learning speed, stability penalty, energy efficiency
    fitness = metrics.episode_return

    # Adaptability bonus/penalty: reward improvement over lifetime learning
    adaptability_weight = _get_adaptability_weight()
    fitness += metrics.adaptability * adaptability_weight

    # Success bonus
    if metrics.task_success:
        fitness += 1.0

    # TEMPORARY FIX: Learning speed bonus with strong weight
    # Issue: Adaptability collapsed to 0.002 while LearnSpeed is 1.493
    # Learning happens but is not rewarded enough in fitness
    # Fix: Enforce fitness = reward + 2.0 * normalized_learn_speed
    normalized_learn_speed = _normalize_learning_speed(metrics.learning_speed)
    fitness += normalized_learn_speed * _TEMPORARY_LEARN_SPEED_WEIGHT

    # Legacy learning speed bonus (kept for backward compatibility, now redundant with temp fix)
    # fitness += metrics.learning_speed * 0.1

    # Stability penalty (prefer consistent performance)
    fitness -= metrics.stability * 0.05

    # Energy efficiency bonus (lower cost is better)
    fitness += (1.0 / (1.0 + metrics.energy_cost)) * 0.5

    # Complexity penalty
    fitness -= metrics.complexity_penalty

    # Novelty bonus
    fitness += metrics.novelty * 0.2

    # Penalize networks with high saturation or dead units
    if metrics.saturation_penalty is not None:
        fitness -= float(metrics.saturation_penalty)
    if metrics.dead_unit_penalty is not None:
        fitness -= float(metrics.dead_unit_penalty)

    # NEURAL HEALTH CONTROLLER: Apply heavy penalties for dead neurons
    # This turns dead neurons into evolutionary pressure rather than runtime errors
    if brain is not None and hasattr(brain, 'neural_health_controller'):
        torch_brain = cast(TorchBrain, brain)
        health_penalty = torch_brain.neural_health_controller.get_fitness_penalty(torch_brain)  # type: ignore[attr-defined]
        fitness *= health_penalty  # Multiplicative penalty for dead neurons

    return float(fitness)

def evaluate_with_plasticity(
    prey_genome,
    predator_genome,
    arena,
    max_steps,
    seed=None,
    stage_name="unknown",
    early_stopping_enabled: bool = False,
    early_stopping_threshold: float = -30.0,
    early_stopping_patience: int = 5,
):
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

    prey_action_counts = np.zeros(int(prey_genome.output_size), dtype=np.int64)
    pred_action_counts = np.zeros(int(predator_genome.output_size), dtype=np.int64)

    predator_brain = PredatorPackBrain(predator_genome)

    catastrophic_steps = 0

    for step in range(max_steps):
        # Get actions from genomes
        prey_actions = prey_genome.act_batch(prey_state)

        # Use predator pack brain for coordinated actions
        pred_actions = predator_brain.act(pred_state)

        prey_action_counts += np.bincount(np.ravel(prey_actions), minlength=prey_action_counts.size)
        pred_action_counts += np.bincount(np.ravel(pred_actions), minlength=pred_action_counts.size)

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
        if 'prey_energy' in info and len(info['prey_energy']) > 0:
            prey_energy_cost += np.mean(info['prey_energy'])
        if 'predator_energy' in info and len(info['predator_energy']) > 0:
            predator_energy_cost += np.mean(info['predator_energy'])

        steps_survived += 1

        if early_stopping_enabled:
            if prey_total_reward <= early_stopping_threshold and predator_total_reward <= early_stopping_threshold:
                catastrophic_steps += 1
            else:
                catastrophic_steps = 0
            if catastrophic_steps >= early_stopping_patience:
                break

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
        if prey_genome.plastic_diagnostics is None:
            prey_genome.plastic_diagnostics = {}
        prey_genome.plastic_diagnostics.update(plastic_diag)
        if 'mean_plastic_delta' in plastic_diag:
            prey_genome.plastic_diagnostics['mean_final_plastic_delta'] = float(plastic_diag['mean_plastic_delta'])
        if 'total_plastic_delta' in plastic_diag:
            prey_learning_speed = float(plastic_diag['total_plastic_delta'])

    # Compute metrics
    prey_action_entropy = _action_entropy(prey_action_counts)
    pred_action_entropy = _action_entropy(pred_action_counts)

    prey_genome.behavior_stats = {
        'mean_reward': float(np.mean(prey_rewards)) if prey_rewards else 0.0,
        'reward_std': float(np.std(prey_rewards)) if prey_rewards else 0.0,
        'steps_survived': int(steps_survived),
        'action_entropy': float(prey_action_entropy),
        'energy_cost': float(prey_energy_cost),
        'episode_return': float(prey_total_reward),
    }
    predator_genome.behavior_stats = {
        'mean_reward': float(np.mean(predator_rewards)) if predator_rewards else 0.0,
        'reward_std': float(np.std(predator_rewards)) if predator_rewards else 0.0,
        'steps_survived': int(steps_survived),
        'action_entropy': float(pred_action_entropy),
        'energy_cost': float(predator_energy_cost),
        'episode_return': float(predator_total_reward),
    }

    prey_stability = prey_genome.brain.get_stability_diagnostics()
    predator_stability = predator_genome.brain.get_stability_diagnostics()
    prey_saturation_penalty = float(prey_stability.get('avg_saturation_fraction', 0.0)) * 0.5
    prey_dead_unit_penalty = float(prey_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3
    predator_saturation_penalty = float(predator_stability.get('avg_saturation_fraction', 0.0)) * 0.5
    predator_dead_unit_penalty = float(predator_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3

    prey_metrics = EpisodeMetrics(
        task_success=prey_total_reward > 0,  # Basic success: positive reward
        episode_return=prey_total_reward,
        learning_speed=prey_learning_speed,
        stability=float(np.std(prey_rewards)) if prey_rewards else 0.0,
        energy_cost=prey_energy_cost,
        complexity_penalty=0.0,  # implement complexity measure
        novelty=prey_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=predator_genome.genome_id if hasattr(predator_genome, 'genome_id') else None,
        saturation_penalty=prey_saturation_penalty,
        dead_unit_penalty=prey_dead_unit_penalty
    )

    # Compute learning speed from plasticity diagnostics
    predator_learning_speed = 0.0
    if hasattr(predator_genome.brain, 'get_plastic_diagnostics'):
        plastic_diag = predator_genome.brain.get_plastic_diagnostics()
        if predator_genome.plastic_diagnostics is None:
            predator_genome.plastic_diagnostics = {}
        predator_genome.plastic_diagnostics.update(plastic_diag)
        if 'mean_plastic_delta' in plastic_diag:
            predator_genome.plastic_diagnostics['mean_final_plastic_delta'] = float(plastic_diag['mean_plastic_delta'])
        if 'total_plastic_delta' in plastic_diag:
            predator_learning_speed = float(plastic_diag['total_plastic_delta'])

    predator_metrics = EpisodeMetrics(
        task_success=predator_total_reward > 0,
        episode_return=predator_total_reward,
        learning_speed=predator_learning_speed,
        stability=float(np.std(predator_rewards)) if predator_rewards else 0.0,
        energy_cost=predator_energy_cost,
        complexity_penalty=0.0,
        novelty=pred_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=prey_genome.genome_id if hasattr(prey_genome, 'genome_id') else None,
        saturation_penalty=predator_saturation_penalty,
        dead_unit_penalty=predator_dead_unit_penalty
    )

    return (prey_total_reward, prey_metrics), (predator_total_reward, predator_metrics)

def evaluate_without_plasticity(
    prey_genome,
    predator_genome,
    arena,
    max_steps,
    seed=None,
    stage_name="unknown",
    early_stopping_enabled: bool = False,
    early_stopping_threshold: float = -30.0,
    early_stopping_patience: int = 5,
):
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

    prey_action_counts = np.zeros(int(prey_genome.output_size), dtype=np.int64)
    pred_action_counts = np.zeros(int(predator_genome.output_size), dtype=np.int64)

    # Create predator brain once, outside the loop
    predator_brain = PredatorPackBrain(predator_genome)

    catastrophic_steps = 0

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

        prey_total_reward += float(np.sum(prey_reward))
        predator_total_reward += float(np.sum(pred_reward))
        prey_rewards.append(float(np.mean(prey_reward)))
        predator_rewards.append(float(np.mean(pred_reward)))

        # Track energy costs from info
        if 'prey_energy' in info and len(info['prey_energy']) > 0:
            prey_energy_cost += np.mean(info['prey_energy'])
        if 'predator_energy' in info and len(info['predator_energy']) > 0:
            predator_energy_cost += np.mean(info['predator_energy'])

        steps_survived += 1

        if early_stopping_enabled:
            if prey_total_reward <= early_stopping_threshold and predator_total_reward <= early_stopping_threshold:
                catastrophic_steps += 1
            else:
                catastrophic_steps = 0
            if catastrophic_steps >= early_stopping_patience:
                break

        if np.any(info['env_done']):
            break

    # Finalize episode plasticity logging (log once per episode, not per step)
    if hasattr(prey_genome.brain, 'finalize_episode_plastic_norms'):
        prey_genome.brain.finalize_episode_plastic_norms()
    if hasattr(predator_genome.brain, 'finalize_episode_plastic_norms'):
        predator_genome.brain.finalize_episode_plastic_norms()

    # Compute metrics (no learning speed for non-plastic)
    prey_action_entropy = _action_entropy(prey_action_counts)
    pred_action_entropy = _action_entropy(pred_action_counts)
    prey_stability = prey_genome.brain.get_stability_diagnostics()
    predator_stability = predator_genome.brain.get_stability_diagnostics()
    prey_saturation_penalty = float(prey_stability.get('avg_saturation_fraction', 0.0)) * 0.5
    prey_dead_unit_penalty = float(prey_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3
    predator_saturation_penalty = float(predator_stability.get('avg_saturation_fraction', 0.0)) * 0.5
    predator_dead_unit_penalty = float(predator_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3

    prey_metrics = EpisodeMetrics(
        task_success=prey_total_reward > 0,
        episode_return=prey_total_reward,
        learning_speed=0.0,  # No plasticity updates
        stability=float(np.std(prey_rewards)) if prey_rewards else 0.0,
        energy_cost=prey_energy_cost,
        complexity_penalty=0.0,
        novelty=prey_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=predator_genome.genome_id if hasattr(predator_genome, 'genome_id') else None,
        saturation_penalty=prey_saturation_penalty,
        dead_unit_penalty=prey_dead_unit_penalty
    )

    predator_metrics = EpisodeMetrics(
        task_success=predator_total_reward > 0,
        episode_return=predator_total_reward,
        learning_speed=0.0,
        stability=float(np.std(predator_rewards)) if predator_rewards else 0.0,
        energy_cost=predator_energy_cost,
        complexity_penalty=0.0,
        novelty=pred_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=prey_genome.genome_id if hasattr(prey_genome, 'genome_id') else None,
        saturation_penalty=predator_saturation_penalty,
        dead_unit_penalty=predator_dead_unit_penalty
    )

    return (prey_total_reward, prey_metrics), (predator_total_reward, predator_metrics)

def log_coevolution_generation(stats: Dict[str, Any]):
    """Log co-evolution generation statistics with metrics decomposition"""
    logger = logging.getLogger('coevolution')
    print(f"{'='*80}")
    print(f"Generation {stats['generation']:04d} - {stats['stage']}")
    print(f"{'-'*80}")
    print(f"Prey Fitness:    Best: {stats['best_prey_fitness']:8.2f} | Mean: {stats['mean_prey_fitness']:8.2f}")
    print(f"Predator Fitness: Best: {stats['best_predator_fitness']:8.2f} | Mean: {stats['mean_predator_fitness']:8.2f}")
    print(f"Evaluation Time: {stats['eval_time']:6.2f}s")
    print(f"Population: {stats['prey_population_size']} prey, {stats['predator_population_size']} predators")

    # Log metrics decomposition summary
    if 'avg_adaptability_score' in stats:
        print(
            "Adaptability: "
            f"{stats['avg_adaptability_score']:.3f} | "
            f"Meta Effectiveness: {stats['avg_meta_effectiveness']:.3f} | "
            f"Delta Reward: {stats.get('avg_reward_delta', 0.0):.3f} | "
            f"Instability: {stats.get('avg_instability', 0.0):.3f}"
        )

    if 'mean_plastic_norm' in stats:
        print(f"Plastic Norms: Mean {stats['mean_plastic_norm']:.4f} | Max {stats['max_plastic_norm']:.4f} | 95th {stats['p95_plastic_norm']:.4f}")

    if 'avg_energy_cost' in stats:
        print(
            "Evaluator: "
            f"Energy {stats['avg_energy_cost']:.3f} | "
            f"LearnSpeed {stats['avg_learning_speed']:.3f} | "
            f"Stability {stats['avg_stability']:.3f} | "
            f"Novelty {stats['avg_novelty']:.3f} | "
            f"Success {stats['avg_success_rate']:.3f}"
        )

    # Neural health summary (replaces per-layer spam)
    neural_health = stats.get('neural_health')
    if neural_health and neural_health.get('genomes_analyzed', 0) > 0:
        dead = neural_health['dead_layers']
        saturated = neural_health['saturated_layers']
        print(f"Neural Health: {dead} dead layers, {saturated} saturated (evolutionary pressure active)")

    # speciation + novelty summary
    prey_species = stats.get('prey_species')
    predator_species = stats.get('predator_species')
    if isinstance(prey_species, dict) and isinstance(predator_species, dict):
        print(
            "Speciation: "
            f"prey {prey_species.get('num_species', 0)} species (avg {prey_species.get('avg_species_size', 0.0):.1f}), "
            f"pred {predator_species.get('num_species', 0)} species (avg {predator_species.get('avg_species_size', 0.0):.1f})"
        )

    # Log speciation thresholds (already set in train_coevolution_async)
    if 'prey_speciation_threshold' in stats and 'predator_speciation_threshold' in stats:
        print(
            "Speciation Thresholds: "
            f"prey {stats['prey_speciation_threshold']:.3f} | "
            f"pred {stats['predator_speciation_threshold']:.3f}"
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

    arch_clusters = stats.get('architecture_clusters')
    if isinstance(arch_clusters, dict):
        print(
            "Architecture Clusters: "
            f"{arch_clusters.get('num_clusters', 0)} clusters | "
            f"silhouette {arch_clusters.get('silhouette', 0.0):.3f} | "
            f"diversity {arch_clusters.get('diversity', 0.0):.3f}"
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
    plt.savefig(f'output_logs/meta_gene_distribution_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"META gene histograms saved: output_logs/meta_gene_distribution_gen_{generation:04d}.png")

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

    print("Plastic norm evolution plot saved: output_logs/plastic_norm_evolution.png")

def plot_learning_rule_stats(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]):
    """Plot learning rule parameter distributions per generation"""
    population = prey_population + predator_population 
    rules = ["A", "B", "C", "D", "E"]

    for k in rules:
        vals = [g.learning_rule[k] for g in population if g.learning_rule is not None]
        plt.hist(vals, bins=30)
        plt.axvline(float(np.mean(vals).item()), color='r')
        plt.title(f"Learning Rule {k} — Gen {generation}")
    plt.tight_layout()
    plt.savefig(f'output_logs/learning_rule_stats_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

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
    plt.savefig(f'output_logs/learning_rule_vs_fitness_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Learning rule vs fitness scatter plots saved: output_logs/learning_rule_vs_fitness_gen_{generation:04d}.png")

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
    plt.savefig(f'output_logs/strategy_clustering_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Strategy clustering plot saved: output_logs/strategy_clustering_gen_{generation:04d}.png")

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

def plot_architecture_clustering(generation: int, prey_population: List[PreyGenome], predator_population: List[PredatorGenome]):
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

    ax.boxplot(fitness_data, label=cluster_names, patch_artist=True,
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

    print(f"Architecture clustering plot saved: output_logs/architecture_clustering_gen_{generation:04d}.png")

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
    plt.savefig(f'output_logs/in_lifetime_learning_curve_gen_{generation:04d}.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"In-lifetime learning curve plot saved: output_logs/in_lifetime_learning_curve_gen_{generation:04d}.png")

# Meta-scientist experiment runners
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

def save_coevolution_state(training_state: TrainingState, filename: str = "data/coevolution_state.json"):
    """Save complete co-evolution training state"""
    state = {
        'generation': training_state.generation,
        'config': training_state.config.__dict__,
        'best_prey_fitness_history': training_state.best_prey_fitness_history,
        'best_predator_fitness_history': training_state.best_predator_fitness_history,
        'generation_stats': training_state.generation_stats,
        'experiment_reports': [exp.__dict__ for exp in training_state.experiment_reports],
        'generalization_reports': [r.to_dict() for r in training_state.generalization_reports],
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

def load_coevolution_state(filename: str = "data/coevolution_state.json") -> TrainingState:
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
    training_state.generalization_reports = []
    for r in state.get('generalization_reports', []):
        if isinstance(r, GeneralizationReport):
            training_state.generalization_reports.append(r)
            continue

        benchmark_results = [
            b if isinstance(b, BenchmarkResult) else BenchmarkResult.from_dict(b)
            for b in r.get('benchmark_results', [])
        ]
        training_state.generalization_reports.append(
            GeneralizationReport(
                generation=r.get('generation', training_state.generation),
                genome_id=r.get('genome_id', 'unknown'),
                benchmark_results=benchmark_results,
            )
        )
    
    print(f"Co-evolution state loaded: {filename}")
    print(f"Generation: {training_state.generation}")
    print(f"Prey: {len(training_state.prey_population)}, Predators: {len(training_state.predator_population)}")
    
    return training_state

async def main_coevolution_async():
    """Main async co-evolution training loop with adaptive curriculum"""
    # Configure logging for immediate visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Output to console
        ]
    )

    print("Starting Co-Evolution Training with Adaptive Curriculum")
    print("=" * 60)

    # Add watchdog thread (critical)
    def watchdog():
        while True:
            print("[WATCHDOG] main loop alive")
            time.sleep(10)

    threading.Thread(target=watchdog, daemon=True).start()

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
        use_gpu=torch.cuda.is_available(),
        envs_per_genome=config.envs_per_genome,
        max_steps=config.max_steps
    )

    # Load checkpoint if exists
    if os.path.exists("data/coevolution_state.json"):
        response = os.getenv("AUTO_LOAD_COEVOLUTION_STATE")
        if response is None:
            response = input("Co-evolution state found. Load? (y/n): ")
        if response.lower() == 'y':
            training_state = load_coevolution_state()
            evaluator.load_seeds("data/seed_registry.json")

    # Initialize meta-evolution populations
    architect_population = ArchitectPopulation(population_size=20)
    mutator_population = MutatorPopulation(population_size=15)

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
        compatibility_threshold=config.prey_speciation_compatibility_threshold,
        compatibility_threshold_decay_rate=config.speciation_compatibility_decay_rate,
        speciation_architecture_weight=config.speciation_architecture_weight,
        speciation_behavior_weight=config.speciation_behavior_weight,
        speciation_param_weight=config.speciation_param_weight,
        min_species_size=config.speciation_min_species_size,
        max_species_stagnation=config.speciation_max_stagnation,
        min_offspring_per_species=config.speciation_min_offspring_per_species,
        novelty_threshold=config.novelty_threshold,
        max_archive_size=config.novelty_max_archive_size,
        immigration_rate=config.novelty_immigration_rate,
        novelty_weight=config.prey_novelty_weight,
        novelty_fitness_beta=config.novelty_fitness_beta,
        cross_species_reproduction_rate=config.cross_species_reproduction_rate,
        architect_population=architect_population,
        mutator_population=mutator_population,
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
        compatibility_threshold=config.predator_speciation_compatibility_threshold,
        compatibility_threshold_decay_rate=config.speciation_compatibility_decay_rate,

        speciation_architecture_weight=config.speciation_architecture_weight,
        speciation_behavior_weight=config.speciation_behavior_weight,
        speciation_param_weight=config.speciation_param_weight,
        min_species_size=config.speciation_min_species_size,
        max_species_stagnation=config.speciation_max_stagnation,
        min_offspring_per_species=config.speciation_min_offspring_per_species,
        novelty_threshold=config.novelty_threshold,
        max_archive_size=config.novelty_max_archive_size,
        immigration_rate=config.novelty_immigration_rate,
        novelty_weight=config.predator_novelty_weight,
        novelty_fitness_beta=config.novelty_fitness_beta,
        cross_species_reproduction_rate=config.cross_species_reproduction_rate,
        architect_population=architect_population,
        mutator_population=mutator_population,
    )


    # Initialize meta-scientist systems
    meta_scientist = MetaScientist()
    evolution_modifier = EvolutionModifier()
    diagnostic_task_generator = DiagnosticTaskGenerator()

    def evolve_all_populations(gen: int) -> None:
        print(f"[EVOLVE] Generation {gen} START")

        if prey_engine:
            start = time.time()
            prey_population = prey_engine.create_next_generation(
                training_state.prey_population, gen, pop_name="prey"
            )
            if time.time() - start > 30:
                print("[WARN] Evolution step slow")
            training_state.prey_population = prey_population.genomes

        if predator_engine:
            start = time.time()
            predator_population = predator_engine.create_next_generation(
                training_state.predator_population, gen, pop_name="predator"
            )
            if time.time() - start > 30:
                print("[WARN] Evolution step slow")
            training_state.predator_population = predator_population.genomes

        print(f"[EVOLVE] Generation {gen} END")

    # Training loop
    MAX_GEN_TIME = 300  # Maximum time per generation in seconds
    stagnation_generations = 0
    last_combined_best = None
    stagnation_epsilon = 1e-3
    for generation in range(training_state.generation, config.generations):
        gen_start = time.time()

        # Generation-level circuit breaker
        if generation == 0:
            MAX_GENOMES_EVALUATED = 10

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
        print(f"[Heartbeat] Gen {generation} still alive at {time.time()}")

        skip_diagnostics = False
        gen_elapsed = time.time() - gen_start
        if gen_elapsed > MAX_GEN_TIME:
            print(f"[WARN] Generation exceeded MAX_GEN_TIME ({MAX_GEN_TIME}s) at {gen_elapsed:.1f}s — skipping diagnostics")
            skip_diagnostics = True

        # Update hall of fame
        training_state.update_hall_of_fame()

        # Meta-stagnation: if no improvement for several generations, boost exploration
        if training_state.best_prey_fitness_history and training_state.best_predator_fitness_history:
            current_best = (
                training_state.best_prey_fitness_history[-1]
                + training_state.best_predator_fitness_history[-1]
            ) / 2.0
            if last_combined_best is None:
                last_combined_best = current_best
            elif current_best <= last_combined_best + stagnation_epsilon:
                stagnation_generations += 1
            else:
                last_combined_best = current_best
                stagnation_generations = 0

            if stagnation_generations > 5:
                old_mutation_rate = config.mutation_rate
                old_arch_rate = config.architecture_mutation_rate
                config.mutation_rate = min(config.mutation_rate * 1.2, 1.0)
                config.architecture_mutation_rate = min(config.architecture_mutation_rate + 0.05, 1.0)

                if prey_engine:
                    prey_engine.mutation_rate = config.mutation_rate
                    prey_engine.architecture_mutation_rate = config.architecture_mutation_rate
                if predator_engine:
                    predator_engine.mutation_rate = config.mutation_rate
                    predator_engine.architecture_mutation_rate = config.architecture_mutation_rate

                print(
                    "Meta-Scientist: Stagnation > 5 generations. "
                    f"Increasing mutation rate {old_mutation_rate:.4f} -> {config.mutation_rate:.4f} "
                    f"and architecture mutation {old_arch_rate:.4f} -> {config.architecture_mutation_rate:.4f}"
                )

                stagnation_generations = 0

        # Compute diversity score from population
        from core.population import Population
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

        print(f"[Heartbeat] Gen {generation} still alive at {time.time()}")

        # Evolve populations (single controller)
        print("Evolving populations...")
        evolve_all_populations(generation)

        print(f"[Heartbeat] Gen {generation} still alive at {time.time()}")

        # Save checkpoint and run diagnostics
        top_prey_genome = None
        if not skip_diagnostics and config.plot_every > 0 and generation % config.plot_every == 0:
            save_coevolution_state(training_state)
            evaluator.save_seeds()
            plot_meta_gene_histograms(stats)
            plot_plastic_norm_evolution(training_state.generation_stats)
            plot_learning_rule_stats(generation, training_state.prey_population, training_state.predator_population)
            plot_learning_rule_vs_fitness(generation, training_state.prey_population, training_state.predator_population)
            plot_strategy_clustering(generation, training_state.prey_population, training_state.predator_population)
            plot_architecture_clustering(generation, training_state.prey_population, training_state.predator_population)

            # In-Lifetime Learning Curve: Pick top genome and log during single episode
            top_prey_genome = max(training_state.prey_population, key=lambda g: g.fitness)
            episode_data = evaluate_single_episode_with_logging(top_prey_genome, seed=generation, max_steps=config.max_steps)
            plot_in_lifetime_learning_curve(generation, episode_data)

        # Milestone 7: Run integrated meta-scientist experiments

        # Reduced frequency: every 10 generations instead of 20, skip after gen 300
        if generation % 10 == 0 and generation <= 300:
            print("Running integrated meta-scientist experiments...")
            
            # Analyze population failures and generate hypotheses
            combined_population = cast(List[EvolvableGenome], training_state.prey_population + training_state.predator_population)
            task_info = {'name': stage_name, 'generation': generation}
            
            analysis_results = meta_scientist.analyze_population_failures(
                combined_population,
                task_info
            )

            # Build diagnostic task suite based on primary failure mode
            worst_diagnosis = _select_primary_failure_diagnosis(analysis_results.get('failure_data', []))
            if worst_diagnosis:
                diagnostic_tasks = diagnostic_task_generator.generate_task_suite(
                    worst_diagnosis.get('diagnosis', worst_diagnosis)
                )
                target_capability = diagnostic_tasks[0].target_capability if diagnostic_tasks else None

                # Reduced from 6 to 3 tasks for performance
                diagnostic_suite = _build_targeted_task_suite(target_capability, generation, max_tasks=3)

                diagnostic_evaluator = MultiTaskEvaluator(diagnostic_suite, base_seed=config.base_seed)

                # Reduced from 6 to 3 tasks for performance
                # Ensure top_prey_genome is available for meta-scientist experiments
                if top_prey_genome is None and training_state.prey_population:
                    top_prey_genome = max(training_state.prey_population, key=lambda g: g.fitness)
                diagnostic_report = diagnostic_evaluator.run_subset_evaluation(
                    top_prey_genome,
                    num_tasks=min(3, len(diagnostic_suite.tasks)),

                    generation=generation,
                    hall_of_fame_prey=training_state.prey_hall_of_fame,
                    hall_of_fame_pred=training_state.predator_hall_of_fame,
                    current_prey=training_state.prey_population,
                    current_pred=training_state.predator_population,
                )

                training_state.generalization_reports.append(diagnostic_report)

                stage_candidates = [
                    r for r in diagnostic_report.benchmark_results
                    if r.metadata.get('task_type') == TaskType.CURRICULUM_STAGE.value
                ]
                if stage_candidates:
                    worst_stage = min(stage_candidates, key=lambda r: r.fitness)
                    stage_name_candidate = worst_stage.metadata.get('curriculum_stage')
                    if stage_name_candidate and population_stats['mean'] > 0:
                        threshold = population_stats['mean'] * 0.5
                        if worst_stage.fitness < threshold:
                            print(
                                f"[Curriculum] Diagnostic focus: {stage_name_candidate} "
                                f"(fitness {worst_stage.fitness:.2f} < {threshold:.2f})"
                            )
                            curriculum_controller.reset_to_stage(CurriculumStage[stage_name_candidate])

            # Run automated experiments based on hypotheses
            experiment_results = meta_scientist.run_automated_experiments(
                analysis_results['hypotheses'],
                combined_population,
                task_info,
                generation
            )

            # Learn from experiments and update knowledge base
            meta_scientist.learn_from_experiments(experiment_results)

            # NEUROGENESIS: Detect failure patterns with additional health metrics
            # Prepare adaptability stats from generation stats
            adaptability_stats = {
                'avg_adaptability_score': stats.get('avg_adaptability_score', 0.0),
                'avg_meta_effectiveness': stats.get('avg_meta_effectiveness', 0.0),
                'avg_reward_delta': stats.get('avg_reward_delta', 0.0),
            }
            
            # Get neural health from stats
            neural_health = stats.get('neural_health', {
                'dead_layers': 0,
                'saturated_layers': 0,
                'genomes_analyzed': 1
            })
            
            # Get species stats for both populations
            prey_species_stats = stats.get('prey_species', {'num_species': 0})
            predator_species_stats = stats.get('predator_species', {'num_species': 0})
            
            # Detect failure patterns with NeuroGenesis metrics
            failure_patterns = meta_scientist.detect_failure_patterns(
                analysis_results,
                experiment_results,
                population=combined_population,
                evolution_engine=prey_engine,  # Use prey engine as reference for chaos risk
                species_stats=prey_species_stats,
                generation=generation,
                neural_health=neural_health,
                prey_species_stats=prey_species_stats,
                predator_species_stats=predator_species_stats,
                adaptability_stats=adaptability_stats
            )
            
            # NEUROGENESIS: Execute interventions based on detected patterns
            if failure_patterns:
                interventions = meta_scientist.intervene_in_evolution(
                    failure_patterns,
                    evolution_engine=None,  # Not used - we pass prey/predator engines directly
                    curriculum_controller=curriculum_controller,
                    generation=generation,
                    prey_engine=prey_engine,
                    predator_engine=predator_engine
                )
                
                # Log intervention summary
                total_interventions = (
                    len(interventions.get('plasticity_boosts', [])) +
                    len(interventions.get('architecture_prunes', [])) +
                    len(interventions.get('species_rebalances', []))
                )
                if total_interventions > 0:
                    print(f"Meta-Scientist NeuroGenesis: Applied {total_interventions} specialized interventions")

            # Apply meta-optimizer changes to evolution engines
            if experiment_results:
                experiment_payload = {
                    'experiments': [
                        {
                            'fitness': exp.get('result', {}).get('effect_size', 0.0),
                            'parameters': {
                                'mutation_rate': prey_engine.mutation_rate,
                                'mutation_strength': prey_engine.mutation_strength,
                                'selection_pressure': getattr(prey_engine.selector, 'selection_pressure', 1.0),
                                'novelty_weight': getattr(prey_engine, 'novelty_weight', 0.5),
                            }
                        }
                        for exp in experiment_results
                    ]
                }

                modifications = evolution_modifier.optimize_evolution(experiment_payload)

                for mod in modifications:
                    if hasattr(prey_engine, mod['parameter']):
                        setattr(prey_engine, mod['parameter'], mod['new_value'])
                    if hasattr(predator_engine, mod['parameter']):
                        setattr(predator_engine, mod['parameter'], mod['new_value'])
                    if hasattr(config, mod['parameter']):
                        setattr(config, mod['parameter'], mod['new_value'])

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
        else:
            print(f"Meta-scientist: Skipped (gen {generation} > 300 or not multiple of 10)")



        training_state.generation += 1

    # Final save
    save_coevolution_state(training_state, "data/final_coevolution_state.json")
    evaluator.save_seeds("data/final_seed_registry.json")

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
