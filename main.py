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
import csv
import json
import time
import subprocess
import sys
import concurrent.futures
import threading
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, cast
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
    if config.reduce_episode_length_early:
        reduced_steps = max(1, int(round(max_steps * config.early_curriculum_reduction_factor)))
        ramp_generations = max(10, int(config.early_curriculum_generations))
        progress = min(1.0, max(0.0, float(generation) / float(ramp_generations)))
        smoothed_steps = reduced_steps + (max_steps - reduced_steps) * progress
        max_steps = int(round(smoothed_steps))
    return max_steps

def _get_opponent_sample_size(num_opponents: int, partial_enabled: bool, partial_fraction: float) -> int:
    if not partial_enabled:
        return num_opponents
    return max(1, int(round(num_opponents * partial_fraction)))

def _get_effective_partial_eval_fraction(config: "EvolutionConfig", max_steps: int, generation: int) -> float:
    """Adapt partial-evaluation fraction for expensive early generations."""
    configured = float(config.partial_eval_fraction)
    floor = 0.55 if max_steps >= 40 else 0.65
    ramp_generations = max(10, int(config.early_curriculum_generations))
    progress = min(1.0, max(0.0, float(generation) / float(ramp_generations)))
    effective = floor + (configured - floor) * progress
    return float(np.clip(effective, min(floor, configured), max(floor, configured)))

def _get_effective_opponents_per_eval(config: "EvolutionConfig", max_steps: int, generation: int) -> int:
    """Use fewer opponents in the earliest expensive rounds to bound eval time."""
    base = max(1, int(config.num_opponents_per_eval))
    if base <= 1:
        return base
    if generation < max(8, int(config.early_curriculum_generations * 0.25)) and max_steps >= 40:
        return max(1, base - 1)
    return base

def _safe_paired_correlation(x_values: List[float], y_values: List[float]) -> float:
    """Return finite correlation for paired vectors, robust to size/variance edge cases."""
    n = min(len(x_values), len(y_values))
    if n < 2:
        return 0.0
    x = np.asarray(x_values[:n], dtype=np.float64)
    y = np.asarray(y_values[:n], dtype=np.float64)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite_mask)) < 2:
        return 0.0
    x = x[finite_mask]
    y = y[finite_mask]
    if float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else 0.0

def _extract_energy_efficiency_cost(
    final_info: Optional[Dict[str, Any]],
    role: str,
    steps_survived: int,
    fallback_total: float = 0.0,
) -> float:
    """Return comparable per-step energy cost using final episode totals when available."""
    total_used: Optional[float] = None
    if isinstance(final_info, dict):
        success_signals = final_info.get('success_signals', {})
        if isinstance(success_signals, dict):
            energy_key = f"{role}_energy_used"
            if energy_key in success_signals:
                try:
                    total_used = float(success_signals.get(energy_key, 0.0))
                except Exception:
                    total_used = None

        if total_used is None:
            energy_usage = final_info.get('energy_usage', {})
            if isinstance(energy_usage, dict):
                energy_key = f"{role}_energy_used"
                if energy_key in energy_usage:
                    try:
                        total_used = float(energy_usage.get(energy_key, 0.0))
                    except Exception:
                        total_used = None

    if total_used is None or not np.isfinite(total_used):
        total_used = float(fallback_total)

    total_used = max(0.0, float(total_used))
    return float(total_used / max(1, int(steps_survived)))

def _get_effective_nonplastic_fraction(config: "EvolutionConfig", max_steps: int, generation: int) -> float:
    """Scale non-plastic comparison budget to keep early evaluation time bounded.

    Long episodes and early generations are the most expensive phase. We keep a
    floor so adaptability remains measured, then ramp to the configured target.
    """
    configured = float(config.nonplastic_check_fraction)
    if configured <= 0.0:
        return 0.0

    floor = 0.35
    # Heavier discount for longer episodes.
    if max_steps >= 80:
        floor = 0.30
    elif max_steps >= 60:
        floor = 0.33
    elif max_steps >= 40:
        floor = 0.28

    # Ramp to configured value over early curriculum.
    ramp_generations = max(10, int(config.early_curriculum_generations))
    progress = min(1.0, max(0.0, float(generation) / float(ramp_generations)))
    # When users explicitly ask for full comparisons (e.g., 1.0), ramp faster so
    # adaptability pressure is present early enough to shape evolution.
    if configured >= 0.9:
        progress = float(np.sqrt(progress))

    effective = floor + (configured - floor) * progress
    if configured >= 0.9 and generation < int(max(1, config.adaptability_boost_generations)):
        effective = max(effective, 0.55)

    return float(np.clip(effective, floor, configured))

def _ensure_seed_coverage(
    evaluator: Optional["AsyncDeterministicEvaluator"],
    generation: int,
    expected_evals: int,
) -> None:
    """
    Ensure deterministic seed coverage is recorded even when using custom
    serial/threaded evaluation paths that bypass evaluator batch methods.
    """
    if evaluator is None or expected_evals <= 0:
        return
    if not hasattr(evaluator, "seed_manager") or not hasattr(evaluator, "summarize_seed_coverage"):
        return

    try:
        summary = evaluator.summarize_seed_coverage(generation, max_examples=1)
        if int(summary.get("total", 0)) > 0:
            return

        # Register deterministic fallback seeds for this generation.
        for i in range(int(expected_evals)):
            evaluator.seed_manager.get_seed(f"g{generation}_fallback_eval{i}", offset=0)
    except Exception:
        return


def _log_heartbeat(generation: int) -> None:
    """Print a compact heartbeat line with human-readable timestamp."""
    stamp = time.strftime("%H:%M:%S", time.localtime())
    print(f"💓 HEARTBEAT | Gen {generation:04d} | {stamp}")

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

try:
    import torch_directml  # type: ignore
except Exception:
    torch_directml = None  # type: ignore

print("Event loop:", asyncio.get_event_loop_policy())

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

# Import plotting functions from diagnostics
from diagnostics.plotting import (
    compute_architecture_clustering_stats,
    plot_meta_gene_histograms,
    plot_plastic_norm_evolution,
    plot_learning_rule_stats,
    plot_learning_rule_vs_fitness,
    plot_strategy_clustering,
    plot_architecture_clustering,
    plot_in_lifetime_learning_curve,
    evaluate_single_episode_with_logging,
    evaluate_recovery_after_perturbation,
)
from diagnostics.meta_gene_entropy import MetaGeneEntropyLogger

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
from evolution.evolution import ArchitectPopulation, MutatorPopulation, BehaviorEmbedding

@dataclass
class EpisodeMetrics:
    """Structured decomposition of evaluation metrics for evolution and curriculum reasoning"""
    task_success: bool  # Whether the agent achieved the primary objective (e.g., survived, captured prey, collected food)
    episode_return: float  # Total accumulated reward over the episode
    learning_speed: float  # Rate of adaptation (plasticity effectiveness over time)
    stability: float  # Consistency of performance (variance in rewards/actions)
    energy_cost: float  # Per-step energy expenditure for cross-stage comparability
    complexity_penalty: float  # Penalty for overly complex behaviors (optional)
    novelty: float  # Novelty score (exploration of new strategies/behaviors)
    seed: int  # Random seed used for this evaluation
    stage: str  # Curriculum stage name
    adaptability: float = 0.0  # Reward delta from lifetime learning (plastic vs non-plastic)
    adaptability_measured: bool = True  # False when non-plastic baseline was skipped for this rollout
    opponent_id: Optional[str] = None  # ID of opponent genome (for co-evolution)
    instability: float = 0.0  # Variance/spikiness of plastic updates
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
    tournament_size: int = 6
    elite_count: int = 2
    mutation_rate: float = 0.25
    mutation_strength: float = 0.4
    architecture_mutation_rate: float = 0.15
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
    nonplastic_check_fraction: float = 1.0
    # Plotting/diagnostics are expensive; do them every N generations.
    plot_every: int = 5  # Reduced from 20 to 50 for performance

    # Multi-agent specific
    num_prey_per_arena: int = 10
    num_predators_per_arena: int = 3
    # Co-evolution parameters
    num_opponents_per_eval: int = 2
    hall_of_fame_size: int = 10

    # Milestone 4: Speciation + novelty archive knobs
    speciation_enabled: bool = True
    speciation_compatibility_threshold: float = 2.0
    speciation_compatibility_decay_rate: float = 400.0
    # CRITICAL FIX: Increase architecture weight in distance calculation
    # behavior_weight was dominating (0.5), causing fitness convergence to cluster species
    # Now: architecture (0.8) > behavior (0.1) = more structural diversity required
    speciation_architecture_weight: float = 0.8  # Increased from 0.6
    speciation_behavior_weight: float = 0.1  # Decreased from 0.2
    speciation_param_weight: float = 0.15  # Decreased from 0.2
    speciation_min_species_size: int = 5
    speciation_max_stagnation: int = 15
    speciation_min_offspring_per_species: int = 5
    speciation_target_species_min: int = 5
    speciation_target_species_max: int = 10
    speciation_adjust_rate: float = 0.05
    speciation_threshold_min: float = 0.1
    speciation_threshold_max: float = 5.0

    # Prey-specific diversity controls
    prey_speciation_compatibility_threshold: float = 0.75
    prey_novelty_weight: float = 0.7
    prey_min_species_enforcement: int = 3
    prey_min_species_adjust_rate: float = 0.2

    # Predator-specific diversity controls (Issue 3: Fix predator monoculture)
    # Lower compatibility threshold = stricter speciation = more species maintained.
    predator_speciation_compatibility_threshold: float = 0.75
    predator_novelty_weight: float = 1.0  # Increased from 0.7 to strongly reward diversity
    predator_min_species_enforcement: int = 3
    predator_min_species_adjust_rate: float = 0.2

    # Cross-species reproduction rate (increased to encourage diversity mixing)
    # Higher rate helps prevent any single species from dominating
    cross_species_reproduction_rate: float = 0.3  # Increased from 0.15 (15% to 30%)

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
    adaptability_weight_boost: float = 2.0
    adaptability_boost_generations: int = 50
    adaptability_taper_generations: int = 10
    # PPO inner-loop training - DISABLED: Contradicts NeuroGenesis philosophy
    ppo_training_steps: int = 100  # Number of PPO training steps per genome
    enable_ppo_inner_loop: bool = False  # PERMANENTLY DISABLED: Evolution discovers learning rules, not gradients

    # === Performance Optimization Parameters ===
    # Early stopping: terminate episode if score is catastrophically low
    early_stopping_enabled: bool = True  # Enable early stopping
    early_stopping_threshold: float = -100.0  # Catastrophic low score threshold
    early_stopping_patience: int = 5  # Steps to wait before terminating

    # Curriculum-aware episode length: reduce steps in early generations
    reduce_episode_length_early: bool = True  # Enable reduced episode length
    early_curriculum_reduction_factor: float = 0.5  # Multiply max_steps by this in early generations
    early_curriculum_generations: int = 50  # Number of generations to apply reduction

    # Partial evaluation: evaluate on subset of seeds
    partial_evaluation_enabled: bool = True  # Enable partial evaluation
    partial_eval_fraction: float = 0.75  # Fraction of seeds to evaluate (0.0-1.0)

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
        assert 0 < self.nonplastic_check_fraction <= 1, "Nonplastic check fraction must be in (0, 1]"
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
    curriculum_state: Optional[Dict[str, Any]] = None
    meta_evolution_state: Optional[Dict[str, Any]] = None
    engine_runtime_state: Optional[Dict[str, Any]] = None
    loop_runtime_state: Optional[Dict[str, Any]] = None
    
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


@dataclass(frozen=True)
class RuntimeOverrides:
    """Explicit runtime config overrides from CLI or environment."""
    base_seed: Optional[int] = None
    population_size: Optional[int] = None
    predator_population_size: Optional[int] = None
    generations: Optional[int] = None

    def has_seed_override(self) -> bool:
        return self.base_seed is not None


def _set_global_random_seeds(seed: int) -> None:
    """Apply a deterministic seed before creating random populations or controllers."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _apply_runtime_overrides_to_config(
    config: EvolutionConfig,
    runtime_overrides: Optional[RuntimeOverrides],
) -> List[str]:
    """Apply explicit runtime overrides to a config and return labels for logging."""
    if runtime_overrides is None:
        return []

    applied: List[str] = []
    if runtime_overrides.base_seed is not None:
        config.base_seed = runtime_overrides.base_seed
        applied.append(f"seed={config.base_seed}")
    if runtime_overrides.population_size is not None:
        config.population_size = runtime_overrides.population_size
        applied.append(f"population={config.population_size}")
    if runtime_overrides.predator_population_size is not None:
        config.predator_population_size = runtime_overrides.predator_population_size
        applied.append(f"predator_population={config.predator_population_size}")
    if runtime_overrides.generations is not None:
        config.generations = runtime_overrides.generations
        applied.append(f"generations={config.generations}")

    config.__post_init__()
    return applied


def _merge_runtime_overrides(
    primary: Optional[RuntimeOverrides],
    fallback: Optional[RuntimeOverrides],
) -> Optional[RuntimeOverrides]:
    """Merge two override sets with primary taking precedence field-by-field."""
    if primary is None and fallback is None:
        return None

    primary = primary or RuntimeOverrides()
    fallback = fallback or RuntimeOverrides()
    merged = RuntimeOverrides(
        base_seed=primary.base_seed if primary.base_seed is not None else fallback.base_seed,
        population_size=(
            primary.population_size
            if primary.population_size is not None
            else fallback.population_size
        ),
        predator_population_size=(
            primary.predator_population_size
            if primary.predator_population_size is not None
            else fallback.predator_population_size
        ),
        generations=primary.generations if primary.generations is not None else fallback.generations,
    )
    if (
        merged.base_seed is None
        and merged.population_size is None
        and merged.predator_population_size is None
        and merged.generations is None
    ):
        return None
    return merged


def _resize_population(
    population: List[Any],
    target_size: int,
    factory: Callable[[], Any],
    label: str,
) -> None:
    """Resize a population while keeping the strongest loaded genomes when shrinking."""
    current_size = len(population)
    if current_size == target_size:
        return

    if current_size > target_size:
        population.sort(key=lambda genome: float(getattr(genome, "fitness", 0.0)), reverse=True)
        del population[target_size:]
        print(f"[Config] Trimmed {label} population from {current_size} to {target_size}")
        return

    additions = target_size - current_size
    population.extend(factory() for _ in range(additions))
    print(f"[Config] Expanded {label} population from {current_size} to {target_size}")


def _initialize_or_resize_populations(
    training_state: TrainingState,
    runtime_overrides: Optional[RuntimeOverrides] = None,
) -> List[str]:
    """Apply config overrides and keep current populations aligned to that config."""
    applied = _apply_runtime_overrides_to_config(training_state.config, runtime_overrides)
    _set_global_random_seeds(training_state.config.base_seed)

    _resize_population(
        training_state.prey_population,
        training_state.config.population_size,
        PreyGenome.random_initialization,
        "prey",
    )
    _resize_population(
        training_state.predator_population,
        training_state.config.predator_population_size,
        PredatorGenome.random_initialization,
        "predator",
    )
    return applied


def _format_architecture_string(genome: Any) -> str:
    """Return a compact architecture string like "8-32-16-4"."""
    input_size = int(getattr(genome, "input_size", 0) or 0)
    genes = getattr(genome, "genes", None) or []

    if genes:
        dims = [input_size] + [int(getattr(gene, "output_dim", 0) or 0) for gene in genes]
        # Keep order while removing accidental duplicate consecutive dims from malformed genes.
        compact_dims: List[int] = []
        for dim in dims:
            if not compact_dims or compact_dims[-1] != dim:
                compact_dims.append(dim)
        return "-".join(str(dim) for dim in compact_dims if dim > 0)

    architecture = getattr(genome, "architecture", None)
    if isinstance(architecture, list) and architecture:
        return "-".join(str(int(dim)) for dim in architecture)

    output_size = int(getattr(genome, "output_size", 0) or 0)
    if input_size > 0 and output_size > 0:
        return f"{input_size}-{output_size}"
    return "unknown"


def _extract_species_index(species_id: Optional[str]) -> Optional[int]:
    if not species_id:
        return None
    tail = str(species_id).split("_")[-1]
    return int(tail) if tail.isdigit() else None


def _find_genome_species_index(genome: Any, engine: Optional[EvolutionEngine]) -> Optional[int]:
    """Best-effort species lookup for metadata."""
    if engine is None:
        return None
    speciation_manager = getattr(engine, "speciation_manager", None)
    if speciation_manager is None:
        return None

    genome_id = getattr(genome, "genome_id", None)
    genome_signature = getattr(genome, "signature", None)

    for species in getattr(speciation_manager, "species", []):
        for member in getattr(species, "members", []):
            if member is genome:
                return _extract_species_index(getattr(species, "species_id", None))

            if genome_id is not None and getattr(member, "genome_id", None) == genome_id:
                return _extract_species_index(getattr(species, "species_id", None))

            member_signature = getattr(member, "signature", None)
            if genome_signature is not None and member_signature == genome_signature:
                return _extract_species_index(getattr(species, "species_id", None))

    return None


def _read_existing_best_fitness(metadata_path: str) -> Optional[float]:
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "fitness" in data:
            return float(data["fitness"])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None

    return None


def _save_best_agent_artifacts(
    genome: Any,
    role: str,
    generation: int,
    species_index: Optional[int],
    models_dir: str = "models",
) -> bool:
    """Save best model (.pt) and metadata (.json). Returns True when updated."""
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"best_{role}.pt")
    metadata_path = os.path.join(models_dir, f"best_{role}.json")

    current_fitness = float(getattr(genome, "fitness", 0.0))
    existing_fitness = _read_existing_best_fitness(metadata_path)

    # Only overwrite when a new best fitness is found.
    if existing_fitness is not None and current_fitness <= existing_fitness:
        return False

    brain = genome.get_brain()
    brain.save_model(model_path)

    architecture_summary = None
    if hasattr(brain, "get_architecture_summary"):
        try:
            architecture_summary = brain.get_architecture_summary()
        except Exception:
            architecture_summary = None

    metadata_obj = getattr(genome, "metadata", None)
    if metadata_obj is not None:
        parent_ids = list(getattr(metadata_obj, "parent_ids", []))
        birth_generation = int(getattr(metadata_obj, "birth_generation", 0) or 0)
        origin_population = str(getattr(metadata_obj, "origin_population", "unknown"))
        mutation_history = list(getattr(metadata_obj, "mutation_history", []))
        last_eval_metrics = getattr(metadata_obj, "last_eval_metrics", None)
    else:
        parent_ids = []
        birth_generation = 0
        origin_population = "unknown"
        mutation_history = []
        last_eval_metrics = None

    learning_rule_net = getattr(genome, "learning_rule_net", None)
    learning_rule_info = None
    if learning_rule_net is not None:
        learning_rule_info = {
            "input_dim": int(getattr(learning_rule_net, "input_dim", 0) or 0),
            "output_dim": int(getattr(learning_rule_net, "output_dim", 0) or 0),
            "hidden_dim": int(getattr(learning_rule_net, "hidden_dim", 0) or 0),
        }

    species_id = f"species_{species_index}" if species_index is not None else None
    role_prefix = "best_prey" if role == "prey" else "best_predator"

    metadata = {
        "generation": int(generation),
        "fitness": current_fitness,
        "species": species_index,
        "architecture": _format_architecture_string(genome),
        "role": role,
        "genome_id": str(getattr(genome, "genome_id", "unknown")),
        "genome_signature": str(getattr(genome, "signature", "unknown")),
        "species_id": species_id,
        "artifact_name": role_prefix,
        "model_file": os.path.basename(model_path),
        "model_path": model_path,
        "metadata_file": os.path.basename(metadata_path),
        "metadata_path": metadata_path,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_epoch_seconds": time.time(),
        "input_size": int(getattr(genome, "input_size", 0) or 0),
        "output_size": int(getattr(genome, "output_size", 0) or 0),
        "gene_count": int(len(getattr(genome, "genes", []))),
        "module_count": int(len(getattr(genome, "modules", []))),
        "role_architecture_default": getattr(genome, "architecture", None),
        "architecture_summary": architecture_summary,
        "age": int(getattr(genome, "age", 0) or 0),
        "norm_fitness": float(getattr(genome, "norm_fitness", 0.0) or 0.0),
        "novelty_score": float(getattr(genome, "novelty_score", 0.0) or 0.0),
        "novelty_score_norm": float(getattr(genome, "novelty_score_norm", 0.0) or 0.0),
        "meta_parameters": dict(getattr(genome, "meta", {})),
        "learning_rule": getattr(genome, "learning_rule", None),
        "learning_rule_net": learning_rule_info,
        "lineage": {
            "parent_ids": parent_ids,
            "birth_generation": birth_generation,
            "origin_population": origin_population,
            "mutation_history_count": int(len(mutation_history)),
            "last_eval_metrics": last_eval_metrics,
        },
        "export": {
            "save_condition": "new_best_fitness",
            "previous_best_fitness": existing_fitness,
            "fitness_delta_vs_previous": (
                None if existing_fitness is None else current_fitness - float(existing_fitness)
            ),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return True


def save_best_agents(
    training_state: TrainingState,
    generation: int,
    prey_engine: Optional[EvolutionEngine],
    predator_engine: Optional[EvolutionEngine],
    models_dir: str = "models",
) -> None:
    """Persist best prey and predator agents to models/ as .pt + metadata JSON."""
    if training_state.prey_hall_of_fame:
        best_prey = training_state.prey_hall_of_fame[0]
    elif training_state.prey_population:
        best_prey = max(training_state.prey_population, key=lambda g: g.fitness)
    else:
        best_prey = None

    if training_state.predator_hall_of_fame:
        best_predator = training_state.predator_hall_of_fame[0]
    elif training_state.predator_population:
        best_predator = max(training_state.predator_population, key=lambda g: g.fitness)
    else:
        best_predator = None

    if best_prey is not None:
        prey_species_index = _find_genome_species_index(best_prey, prey_engine)
        updated = _save_best_agent_artifacts(
            genome=best_prey,
            role="prey",
            generation=generation,
            species_index=prey_species_index,
            models_dir=models_dir,
        )
        if updated:
            print(f"[ModelExport] Updated best prey model at {os.path.join(models_dir, 'best_prey.pt')}")

    if best_predator is not None:
        predator_species_index = _find_genome_species_index(best_predator, predator_engine)
        updated = _save_best_agent_artifacts(
            genome=best_predator,
            role="predator",
            generation=generation,
            species_index=predator_species_index,
            models_dir=models_dir,
        )
        if updated:
            print(f"[ModelExport] Updated best predator model at {os.path.join(models_dir, 'best_predator.pt')}")

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

    print(f"\n{'─'*90}")
    print(f"🚀 TRAINING ROUND {generation:04d} STARTING")
    print(f"  Stage: {stage.name}")
    print(f"  Prey Agents Ready:      {len(training_state.prey_population):4d}")
    print(f"  Predator Agents Ready:  {len(training_state.predator_population):4d}", flush=True)
    
    # Get stage configuration
    stage_config = get_stage_config(stage)

    # Reset per-generation adaptability aggregates so summary stats use this generation only.
    for genome in training_state.prey_population + training_state.predator_population:
        _reset_generation_plastic_diagnostics(genome)
    
    # Evaluate co-evolution
    start_time = time.time()
    
    # Combine hall of fame with current population for more challenging opponents
    all_prey = training_state.prey_population + training_state.prey_hall_of_fame
    all_predators = training_state.predator_population + training_state.predator_hall_of_fame
    
    # Reuse a single arena for serial evaluation (batch_size=1 for single prey-predator pair).
    arena = MultiAgentArena(
        batch_size=1,
        num_prey_per_env=1,
        num_predators_per_env=1,
    )

    effective_max_steps = _get_effective_max_steps(training_state.config, stage_config, generation)
    effective_nonplastic_fraction = _get_effective_nonplastic_fraction(
        training_state.config,
        effective_max_steps,
        generation,
    )
    effective_partial_eval_fraction = _get_effective_partial_eval_fraction(
        training_state.config,
        effective_max_steps,
        generation,
    )
    effective_opponents_per_eval = _get_effective_opponents_per_eval(
        training_state.config,
        effective_max_steps,
        generation,
    )

    # Sample opponents for evaluation after adaptive scheduling is applied.
    num_prey_opponents = min(effective_opponents_per_eval, len(all_predators))
    num_pred_opponents = min(effective_opponents_per_eval, len(all_prey))

    if effective_nonplastic_fraction < float(training_state.config.nonplastic_check_fraction):
        print(
            f"  ⏱️  Adaptive non-plastic checks: {effective_nonplastic_fraction:.2f} "
            f"(base={training_state.config.nonplastic_check_fraction:.2f}, steps={effective_max_steps})",
            flush=True,
        )
    if (
        effective_partial_eval_fraction < float(training_state.config.partial_eval_fraction)
        or effective_opponents_per_eval < int(training_state.config.num_opponents_per_eval)
    ):
        print(
            f"  ⏱️  Adaptive eval budget: partial={effective_partial_eval_fraction:.2f} "
            f"(base={training_state.config.partial_eval_fraction:.2f}), "
            f"opponents={effective_opponents_per_eval}",
            flush=True,
        )
    
    # Evaluate prey population
    print("  ⏱️  Testing prey agents against predators...", flush=True)
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
            nonplastic_check_fraction=effective_nonplastic_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=effective_partial_eval_fraction,
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
            nonplastic_check_fraction=effective_nonplastic_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=effective_partial_eval_fraction,
            fitness_cache_enabled=training_state.config.fitness_cache_enabled,
            fitness_cache_max_size=training_state.config.fitness_cache_max_size,
            early_stopping_enabled=training_state.config.early_stopping_enabled,
            early_stopping_threshold=training_state.config.early_stopping_threshold,
            early_stopping_patience=training_state.config.early_stopping_patience,
        )

    # Evaluate predator population
    print("  ⏱️  Testing predator agents against prey...", flush=True)
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
            nonplastic_check_fraction=effective_nonplastic_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=effective_partial_eval_fraction,
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
            nonplastic_check_fraction=effective_nonplastic_fraction,
            partial_evaluation_enabled=training_state.config.partial_evaluation_enabled,
            partial_eval_fraction=effective_partial_eval_fraction,
            fitness_cache_enabled=training_state.config.fitness_cache_enabled,
            fitness_cache_max_size=training_state.config.fitness_cache_max_size,
            early_stopping_enabled=training_state.config.early_stopping_enabled,
            early_stopping_threshold=training_state.config.early_stopping_threshold,
            early_stopping_patience=training_state.config.early_stopping_patience,
        )
    
    eval_time = time.time() - start_time

    expected_prey_evals = len(training_state.prey_population) * int(num_prey_opponents)
    expected_pred_evals = len(training_state.predator_population) * int(num_pred_opponents)
    _ensure_seed_coverage(
        evaluator,
        generation,
        expected_evals=expected_prey_evals + expected_pred_evals,
    )
    
    # Update best fitness history
    best_prey_fitness = max(prey_fitnesses) if prey_fitnesses else 0.0
    best_predator_fitness = max(predator_fitnesses) if predator_fitnesses else 0.0
    
    training_state.best_prey_fitness_history.append(best_prey_fitness)
    training_state.best_predator_fitness_history.append(best_predator_fitness)

    # Log META gene distribution
    combined_population = cast(List[Any], training_state.prey_population + training_state.predator_population)
    meta_gain = [g.meta["reward_gain"] for g in combined_population]
    meta_bias = [g.meta["reward_bias"] for g in combined_population]

    # Log plasticity magnitudes per generation (sample once, not per update).
    # `plastic_norms` tracks cumulative RMS(DeltaW) per episode.
    from core.torch_brain import PlasticLinear
    # Only collect from the last evaluation to avoid O(steps × layers × population) growth
    plastic_norms = PlasticLinear.plastic_norms.copy()
    plastic_weight_norms = PlasticLinear.plastic_weight_rms_norms.copy()
    PlasticLinear.plastic_norms.clear()  # Clear for next generation
    PlasticLinear.plastic_weight_rms_norms.clear()  # Clear for next generation

    if plastic_norms:
        mean_plastic_norm = float(np.mean(plastic_norms))
        max_plastic_norm = float(np.max(plastic_norms))
        p95_plastic_norm = float(np.percentile(plastic_norms, 95))
        mean_plastic_weight_norm = float(np.mean(plastic_weight_norms)) if plastic_weight_norms else 0.0
        max_plastic_weight_norm = float(np.max(plastic_weight_norms)) if plastic_weight_norms else 0.0
        p95_plastic_weight_norm = float(np.percentile(plastic_weight_norms, 95)) if plastic_weight_norms else 0.0
        print("\n⚡ LEARNING MECHANISMS (Weight Modifications)")
        print(f"  {'─'*86}")
        print(f"  Plastic Delta RMS (episode cumulative): Mean {mean_plastic_norm:.4f}  |  Max {max_plastic_norm:.4f}  |  95th {p95_plastic_norm:.4f}")
        print(f"  Plastic Weight RMS: Mean {mean_plastic_weight_norm:.4f}  |  Max {max_plastic_weight_norm:.4f}  |  95th {p95_plastic_weight_norm:.4f}")
    else:
        mean_plastic_norm = 0.0
        max_plastic_norm = 0.0
        p95_plastic_norm = 0.0
        mean_plastic_weight_norm = 0.0
        max_plastic_weight_norm = 0.0
        p95_plastic_weight_norm = 0.0

    adaptability_scores = []
    meta_effectiveness_scores = []
    reward_before_learning = []
    reward_after_learning = []
    reward_delta_learning = []
    reward_delta_raw_learning = []
    energy_costs = []
    learning_speeds = []
    improving_learning_speeds = []
    stabilities = []
    novelties = []
    success_rates = []
    instability_scores = []  # CRITICAL FIX 2: Track instability
    paired_learning_speeds = []
    paired_reward_deltas = []
    metrics_missing = 0

    for genome in training_state.prey_population + training_state.predator_population:
        plastic_advantage: Optional[float] = None
        adaptability_score: Optional[float] = None
        reward_before: float = 0.0
        reward_after: float = 0.0
        instability: float = 0.0
        plastic_diag = getattr(genome, 'plastic_diagnostics', None)
        if isinstance(plastic_diag, dict):
            gen_count = int(plastic_diag.get('gen_plastic_adv_count', 0))
            if gen_count > 0:
                plastic_advantage = float(plastic_diag.get('gen_plastic_adv_sum', 0.0)) / float(gen_count)
                reward_before = float(plastic_diag.get('gen_reward_before_sum', 0.0)) / float(gen_count)
                reward_after = float(plastic_diag.get('gen_reward_after_sum', 0.0)) / float(gen_count)
                instability = float(plastic_diag.get('gen_instability_sum', 0.0)) / float(gen_count)
                if 'gen_adaptability_score_sum' in plastic_diag:
                    adaptability_score = float(plastic_diag.get('gen_adaptability_score_sum', 0.0)) / float(gen_count)
                else:
                    adaptability_score = _compute_adaptability_score(reward_before, reward_after)
            elif plastic_diag.get('plastic_advantage_measured', False):
                # Backward-compatible fallback for older checkpoints/paths.
                plastic_advantage = float(plastic_diag.get('plastic_advantage', 0.0))
                reward_before = float(plastic_diag.get('reward_before_learning', 0.0))
                reward_after = float(plastic_diag.get('reward_after_learning', 0.0))
                instability = float(plastic_diag.get('instability', 0.0))
                adaptability_score = _compute_adaptability_score(reward_before, reward_after)

        if plastic_advantage is not None:
            instability_scores.append(float(instability))

            # Use robust bounded adaptability score from paired before/after rewards.
            if adaptability_score is None:
                adaptability_score = _compute_adaptability_score(reward_before, reward_after)
            adaptability_scores.append(float(np.clip(adaptability_score, 0.0, 1.0)))
            reward_before_learning.append(float(reward_before))
            reward_after_learning.append(float(reward_after))
            reward_delta_learning.append(float(max(0.0, reward_after - reward_before)))
            reward_delta_raw_learning.append(float(plastic_advantage))

            # Meta-parameter effectiveness based on how well they enable plasticity
            local_meta_gain = genome.meta.get('reward_gain', 1.0)
            local_meta_bias = genome.meta.get('reward_bias', 0.0)
            plastic_lr = genome.meta.get('plastic_lr', 1.0)

            # Meta-effectiveness should reflect both observed learning outcomes and
            # whether meta-parameters are in healthy operating ranges.
            gain_effectiveness = 1.0 - min(abs(float(local_meta_gain) - 1.5) / 4.0, 1.0)
            lr_effectiveness = 1.0 - min(abs(float(plastic_lr) - 1.2) / 6.0, 1.0)
            bias_effectiveness = 1.0 - min(abs(float(local_meta_bias)) / 3.5, 1.0)
            param_quality = float(np.clip(
                0.45 * gain_effectiveness + 0.40 * lr_effectiveness + 0.15 * bias_effectiveness,
                0.0,
                1.0,
            ))

            delta_scale = max(1.0, abs(float(reward_before)), abs(float(reward_after)))
            signed_delta = float(plastic_advantage) / delta_scale
            delta_effectiveness = float(np.clip(0.5 + 0.5 * np.tanh(3.0 * signed_delta), 0.0, 1.0))
            adapt_effectiveness = float(np.clip(adaptability_score if adaptability_score is not None else 0.0, 0.0, 1.0))
            stability_effectiveness = float(np.clip(1.0 - float(instability), 0.0, 1.0))

            meta_effectiveness = float(np.clip(
                0.45 * adapt_effectiveness +
                0.25 * delta_effectiveness +
                0.20 * param_quality +
                0.10 * stability_effectiveness,
                0.0,
                1.0,
            ))

            meta_effectiveness_scores.append(float(meta_effectiveness))
        else:
            instability_scores.append(0.0)  # CRITICAL FIX 2: Default instability


        last_metrics = getattr(genome, 'last_eval_metrics', None)
        if isinstance(last_metrics, dict):
            current_learning_speed = float(last_metrics.get('learning_speed', 0.0))
            if isinstance(plastic_diag, dict):
                ls_count = int(plastic_diag.get('gen_learning_speed_count', 0))
                if ls_count > 0:
                    current_learning_speed = float(plastic_diag.get('gen_learning_speed_sum', 0.0)) / float(ls_count)
            energy_costs.append(float(last_metrics.get('energy_cost', 0.0)))
            learning_speeds.append(current_learning_speed)
            if plastic_advantage is not None and float(plastic_advantage) > 0.0:
                improving_learning_speeds.append(current_learning_speed)
            stabilities.append(float(last_metrics.get('stability', 0.0)))
            novelties.append(float(last_metrics.get('novelty', 0.0)))
            success_rates.append(1.0 if last_metrics.get('task_success', False) else 0.0)
            if plastic_advantage is not None:
                paired_learning_speeds.append(current_learning_speed)
                paired_reward_deltas.append(float(plastic_advantage))
        else:
            metrics_missing += 1

    if metrics_missing == len(training_state.prey_population) + len(training_state.predator_population):
        print("⚠️  Evaluator metrics missing for all genomes; check last_eval_metrics wiring")
    elif energy_costs and learning_speeds and stabilities and novelties and success_rates:
        print("\n📈 RAW EVALUATION SNAPSHOT (min/avg/max)")
        print(f"  {'─'*86}")
        print(
            f"  Energy:    {np.min(energy_costs):.3f} / {np.mean(energy_costs):.3f} / {np.max(energy_costs):.3f}"
            f"  │  Learn: {np.min(learning_speeds):.3f} / {np.mean(learning_speeds):.3f} / {np.max(learning_speeds):.3f}"
        )
        print(
            f"  Stability: {np.min(stabilities):.3f} / {np.mean(stabilities):.3f} / {np.max(stabilities):.3f}"
            f"  │  Novelty: {np.min(novelties):.3f} / {np.mean(novelties):.3f} / {np.max(novelties):.3f}"
            f"  │  Success: {np.mean(success_rates):.3f}"
        )
        # PLASTICITY EFFECTIVENESS DIAGNOSTIC: Track if learning is translating to performance
        if reward_delta_learning:
            correlation = _safe_paired_correlation(paired_learning_speeds, paired_reward_deltas)
            print(
                f"  Plasticity: Adaptability {np.min(adaptability_scores):.4f} / {np.mean(adaptability_scores):.4f} / {np.max(adaptability_scores):.4f}"
                f"  │  Reward Delta(raw) {np.min(reward_delta_raw_learning):.4f} / {np.mean(reward_delta_raw_learning):.4f} / {np.max(reward_delta_raw_learning):.4f}"
                f"  │  Reward Delta(realized) {np.min(reward_delta_learning):.4f} / {np.mean(reward_delta_learning):.4f} / {np.max(reward_delta_learning):.4f}"
                f"  │  Learn↔Adapt Corr {correlation:.3f}"
            )

    avg_adaptability = float(np.mean(adaptability_scores)) if adaptability_scores else 0.0
    avg_meta_effectiveness = float(np.mean(meta_effectiveness_scores)) if meta_effectiveness_scores else 0.0
    avg_reward_before = float(np.mean(reward_before_learning)) if reward_before_learning else 0.0
    avg_reward_after = float(np.mean(reward_after_learning)) if reward_after_learning else 0.0
    avg_reward_delta = float(np.mean(reward_delta_learning)) if reward_delta_learning else 0.0
    avg_reward_delta_raw = float(np.mean(reward_delta_raw_learning)) if reward_delta_raw_learning else 0.0
    avg_energy_cost = float(np.mean(energy_costs)) if energy_costs else 0.0
    if improving_learning_speeds:
        avg_learning_speed = float(np.mean(improving_learning_speeds))
    else:
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
        'mean_plastic_weight_norm': mean_plastic_weight_norm,
        'max_plastic_weight_norm': max_plastic_weight_norm,
        'p95_plastic_weight_norm': p95_plastic_weight_norm,
        'avg_adaptability_score': avg_adaptability,
        'avg_meta_effectiveness': avg_meta_effectiveness,
        'avg_reward_before_learning': avg_reward_before,
        'avg_reward_after_learning': avg_reward_after,
        'avg_reward_delta': avg_reward_delta,
        'avg_reward_delta_raw': avg_reward_delta_raw,
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
            
            # CRITICAL ALERT: Detect predator collapse early
            if predator_species_count == 1 and len(training_state.predator_population) > 1:
                largest_species_size = max(
                    predator_species.get('species_sizes', [0]) if isinstance(predator_species.get('species_sizes'), list) else [0]
                )
                collapse_ratio = largest_species_size / max(len(training_state.predator_population), 1)
                
                if collapse_ratio > 0.95:
                    print(f"\n❌ PREDATOR COLLAPSE DETECTED at Gen {generation}!")
                    print(f"   {largest_species_size}/{len(training_state.predator_population)} agents in single species")
                    print(f"   This will KILL co-evolution. Applying emergency recovery...")
                    
                    # Trigger emergency speciation reset
                    if predator_engine.speciation_manager is not None:
                        old_threshold = predator_engine.speciation_manager.compatibility_threshold
                        new_threshold = max(
                            training_state.config.speciation_threshold_min,
                            float(old_threshold) * 0.7,
                        )
                        predator_engine.speciation_manager.compatibility_threshold = new_threshold
                        print(
                            f"   Emergency: compatibility_threshold reduced to "
                            f"{new_threshold:.3f} (was {old_threshold:.3f})"
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

    # Fold mutation-strategy quality into the reported meta effectiveness.
    # This keeps the dashboard aligned with what the mutator population is doing.
    _best_mutator_for_metrics = mutator_population.get_best_strategy()
    if _best_mutator_for_metrics:
        mutator_effectiveness = float(np.clip(_best_mutator_for_metrics.get('current_effectiveness', 0.5), 0.0, 1.0))
        mutator_meta_raw = float(_best_mutator_for_metrics.get('meta_fitness', 0.0))
        mutator_meta_score = float(np.clip(0.5 + 0.5 * np.tanh(mutator_meta_raw / 5.0), 0.0, 1.0))
        success_values = [float(v) for v in mutation_success_rates.values()] if mutation_success_rates else []
        mutation_success_score = float(np.clip(np.mean(success_values), 0.0, 1.0)) if success_values else 0.0

        base_meta_effectiveness = float(np.clip(stats.get('avg_meta_effectiveness', 0.0), 0.0, 1.0))
        stats['avg_meta_effectiveness'] = float(np.clip(
            0.55 * base_meta_effectiveness +
            0.30 * mutator_effectiveness +
            0.10 * mutator_meta_score +
            0.05 * mutation_success_score,
            0.0,
            1.0,
        ))

    # Use evolved mutation strategies to adapt main evolution engines
    adaptive_rates = mutator_population.get_adaptive_rates()
    used_strategy = mutator_population.get_best_strategy()  # Track which strategy was used for effectiveness update
    if adaptive_rates:
        target_weight = float(adaptive_rates.get('weight_rate', training_state.config.mutation_rate))
        target_arch = float(adaptive_rates.get('arch_rate', training_state.config.architecture_mutation_rate))

        # Smooth and bound mutator-driven rate updates to avoid oscillations.
        blend = 0.25
        stable_weight = float(np.clip(
            (1.0 - blend) * training_state.config.mutation_rate + blend * target_weight,
            0.1,
            0.3,
        ))
        stable_arch = float(np.clip(
            (1.0 - blend) * training_state.config.architecture_mutation_rate + blend * target_arch,
            0.01,
            0.10,
        ))

        training_state.config.mutation_rate = stable_weight
        training_state.config.architecture_mutation_rate = stable_arch

        if prey_engine:
            prey_engine.mutation_rate = stable_weight
            prey_engine.architecture_mutation_rate = stable_arch

        if predator_engine:
            predator_engine.mutation_rate = stable_weight
            predator_engine.architecture_mutation_rate = stable_arch

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
        print(f"   🔬 Meta-Evolution: Best network architecture design score: {best_architect.get('meta_fitness', 0.0):.3f}")
    if best_mutator:
        print(f"   🔬 Meta-Evolution: Best mutation strategy effectiveness: {best_mutator.get('current_effectiveness', 0.0):.3f}")

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
        # Mark adaptability comparison as not measured for this evaluation pass.
        if prey_genome.plastic_diagnostics is None:
            prey_genome.plastic_diagnostics = {}
        prey_genome.plastic_diagnostics['plastic_advantage_measured'] = False
        prey_genome.plastic_diagnostics['plastic_advantage'] = 0.0
        prey_genome.plastic_diagnostics['reward_before_learning'] = float(plastic_prey_reward)
        prey_genome.plastic_diagnostics['reward_after_learning'] = float(plastic_prey_reward)

        if predator_genome.plastic_diagnostics is None:
            predator_genome.plastic_diagnostics = {}
        predator_genome.plastic_diagnostics['plastic_advantage_measured'] = False
        predator_genome.plastic_diagnostics['plastic_advantage'] = 0.0
        predator_genome.plastic_diagnostics['reward_before_learning'] = float(plastic_pred_reward)
        predator_genome.plastic_diagnostics['reward_after_learning'] = float(plastic_pred_reward)

        # Compute fitness from metrics (weighted combination)
        plastic_prey_metrics.adaptability = 0.0
        plastic_pred_metrics.adaptability = 0.0
        plastic_prey_metrics.adaptability_measured = False
        plastic_pred_metrics.adaptability_measured = False
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
    prey_adaptability_score = _compute_adaptability_score(
        reward_before=nonplastic_prey_reward,
        reward_after=plastic_prey_reward,
        learning_speed=plastic_prey_metrics.learning_speed,
    )
    pred_adaptability_score = _compute_adaptability_score(
        reward_before=nonplastic_pred_reward,
        reward_after=plastic_pred_reward,
        learning_speed=plastic_pred_metrics.learning_speed,
    )

    # Use the paired non-plastic evaluation as a safety floor. Plasticity should
    # improve the policy, not be allowed to drag the base task return below the
    # known non-plastic performance for the same arena seed.
    final_prey_reward = max(plastic_prey_reward, nonplastic_prey_reward)
    final_pred_reward = max(plastic_pred_reward, nonplastic_pred_reward)

    # Store plastic advantage in diagnostics for adaptability calculation
    _accumulate_generation_plastic_diagnostics(
        prey_genome,
        plastic_advantage=float(plastic_advantage_prey),
        reward_before=float(nonplastic_prey_reward),
        reward_after=float(final_prey_reward),
        adaptability_score=float(prey_adaptability_score),
    )
    _accumulate_generation_plastic_diagnostics(
        predator_genome,
        plastic_advantage=float(plastic_advantage_pred),
        reward_before=float(nonplastic_pred_reward),
        reward_after=float(final_pred_reward),
        adaptability_score=float(pred_adaptability_score),
    )

    # Update metrics with final rewards and compute fitness
    plastic_prey_metrics.episode_return = final_prey_reward
    plastic_pred_metrics.episode_return = final_pred_reward
    plastic_prey_metrics.task_success = final_prey_reward > 0
    plastic_pred_metrics.task_success = final_pred_reward > 0
    plastic_prey_metrics.adaptability = float(prey_adaptability_score)
    plastic_pred_metrics.adaptability = float(pred_adaptability_score)
    plastic_prey_metrics.adaptability_measured = True
    plastic_pred_metrics.adaptability_measured = True

    prey_fitness = compute_fitness_from_metrics(plastic_prey_metrics, brain=prey_genome.brain)
    predator_fitness = compute_fitness_from_metrics(plastic_pred_metrics, brain=predator_genome.brain)

    _store_last_eval_metrics(prey_genome, plastic_prey_metrics, prey_fitness)
    _store_last_eval_metrics(predator_genome, plastic_pred_metrics, predator_fitness)

    return (prey_fitness, plastic_prey_metrics), (predator_fitness, plastic_pred_metrics)


def _to_numpy(x):
    """Convert torch tensors or other array-likes to numpy arrays."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

def _reset_generation_plastic_diagnostics(genome) -> None:
    """Reset per-generation plasticity aggregates while preserving other diagnostics."""
    if getattr(genome, "plastic_diagnostics", None) is None:
        genome.plastic_diagnostics = {}

    diag = genome.plastic_diagnostics
    diag["gen_plastic_adv_sum"] = 0.0
    diag["gen_reward_before_sum"] = 0.0
    diag["gen_reward_after_sum"] = 0.0
    diag["gen_adaptability_score_sum"] = 0.0
    diag["gen_instability_sum"] = 0.0
    diag["gen_learning_speed_sum"] = 0.0
    diag["gen_plastic_adv_count"] = 0
    diag["gen_learning_speed_count"] = 0
    diag["plastic_advantage_measured"] = False
    diag["plastic_advantage"] = 0.0

def _accumulate_generation_plastic_diagnostics(
    genome,
    plastic_advantage: float,
    reward_before: float,
    reward_after: float,
    adaptability_score: Optional[float] = None,
) -> None:
    """Accumulate plasticity comparison outcomes for robust generation-level reporting."""
    if getattr(genome, "plastic_diagnostics", None) is None:
        genome.plastic_diagnostics = {}

    diag = genome.plastic_diagnostics
    count = int(diag.get("gen_plastic_adv_count", 0))
    count += 1
    diag["gen_plastic_adv_count"] = count
    diag["gen_plastic_adv_sum"] = float(diag.get("gen_plastic_adv_sum", 0.0)) + float(plastic_advantage)
    diag["gen_reward_before_sum"] = float(diag.get("gen_reward_before_sum", 0.0)) + float(reward_before)
    diag["gen_reward_after_sum"] = float(diag.get("gen_reward_after_sum", 0.0)) + float(reward_after)
    if adaptability_score is not None:
        score_val = float(adaptability_score)
        if np.isfinite(score_val):
            diag["gen_adaptability_score_sum"] = float(diag.get("gen_adaptability_score_sum", 0.0)) + score_val

    instability = diag.get("instability", None)
    if isinstance(instability, (int, float)):
        diag["gen_instability_sum"] = float(diag.get("gen_instability_sum", 0.0)) + float(instability)

    # Keep latest comparison for backward compatibility with existing consumers.
    diag["plastic_advantage_measured"] = True
    diag["plastic_advantage"] = float(plastic_advantage)
    diag["reward_before_learning"] = float(reward_before)
    diag["reward_after_learning"] = float(reward_after)

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


def _safe_array_mean(values: Any, default: float = 0.0) -> float:
    """Return mean for non-empty arrays/lists, else default."""
    arr = np.asarray(values)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr))


def _compute_adaptability_score(
    reward_before: float,
    reward_after: float,
    learning_speed: Optional[float] = None,
) -> float:
    """
    Compute a bounded 0-1 adaptability score from paired non-plastic/plastic rewards.

    Keeps adaptability comparable across stages where reward magnitudes differ,
    and lightly boosts the score when in-episode learning speed is higher.
    """
    before = float(reward_before)
    after = float(reward_after)
    if not (np.isfinite(before) and np.isfinite(after)):
        return 0.0

    improvement = after - before
    if improvement <= 0.0:
        return 0.0

    # Use reward-scale-aware normalization. A hard 1.0 floor can suppress
    # adaptability in early stages where reward magnitudes are naturally < 1.
    scale = max(0.5, abs(before), abs(after))
    gain_ratio = float(max(0.0, improvement / scale))
    gain_score = float(gain_ratio / (gain_ratio + 0.2)) if gain_ratio > 0.0 else 0.0

    if learning_speed is None:
        return float(np.clip(gain_score, 0.0, 1.0))

    speed_val = float(learning_speed)
    if not np.isfinite(speed_val):
        return float(np.clip(gain_score, 0.0, 1.0))

    speed = float(np.clip(speed_val, 0.0, 1.0))
    # Blend realized reward improvement with in-episode learning speed.
    return float(np.clip(0.85 * gain_score + 0.15 * speed, 0.0, 1.0))

def _compute_episode_learning_speed(rewards: List[float]) -> float:
    """
    Measure in-lifetime adaptation from reward improvement within one episode.
    Positive values mean the agent performed better near the end than the start.
    """
    # Handle empty or very short reward lists to avoid "Mean of empty slice" warning
    if rewards is None or len(rewards) < 2:
        return 0.0
    
    if len(rewards) < 6:
        return 0.0

    arr = np.asarray(rewards, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size < 6:
        return 0.0

    window = max(3, len(arr) // 5)
    early = float(np.mean(arr[:window]))
    late = float(np.mean(arr[-window:]))
    if not np.isfinite(early) or not np.isfinite(late):
        return 0.0

    delta = late - early
    x = np.arange(len(arr), dtype=np.float32)
    slope = float(np.polyfit(x, arr, 1)[0]) if len(arr) >= 3 else 0.0

    # Use reward variability (not absolute reward level) so high baseline reward
    # does not suppress measurable in-episode improvement.
    reward_std = float(np.std(arr))
    reward_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    step_deltas = np.diff(arr)
    step_noise = float(np.std(step_deltas)) if step_deltas.size > 0 else reward_std
    adaptive_floor = max(1e-5, step_noise * 0.25)
    scale = max(adaptive_floor, reward_std, reward_iqr)

    delta_ratio = max(0.0, delta / scale)
    trend_ratio = max(0.0, slope * len(arr) / scale)

    # Bounded index with better sensitivity in low-improvement regimes.
    delta_score = delta_ratio / (delta_ratio + 0.08) if delta_ratio > 0.0 else 0.0
    trend_score = trend_ratio / (trend_ratio + 0.08) if trend_ratio > 0.0 else 0.0

    # Sparse-reward uplift: increased positive-reward hit rate later in episode.
    hit_eps = max(1e-5, scale * 0.1)
    early_hits = float(np.mean(arr[:window] > hit_eps))
    late_hits = float(np.mean(arr[-window:] > hit_eps))
    hit_improvement = max(0.0, late_hits - early_hits)

    return float(np.clip(0.65 * delta_score + 0.25 * trend_score + 0.10 * hit_improvement, 0.0, 1.0))

def _compute_behavioral_stability(rewards: List[float]) -> float:
    """
    Convert reward volatility into a 0-1 consistency score (higher is better).
    """
    # Handle empty or very short reward lists to avoid "Mean of empty slice" warning
    if rewards is None or len(rewards) < 2:
        return 0.5

    if len(rewards) < 3:
        return 0.5

    arr = np.asarray(rewards, dtype=np.float32)
    # Check for empty array after conversion
    if arr.size == 0:
        return 0.5
    
    reward_volatility = float(np.std(arr))
    # Handle potential NaN in np.diff for very short arrays
    diff_arr = np.diff(arr)
    if diff_arr.size == 0:
        return 0.5
    delta_volatility = float(np.std(diff_arr))
    volatility = reward_volatility + 0.7 * delta_volatility
    return float(1.0 / (1.0 + volatility))

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
        'adaptability_measured': bool(metrics.adaptability_measured),
        'instability': float(metrics.instability),
    }

    # Aggregate learning-speed signal across all opponent rollouts in this
    # generation so reporting is not biased by the last evaluated opponent.
    if getattr(genome, 'plastic_diagnostics', None) is None:
        genome.plastic_diagnostics = {}
    diag = genome.plastic_diagnostics
    ls_val = float(metrics.learning_speed)
    if np.isfinite(ls_val):
        diag['gen_learning_speed_sum'] = float(diag.get('gen_learning_speed_sum', 0.0)) + ls_val
        diag['gen_learning_speed_count'] = int(diag.get('gen_learning_speed_count', 0)) + 1

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
    
    speeds = np.asarray(learning_speeds, dtype=np.float32)
    speeds = speeds[np.isfinite(speeds)]
    if speeds.size == 0:
        return
    speeds = np.clip(speeds, 0.0, None)

    _LEARNING_SPEED_STATS['min'] = float(np.min(speeds))
    _LEARNING_SPEED_STATS['max'] = float(np.max(speeds))
    _LEARNING_SPEED_STATS['mean'] = float(np.mean(speeds))
    _LEARNING_SPEED_STATS['std'] = float(np.std(speeds)) if speeds.size > 1 else 1.0
    _LEARNING_SPEED_STATS['initialized'] = True


def _normalize_learning_speed(learning_speed: float) -> float:
    """Normalize learning speed using z-score normalization"""
    global _LEARNING_SPEED_STATS
    
    if not _LEARNING_SPEED_STATS['initialized']:
        # Fallback: if not initialized, use raw value capped at reasonable range
        return float(np.clip(learning_speed, 0.0, 2.0))
    
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

    # Learning speed should only help if it translates into better reward.
    # Otherwise, fast but harmful plastic updates are selected and avg_reward_delta goes negative.
    normalized_learn_speed = _normalize_learning_speed(metrics.learning_speed)
    instability_penalty = float(max(0.0, metrics.instability))
    if not metrics.adaptability_measured:
        # Neutral path: comparison was intentionally skipped, so avoid injecting
        # either reward or penalty from adaptability-coupled learning-speed terms.
        pass
    elif metrics.adaptability > 0.0:
        # Reward learning speed only when plasticity is helping reward within-lifetime.
        helpfulness = float(np.clip(metrics.adaptability, 0.0, 1.0))
        fitness += normalized_learn_speed * (0.8 + 0.7 * helpfulness)
        fitness += metrics.learning_speed * (0.8 + 1.2 * helpfulness)
    else:
        # Penalize "learning" signatures that coincide with non-positive adaptability.
        fitness -= normalized_learn_speed * 0.8
        fitness -= max(0.0, metrics.learning_speed) * 0.6
        fitness -= abs(min(0.0, metrics.adaptability)) * 0.6
        fitness -= instability_penalty * 0.45

    # Even helpful plasticity should prefer stable learners over spiky ones.
    fitness -= instability_penalty * 0.2

    # Stability bonus (metrics.stability is a 0-1 consistency score).
    fitness += metrics.stability * 0.2

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
        # Multiplicative penalty: for positive fitness, shrink it; for negative fitness,
        # amplify the magnitude so dead neurons always make fitness worse regardless of sign.
        if fitness >= 0:
            fitness *= health_penalty
        else:
            fitness *= (2.0 - health_penalty)

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
    prey_reward_ema = 0.0
    pred_reward_ema = 0.0
    ema_alpha = 0.1

    predator_brain = PredatorPackBrain(predator_genome)

    catastrophic_steps = 0
    final_info: Optional[Dict[str, Any]] = None

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
        final_info = info

        # Use rewards directly (no inverted multiplier that destroys signal)
        prey_reward = _to_numpy(r_prey)
        pred_reward = _to_numpy(r_pred)

        # Plasticity learns from reward surprise (advantage), not raw reward level.
        prey_r = _safe_array_mean(prey_reward, default=0.0)
        pred_r = _safe_array_mean(pred_reward, default=0.0)
        prey_adv = prey_r - prey_reward_ema
        pred_adv = pred_r - pred_reward_ema
        prey_reward_ema = (1.0 - ema_alpha) * prey_reward_ema + ema_alpha * prey_r
        pred_reward_ema = (1.0 - ema_alpha) * pred_reward_ema + ema_alpha * pred_r

        # Conservative plasticity gate: update on clear positive surprise.
        # Keep the threshold slightly permissive to avoid learning stalls when
        # rewards are low-magnitude, while still avoiding noisy micro-updates.
        prey_signal = float(np.clip(prey_adv, 0.0, 1.0))
        pred_signal = float(np.clip(pred_adv, 0.0, 1.0))
        if prey_signal > 0.002:
            prey_genome.brain.update_plasticity(prey_signal, done=False)
        if pred_signal > 0.002:
            predator_genome.brain.update_plasticity(pred_signal, done=False)

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

    # Learning speed must represent behavioral improvement, not weight motion magnitude.
    prey_learning_speed = _compute_episode_learning_speed(prey_rewards)
    if hasattr(prey_genome.brain, 'get_plastic_diagnostics'):
        plastic_diag = prey_genome.brain.get_plastic_diagnostics()
        if prey_genome.plastic_diagnostics is None:
            prey_genome.plastic_diagnostics = {}
        prey_genome.plastic_diagnostics.update(plastic_diag)
        if 'mean_plastic_delta' in plastic_diag:
            prey_genome.plastic_diagnostics['mean_final_plastic_delta'] = float(plastic_diag['mean_plastic_delta'])
        prey_genome.plastic_diagnostics['episode_reward_learning_speed'] = float(prey_learning_speed)
    prey_instability = float(prey_genome.plastic_diagnostics.get('instability', 0.0)) if prey_genome.plastic_diagnostics else 0.0

    # Compute metrics
    prey_action_entropy = _action_entropy(prey_action_counts)
    pred_action_entropy = _action_entropy(pred_action_counts)

    prey_energy_cost = _extract_energy_efficiency_cost(final_info, "prey", steps_survived, prey_energy_cost)
    predator_energy_cost = _extract_energy_efficiency_cost(final_info, "predator", steps_survived, predator_energy_cost)

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

    success_signals = final_info.get('success_signals', {}) if isinstance(final_info, dict) else {}
    predator_captures = int(success_signals.get('predator_captures', 0)) if isinstance(success_signals, dict) else 0
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
        stability=_compute_behavioral_stability(prey_rewards),
        energy_cost=prey_energy_cost,
        complexity_penalty=0.0,  # implement complexity measure
        novelty=prey_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=predator_genome.genome_id if hasattr(predator_genome, 'genome_id') else None,
        instability=prey_instability,
        saturation_penalty=prey_saturation_penalty,
        dead_unit_penalty=prey_dead_unit_penalty
    )

    predator_learning_speed = _compute_episode_learning_speed(predator_rewards)
    if hasattr(predator_genome.brain, 'get_plastic_diagnostics'):
        plastic_diag = predator_genome.brain.get_plastic_diagnostics()
        if predator_genome.plastic_diagnostics is None:
            predator_genome.plastic_diagnostics = {}
        predator_genome.plastic_diagnostics.update(plastic_diag)
        if 'mean_plastic_delta' in plastic_diag:
            predator_genome.plastic_diagnostics['mean_final_plastic_delta'] = float(plastic_diag['mean_plastic_delta'])
        predator_genome.plastic_diagnostics['episode_reward_learning_speed'] = float(predator_learning_speed)
    predator_instability = float(predator_genome.plastic_diagnostics.get('instability', 0.0)) if predator_genome.plastic_diagnostics else 0.0

    predator_metrics = EpisodeMetrics(
        task_success=(predator_total_reward > 0) or (predator_captures > 0),
        episode_return=predator_total_reward,
        learning_speed=predator_learning_speed,
        stability=_compute_behavioral_stability(predator_rewards),
        energy_cost=predator_energy_cost,
        complexity_penalty=0.0,
        novelty=pred_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=prey_genome.genome_id if hasattr(prey_genome, 'genome_id') else None,
        instability=predator_instability,
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
    final_info: Optional[Dict[str, Any]] = None

    for step in range(max_steps):
        # Get actions from genomes (plasticity not updated)
        prey_actions = prey_genome.act_batch(prey_state)

        # Use predator pack brain for coordinated actions
        pred_actions = predator_brain.act(pred_state)

        # Step the arena
        (prey_state, pred_state), r_prey, r_pred, info = arena.step(
            prey_actions, pred_actions
        )
        final_info = info

        # Use rewards directly (no pressure injection multiplier)
        prey_reward = _to_numpy(r_prey)
        pred_reward = _to_numpy(r_pred)

        prey_total_reward += float(np.sum(prey_reward))
        predator_total_reward += float(np.sum(pred_reward))
        prey_rewards.append(_safe_array_mean(prey_reward, default=0.0))
        predator_rewards.append(_safe_array_mean(pred_reward, default=0.0))

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

    prey_energy_cost = _extract_energy_efficiency_cost(final_info, "prey", steps_survived, prey_energy_cost)
    predator_energy_cost = _extract_energy_efficiency_cost(final_info, "predator", steps_survived, predator_energy_cost)

    # Compute metrics (no learning speed for non-plastic)
    prey_action_entropy = _action_entropy(prey_action_counts)
    pred_action_entropy = _action_entropy(pred_action_counts)
    prey_stability = prey_genome.brain.get_stability_diagnostics()
    predator_stability = predator_genome.brain.get_stability_diagnostics()
    success_signals = final_info.get('success_signals', {}) if isinstance(final_info, dict) else {}
    predator_captures = int(success_signals.get('predator_captures', 0)) if isinstance(success_signals, dict) else 0
    prey_saturation_penalty = float(prey_stability.get('avg_saturation_fraction', 0.0)) * 0.5
    prey_dead_unit_penalty = float(prey_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3
    predator_saturation_penalty = float(predator_stability.get('avg_saturation_fraction', 0.0)) * 0.5
    predator_dead_unit_penalty = float(predator_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3

    prey_metrics = EpisodeMetrics(
        task_success=prey_total_reward > 0,
        episode_return=prey_total_reward,
        learning_speed=0.0,  # No plasticity updates
        stability=_compute_behavioral_stability(prey_rewards),
        energy_cost=prey_energy_cost,
        complexity_penalty=0.0,
        novelty=prey_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=predator_genome.genome_id if hasattr(predator_genome, 'genome_id') else None,
        instability=0.0,
        saturation_penalty=prey_saturation_penalty,
        dead_unit_penalty=prey_dead_unit_penalty
    )

    predator_metrics = EpisodeMetrics(
        task_success=(predator_total_reward > 0) or (predator_captures > 0),
        episode_return=predator_total_reward,
        learning_speed=0.0,
        stability=_compute_behavioral_stability(predator_rewards),
        energy_cost=predator_energy_cost,
        complexity_penalty=0.0,
        novelty=pred_action_entropy,
        seed=seed or 0,
        stage=stage_name,
        opponent_id=prey_genome.genome_id if hasattr(prey_genome, 'genome_id') else None,
        instability=0.0,
        saturation_penalty=predator_saturation_penalty,
        dead_unit_penalty=predator_dead_unit_penalty
    )

    return (prey_total_reward, prey_metrics), (predator_total_reward, predator_metrics)

def log_coevolution_generation(stats: Dict[str, Any]):
    """Log co-evolution generation statistics with human-readable formatting"""
    logger = logging.getLogger('coevolution')
    
    generation = stats['generation']
    stage = stats['stage']
    
    # Main header
    print(f"\n{'='*90}")
    print(f"  GENERATION {generation:04d} - {stage}")
    print(f"{'='*90}")
    
    # --- FITNESS PERFORMANCE ---
    print(f"\n📊 FITNESS PERFORMANCE")
    print(f"  {'─'*86}")
    best_prey = stats['best_prey_fitness']
    mean_prey = stats['mean_prey_fitness']
    best_pred = stats['best_predator_fitness']
    mean_pred = stats['mean_predator_fitness']
    
    prey_indicator = "🟢" if best_prey > mean_prey * 1.2 else "🟡" if best_prey > mean_prey else "🔴"
    pred_indicator = "🟢" if best_pred > mean_pred * 1.2 else "🟡" if best_pred > mean_pred else "🔴"
    
    print(f"  🎯 Prey Agents:     Best: {best_prey:8.2f}  │  Average: {mean_prey:8.2f}  {prey_indicator}")
    print(f"  🎯 Predator Agents: Best: {best_pred:8.2f}  │  Average: {mean_pred:8.2f}  {pred_indicator}")
    
    # --- POPULATION STATUS ---
    print(f"\n👥 POPULATION STATUS")
    print(f"  {'─'*86}")
    print(f"  Prey Population:      {stats['prey_population_size']:4d} agents", end="")
    if 'prey_species' in stats:
        prey_num_species = int(stats['prey_species'].get('num_species', 0))
        print(f"  │  Species: {prey_num_species}", end="")
    print()
    print(f"  Predator Population:  {stats['predator_population_size']:4d} agents", end="")
    if 'predator_species' in stats:
        predator_num_species = int(stats['predator_species'].get('num_species', 0))
        predator_size = int(stats['predator_population_size'])
        if predator_num_species == 1 and predator_size > 1:
            print(f"  │  Species: {predator_num_species} ⚠️ COLLAPSE!", end="")
        else:
            print(f"  │  Species: {predator_num_species}", end="")
        
        # Show species breakdown for predators
        species_sizes = stats['predator_species'].get('species_sizes', [])
        if species_sizes and len(species_sizes) <= 5:
            breakdown = ", ".join(str(s) for s in sorted(species_sizes, reverse=True))
            print(f"  ({breakdown})", end="")
    print()
    print(f"  Evaluation Time:      {stats['eval_time']:6.2f} seconds")
    
    # --- LEARNING & ADAPTATION ---
    if 'avg_adaptability_score' in stats:
        print(f"\n🧠 LEARNING & ADAPTATION")
        print(f"  {'─'*86}")
        adapt_score = stats['avg_adaptability_score']
        # Practical bands tuned to observed project scale (roughly p25~0.14, p75~0.20).
        adapt_indicator = "🟢" if adapt_score >= 0.20 else "🟡" if adapt_score >= 0.14 else "🔴"
        print(f"  Adaptability Score:   {adapt_score:.3f}  (How quickly agents learn)  {adapt_indicator}")
        print(f"  Meta Effectiveness:   {stats['avg_meta_effectiveness']:.3f}  (Quality of evolution)")
        print(f"  Performance Change:   {stats.get('avg_reward_delta', 0.0):.3f}  (Reward improvement)")
        print(f"  Instability:          {stats.get('avg_instability', 0.0):.3f}  (Consistency of behavior)")
    
    # --- WEIGHT PLASTICITY (Learning Mechanisms) ---
    if 'mean_plastic_norm' in stats:
        print(f"\n⚡ LEARNING MECHANISMS (Weight Modifications)")
        print(f"  {'─'*86}")
        mean_norm = stats['mean_plastic_norm']
        norm_indicator = "🟢" if 0.05 < mean_norm < 1.50 else "🟡" if mean_norm >= 0.01 else "🔴"
        print(f"  Average Plasticity:   {mean_norm:.4f}  (Per-episode cumulative RMS ΔW)  {norm_indicator}")
        print(f"  Maximum Plasticity:   {stats['max_plastic_norm']:.4f}  (Largest episode cumulative RMS ΔW)")
        print(f"  95th Percentile:      {stats['p95_plastic_norm']:.4f}")
        if 'mean_plastic_weight_norm' in stats:
            print(f"  Plastic Weight RMS:   {stats['mean_plastic_weight_norm']:.4f}  (Accumulated plastic state)")
    
    # --- BEHAVIORAL METRICS ---
    if 'avg_energy_cost' in stats:
        print(f"\n🎮 BEHAVIORAL QUALITY")
        print(f"  {'─'*86}")
        energy = stats['avg_energy_cost']
        speed = stats['avg_learning_speed']
        stability = stats['avg_stability']
        novelty = stats['avg_novelty']
        success = stats['avg_success_rate']
        
        energy_indicator = "🟢" if energy < 0.5 else "🟡" if energy < 0.7 else "🔴"
        # Learning speed is bounded [0,1], but practical early-training values are much smaller.
        speed_indicator = "🟢" if speed > 0.08 else "🟡" if speed > 0.03 else "🔴"
        stability_indicator = "🟢" if stability > 0.7 else "🟡" if stability > 0.4 else "🔴"
        novelty_indicator = "🟢" if novelty > 0.5 else "🟡"
        success_indicator = "🟢" if success > 0.7 else "🟡" if success > 0.4 else "🔴"
        
        print(f"  Energy Efficiency:    {energy:.3f}  (Lower is better - less wasted energy)  {energy_indicator}")
        print(f"  Learning Speed:       {speed:.3f}  (How fast agents improve mid-episode)  {speed_indicator}")
        print(f"  Behavioral Stability: {stability:.3f}  (Consistency of actions)  {stability_indicator}")
        print(f"  Strategy Novelty:     {novelty:.3f}  (Variety in discovered strategies)  {novelty_indicator}")
        print(f"  Task Success Rate:    {success:.3f}  (% of successful episodes)  {success_indicator}")
    
    # --- GENETIC DIVERSITY ---
    prey_species = stats.get('prey_species')
    predator_species = stats.get('predator_species')
    if isinstance(prey_species, dict) and isinstance(predator_species, dict):
        print(f"\n🧬 GENETIC DIVERSITY (Speciation)")
        print(f"  {'─'*86}")
        prey_sp = prey_species.get('num_species', 0)
        prey_size = prey_species.get('avg_species_size', 0.0)
        pred_sp = predator_species.get('num_species', 0)
        pred_size = predator_species.get('avg_species_size', 0.0)
        
        print(f"  Prey Species Groups:      {prey_sp:3d} groups  (avg size: {prey_size:5.1f} agents)")
        print(f"  Predator Species Groups:  {pred_sp:3d} groups  (avg size: {pred_size:5.1f} agents)")
        print(f"  └─ Higher = More genetic diversity (helps avoid getting stuck in local patterns)")
    
    # Neural health summary
    neural_health = stats.get('neural_health')
    if neural_health and neural_health.get('genomes_analyzed', 0) > 0:
        dead = neural_health['dead_layers']
        saturated = neural_health['saturated_layers']
        health_indicator = "🟢" if dead + saturated < 10 else "🟡" if dead + saturated < 30 else "🔴"
        print(f"\n🔧 NEURAL NETWORK HEALTH")
        print(f"  {'─'*86}")
        print(f"  Dead Neural Connections:   {dead:3d}  (Inactive units that aren't learning)  {health_indicator}")
        print(f"  Saturated Units:           {saturated:3d}  (Neurons at max activation)")
        print(f"  └─ Evolution is removing ineffective connections automatically")
    
    # Novelty archive
    prey_novelty = stats.get('prey_novelty')
    predator_novelty = stats.get('predator_novelty')
    if isinstance(prey_novelty, dict) and isinstance(predator_novelty, dict):
        print(f"\n🔍 STRATEGY DISCOVERY (Novelty Archive)")
        print(f"  {'─'*86}")
        prey_arch = prey_novelty.get('archive', {}) if isinstance(prey_novelty.get('archive', {}), dict) else {}
        pred_arch = predator_novelty.get('archive', {}) if isinstance(predator_novelty.get('archive', {}), dict) else {}
        
        prey_stored = prey_arch.get('size', 0)
        pred_stored = pred_arch.get('size', 0)
        
        print(f"  Unique Prey Strategies:     {prey_stored:4d} stored in memory")
        print(f"  Unique Predator Strategies: {pred_stored:4d} stored in memory")
        print(f"  └─ Archive preserves diverse strategies found so far")
    
    # Architecture clusters
    arch_clusters = stats.get('architecture_clusters')
    if isinstance(arch_clusters, dict) and not arch_clusters.get('skipped', False):
        clusters = arch_clusters.get('num_clusters', 0)
        silhouette = arch_clusters.get('silhouette', 0.0)
        diversity = arch_clusters.get('diversity', 0.0)
        cluster_indicator = "🟢" if clusters > 3 else "🟡" if clusters > 1 else "🔴"
        
        print(f"\n🏗️  NETWORK ARCHITECTURE PATTERNS")
        print(f"  {'─'*86}")
        print(f"  Distinct Network Types:  {clusters:2d}  (Different brain structures discovered)  {cluster_indicator}")
        print(f"  Quality Score:           {silhouette:.3f}  (How well separated the groups are)")
        print(f"  Structure Diversity:     {diversity:.3f}  (Variation in network designs)")
    
    print(f"\n{'='*90}\n")

def print_metric_explanations():
    """Print human-readable explanations of key metrics (optional reference guide)"""
    guide = """
╔════════════════════════════════════════════════════════════════════════════════════╗
║                        📚 METRIC EXPLANATIONS FOR NON-TECHNICAL USERS              ║
╚════════════════════════════════════════════════════════════════════════════════════╝

🎮 FITNESS PERFORMANCE
  • Best/Average Fitness: Score indicating how well agents complete their tasks
    - Higher numbers = Better performance
    - Red indicator (🔴): Performance needs improvement
    - Yellow (🟡): Moderate performance
    - Green (🟢): Good performance

🧠 LEARNING & ADAPTATION
  • Adaptability Score: How quickly agents improve during episodes (0-1 scale)
    - Measures if agents learn from experience within single episodes
    - High = Agents adapt quickly to new situations
  
  • Meta Effectiveness: Quality of the evolutionary algorithm itself
    - Measures if evolution is producing better agents over time
  
  • Performance Change (Reward Delta): Improvement in scores from previous generation
    - Positive = Getting better, Negative = Getting worse or stuck

👥 POPULATION STATUS
  • Prey/Predator Population: Number of agents in each category
    - System maintains populations to promote diversity

⚡ LEARNING MECHANISMS (Weight Modifications)
  • Plasticity: Changes to neural network weights during training
    - Shows how much the agent's brain is being modified in real-time
    - Essential for life-long learning within episodes

🎮 BEHAVIORAL QUALITY
  • Energy Efficiency: How well agents use energy (lower is better)
  • Learning Speed: How fast agents improve mid-episode
  • Behavioral Stability: Consistency of agent actions
  • Strategy Novelty: Diversity in discovered strategies (higher = more creative)
  • Task Success Rate: Percentage of episodes where agent succeeds

🧬 GENETIC DIVERSITY (Speciation)
  • Species Groups: Number of distinct genetic "families"
    - Higher = More diversity = Better chance of finding novel solutions
    - Lower = Risk of everything being similar (monoculture problem)

🔧 NEURAL NETWORK HEALTH
  • Dead Neural Connections: Neurons that aren't participating
    - Evolution naturally removes useless neurons
  • Saturated Units: Neurons at maximum activation
    - Can indicate overfitting or unnecessary complexity

🔍 STRATEGY DISCOVERY (Novelty Archive)
  • Unique Strategies Stored: Count of diverse behaviors discovered
    - Higher = More strategies explored = Better long-term learning

🏗️ NETWORK ARCHITECTURE PATTERNS
  • Distinct Network Types: Different brain structure designs discovered
    - Shows variety in how agents solve problems
  • Quality Score (Silhouette): How well separated different types are
  • Structure Diversity: Variation in network designs

════════════════════════════════════════════════════════════════════════════════════════
💡 KEY INSIGHTS:
  ✓ Green indicators everywhere = System is working well
  ✓ Some red/yellow = Normal exploration phase - check again in ~10 generations
  ✓ Consistent red across generations = May need to adjust training parameters
  ✓ Genetic diversity increasing = Good! System will avoid getting stuck
  ✓ Plasticity too high/low = Agent learning mechanisms may need tuning
════════════════════════════════════════════════════════════════════════════════════════
"""
    print(guide)

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

def _log_meta_gene_entropy(population: list) -> None:
    """Compute and log meta-gene entropy statistics from the combined population."""
    if not population:
        return

    plastic_lrs: list = []
    meta_values: list = []
    all_plasticity: list = []

    for genome in population:
        meta = getattr(genome, 'meta', None) or {}
        plastic_lrs.append(float(meta.get('plastic_lr', 0.0)))
        meta_values.extend([
            float(meta.get('reward_gain', 0.0)),
            float(meta.get('reward_bias', 0.0)),
            float(meta.get('plastic_lr', 0.0)),
        ])
        for gene in getattr(genome, 'genes', []):
            p = getattr(gene, 'plasticity', None)
            if p is not None:
                all_plasticity.append(p.flatten())

    def _shannon_entropy(values: list, bins: int = 10) -> float:
        if len(values) < 2:
            return 0.0
        counts, _ = np.histogram(values, bins=bins)
        total = counts.sum()
        if total == 0:
            return 0.0
        probs = counts[counts > 0] / total
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    meta_gene_entropy = _shannon_entropy(meta_values)
    learning_rate_entropy = _shannon_entropy(plastic_lrs)

    if all_plasticity:
        all_flat = np.concatenate(all_plasticity)
        if all_flat.size > 0:
            plasticity_weight_variance = float(np.var(all_flat))
            plastic_neuron_fraction = float(np.mean(np.abs(all_flat) > 0.05))
        else:
            plasticity_weight_variance = 0.0
            plastic_neuron_fraction = 0.0
    else:
        plasticity_weight_variance = 0.0
        plastic_neuron_fraction = 0.0

    MetaGeneEntropyLogger.log_generation_meta_entropy(
        meta_gene_entropy=meta_gene_entropy,
        plasticity_weight_variance=plasticity_weight_variance,
        learning_rate_entropy=learning_rate_entropy,
        plastic_neuron_fraction=plastic_neuron_fraction,
    )


def _flatten_metrics_row(value: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a generation-stats dict into a single-level dict for CSV output."""
    flat: Dict[str, Any] = {}
    if not isinstance(value, dict):
        return flat

    for key, item in value.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)

        if item is None or isinstance(item, (str, int, float, bool)):
            flat[full_key] = item
            continue

        if isinstance(item, list):
            if len(item) <= 5 and all(v is None or isinstance(v, (str, int, float, bool)) for v in item):
                flat[full_key] = ";".join("" if v is None else str(v) for v in item)
            continue

        if isinstance(item, dict):
            flat.update(_flatten_metrics_row(item, full_key))

    return flat


def append_metrics_row(stats: Dict[str, Any], metrics_path: str = "data/metrices.csv") -> None:
    """Append a single generation's flattened stats to the metrics CSV every generation."""
    from io import StringIO

    row = _flatten_metrics_row(stats)
    if not row:
        return

    path = Path(metrics_path)
    os.makedirs(path.parent if str(path.parent) else Path("."), exist_ok=True)

    file_exists = path.exists() and path.stat().st_size > 0

    # Build consistent ordered field list: generation first, stage second, rest alphabetical.
    fieldnames: List[str] = []
    remaining = set(row.keys())
    for priority in ("generation", "stage"):
        if priority in remaining:
            fieldnames.append(priority)
            remaining.remove(priority)
    fieldnames.extend(sorted(remaining))

    if file_exists:
        # Read existing header to keep column order stable across generations.
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            existing_fields = next(csv.reader(f), None) or []
        # Merge: keep existing columns first, append any new ones.
        merged_fields = existing_fields[:]
        for col in fieldnames:
            if col not in merged_fields:
                merged_fields.append(col)
        fieldnames = merged_fields

    with tempfile.NamedTemporaryFile(
        'w', delete=False,
        dir=path.parent if str(path.parent) else Path("."),
        encoding='utf-8', newline=''
    ) as tmp:
        tmp_path = tmp.name
        # Write header + existing rows + new row
        if file_exists:
            with open(path, 'r', encoding='utf-8-sig', newline='') as src:
                existing_content = src.read()
            tmp.write(existing_content.rstrip('\n'))
            tmp.write('\n')
            buf = StringIO()
            dict_writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
            dict_writer.writerow({k: row.get(k, '') for k in fieldnames})
            tmp.write(buf.getvalue())
        else:
            buf = StringIO()
            dict_writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction='ignore')
            dict_writer.writeheader()
            dict_writer.writerow({k: row.get(k, '') for k in fieldnames})
            tmp.write(buf.getvalue())

    os.replace(tmp_path, path)


def _serialize_curriculum_controller_state(curriculum_controller: CurriculumController) -> Dict[str, Any]:
    """Serialize curriculum controller to JSON-safe checkpoint payload."""
    adapted_thresholds: Dict[str, Dict[str, float]] = {}
    for stage, thresholds in curriculum_controller.adapted_thresholds.items():
        stage_name = stage.name if hasattr(stage, "name") else str(stage)
        adapted_thresholds[stage_name] = {
            str(k): float(v) for k, v in dict(thresholds).items()
        }

    transition_history = []
    for t in curriculum_controller.transition_history[-200:]:
        transition_history.append(
            {
                "from_stage": t.from_stage.name,
                "to_stage": t.to_stage.name,
                "reason": t.reason.value,
                "timestamp": float(t.timestamp),
                "generation": int(t.generation),
                "performance_stats": {str(k): float(v) for k, v in dict(t.performance_stats).items()},
            }
        )

    current_perf = curriculum_controller.current_performance
    return {
        "current_stage": curriculum_controller.current_stage.name,
        "generation": int(curriculum_controller.generation),
        "last_transition_generation": int(curriculum_controller.last_transition_generation),
        "min_generations_per_stage": int(curriculum_controller.min_generations_per_stage),
        "transition_cooldown_generations": int(curriculum_controller.transition_cooldown_generations),
        "current_performance": {
            "generations_in_stage": int(current_perf.generations_in_stage),
            "best_fitness_achieved": float(current_perf.best_fitness_achieved),
            "stagnation_count": int(current_perf.stagnation_count),
            "fitness_history": [float(x) for x in current_perf.fitness_history[-200:]],
            "diversity_history": [float(x) for x in current_perf.diversity_history[-200:]],
            "success_rate_history": [float(x) for x in current_perf.success_rate_history[-200:]],
        },
        "adapted_thresholds": adapted_thresholds,
        "transition_history": transition_history,
    }


def _restore_curriculum_controller_state(
    curriculum_controller: CurriculumController,
    state: Optional[Dict[str, Any]],
) -> None:
    """Restore curriculum controller from a serialized checkpoint payload."""
    if not state:
        return

    try:
        stage_name = str(state.get("current_stage", curriculum_controller.current_stage.name))
        curriculum_controller.current_stage = CurriculumStage[stage_name]
    except Exception:
        pass

    try:
        curriculum_controller.generation = int(state.get("generation", curriculum_controller.generation))
    except Exception:
        pass

    try:
        curriculum_controller.last_transition_generation = int(
            state.get("last_transition_generation", curriculum_controller.last_transition_generation)
        )
    except Exception:
        pass

    try:
        curriculum_controller.min_generations_per_stage = int(
            state.get("min_generations_per_stage", curriculum_controller.min_generations_per_stage)
        )
    except Exception:
        pass

    try:
        curriculum_controller.transition_cooldown_generations = int(
            state.get("transition_cooldown_generations", curriculum_controller.transition_cooldown_generations)
        )
    except Exception:
        pass

    # Restore threshold adaptations
    adapted_thresholds = state.get("adapted_thresholds", {})
    if isinstance(adapted_thresholds, dict):
        for stage_name, thresholds in adapted_thresholds.items():
            if not isinstance(thresholds, dict):
                continue
            try:
                stage = CurriculumStage[str(stage_name)]
            except Exception:
                continue
            curriculum_controller.adapted_thresholds[stage] = {
                str(k): float(v) for k, v in thresholds.items()
            }

    # Restore current stage performance context
    current_perf = state.get("current_performance", {})
    if isinstance(current_perf, dict):
        cp = curriculum_controller.current_performance
        cp.stage = curriculum_controller.current_stage
        cp.generations_in_stage = int(current_perf.get("generations_in_stage", cp.generations_in_stage))
        cp.best_fitness_achieved = float(current_perf.get("best_fitness_achieved", cp.best_fitness_achieved))
        cp.stagnation_count = int(current_perf.get("stagnation_count", cp.stagnation_count))
        cp.fitness_history = [float(x) for x in current_perf.get("fitness_history", [])]
        cp.diversity_history = [float(x) for x in current_perf.get("diversity_history", [])]
        cp.success_rate_history = [float(x) for x in current_perf.get("success_rate_history", [])]


def _serialize_architect_population_state(architect_population: ArchitectPopulation) -> Dict[str, Any]:
    """Serialize architect meta-population state for checkpointing."""
    return {
        "generation": int(getattr(architect_population, "generation", 0)),
        "meta_fitness_history": [float(x) for x in getattr(architect_population, "meta_fitness_history", [])],
        "architecture_templates": list(getattr(architect_population, "architecture_templates", [])),
        "shared_templates": list(getattr(architect_population, "shared_templates", [])),
    }


def _restore_architect_population_state(
    architect_population: ArchitectPopulation,
    state: Optional[Dict[str, Any]],
) -> None:
    """Restore architect meta-population state from checkpoint payload."""
    if not isinstance(state, dict):
        return

    architect_population.generation = int(state.get("generation", getattr(architect_population, "generation", 0)))
    architect_population.meta_fitness_history = [
        float(x) for x in state.get("meta_fitness_history", [])
    ]
    architect_population.architecture_templates = list(state.get("architecture_templates", []))
    architect_population.shared_templates = list(state.get("shared_templates", []))


def _serialize_mutator_population_state(mutator_population: MutatorPopulation) -> Dict[str, Any]:
    """Serialize mutator meta-population state for checkpointing."""
    return {
        "generation": int(getattr(mutator_population, "generation", 0)),
        "meta_fitness_history": [float(x) for x in getattr(mutator_population, "meta_fitness_history", [])],
        "mutation_strategies": list(getattr(mutator_population, "mutation_strategies", [])),
    }


def _restore_mutator_population_state(
    mutator_population: MutatorPopulation,
    state: Optional[Dict[str, Any]],
) -> None:
    """Restore mutator meta-population state from checkpoint payload."""
    if not isinstance(state, dict):
        return

    mutator_population.generation = int(state.get("generation", getattr(mutator_population, "generation", 0)))
    mutator_population.meta_fitness_history = [
        float(x) for x in state.get("meta_fitness_history", [])
    ]
    mutator_population.mutation_strategies = list(state.get("mutation_strategies", []))


def _serialize_engine_runtime_state(engine: EvolutionEngine) -> Dict[str, Any]:
    """Serialize mutable engine runtime state that is not part of TrainingState genomes."""
    runtime: Dict[str, Any] = {
        "last_speciated_generation": int(getattr(engine, "_last_speciated_generation", -1)),
    }

    speciation_manager = getattr(engine, "speciation_manager", None)
    if speciation_manager is not None:
        runtime["speciation"] = {
            "compatibility_threshold": float(speciation_manager.compatibility_threshold),
            "next_species_id": int(getattr(speciation_manager, "next_species_id", 0)),
        }

    novelty_archive = getattr(engine, "novelty_archive", None)
    if novelty_archive is not None:
        archive_items = []
        for emb in novelty_archive.archive:
            archive_items.append(
                {
                    "genome_id": str(getattr(emb, "genome_id", "unknown")),
                    "fitness": float(getattr(emb, "fitness", 0.0)),
                    "generation": int(getattr(emb, "generation", 0)),
                    "embedding": np.asarray(getattr(emb, "embedding", []), dtype=np.float32).tolist(),
                }
            )
        runtime["novelty_archive"] = {
            "archive": archive_items,
            "max_size": int(getattr(novelty_archive, "max_size", len(archive_items))),
            "novelty_threshold": float(getattr(novelty_archive, "novelty_threshold", 0.1)),
        }

    return runtime


def _restore_engine_runtime_state(engine: EvolutionEngine, state: Optional[Dict[str, Any]]) -> None:
    """Restore mutable engine runtime state from checkpoint payload."""
    if not isinstance(state, dict):
        return

    engine._last_speciated_generation = int(state.get("last_speciated_generation", -1))

    speciation = state.get("speciation")
    speciation_manager = getattr(engine, "speciation_manager", None)
    if isinstance(speciation, dict) and speciation_manager is not None:
        if "compatibility_threshold" in speciation:
            speciation_manager.compatibility_threshold = float(speciation["compatibility_threshold"])
        if "next_species_id" in speciation:
            speciation_manager.next_species_id = int(speciation["next_species_id"])

    novelty_archive_state = state.get("novelty_archive")
    novelty_archive = getattr(engine, "novelty_archive", None)
    if isinstance(novelty_archive_state, dict) and novelty_archive is not None:
        archive_items = novelty_archive_state.get("archive", [])
        restored_archive: List[BehaviorEmbedding] = []
        for item in archive_items:
            if not isinstance(item, dict):
                continue
            vec = np.asarray(item.get("embedding", []), dtype=np.float32)
            if vec.size == 0:
                continue
            restored_archive.append(
                BehaviorEmbedding(
                    genome_id=str(item.get("genome_id", "unknown")),
                    fitness=float(item.get("fitness", 0.0)),
                    embedding=vec,
                    generation=int(item.get("generation", 0)),
                    genome=None,
                )
            )

        max_size = int(novelty_archive_state.get("max_size", novelty_archive.max_size))
        novelty_archive.max_size = max(1, max_size)
        novelty_archive.novelty_threshold = float(
            novelty_archive_state.get("novelty_threshold", novelty_archive.novelty_threshold)
        )
        novelty_archive.archive = restored_archive[-novelty_archive.max_size:]


def save_coevolution_state(
    training_state: TrainingState,
    filename: Optional[str] = None,
    curriculum_controller: Optional[CurriculumController] = None,
    architect_population: Optional[ArchitectPopulation] = None,
    mutator_population: Optional[MutatorPopulation] = None,
    prey_engine: Optional[EvolutionEngine] = None,
    predator_engine: Optional[EvolutionEngine] = None,
    loop_runtime_state: Optional[Dict[str, Any]] = None,
):
    """Save complete co-evolution training state to split files."""
    from diagnostics.strategy_clustering import StrategyClusteringLogger
    from diagnostics.reward_recovery import RewardRecoveryLogger

    def _write_text_atomic(path: str, text: str) -> None:
        target = Path(path)
        parent = target.parent if str(target.parent) else Path(".")
        os.makedirs(parent, exist_ok=True)

        with tempfile.NamedTemporaryFile('w', delete=False, dir=parent, encoding='utf-8', newline='') as tmp:
            tmp.write(text)
            tmp_path = tmp.name

        os.replace(tmp_path, target)

    def _write_json_atomic(path: str, payload: Any) -> None:
        _write_text_atomic(path, json.dumps(payload, indent=2, default=str))

    def _derive_split_paths(base_filename: str) -> Dict[str, str]:
        base_path = Path(base_filename)
        target_dir = base_path.parent if str(base_path.parent) else Path(".")
        stem = base_path.stem

        if stem == "coevolution_state":
            config_name = "config.json"
            metrics_name = "metrices.csv"
            experiment_name = "expirement_state.json"
        elif stem == "final_coevolution_state":
            config_name = "final_config.json"
            metrics_name = "final_metrices.csv"
            experiment_name = "final_expirement_state.json"
        else:
            config_name = f"{stem}_config.json"
            metrics_name = f"{stem}_metrices.csv"
            experiment_name = f"{stem}_expirement_state.json"

        return {
            "config": str(target_dir / config_name),
            "metrics": str(target_dir / metrics_name),
            "experiment": str(target_dir / experiment_name),
        }

    state = {
        'generation': training_state.generation,
        'config': training_state.config.__dict__,
        'best_prey_fitness_history': training_state.best_prey_fitness_history,
        'best_predator_fitness_history': training_state.best_predator_fitness_history,
        'generation_stats': training_state.generation_stats,
        'experiment_reports': [exp.__dict__ for exp in training_state.experiment_reports],
        'generalization_reports': [r.to_dict() for r in training_state.generalization_reports],
        # Diagnostic logger states (restored on load so plots span the full run history)
        'meta_gene_entropy_logger': {
            'meta_gene_entropies': MetaGeneEntropyLogger.meta_gene_entropies[:],
            'plasticity_weight_variances': MetaGeneEntropyLogger.plasticity_weight_variances[:],
            'learning_rate_entropies': MetaGeneEntropyLogger.learning_rate_entropies[:],
            'plastic_neuron_fractions': MetaGeneEntropyLogger.plastic_neuron_fractions[:],
        },
        'strategy_clustering_logger': {
            'cluster_centers': [
                c.tolist() if hasattr(c, 'tolist') else c
                for c in StrategyClusteringLogger.cluster_centers
            ],
            'cluster_labels': StrategyClusteringLogger.cluster_labels[:],
            'silhouette_scores': StrategyClusteringLogger.silhouette_scores[:],
            'strategy_diversities': StrategyClusteringLogger.strategy_diversities[:],
        },
        'reward_recovery_logger': {
            'recovery_records': [list(r) for r in RewardRecoveryLogger.recovery_records],
        },
    }

    if curriculum_controller is not None:
        state['curriculum_controller_state'] = _serialize_curriculum_controller_state(curriculum_controller)
    elif training_state.curriculum_state is not None:
        state['curriculum_controller_state'] = training_state.curriculum_state

    if architect_population is not None or mutator_population is not None:
        state['meta_evolution_state'] = {
            'architect_population': (
                _serialize_architect_population_state(architect_population)
                if architect_population is not None
                else None
            ),
            'mutator_population': (
                _serialize_mutator_population_state(mutator_population)
                if mutator_population is not None
                else None
            ),
        }
    elif training_state.meta_evolution_state is not None:
        state['meta_evolution_state'] = training_state.meta_evolution_state

    if prey_engine is not None or predator_engine is not None:
        state['engine_runtime_state'] = {
            'prey_engine': _serialize_engine_runtime_state(prey_engine) if prey_engine is not None else None,
            'predator_engine': _serialize_engine_runtime_state(predator_engine) if predator_engine is not None else None,
        }
    elif training_state.engine_runtime_state is not None:
        state['engine_runtime_state'] = training_state.engine_runtime_state

    if loop_runtime_state is not None:
        _last_combined_best_raw = loop_runtime_state.get('last_combined_best')
        _last_combined_best = None if _last_combined_best_raw is None else float(_last_combined_best_raw)
        state['loop_runtime_state'] = {
            'stagnation_generations': int(loop_runtime_state.get('stagnation_generations', 0)),
            'last_combined_best': _last_combined_best,
        }
    elif training_state.loop_runtime_state is not None:
        state['loop_runtime_state'] = training_state.loop_runtime_state

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

    if filename is None:
        split_paths = {
            "config": "data/config.json",
            "metrics": "data/metrices.csv",
            "experiment": "data/expirement_state.json",
        }
    else:
        split_paths = _derive_split_paths(filename)
    for path in split_paths.values():
        parent = Path(path).parent
        if str(parent):
            os.makedirs(parent, exist_ok=True)

    # 1) Config-only log
    _write_json_atomic(split_paths["config"], state['config'])

    # 2) Metrics-only log (CSV) — rebuild full history from all generation_stats
    metrics_rows = []
    for stat in training_state.generation_stats:
        if isinstance(stat, dict):
            metrics_rows.append(_flatten_metrics_row(stat))

    if metrics_rows:
        fieldnames_set = set()
        for row in metrics_rows:
            fieldnames_set.update(row.keys())

        ordered_fields: List[str] = []
        if 'generation' in fieldnames_set:
            ordered_fields.append('generation')
            fieldnames_set.remove('generation')
        if 'stage' in fieldnames_set:
            ordered_fields.append('stage')
            fieldnames_set.remove('stage')
        ordered_fields.extend(sorted(fieldnames_set))

        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=ordered_fields)
        writer.writeheader()
        writer.writerows(metrics_rows)
        _write_text_atomic(split_paths["metrics"], buffer.getvalue())
    else:
        from io import StringIO

        buffer = StringIO()
        writer = csv.writer(buffer)
        writer.writerow(['generation', 'stage'])
        _write_text_atomic(split_paths["metrics"], buffer.getvalue())

    # 3) Experiment/detailed state log (everything except config)
    experiment_state = {k: v for k, v in state.items() if k != 'config'}
    _write_json_atomic(split_paths["experiment"], experiment_state)

    print("\n💾 CHECKPOINT SAVE")
    print(f"  {'─'*86}")
    print(f"  ✅ Config:            {split_paths['config']}")
    print(f"  ✅ Metrics:           {split_paths['metrics']}")
    print(f"  ✅ Experiment State:  {split_paths['experiment']}")
    print(
        f"  📊 Generation {training_state.generation} | "
        f"Populations: prey={len(training_state.prey_population)}, predator={len(training_state.predator_population)}"
    )
    print(
        f"  🏆 Hall of Fame: prey={len(training_state.prey_hall_of_fame)}, "
        f"predator={len(training_state.predator_hall_of_fame)}"
    )
    print("  💾 Preserved: lineage, mutation history, learning parameters\n")

def load_coevolution_state(filename: Optional[str] = None) -> TrainingState:
    """Load co-evolution training state from split files by default."""
    legacy_file: Optional[Path] = None

    def _load_json_file(path: Path) -> Dict[str, Any]:
        try:
            with open(path, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in checkpoint file {path}: {exc}") from exc

    if filename is None:
        config_file = Path("data/config.json")
        experiment_file = Path("data/expirement_state.json")
    else:
        base_path = Path(filename)
        target_dir = base_path.parent if str(base_path.parent) else Path(".")
        stem = base_path.stem

        if stem == "coevolution_state":
            config_file = target_dir / "config.json"
            experiment_file = target_dir / "expirement_state.json"
        elif stem == "final_coevolution_state":
            config_file = target_dir / "final_config.json"
            experiment_file = target_dir / "final_expirement_state.json"
        else:
            config_file = target_dir / f"{stem}_config.json"
            experiment_file = target_dir / f"{stem}_expirement_state.json"
        legacy_file = base_path

    if config_file.exists() and experiment_file.exists():
        config_dict = _load_json_file(config_file)
        state = _load_json_file(experiment_file)
        state['config'] = config_dict
    elif legacy_file is not None and legacy_file.exists():
        # Backward compatibility when an explicit legacy checkpoint path is provided.
        state = _load_json_file(legacy_file)
    else:
        raise FileNotFoundError(
            f"Split checkpoint files not found: {config_file} and {experiment_file}."
        )
    
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
    training_state.best_prey_fitness_history = state.get('best_prey_fitness_history', [])
    training_state.best_predator_fitness_history = state.get('best_predator_fitness_history', [])
    training_state.generation_stats = state.get('generation_stats', [])
    training_state.curriculum_state = state.get('curriculum_controller_state')
    training_state.meta_evolution_state = state.get('meta_evolution_state')
    training_state.engine_runtime_state = state.get('engine_runtime_state')
    training_state.loop_runtime_state = state.get('loop_runtime_state')
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
    
    # Restore diagnostic logger states so plots include pre-checkpoint history
    if 'meta_gene_entropy_logger' in state:
        d = state['meta_gene_entropy_logger']
        MetaGeneEntropyLogger.meta_gene_entropies = d.get('meta_gene_entropies', [])
        MetaGeneEntropyLogger.plasticity_weight_variances = d.get('plasticity_weight_variances', [])
        MetaGeneEntropyLogger.learning_rate_entropies = d.get('learning_rate_entropies', [])
        MetaGeneEntropyLogger.plastic_neuron_fractions = d.get('plastic_neuron_fractions', [])

    if 'strategy_clustering_logger' in state:
        from diagnostics.strategy_clustering import StrategyClusteringLogger
        import numpy as _np
        d = state['strategy_clustering_logger']
        StrategyClusteringLogger.cluster_centers = [
            _np.array(c) for c in d.get('cluster_centers', [])
        ]
        StrategyClusteringLogger.cluster_labels = d.get('cluster_labels', [])
        StrategyClusteringLogger.silhouette_scores = d.get('silhouette_scores', [])
        StrategyClusteringLogger.strategy_diversities = d.get('strategy_diversities', [])

    if 'reward_recovery_logger' in state:
        from diagnostics.reward_recovery import RewardRecoveryLogger
        d = state['reward_recovery_logger']
        RewardRecoveryLogger.recovery_records = [
            tuple(r) for r in d.get('recovery_records', [])
        ]

    print("\n📂 CHECKPOINT LOAD")
    print(f"  {'─'*86}")
    if config_file.exists() and experiment_file.exists():
        print("  ✅ Loaded split checkpoint files")
        print(f"  • {config_file}")
        print(f"  • {experiment_file}")
    else:
        print(f"  ✅ Loaded legacy checkpoint: {legacy_file}")
    print(f"  📍 Resume Generation: {training_state.generation}")
    print(
        f"  👥 Populations: prey={len(training_state.prey_population)}, "
        f"predator={len(training_state.predator_population)}"
    )
    
    return training_state

async def main_coevolution_async(runtime_overrides: Optional[RuntimeOverrides] = None):
    """Main async co-evolution training loop with adaptive curriculum"""
    # Configure logging for immediate visibility
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Output to console
        ]
    )

    print("\n🧬 EVOLUTIONARY AI TRAINING SYSTEM")
    print(f"  {'─'*86}")
    print("  Co-evolution training with adaptive curriculum and plasticity\n")

    # Add watchdog thread (critical)
    # def watchdog():
    #     while True:
    #         print("[WATCHDOG] main loop alive")
    #         time.sleep(10)

    # threading.Thread(target=watchdog, daemon=True).start()

    # Initialize configuration
    config = EvolutionConfig()

    # Optional runtime overrides for quick perf testing / CI runs.
    # Example (PowerShell):
    #   $env:MAX_GENERATIONS=1; $env:AUTO_LOAD_COEVOLUTION_STATE='n'; python main.py
    max_gens_env = os.getenv("MAX_GENERATIONS")
    env_runtime_overrides: Optional[RuntimeOverrides] = None
    if max_gens_env:
        try:
            env_runtime_overrides = RuntimeOverrides(generations=int(max_gens_env))
        except ValueError:
            pass
    runtime_overrides = _merge_runtime_overrides(runtime_overrides, env_runtime_overrides)

    # Initialize training state
    training_state = TrainingState(config=config)

    # Initialize populations
    print("\n📝 POPULATION INITIALIZATION")
    print(f"  {'─'*86}")
    print(f"  🎯 Generating prey agents: {config.population_size}")
    training_state.prey_population = [
        PreyGenome.random_initialization()
        for _ in range(config.population_size)
    ]
    print(f"  🎯 Generating predator agents: {config.predator_population_size}")
    training_state.predator_population = [
        PredatorGenome.random_initialization()
        for _ in range(config.predator_population_size)
    ]
    print(f"  ✅ Total initialized: {len(training_state.prey_population) + len(training_state.predator_population)} agents\n")

    # Initialize curriculum controller
    curriculum_controller = CurriculumController()

    # Initialize evaluator
    requested_device = os.getenv("EVOMIND_DEVICE", "auto").strip().lower()
    use_gpu = torch.cuda.is_available() and requested_device in ("auto", "cuda")
    evaluator = AsyncDeterministicEvaluator(
        base_seed=config.base_seed,
        num_workers=config.num_workers,
        use_gpu=use_gpu,
        envs_per_genome=config.envs_per_genome,
        max_steps=config.max_steps
    )

    # Load checkpoint if split files exist
    split_checkpoint_exists = os.path.exists("data/config.json") and os.path.exists("data/expirement_state.json")
    loaded_checkpoint = False
    if split_checkpoint_exists:
        response = os.getenv("AUTO_LOAD_COEVOLUTION_STATE")
        if response is None:
            # Flush any leftover stdin content on Windows (e.g. from terminal activation scripts)
            try:
                import msvcrt
                while msvcrt.kbhit():
                    msvcrt.getwch()
            except ImportError:
                pass
            while True:
                response = input("Co-evolution state found. Load? (y/n): ").strip().lower()
                if response in ('y', 'n', 'yes', 'no'):
                    response = response[0]  # normalise to 'y' or 'n'
                    break
                print(f"  Invalid input '{response}'. Please enter 'y' or 'n'.")
        if response.lower().startswith('y'):
            try:
                training_state = load_coevolution_state()
                loaded_checkpoint = True
                # Keep runtime config alias consistent with restored checkpoint config.
                config = training_state.config
                evaluator.load_seeds("data/seed_registry.json")
                _restore_curriculum_controller_state(curriculum_controller, training_state.curriculum_state)
                print(
                    f"🧭 Curriculum restored | stage={curriculum_controller.current_stage.name}, "
                    f"controller_gen={curriculum_controller.generation}"
                )
            except (FileNotFoundError, ValueError) as exc:
                print(f"⚠️  Checkpoint load skipped: {exc}")
                print("  Starting from a fresh training state instead.")

    if not loaded_checkpoint:
        # Discard the speculative fresh populations created before runtime overrides were resolved.
        training_state.prey_population = []
        training_state.predator_population = []

    # Re-apply runtime overrides after any restore so CLI values remain authoritative.
    config = training_state.config
    applied_runtime_overrides = _initialize_or_resize_populations(training_state, runtime_overrides)
    config = training_state.config

    if applied_runtime_overrides:
        print(f"⚙️  Runtime overrides applied: {', '.join(applied_runtime_overrides)}")

    print("\n👥 POPULATION READINESS")
    print(f"  {'─'*86}")
    print(f"  Prey Population:      {len(training_state.prey_population)}")
    print(f"  Predator Population:  {len(training_state.predator_population)}")
    print(
        f"  Seed {config.base_seed} | Generation cap {config.generations} | "
        f"Total agents {len(training_state.prey_population) + len(training_state.predator_population)}\n"
    )

    # Rebuild evaluator from the final resolved config so seed and generation overrides stick.
    evaluator.close()
    requested_device = os.getenv("EVOMIND_DEVICE", "auto").strip().lower()
    use_gpu = torch.cuda.is_available() and requested_device in ("auto", "cuda")
    evaluator = AsyncDeterministicEvaluator(
        base_seed=config.base_seed,
        num_workers=config.num_workers,
        use_gpu=use_gpu,
        envs_per_genome=config.envs_per_genome,
        max_steps=config.max_steps
    )
    if loaded_checkpoint and not (runtime_overrides and runtime_overrides.has_seed_override()):
        evaluator.load_seeds("data/seed_registry.json")
    elif loaded_checkpoint and runtime_overrides and runtime_overrides.has_seed_override():
        print("⚙️  Seed registry load skipped: runtime seed override is active")

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

    # Restore additional resume state for continuity across restarts.
    if training_state.meta_evolution_state:
        _restore_architect_population_state(
            architect_population,
            training_state.meta_evolution_state.get('architect_population')
            if isinstance(training_state.meta_evolution_state, dict)
            else None,
        )
        _restore_mutator_population_state(
            mutator_population,
            training_state.meta_evolution_state.get('mutator_population')
            if isinstance(training_state.meta_evolution_state, dict)
            else None,
        )
        print(
            f"[Meta] Restored architect_gen={architect_population.generation}, "
            f"mutator_gen={mutator_population.generation}"
        )

    if training_state.engine_runtime_state and isinstance(training_state.engine_runtime_state, dict):
        _restore_engine_runtime_state(prey_engine, training_state.engine_runtime_state.get('prey_engine'))
        _restore_engine_runtime_state(predator_engine, training_state.engine_runtime_state.get('predator_engine'))
        print(
            f"[Engine] Restored thresholds: prey={prey_engine.get_speciation_threshold():.4f}, "
            f"predator={predator_engine.get_speciation_threshold():.4f}"
        )


    # Initialize meta-scientist systems
    meta_scientist = MetaScientist()
    evolution_modifier = EvolutionModifier()
    diagnostic_task_generator = DiagnosticTaskGenerator()

    def evolve_all_populations(gen: int) -> None:
        print(f"\n🧬 EVOLUTION PHASE - GENERATION {gen:04d}")
        print(f"  {'─'*86}")
        print("  Applying mutations and crossover for prey + predator populations...")

        if prey_engine:
            start = time.time()
            prey_population = prey_engine.create_next_generation(
                training_state.prey_population, gen, pop_name="prey"
            )
            if time.time() - start > 30:
                print("  ⚠️  Prey evolution step is taking longer than expected")
            training_state.prey_population = prey_population.genomes

        if predator_engine:
            start = time.time()
            predator_population = predator_engine.create_next_generation(
                training_state.predator_population, gen, pop_name="predator"
            )
            if time.time() - start > 30:
                print("  ⚠️  Predator evolution step is taking longer than expected")
            training_state.predator_population = predator_population.genomes

        print("  ✅ Evolution complete - new generation created")

    # Training loop
    MAX_GEN_TIME = 300  # Maximum time per generation in seconds
    restored_loop_state = training_state.loop_runtime_state if isinstance(training_state.loop_runtime_state, dict) else {}
    stagnation_generations = int(restored_loop_state.get('stagnation_generations', 0))
    last_combined_best = restored_loop_state.get('last_combined_best')
    if last_combined_best is not None:
        try:
            last_combined_best = float(last_combined_best)
        except Exception:
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
        append_metrics_row(stats)  # Write this generation's metrics immediately
        _log_heartbeat(generation)

        # Stability guardrail: preserve exploration while easing selection pressure on instability.
        avg_instability = float(stats.get('avg_instability', 0.0))
        if avg_instability > 0.6:
            prey_selection_pressure = None
            predator_selection_pressure = None

            if prey_engine and hasattr(prey_engine, 'reduce_selection_pressure'):
                prey_selection_pressure = prey_engine.reduce_selection_pressure(factor=0.9, min_pressure=0.5)
            if predator_engine and hasattr(predator_engine, 'reduce_selection_pressure'):
                predator_selection_pressure = predator_engine.reduce_selection_pressure(factor=0.9, min_pressure=0.5)

            print(
                f"⚖️  Stability Guard: high instability={avg_instability:.3f} | "
                f"selection pressure reduced (prey={prey_selection_pressure}, predator={predator_selection_pressure}) | "
                f"mutation_rate={config.mutation_rate:.4f}, mutation_strength={config.mutation_strength:.4f}"
            )

        skip_diagnostics = False
        gen_elapsed = time.time() - gen_start
        if gen_elapsed > MAX_GEN_TIME:
            print(
                f"⚠️  Diagnostics Skipped: generation time {gen_elapsed:.1f}s exceeded limit {MAX_GEN_TIME}s"
            )
            skip_diagnostics = True

        # Update hall of fame
        training_state.update_hall_of_fame()
        save_best_agents(
            training_state=training_state,
            generation=generation,
            prey_engine=prey_engine,
            predator_engine=predator_engine,
            models_dir="models",
        )

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
                prey_species_now = int(stats.get('prey_species', {}).get('num_species', 0))
                predator_species_now = int(stats.get('predator_species', {}).get('num_species', 0))
                total_species_now = prey_species_now + predator_species_now
                avg_instability_now = float(stats.get('avg_instability', 0.0))
                dead_units_now = int(stats.get('neural_health', {}).get('dead_layers', 0))

                # If the system is already fragmented or unstable, do not answer
                # stagnation by blindly increasing mutation. That worsens species
                # spread, dead units, and architecture chaos.
                if avg_instability_now > 0.16 or total_species_now > 20 or dead_units_now > 20:
                    old_mutation_rate = config.mutation_rate
                    old_arch_rate = config.architecture_mutation_rate
                    config.mutation_rate = max(config.mutation_rate * 0.95, 0.08)
                    config.architecture_mutation_rate = max(config.architecture_mutation_rate * 0.95, 0.03)

                    if prey_engine:
                        prey_engine.mutation_rate = config.mutation_rate
                        prey_engine.architecture_mutation_rate = config.architecture_mutation_rate
                    if predator_engine:
                        predator_engine.mutation_rate = config.mutation_rate
                        predator_engine.architecture_mutation_rate = config.architecture_mutation_rate

                    print(
                        "🧪 Meta-scientist: stagnation under high instability/diversity. "
                        f"Reducing mutation rate {old_mutation_rate:.4f} -> {config.mutation_rate:.4f} "
                        f"and architecture mutation {old_arch_rate:.4f} -> {config.architecture_mutation_rate:.4f}"
                    )
                    stagnation_generations = 0
                else:
                    old_mutation_rate = config.mutation_rate
                    old_arch_rate = config.architecture_mutation_rate
                    # Keep exploration increases bounded to avoid destabilizing behavior.
                    config.mutation_rate = min(config.mutation_rate * 1.1, 0.3)
                    config.architecture_mutation_rate = min(config.architecture_mutation_rate * 1.1, 0.08)

                    if prey_engine:
                        prey_engine.mutation_rate = config.mutation_rate
                        prey_engine.architecture_mutation_rate = config.architecture_mutation_rate
                    if predator_engine:
                        predator_engine.mutation_rate = config.mutation_rate
                        predator_engine.architecture_mutation_rate = config.architecture_mutation_rate

                    print(
                        "🧪 Meta-scientist: stagnation > 5 generations. "
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
        success_rate = float(np.mean([1.0 if f > 0 else 0.0 for f in all_fitnesses])) if all_fitnesses else 0.0
        success_rate = float(stats.get('avg_success_rate', success_rate))

        # Update curriculum controller with performance metrics
        if all_fitnesses:
            population_stats = {
                'mean': float(np.mean(all_fitnesses)),
                'max': float(max(all_fitnesses)),
                'min': float(min(all_fitnesses)),
                'std': float(np.std(all_fitnesses))
            }
        else:
            population_stats = {
                'mean': 0.0,
                'max': 0.0,
                'min': 0.0,
                'std': 0.0
            }

        new_stage = curriculum_controller.update(population_stats, diversity_score, success_rate)

        # Stagnation detection
        if len(training_state.best_prey_fitness_history) > 50:
            recent_prey = training_state.best_prey_fitness_history[-50:]
            recent_predator = training_state.best_predator_fitness_history[-50:]

            if max(recent_prey) - min(recent_prey) < 0.1:
                print("📉 Prey stagnation detected: adjusting mutation parameters")
                config.mutation_rate = min(config.mutation_rate * 1.25, 0.3)
                config.architecture_mutation_rate = min(config.architecture_mutation_rate * 1.25, 0.08)

            if max(recent_predator) - min(recent_predator) < 0.1:
                print("📉 Predator stagnation detected: adjusting mutation strength")
                config.mutation_strength = min(config.mutation_strength * 1.1, 0.4)

        # Critical floor guardrail: never allow mutation exploration to collapse.
        config.mutation_rate = max(0.1, float(config.mutation_rate))
        config.mutation_strength = max(0.15, float(config.mutation_strength))

        if prey_engine:
            prey_engine.mutation_rate = config.mutation_rate
            prey_engine.mutation_strength = config.mutation_strength
        if predator_engine:
            predator_engine.mutation_rate = config.mutation_rate
            predator_engine.mutation_strength = config.mutation_strength

        # Log meta-gene entropy on the evaluated (pre-evolution) population so the data
        # is available immediately when diagnostics/plots are generated this same generation.
        _log_meta_gene_entropy(training_state.prey_population + training_state.predator_population)

        # Save checkpoint and run diagnostics on EVALUATED population (before evolution)
        # Must run before evolve_all_populations() — after evolution all genomes reset to fitness=0.0
        top_prey_genome = None
        if not skip_diagnostics and config.plot_every > 0 and generation % config.plot_every == 0:
            save_coevolution_state(
                training_state,
                curriculum_controller=curriculum_controller,
                architect_population=architect_population,
                mutator_population=mutator_population,
                prey_engine=prey_engine,
                predator_engine=predator_engine,
                loop_runtime_state={
                    'stagnation_generations': stagnation_generations,
                    'last_combined_best': last_combined_best,
                },
            )
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

            # Meta-gene entropy plots
            MetaGeneEntropyLogger.plot_meta_gene_entropy(
                filename=f"diagnostics/meta_gene_entropy_gen{generation:04d}.png"
            )

            # Plastic weight activation timing plot
            if top_prey_genome is not None:
                brain = top_prey_genome.get_brain()
                if brain is not None:
                    brain.plot_plastic_weight_activation_timing(
                        filename=f"diagnostics/plastic_weight_activation_timing_gen{generation:04d}.png"
                    )

            # Reward recovery (learning speed compression) diagnostic
            if top_prey_genome is not None:
                evaluate_recovery_after_perturbation(
                    top_prey_genome, seed=generation, max_steps=config.max_steps, generation=generation
                )
                brain = top_prey_genome.get_brain()
                if brain is not None:
                    brain.plot_learning_speed_compression(
                        filename=f"diagnostics/learning_speed_compression_gen{generation:04d}.png"
                    )

            # Strategy clustering longitudinal trends
            from diagnostics.strategy_clustering import StrategyClusteringLogger
            StrategyClusteringLogger.plot_strategy_clustering(
                filename=f"diagnostics/strategy_clustering_trends_gen{generation:04d}.png"
            )

        _log_heartbeat(generation)

        # Evolve populations (single controller)
        print("\n🔁 Running population evolution step...")
        evolve_all_populations(generation)

        _log_heartbeat(generation)

        # Milestone 7: Run integrated meta-scientist experiments

        # Reduced frequency: every 10 generations instead of 20, skip after gen 300
        if generation % 10 == 0 and generation <= 300:
            print("\n🧪 Meta-scientist: running integrated experiments")
            
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
                                f"🧭 Curriculum diagnostic focus: {stage_name_candidate} "
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
                    print(f"🧠 NeuroGenesis: applied {total_interventions} specialized interventions")

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
            print(f"🧪 Meta-scientist completed (generation {generation})")
            for exp in experiment_results:
                hypothesis = exp.get('hypothesis', 'unknown')[:50]
                result = exp.get('result', {}).get('hypothesis_supported', False)
                print(f"  • {hypothesis}... -> {'SUPPORTED' if result else 'NOT SUPPORTED'}")
        else:
            print(f"🧪 Meta-scientist: skipped (generation {generation} not scheduled)")



        training_state.generation += 1

    # Final save
    save_coevolution_state(
        training_state,
        "data/final_coevolution_state.json",
        curriculum_controller=curriculum_controller,
        architect_population=architect_population,
        mutator_population=mutator_population,
        prey_engine=prey_engine,
        predator_engine=predator_engine,
        loop_runtime_state={
            'stagnation_generations': stagnation_generations,
            'last_combined_best': last_combined_best,
        },
    )
    evaluator.save_seeds("data/final_seed_registry.json")

    # Close evaluator
    evaluator.close()

    print("\n✅ CO-EVOLUTION TRAINING COMPLETED")
    print(f"  {'─'*86}")
    if training_state.best_prey_fitness_history:
        print(f"  Best Prey Fitness:      {max(training_state.best_prey_fitness_history):.2f}")
    else:
        print("  Best Prey Fitness:      n/a")
    if training_state.best_predator_fitness_history:
        print(f"  Best Predator Fitness:  {max(training_state.best_predator_fitness_history):.2f}")
    else:
        print("  Best Predator Fitness:  n/a")
    print(f"  Final Curriculum Stage: {curriculum_controller.get_current_config()['name']}")

async def main_async():
    """Placeholder for single-agent async training"""
    print("ℹ️  Single-agent async training is not implemented in this version")
    print("   Use --evolution-type multi for co-evolution")

def main():
    """Placeholder for single-agent sync training"""
    print("ℹ️  Single-agent sync training is not implemented in this version")
    print("   Use --evolution-type multi for co-evolution")


def launch_live_dashboard(
    enabled: bool,
    metrics_path: str = "data/metrices.csv",
    port: int = 8050,
) -> Optional[subprocess.Popen]:
    """Launch the live Plotly dashboard as a background process."""
    if not enabled:
        return None

    workspace_root = Path(__file__).resolve().parent
    dashboard_script = workspace_root / "live_metrics_dashboard.py"
    if not dashboard_script.exists():
        print(f"📈 Dashboard skipped: missing script at {dashboard_script}")
        return None

    cmd = [
        sys.executable,
        str(dashboard_script),
        "--metrics-path",
        metrics_path,
        "--port",
        str(port),
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(workspace_root),
            creationflags=creationflags,
        )
        print(f"📈 Dashboard started: http://127.0.0.1:{port}")
        return proc
    except Exception as exc:
        print(f"📈 Dashboard failed to start: {exc}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evolution Arena Training")
    parser.add_argument("--mode", choices=["async", "sync", "coevolution"], default="coevolution",
                       help="Training mode (async, sync, or coevolution)")
    parser.add_argument("--evolution-type", choices=["single", "multi"], default="multi",
                       help="Evolution type (single-agent or multi-agent)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for determinism")
    parser.add_argument("--population", type=int, default=None,
                       help="Prey population size")
    parser.add_argument("--predator-population", type=int, default=None,
                       help="Predator population size")
    parser.add_argument("--generations", type=int, default=None,
                       help="Number of generations")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml"], default="auto",
                       help="Torch device preference (auto, cpu, cuda, dml)")
    parser.add_argument("--live-dashboard", action="store_true",
                       help="Auto-start live Plotly dashboard during training")
    parser.add_argument("--live-dashboard-port", type=int, default=8050,
                       help="Port for live dashboard (used with --live-dashboard)")
    
    args = parser.parse_args()

    if args.device == "auto":
        os.environ.pop("EVOMIND_DEVICE", None)
    else:
        os.environ["EVOMIND_DEVICE"] = args.device

    cuda_available = torch.cuda.is_available()
    dml_available = torch_directml is not None
    if args.device == "auto":
        selected_runtime_device = "cuda" if cuda_available else ("dml" if dml_available else "cpu")
    else:
        selected_runtime_device = args.device
    print(
        f"🖥️  Device: requested={args.device} "
        f"torch={torch.__version__} "
        f"torch.cuda={torch.version.cuda} "
        f"cuda_available={cuda_available} "
        f"dml_available={dml_available}"
    )
    if cuda_available:
        print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")
    elif args.device == "cuda":
        raise RuntimeError(
            "CUDA was requested but is unavailable in this Python environment. "
            "Install a CUDA-enabled PyTorch build in this venv (for example: "
            "pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio)."
        )
    elif args.device == "dml" and not dml_available:
        raise RuntimeError(
            "DirectML was requested but torch-directml is not installed in this Python environment. "
            "Install it in this venv with: pip install torch-directml"
        )
    else:
        print(
            f"🖥️  Runtime device preference: {selected_runtime_device}"
        )
    
    runtime_overrides = RuntimeOverrides(
        base_seed=args.seed,
        population_size=args.population,
        predator_population_size=args.predator_population,
        generations=args.generations,
    )
    
    dashboard_proc = launch_live_dashboard(
        enabled=args.live_dashboard,
        metrics_path="data/metrices.csv",
        port=args.live_dashboard_port,
    )

    try:
        if args.evolution_type == "multi":
            # Run co-evolution
            if args.mode == "async":
                asyncio.run(main_coevolution_async(runtime_overrides=runtime_overrides))
            else:
                print("ℹ️  Co-evolution currently supports async mode only")
                asyncio.run(main_coevolution_async(runtime_overrides=runtime_overrides))
        else:
            # Run single-agent evolution (original code)
            if args.mode == "async":
                asyncio.run(main_async())
            else:
                main()
    finally:
        if dashboard_proc is not None and dashboard_proc.poll() is None:
            dashboard_proc.terminate()
            try:
                dashboard_proc.wait(timeout=5)
            except Exception:
                dashboard_proc.kill()
