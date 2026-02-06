"""
Curriculum Controller Module
Adaptive curriculum progression based on learning progress.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import time
from scipy import stats
import random

# Import from curriculum.py
from curriculum.curriculum import CurriculumStage, get_stage_config, get_stage_transition_thresholds, get_stage_transition_graph

# Import evolution components
from evolution.evolution import EvolutionEngine
from core.genome import EvolvableGenome


@dataclass
class StagePerformance:
    """Track performance metrics for a curriculum stage"""
    stage: CurriculumStage
    start_time: float
    end_time: Optional[float] = None
    generations_in_stage: int = 0
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    success_rate_history: List[float] = field(default_factory=list)
    best_fitness_achieved: float = 0.0
    stagnation_count: int = 0
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def mean_fitness(self) -> float:
        """Get mean fitness across history"""
        if not self.fitness_history:
            return 0.0
        return float(np.mean(self.fitness_history[-50:]))  # Last 50 generations
    
    @property
    def mean_diversity(self) -> float:
        """Get mean diversity across history"""
        if not self.diversity_history:
            return 0.0
        return float(np.mean(self.diversity_history[-50:]))
    
    @property
    def mean_success_rate(self) -> float:
        """Get mean success rate across history"""
        if not self.success_rate_history:
            return 0.0
        return float(np.mean(self.success_rate_history[-50:]))
    
    @property
    def learning_speed(self) -> float:
        """Calculate learning speed (fitness improvement per generation)"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Calculate slope of fitness improvement
        recent_history = self.fitness_history[-20:]  # Last 20 generations
        if len(recent_history) < 2:
            return 0.0
        
        x = np.arange(len(recent_history))
        slope, _ = np.polyfit(x, recent_history, 1)
        return float(slope)
    
    def is_stagnant(self, threshold: float = 0.01, window: int = 10) -> bool:
        """Check if learning has stagnated"""
        if len(self.fitness_history) < window:
            return False
        
        recent_fitness = self.fitness_history[-window:]
        improvement = max(recent_fitness) - min(recent_fitness)
        return improvement < threshold


class TransitionReason(Enum):
    """Reasons for curriculum stage transition"""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    STAGNATION = "stagnation"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    MANUAL = "manual"
    RESET = "reset"
    FAILURE = "failure"


@dataclass
class TransitionRecord:
    """Record of a curriculum stage transition"""
    from_stage: CurriculumStage
    to_stage: CurriculumStage
    reason: TransitionReason
    timestamp: float
    performance_stats: Dict[str, float]
    generation: int


class CurriculumController:
    """
    Adaptive controller for curriculum progression.
    Monitors learning progress and decides when to advance or regress stages.
    """
    
    def __init__(self,
                 initial_stage: CurriculumStage = CurriculumStage.FORAGING,
                 adaptation_rate: float = 0.1,
                 enable_self_analysis: bool = True):

        self.current_stage = initial_stage
        self.initial_stage = initial_stage
        self.adaptation_rate = adaptation_rate
        self.enable_self_analysis = enable_self_analysis

        # Performance tracking
        self.current_performance = StagePerformance(stage=initial_stage, start_time=time.time())
        self.performance_history: Dict[CurriculumStage, StagePerformance] = {}
        self.transition_history: List[TransitionRecord] = []

        # Stage transition graph
        self.allowed_transitions = get_stage_transition_graph()

        # Adaptive thresholds
        self.base_thresholds = get_stage_transition_thresholds()
        self.adapted_thresholds = self.base_thresholds.copy()

        # Memory of what worked
        self.successful_transitions: List[Tuple[CurriculumStage, CurriculumStage]] = []
        self.failed_transitions: List[Tuple[CurriculumStage, CurriculumStage]] = []
        self.collapse_conditions: Dict[CurriculumStage, Dict[str, Any]] = {}

        # Current generation counter
        self.generation = 0
        
    def update(self, 
               population_stats: Dict[str, float],
               diversity_score: float,
               success_rate: float) -> Optional[CurriculumStage]:
        """
        Update curriculum based on current performance.
        Returns new stage if transition occurred, None otherwise.
        
        Args:
            population_stats: Dictionary with 'mean', 'max', 'min', 'std' of fitness
            diversity_score: Population diversity metric (0-1)
            success_rate: Task success rate (0-1)
        """
        self.generation += 1
        
        # Update current performance tracking
        self.current_performance.generations_in_stage += 1
        self.current_performance.fitness_history.append(population_stats['mean'])
        self.current_performance.diversity_history.append(diversity_score)
        self.current_performance.success_rate_history.append(success_rate)
        
        if population_stats['max'] > self.current_performance.best_fitness_achieved:
            self.current_performance.best_fitness_achieved = population_stats['max']
        
        # Check for stagnation
        if self.current_performance.is_stagnant():
            self.current_performance.stagnation_count += 1
        else:
            self.current_performance.stagnation_count = 0
        
        # Decide on transition
        new_stage = self._evaluate_transition(population_stats, diversity_score, success_rate)
        
        if new_stage and new_stage != self.current_stage:
            self._perform_transition(new_stage, population_stats, diversity_score, success_rate)
            return new_stage
        
        return None
    
    def _evaluate_transition(self,
                            population_stats: Dict[str, float],
                            diversity_score: float,
                            success_rate: float) -> Optional[CurriculumStage]:
        """
        Evaluate whether to transition to a different stage.
        Returns new stage if transition should occur, None otherwise.
        """
        current_thresholds = self.adapted_thresholds.get(self.current_stage, {})
        
        # Check if we should advance
        if self._should_advance(population_stats, diversity_score, success_rate, current_thresholds):
            next_stage = self._get_next_stage(self.current_stage)
            if next_stage:
                return next_stage
        
        # Check if we should regress (learning is too hard)
        if self._should_regress(population_stats, diversity_score, success_rate, current_thresholds):
            prev_stage = self._get_previous_stage(self.current_stage)
            if prev_stage:
                return prev_stage
        
        return None
    
    def _should_advance(self,
                       population_stats: Dict[str, float],
                       diversity_score: float,
                       success_rate: float,
                       thresholds: Dict[str, float]) -> bool:
        """Determine if population should advance to next stage"""
        
        # Check performance thresholds
        if population_stats['mean'] < thresholds.get('min_mean_fitness', 0):
            return False
        
        if success_rate < thresholds.get('min_success_rate', 0):
            return False
        
        if diversity_score < thresholds.get('min_diversity', 0):
            return False
        
        # Check stagnation
        stagnation_limit = thresholds.get('max_stagnation', 30)
        if self.current_performance.stagnation_count > stagnation_limit:
            # Too stagnant, might need to advance or change approach
            return population_stats['mean'] > thresholds.get('min_mean_fitness', 0) * 1.5
        
        # Check learning is still active
        learning_speed = self.current_performance.learning_speed
        if learning_speed > 0:  # Still learning
            return True
        elif learning_speed == 0 and self.current_performance.generations_in_stage > 20:
            # Plateaued but met thresholds
            return True
        
        return False
    
    def _should_regress(self,
                       population_stats: Dict[str, float],
                       diversity_score: float,
                       success_rate: float,
                       thresholds: Dict[str, float]) -> bool:
        """Determine if population should regress to easier stage"""
        
        # Check for collapse
        if diversity_score < 0.05:  # Extreme diversity collapse
            return True
        
        if success_rate < 0.1:  # Complete failure
            return True
        
        # Check if stuck for too long without improvement
        if (self.current_performance.stagnation_count > 
            thresholds.get('max_stagnation', 30) * 2):
            return True
        
        # Too many generations without reaching threshold
        if (self.current_performance.generations_in_stage > 100 and
            population_stats['mean'] < thresholds.get('min_mean_fitness', 0) * 0.5):
            return True
        
        return False
    
    def _get_next_stage(self, current: CurriculumStage) -> Optional[CurriculumStage]:
        """Get next stage in curriculum, if exists"""
        try:
            stages = list(CurriculumStage)
            current_idx = stages.index(current)
            if current_idx + 1 < len(stages):
                return stages[current_idx + 1]
        except ValueError:
            pass
        return None
    
    def _get_previous_stage(self, current: CurriculumStage) -> Optional[CurriculumStage]:
        """Get previous stage in curriculum, if exists"""
        try:
            stages = list(CurriculumStage)
            current_idx = stages.index(current)
            if current_idx - 1 >= 0:
                return stages[current_idx - 1]
        except ValueError:
            pass
        return None
    
    def _perform_transition(self,
                           new_stage: CurriculumStage,
                           population_stats: Dict[str, float],
                           diversity_score: float,
                           success_rate: float):
        """Execute stage transition with proper bookkeeping"""
        
        # Determine transition reason
        if new_stage.value > self.current_stage.value:
            reason = TransitionReason.PERFORMANCE_THRESHOLD
        else:
            reason = TransitionReason.STAGNATION
        
        # Record the transition
        transition_record = TransitionRecord(
            from_stage=self.current_stage,
            to_stage=new_stage,
            reason=reason,
            timestamp=time.time(),
            performance_stats={
                'mean_fitness': population_stats['mean'],
                'max_fitness': population_stats['max'],
                'diversity': diversity_score,
                'success_rate': success_rate,
                'generations_in_stage': self.current_performance.generations_in_stage
            },
            generation=self.generation
        )
        self.transition_history.append(transition_record)
        
        # Finalize current performance record
        self.current_performance.end_time = time.time()
        self.performance_history[self.current_stage] = self.current_performance
        
        # Store memory of transition
        if reason == TransitionReason.PERFORMANCE_THRESHOLD:
            self.successful_transitions.append((self.current_stage, new_stage))
        else:
            self.failed_transitions.append((self.current_stage, new_stage))
        
        # Update to new stage
        old_stage = self.current_stage
        self.current_stage = new_stage
        
        # Create new performance tracker
        self.current_performance = StagePerformance(
            stage=new_stage,
            start_time=time.time()
        )
        
        # Adapt thresholds based on experience
        self._adapt_thresholds(old_stage, new_stage, transition_record)
        
        print(f"[Curriculum] Transition: {old_stage.name} → {new_stage.name} "
              f"(Reason: {reason.value}, Gen: {self.generation})")
    
    def _adapt_thresholds(self,
                         from_stage: CurriculumStage,
                         to_stage: CurriculumStage,
                         transition: TransitionRecord):
        """Adapt thresholds based on transition experience"""
        if not self.enable_self_analysis:
            return
        
        # Get the stage we transitioned to (or from)
        target_stage = to_stage if to_stage.value > from_stage.value else from_stage
        
        if target_stage in self.adapted_thresholds:
            # Adjust based on difficulty of transition
            stats = transition.performance_stats
            
            # If transition was easy, make thresholds stricter
            if stats['generations_in_stage'] < 20:
                for key in ['min_mean_fitness', 'min_success_rate', 'min_diversity']:
                    if key in self.adapted_thresholds[target_stage]:
                        self.adapted_thresholds[target_stage][key] *= 1.1
            
            # If transition was hard, make thresholds easier
            elif stats['generations_in_stage'] > 50:
                for key in ['min_mean_fitness', 'min_success_rate', 'min_diversity']:
                    if key in self.adapted_thresholds[target_stage]:
                        self.adapted_thresholds[target_stage][key] *= 0.9
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get configuration for current stage"""
        return get_stage_config(self.current_stage)
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate progress report"""
        return {
            'current_stage': self.current_stage.name,
            'current_difficulty': self.current_stage.value,
            'generation': self.generation,
            'generations_in_current_stage': self.current_performance.generations_in_stage,
            'current_performance': {
                'mean_fitness': self.current_performance.mean_fitness,
                'best_fitness': self.current_performance.best_fitness_achieved,
                'mean_diversity': self.current_performance.mean_diversity,
                'mean_success_rate': self.current_performance.mean_success_rate,
                'learning_speed': self.current_performance.learning_speed,
                'stagnation_count': self.current_performance.stagnation_count
            },
            'transition_count': len(self.transition_history),
            'successful_transitions': len(self.successful_transitions),
            'failed_transitions': len(self.failed_transitions),
            'recent_transition': (self.transition_history[-1].to_stage.name 
                                 if self.transition_history else None)
        }
    
    def save_state(self, filepath: str):
        """Save controller state to file"""
        state = {
            'current_stage': self.current_stage.name,
            'generation': self.generation,
            'performance_history': {
                stage.name: {
                    'generations': perf.generations_in_stage,
                    'best_fitness': perf.best_fitness_achieved,
                    'duration': perf.duration
                }
                for stage, perf in self.performance_history.items()
            },
            'transition_history': [
                {
                    'from': trans.from_stage.name,
                    'to': trans.to_stage.name,
                    'reason': trans.reason.value,
                    'generation': trans.generation
                }
                for trans in self.transition_history[-50:]  # Last 50 transitions
            ],
            'adapted_thresholds': self.adapted_thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load controller state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_stage = CurriculumStage[state['current_stage']]
            self.generation = state['generation']
            # Note: More complex state restoration would be needed for full recovery
        except Exception as e:
            print(f"[Curriculum] Failed to load state: {e}")
    
    def reset_to_stage(self, stage: CurriculumStage):
        """Manually reset to a specific stage"""
        transition_record = TransitionRecord(
            from_stage=self.current_stage,
            to_stage=stage,
            reason=TransitionReason.MANUAL,
            timestamp=time.time(),
            performance_stats={},
            generation=self.generation
        )
        self.transition_history.append(transition_record)
        
        self.current_performance.end_time = time.time()
        self.performance_history[self.current_stage] = self.current_performance
        
        self.current_stage = stage
        self.current_performance = StagePerformance(
            stage=stage,
            start_time=time.time()
        )
        
        print(f"[Curriculum] Manual reset to stage: {stage.name}")


@dataclass
class ExperimentResult:
    """Results from a curriculum experiment"""
    strategy_name: str
    final_fitness: float
    learning_curve: List[float]
    convergence_generation: Optional[int]
    stability_score: float
    diversity_trajectory: List[float]


class CurriculumExperimentRunner:
    """A/B test curriculum strategies"""

    def __init__(self,
                 max_generations: int = 100,
                 significance_level: float = 0.05,
                 population_size: int = 50,
                 tournament_size: int = 5,
                 elite_count: int = 2,
                 mutation_rate: float = 0.1,
                 mutation_strength: float = 0.1,
                 architecture_mutation_rate: float = 0.05):
        self.max_generations = max_generations
        self.significance_level = significance_level

        # Evolution engine parameters
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.architecture_mutation_rate = architecture_mutation_rate

        # Initialize evolution engine
        self.evolution_engine = EvolutionEngine(
            population_size=population_size,
            tournament_size=tournament_size,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            architecture_mutation_rate=architecture_mutation_rate,
            genome_cls=EvolvableGenome
        )

    def run_experiment(self,
                      strategy_a: Callable,
                      strategy_b: Callable,
                      population,
                      evaluator: Callable) -> str:
        """
        Run A/B test comparing two curriculum strategies.

        Args:
            strategy_a: Function implementing curriculum strategy A
            strategy_b: Function implementing curriculum strategy B
            population: Population object to split and test
            evaluator: Function to evaluate fitness of population

        Returns:
            Name of winning strategy ('strategy_a' or 'strategy_b')
        """
        # Split population
        group_a, group_b = self._split_population(population)

        # Run parallel curricula
        results_a = self._run_curriculum(group_a, strategy_a, evaluator)
        results_b = self._run_curriculum(group_b, strategy_b, evaluator)

        # Statistical comparison
        winner = self._compare_strategies(results_a, results_b)

        return winner

    def _split_population(self, population) -> Tuple[Any, Any]:
        """Split population into two groups for A/B testing"""
        # Create copies of the population
        group_a = population.__class__(size=len(population)//2, name=f"{population.name}_A")
        group_b = population.__class__(size=len(population)//2, name=f"{population.name}_B")

        # Randomly assign genomes to groups
        all_genomes = list(population.genomes)
        random.shuffle(all_genomes)

        mid = len(all_genomes) // 2
        group_a.genomes = all_genomes[:mid]
        group_b.genomes = all_genomes[mid:]

        return group_a, group_b

    def _run_curriculum(self,
                       population,
                       strategy: Callable,
                       evaluator: Callable) -> ExperimentResult:
        """
        Run a single curriculum strategy and collect results.

        Args:
            population: Population to evolve
            strategy: Curriculum strategy function
            evaluator: Fitness evaluation function

        Returns:
            ExperimentResult with performance metrics
        """
        learning_curve = []
        diversity_trajectory = []

        # Initialize curriculum controller
        controller = CurriculumController()

        for generation in range(self.max_generations):
            # Evaluate current population
            fitness_scores = evaluator(population)
            diversity_score = population.get_diversity_score()

            # Update learning curve
            avg_fitness = np.mean(fitness_scores)
            learning_curve.append(avg_fitness)
            diversity_trajectory.append(diversity_score)

            # Update curriculum strategy
            population_stats = {
                'mean': avg_fitness,
                'max': max(fitness_scores),
                'min': min(fitness_scores),
                'std': np.std(fitness_scores)
            }

            # Apply curriculum strategy
            new_stage = strategy(controller, population_stats, diversity_score, generation)
            if new_stage:
                controller.reset_to_stage(new_stage)

            # Evolve population using EvolutionEngine
            population = self._evolve_population(population)

            # Check for convergence
            if self._has_converged(learning_curve):
                break

        # Calculate final metrics
        final_fitness = learning_curve[-1] if learning_curve else 0.0
        convergence_generation = self._find_convergence_generation(learning_curve)
        stability_score = self._calculate_stability_score(learning_curve)

        return ExperimentResult(
            strategy_name=strategy.__name__,
            final_fitness=final_fitness,
            learning_curve=learning_curve,
            convergence_generation=convergence_generation,
            stability_score=stability_score,
            diversity_trajectory=diversity_trajectory
        )

    def _evolve_population(self, population):
        """Placeholder for population evolution step"""
        # This would implement actual evolution logic
        # For now, just return the population unchanged
        return population

    def _has_converged(self, learning_curve: List[float], window: int = 10, threshold: float = 0.001) -> bool:
        """Check if learning has converged"""
        if len(learning_curve) < window:
            return False

        recent = learning_curve[-window:]
        improvement = max(recent) - min(recent)
        return improvement < threshold

    def _find_convergence_generation(self, learning_curve: List[float]) -> Optional[int]:
        """Find generation where learning converged"""
        # Simple heuristic: when improvement drops below threshold
        if len(learning_curve) < 20:
            return None

        for i in range(10, len(learning_curve)):
            recent_improvement = learning_curve[i] - learning_curve[i-10]
            if abs(recent_improvement) < 0.01:  # Convergence threshold
                return i

        return None

    def _calculate_stability_score(self, learning_curve: List[float]) -> float:
        """Calculate stability score (lower variance = more stable)"""
        if len(learning_curve) < 10:
            return 0.0

        recent_curve = learning_curve[-20:]  # Last 20 generations
        variance = np.var(recent_curve)
        # Normalize to 0-1 scale (lower variance = higher stability)
        stability = 1.0 / (1.0 + variance)
        return float(stability)

    def _compare_strategies(self, results_a: ExperimentResult, results_b: ExperimentResult) -> str:
        """
        Statistically compare two experiment results.

        Returns:
            'strategy_a' or 'strategy_b' based on which performed better
        """
        # Primary metric: final fitness
        fitness_a = results_a.final_fitness
        fitness_b = results_b.final_fitness

        # If clear winner by final fitness, return it
        fitness_diff = abs(fitness_a - fitness_b)
        if fitness_diff > 0.1:  # Significant difference threshold
            return 'strategy_a' if fitness_a > fitness_b else 'strategy_b'

        # Secondary metric: convergence speed
        conv_a = results_a.convergence_generation
        conv_b = results_b.convergence_generation

        if conv_a is not None and conv_b is not None:
            if conv_a < conv_b:
                return 'strategy_a'
            elif conv_b < conv_a:
                return 'strategy_b'

        # Tertiary metric: stability
        if results_a.stability_score > results_b.stability_score:
            return 'strategy_a'
        else:
            return 'strategy_b'

    def run_multiple_experiments(self,
                                strategy_a: Callable,
                                strategy_b: Callable,
                                population,
                                evaluator: Callable,
                                num_runs: int = 5) -> Dict[str, Any]:
        """
        Run multiple A/B experiments and aggregate results.

        Returns:
            Dictionary with aggregated statistics and winner determination
        """
        results_a = []
        results_b = []

        for run in range(num_runs):
            winner = self.run_experiment(strategy_a, strategy_b, population, evaluator)
            if winner == 'strategy_a':
                results_a.append(1)
                results_b.append(0)
            else:
                results_a.append(0)
                results_b.append(1)

        # Statistical analysis
        wins_a = sum(results_a)
        wins_b = sum(results_b)

        # Perform binomial test for significance
        result = stats.binomtest(wins_a, n=num_runs, p=0.5, alternative='two-sided')
        p_value = result.pvalue

        winner = 'strategy_a' if wins_a > wins_b else 'strategy_b'
        is_significant = p_value < self.significance_level

        return {
            'winner': winner,
            'strategy_a_wins': wins_a,
            'strategy_b_wins': wins_b,
            'total_runs': num_runs,
            'win_rate_a': wins_a / num_runs,
            'win_rate_b': wins_b / num_runs,
            'p_value': float(p_value),
            'statistically_significant': is_significant,
            'confidence_level': 1.0 - p_value
        }

    def curriculum_reasoning(self, failure_analysis: Dict[str, Any], current_stage: CurriculumStage) -> Dict[str, Any]:
        """
        Curriculum reasoning: not thresholds, but intelligent adaptation
        "Agents fail due to saturation" → "Introduce sparse sensory tasks"
        "Isolate obstacle avoidance"

        Args:
            failure_analysis: Analysis of why agents are failing
            current_stage: Current curriculum stage

        Returns:
            Dict with reasoning and recommended curriculum adjustments
        """
        reasoning = {
            'failure_root_causes': [],
            'recommended_actions': [],
            'curriculum_adjustments': {},
            'confidence': 0.0
        }

        # Analyze failure patterns
        saturation_indicators = failure_analysis.get('saturation_indicators', {})
        plasticity_issues = failure_analysis.get('plasticity_issues', {})
        architectural_problems = failure_analysis.get('architectural_problems', {})

        # Root cause analysis
        if saturation_indicators.get('gradient_saturation', False):
            reasoning['failure_root_causes'].append('gradient_saturation')
            reasoning['recommended_actions'].append('introduce_sparse_sensory_tasks')
            reasoning['curriculum_adjustments']['sensory_sparsity'] = 0.7
            reasoning['curriculum_adjustments']['task_complexity'] = 'reduce'

        if plasticity_issues.get('dead_layers', []) or architectural_problems.get('layer_collapse', False):
            reasoning['failure_root_causes'].append('architectural_collapse')
            reasoning['recommended_actions'].append('isolate_obstacle_avoidance')
            reasoning['curriculum_adjustments']['task_type'] = 'obstacle_avoidance_only'
            reasoning['curriculum_adjustments']['environment_complexity'] = 'minimal'

        if saturation_indicators.get('meta_parameter_saturation', False):
            reasoning['failure_root_causes'].append('meta_parameter_saturation')
            reasoning['recommended_actions'].append('introduce_meta_learning_pause')
            reasoning['curriculum_adjustments']['meta_learning_enabled'] = False
            reasoning['curriculum_adjustments']['plasticity_freeze'] = True

        if plasticity_issues.get('oscillating_plasticity', False):
            reasoning['failure_root_causes'].append('plasticity_instability')
            reasoning['recommended_actions'].append('stabilize_learning_rules')
            reasoning['curriculum_adjustments']['learning_rule_regularization'] = 0.8
            reasoning['curriculum_adjustments']['plasticity_bounds'] = 'tighten'

        # Determine confidence based on evidence strength
        evidence_count = len(reasoning['failure_root_causes'])
        reasoning['confidence'] = min(evidence_count * 0.3, 1.0)

        # Generate specific curriculum transitions
        if reasoning['confidence'] > 0.5:
            reasoning['curriculum_adjustments']['transition_reasoning'] = self._generate_transition_reasoning(
                reasoning['failure_root_causes'], current_stage
            )

        return reasoning

    def _generate_transition_reasoning(self, root_causes: List[str], current_stage: CurriculumStage) -> str:
        """Generate human-readable reasoning for curriculum transitions"""
        if 'gradient_saturation' in root_causes:
            return "Agents showing gradient saturation - introducing sparse sensory tasks to encourage efficient feature learning"
        elif 'architectural_collapse' in root_causes:
            return "Architectural collapse detected - isolating obstacle avoidance to rebuild fundamental navigation skills"
        elif 'meta_parameter_saturation' in root_causes:
            return "Meta-parameters saturated - pausing meta-learning to allow weight consolidation"
        elif 'plasticity_instability' in root_causes:
            return "Plasticity oscillating - tightening learning rule bounds for stability"
        else:
            return "Multiple failure patterns detected - applying conservative curriculum adjustments"
