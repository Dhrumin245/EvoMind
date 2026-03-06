"""
Curriculum Experiment Runner Module
A/B testing harness for comparing curriculum strategies statistically.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from scipy import stats
import random

# Curriculum imports
from curriculum.curriculum import CurriculumStage
from curriculum.curriculum_controller import CurriculumController

# Evolution imports
from evolution.evolution import EvolutionEngine
from core.genome import EvolvableGenome


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
