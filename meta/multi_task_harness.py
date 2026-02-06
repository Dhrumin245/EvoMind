"""
Multi-task generalization harness for evolutionary training.
Defines task suites with varied configurations and handles subset sampling + full benchmarks.
"""
import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from curriculum.curriculum import CurriculumStage, get_stage_config
import random


class TaskType(Enum):
    """Types of tasks in the generalization suite"""
    ARENA_CONFIG = "arena_config"
    CURRICULUM_STAGE = "curriculum_stage"
    PREDATOR_PREY_RATIO = "predator_prey_ratio"
    SENSOR_NOISE = "sensor_noise"
    ENERGY_DYNAMICS = "energy_dynamics"
    FOOD_DISTRIBUTION = "food_distribution"


@dataclass
class TaskConfig:
    """Configuration for a single task in the generalization suite"""
    task_id: str
    task_type: TaskType
    name: str
    description: str

    # Arena configuration variations
    screen_width: int = 800
    screen_height: int = 600
    num_food: int = 10
    food_respawn_rate: float = 0.05
    boundary_margin: int = 20
    max_steps: int = 80

    # Agent configuration variations
    num_prey_per_env: int = 10
    num_predators_per_env: int = 3
    prey_speed: float = 2.0
    predator_speed: float = 2.5
    prey_vision_range: float = 250.0
    predator_vision_range: float = 350.0
    prey_energy_max: float = 80.0
    predator_energy_max: float = 120.0

    # Curriculum stage
    curriculum_stage: CurriculumStage = CurriculumStage.FORAGING

    # Sensor noise parameters
    sensor_noise_std: float = 0.0
    sensor_noise_prob: float = 0.0

    # Energy dynamics
    energy_decay_rate: float = 1.0
    food_energy_variation: float = 0.0

    # Food distribution
    food_clustering: float = 0.0  # 0 = uniform, 1 = clustered
    food_respawn_delay: int = 0

    # Reward scaling
    food_reward_scale: float = 1.0
    capture_reward_scale: float = 1.0
    step_penalty_scale: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'name': self.name,
            'description': self.description,
            'screen_width': self.screen_width,
            'screen_height': self.screen_height,
            'num_food': self.num_food,
            'food_respawn_rate': self.food_respawn_rate,
            'boundary_margin': self.boundary_margin,
            'max_steps': self.max_steps,
            'num_prey_per_env': self.num_prey_per_env,
            'num_predators_per_env': self.num_predators_per_env,
            'prey_speed': self.prey_speed,
            'predator_speed': self.predator_speed,
            'prey_vision_range': self.prey_vision_range,
            'predator_vision_range': self.predator_vision_range,
            'prey_energy_max': self.prey_energy_max,
            'predator_energy_max': self.predator_energy_max,
            'curriculum_stage': self.curriculum_stage.name,
            'sensor_noise_std': self.sensor_noise_std,
            'sensor_noise_prob': self.sensor_noise_prob,
            'energy_decay_rate': self.energy_decay_rate,
            'food_energy_variation': self.food_energy_variation,
            'food_clustering': self.food_clustering,
            'food_respawn_delay': self.food_respawn_delay,
            'food_reward_scale': self.food_reward_scale,
            'capture_reward_scale': self.capture_reward_scale,
            'step_penalty_scale': self.step_penalty_scale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """Create from dictionary"""
        data_copy = data.copy()
        data_copy['task_type'] = TaskType(data_copy['task_type'])
        data_copy['curriculum_stage'] = CurriculumStage[data_copy['curriculum_stage']]
        return cls(**data_copy)


@dataclass
class TaskSuite:
    """Collection of tasks for generalization evaluation"""
    tasks: List[TaskConfig] = field(default_factory=list)
    base_seed: int = 42

    def __post_init__(self):
        """Initialize the task suite with diverse configurations"""
        if not self.tasks:
            self._generate_task_suite()

    def _generate_task_suite(self):
        """Generate a comprehensive task suite"""
        task_id = 0

        # 1. Arena size variations
        for width, height in [(600, 450), (800, 600), (1000, 750), (1200, 900)]:
            self.tasks.append(TaskConfig(
                task_id=f"arena_size_{task_id}",
                task_type=TaskType.ARENA_CONFIG,
                name=f"Arena {width}x{height}",
                description=f"Different arena dimensions: {width}x{height}",
                screen_width=width,
                screen_height=height,
                num_food=max(5, int(10 * (width * height) / (800 * 600))),  # Scale food count
            ))
            task_id += 1

        # 2. Curriculum stage variations
        for stage in CurriculumStage:
            config = get_stage_config(stage)
            self.tasks.append(TaskConfig(
                task_id=f"curriculum_{stage.name.lower()}_{task_id}",
                task_type=TaskType.CURRICULUM_STAGE,
                name=f"Stage: {stage.name}",
                description=f"Curriculum stage: {stage.name} - {config['description']}",
                curriculum_stage=stage,
                num_food=config.get('food_count', 10),
                max_steps=config.get('max_steps', 80),
                food_reward_scale=config.get('food_reward', 10.0) / 10.0,  # Normalize
                step_penalty_scale=abs(config.get('step_penalty', -0.01)) / 0.01,  # Normalize
            ))
            task_id += 1

        # 3. Predator/prey ratio variations
        ratios = [
            (5, 1),   # Many prey, few predators
            (10, 3),  # Balanced
            (15, 5),  # More prey
            (8, 4),   # More predators
            (20, 2),  # Very prey-heavy
            (6, 6),   # Equal numbers
        ]
        for num_prey, num_pred in ratios:
            self.tasks.append(TaskConfig(
                task_id=f"ratio_{num_prey}_{num_pred}_{task_id}",
                task_type=TaskType.PREDATOR_PREY_RATIO,
                name=f"Ratio {num_prey}:{num_pred}",
                description=f"Predator/prey ratio: {num_prey} prey vs {num_pred} predators",
                num_prey_per_env=num_prey,
                num_predators_per_env=num_pred,
            ))
            task_id += 1

        # 4. Sensor noise variations
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        for noise_std in noise_levels:
            self.tasks.append(TaskConfig(
                task_id=f"sensor_noise_{noise_std}_{task_id}",
                task_type=TaskType.SENSOR_NOISE,
                name=f"Sensor Noise {noise_std}",
                description=f"Sensor noise with std={noise_std}",
                sensor_noise_std=noise_std,
                sensor_noise_prob=0.1,  # 10% chance of noise per observation
            ))
            task_id += 1

        # 5. Energy dynamics variations
        energy_configs = [
            (0.5, "Low Energy Decay"),
            (1.0, "Normal Energy Decay"),
            (1.5, "High Energy Decay"),
            (2.0, "Very High Energy Decay"),
        ]
        for decay_rate, desc in energy_configs:
            self.tasks.append(TaskConfig(
                task_id=f"energy_{decay_rate}_{task_id}",
                task_type=TaskType.ENERGY_DYNAMICS,
                name=desc,
                description=f"Energy decay rate: {decay_rate}x normal",
                energy_decay_rate=decay_rate,
                prey_energy_max=int(80 / decay_rate),  # Adjust max energy
                predator_energy_max=int(120 / decay_rate),
            ))
            task_id += 1

        # 6. Food distribution variations
        food_configs = [
            (0.0, 0, "Uniform Distribution"),
            (0.3, 5, "Mild Clustering"),
            (0.7, 10, "Strong Clustering"),
            (1.0, 20, "Extreme Clustering"),
        ]
        for clustering, delay, desc in food_configs:
            self.tasks.append(TaskConfig(
                task_id=f"food_dist_{clustering}_{delay}_{task_id}",
                task_type=TaskType.FOOD_DISTRIBUTION,
                name=desc,
                description=f"Food clustering: {clustering}, respawn delay: {delay}",
                food_clustering=clustering,
                food_respawn_delay=delay,
                num_food=max(5, int(10 * (1 + clustering))),  # More food when clustered
            ))
            task_id += 1

        # 7. Combined challenge tasks (mix of multiple variations)
        combined_configs = [
            {
                'name': 'Sparse Resources + Noise',
                'num_food': 5,
                'sensor_noise_std': 0.05,
                'energy_decay_rate': 1.5,
                'num_predators_per_env': 4,
            },
            {
                'name': 'Abundant + Fast Predators',
                'num_food': 20,
                'predator_speed': 3.5,
                'num_predators_per_env': 5,
                'prey_speed': 1.5,
            },
            {
                'name': 'Large Arena + Low Vision',
                'screen_width': 1200,
                'screen_height': 900,
                'prey_vision_range': 150.0,
                'predator_vision_range': 200.0,
            },
            {
                'name': 'Energy Critical + Clustered Food',
                'energy_decay_rate': 2.0,
                'food_clustering': 0.8,
                'num_food': 8,
            },
        ]

        for i, config in enumerate(combined_configs):
            task_config = TaskConfig(
                task_id=f"combined_{i}_{task_id}",
                task_type=TaskType.ARENA_CONFIG,  # Generic type for combined
                name=config['name'],
                description=f"Combined challenge: {config['name']}",
                **{k: v for k, v in config.items() if k != 'name'}
            )
            self.tasks.append(task_config)
            task_id += 1

        print(f"Generated task suite with {len(self.tasks)} tasks")

    def sample_tasks(self, num_tasks: int, seed: Optional[int] = None) -> List[TaskConfig]:
        """Sample a subset of tasks for evaluation"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if num_tasks >= len(self.tasks):
            return self.tasks.copy()

        return random.sample(self.tasks, num_tasks)

    def get_task_by_type(self, task_type: TaskType) -> List[TaskConfig]:
        """Get all tasks of a specific type"""
        return [task for task in self.tasks if task.task_type == task_type]

    def save_suite(self, filename: str = "task_suite.json"):
        """Save task suite to JSON file"""
        suite_data = {
            'base_seed': self.base_seed,
            'num_tasks': len(self.tasks),
            'tasks': [task.to_dict() for task in self.tasks]
        }
        with open(filename, 'w') as f:
            json.dump(suite_data, f, indent=2)
        print(f"Task suite saved to {filename}")

    def load_suite(self, filename: str = "task_suite.json"):
        """Load task suite from JSON file"""
        with open(filename, 'r') as f:
            suite_data = json.load(f)

        self.base_seed = suite_data['base_seed']
        self.tasks = [TaskConfig.from_dict(task_data) for task_data in suite_data['tasks']]
        print(f"Task suite loaded from {filename} with {len(self.tasks)} tasks")


@dataclass
class BenchmarkResult:
    """Result of evaluating a genome on a task"""
    task_id: str
    fitness: float
    prey_fitness: Optional[float] = None
    predator_fitness: Optional[float] = None
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'fitness': self.fitness,
            'prey_fitness': self.prey_fitness,
            'predator_fitness': self.predator_fitness,
            'evaluation_time': self.evaluation_time,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create from dictionary"""
        return cls(
            task_id=data['task_id'],
            fitness=data['fitness'],
            prey_fitness=data.get('prey_fitness'),
            predator_fitness=data.get('predator_fitness'),
            evaluation_time=data.get('evaluation_time', 0.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class GeneralizationReport:
    """Report on generalization performance across tasks"""
    generation: int
    genome_id: str
    benchmark_results: List[BenchmarkResult]
    summary_stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._compute_summary_stats()

    def _compute_summary_stats(self):
        """Compute summary statistics across all benchmark results"""
        if not self.benchmark_results:
            return

        fitnesses = [r.fitness for r in self.benchmark_results]
        eval_times = [r.evaluation_time for r in self.benchmark_results]

        self.summary_stats = {
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'median_fitness': float(np.median(fitnesses)),
            'total_evaluation_time': float(np.sum(eval_times)),
            'mean_evaluation_time': float(np.mean(eval_times)),
            'num_tasks_evaluated': len(self.benchmark_results),
            'fitness_range': float(np.max(fitnesses) - np.min(fitnesses)),
            'fitness_coefficient_of_variation': float(np.std(fitnesses) / np.mean(fitnesses)) if np.mean(fitnesses) != 0 else 0.0,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'genome_id': self.genome_id,
            'benchmark_results': [r.to_dict() for r in self.benchmark_results],
            'summary_stats': self.summary_stats,
        }


class MultiTaskEvaluator:
    """Evaluator that handles multi-task generalization assessment"""

    def __init__(self, task_suite: TaskSuite, base_seed: int = 42,
                 hall_of_fame_fraction: float = 0.7, recent_fraction: float = 0.3):
        self.task_suite = task_suite
        self.base_seed = base_seed
        self.rng = np.random.RandomState(base_seed)
        self.hall_of_fame_fraction = hall_of_fame_fraction
        self.recent_fraction = recent_fraction

    def evaluate_genome_on_task(self, genome, task: TaskConfig,
                               num_opponents: int = 1, max_steps: int = 80,
                               hall_of_fame_prey: Optional[List] = None,
                               hall_of_fame_pred: Optional[List] = None,
                               current_prey: Optional[List] = None,
                               current_pred: Optional[List] = None) -> BenchmarkResult:
        """
        Evaluate a single genome on a specific task
        """
        start_time = time.time()

        # Create arena configuration from task
        arena_config = self._task_to_arena_config(task)

        # Import here to avoid circular imports
        from environments.arena_multi import MultiAgentArena

        # Create arena with task-specific configuration
        arena = MultiAgentArena(
            batch_size=1,  # Single environment for evaluation
            num_prey_per_env=task.num_prey_per_env,
            num_predators_per_env=task.num_predators_per_env,
            config=arena_config,
            seed=self.rng.randint(0, int(1e9)),  # Deterministic but varied seed
        )

        # Sample opponents from hall of fame and current populations
        opponent_genomes = self._sample_opponents(
            genome, num_opponents, task,
            hall_of_fame_prey or [], hall_of_fame_pred or [],
            current_prey or [], current_pred or []
        )

        total_fitness = 0.0
        prey_fitness = 0.0
        predator_fitness = 0.0

        # Evaluate against multiple opponents
        for opponent in opponent_genomes:
            fitness_prey, fitness_pred = self._evaluate_pair(genome, opponent, arena, max_steps)
            total_fitness += fitness_prey if hasattr(genome, 'num_prey_agents') else fitness_pred
            prey_fitness += fitness_prey
            predator_fitness += fitness_pred

        avg_fitness = total_fitness / len(opponent_genomes)
        avg_prey_fitness = prey_fitness / len(opponent_genomes)
        avg_predator_fitness = predator_fitness / len(opponent_genomes)

        evaluation_time = time.time() - start_time

        return BenchmarkResult(
            task_id=task.task_id,
            fitness=avg_fitness,
            prey_fitness=avg_prey_fitness,
            predator_fitness=avg_predator_fitness,
            evaluation_time=evaluation_time,
            metadata={
                'task_type': task.task_type.value,
                'num_opponents': num_opponents,
                'arena_config': arena_config,
                'curriculum_stage': task.curriculum_stage.name,
            }
        )

    def _task_to_arena_config(self, task: TaskConfig) -> Dict[str, Any]:
        """Convert task config to arena configuration dict"""
        # Get base config from curriculum stage
        config = get_stage_config(task.curriculum_stage).copy()

        # Override with task-specific settings
        config.update({
            'screen_width': task.screen_width,
            'screen_height': task.screen_height,
            'food_count': task.num_food,
            'food_respawn_rate': task.food_respawn_rate,
            'max_steps': task.max_steps,
            'prey_speed': task.prey_speed,
            'predator_speed': task.predator_speed,
            'prey_vision_range': task.prey_vision_range,
            'predator_vision_range': task.predator_vision_range,
            'prey_energy': task.prey_energy_max,
            'predator_energy': task.predator_energy_max,
            'sensor_noise_std': task.sensor_noise_std,
            'sensor_noise_prob': task.sensor_noise_prob,
            'energy_decay_rate': task.energy_decay_rate,
            'food_energy_variation': task.food_energy_variation,
            'food_clustering': task.food_clustering,
            'food_respawn_delay': task.food_respawn_delay,
            'food_reward_scale': task.food_reward_scale,
            'capture_reward_scale': task.capture_reward_scale,
            'step_penalty_scale': task.step_penalty_scale,
        })

        return config

    def _create_random_opponents(self, genome, num_opponents: int, task: TaskConfig):
        """Create random opponent genomes for evaluation"""
        opponents = []

        # Import here to avoid circular imports
        from genome_prey import PreyGenome
        from genome_predator import PredatorGenome

        for _ in range(num_opponents):
            if isinstance(genome, PreyGenome):
                opponents.append(PredatorGenome.random_initialization())
            else:
                opponents.append(PreyGenome.random_initialization())

        return opponents

    def _sample_opponents(self, genome, num_opponents: int, task: TaskConfig,
                         hall_of_fame_prey: List, hall_of_fame_pred: List,
                         current_prey: List, current_pred: List):
        """
        Sample opponents from hall of fame and current populations using deterministic policy
        """
        from genome_prey import PreyGenome
        from genome_predator import PredatorGenome

        # Determine opponent type needed
        is_prey_genome = isinstance(genome, PreyGenome)
        opponent_hof = hall_of_fame_pred if is_prey_genome else hall_of_fame_prey
        opponent_current = current_pred if is_prey_genome else current_prey

        # If no populations available, fall back to random opponents
        if not opponent_hof and not opponent_current:
            return self._create_random_opponents(genome, num_opponents, task)

        opponents = []

        # Calculate sampling counts
        hof_count = int(num_opponents * self.hall_of_fame_fraction)
        recent_count = int(num_opponents * self.recent_fraction)
        random_count = num_opponents - hof_count - recent_count

        # Sample from hall of fame (deterministic by fitness ranking)
        if hof_count > 0 and opponent_hof:
            # Sort by fitness (highest first) and take top
            sorted_hof = sorted(opponent_hof, key=lambda g: g.fitness, reverse=True)
            hof_sample = sorted_hof[:min(hof_count, len(sorted_hof))]
            opponents.extend(hof_sample)

        # Sample from current population (deterministic by fitness ranking)
        if recent_count > 0 and opponent_current:
            # Sort by fitness (highest first) and take top
            sorted_current = sorted(opponent_current, key=lambda g: g.fitness, reverse=True)
            current_sample = sorted_current[:min(recent_count, len(sorted_current))]
            opponents.extend(current_sample)

        # Fill remaining with random opponents
        if len(opponents) < num_opponents:
            remaining = num_opponents - len(opponents)
            random_opponents = self._create_random_opponents(genome, remaining, task)
            opponents.extend(random_opponents)

        # Trim if we have too many (shouldn't happen but safety check)
        opponents = opponents[:num_opponents]

        return opponents

    def _evaluate_pair(self, genome1, genome2, arena, max_steps: int) -> Tuple[float, float]:
        """Evaluate a genome pair in the arena with stability monitoring and penalties"""
        # Reset arena
        prey_state, pred_state = arena.reset()

        # Determine which is prey and which is predator
        from genome_prey import PreyGenome
        from genome_predator import PredatorGenome

        if isinstance(genome1, PreyGenome):
            prey_genome, predator_genome = genome1, genome2
        else:
            prey_genome, predator_genome = genome2, genome1

        total_prey_reward = 0.0
        total_predator_reward = 0.0

        # Milestone 6: Collect stability diagnostics during evaluation
        prey_stability_penalty = 0.0
        predator_stability_penalty = 0.0

        for step in range(max_steps):
            # Get actions
            prey_actions = prey_genome.act_batch(prey_state)
            pred_actions = predator_genome.act_batch(pred_state)

            # Step arena
            (prey_state, pred_state), r_prey, r_pred, info = arena.step(
                prey_actions, pred_actions
            )

            total_prey_reward += float(np.mean(r_prey))
            total_predator_reward += float(np.mean(r_pred))

            if np.any(info['env_done']):
                break

        # Milestone 6: Apply stability penalties based on activation monitoring
        prey_brain = getattr(prey_genome, 'brain', None)
        if prey_brain is not None and hasattr(prey_brain, 'get_stability_diagnostics'):
            prey_stability = prey_brain.get_stability_diagnostics()
            if 'avg_saturation_fraction' in prey_stability:
                saturation_penalty = float(prey_stability.get('avg_saturation_fraction', 0.0)) * 0.5  # Penalty for high saturation
                dead_unit_penalty = float(prey_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3  # Penalty for dead units
                prey_stability_penalty = saturation_penalty + dead_unit_penalty

        predator_brain = getattr(predator_genome, 'brain', None)
        if predator_brain is not None and hasattr(predator_brain, 'get_stability_diagnostics'):
            predator_stability = predator_brain.get_stability_diagnostics()
            if 'avg_saturation_fraction' in predator_stability:
                saturation_penalty = float(predator_stability.get('avg_saturation_fraction', 0.0)) * 0.5
                dead_unit_penalty = float(predator_stability.get('avg_dead_unit_fraction', 0.0)) * 0.3
                predator_stability_penalty = saturation_penalty + dead_unit_penalty

        # Apply penalties to rewards
        final_prey_reward = total_prey_reward - prey_stability_penalty
        final_predator_reward = total_predator_reward - predator_stability_penalty

        return final_prey_reward, final_predator_reward

    def run_full_benchmark(self, genome, generation: int,
                          max_tasks: Optional[int] = None) -> GeneralizationReport:
        """
        Run full benchmark evaluation on all (or subset) tasks
        """
        tasks_to_evaluate = self.task_suite.tasks
        if max_tasks and max_tasks < len(tasks_to_evaluate):
            tasks_to_evaluate = self.task_suite.sample_tasks(max_tasks, seed=generation)

        print(f"Running full benchmark on {len(tasks_to_evaluate)} tasks for generation {generation}")

        benchmark_results = []
        for i, task in enumerate(tasks_to_evaluate):
            if i % 10 == 0:
                print(f"  Evaluating task {i+1}/{len(tasks_to_evaluate)}: {task.name}")

            result = self.evaluate_genome_on_task(genome, task)
            benchmark_results.append(result)

        genome_id = getattr(genome, 'genome_id', f'genome_{id(genome)}')

        return GeneralizationReport(
            generation=generation,
            genome_id=genome_id,
            benchmark_results=benchmark_results
        )

    def run_subset_evaluation(self, genome, num_tasks: int, generation: int,
                             hall_of_fame_prey: Optional[List] = None,
                             hall_of_fame_pred: Optional[List] = None,
                             current_prey: Optional[List] = None,
                             current_pred: Optional[List] = None) -> GeneralizationReport:
        """
        Run evaluation on a sampled subset of tasks
        """
        subset_tasks = self.task_suite.sample_tasks(num_tasks, seed=generation * 1000 + hash(genome) % 1000)

        print(f"Running subset evaluation on {len(subset_tasks)} tasks")

        benchmark_results = []
        for task in subset_tasks:
            result = self.evaluate_genome_on_task(
                genome, task,
                hall_of_fame_prey=hall_of_fame_prey,
                hall_of_fame_pred=hall_of_fame_pred,
                current_prey=current_prey,
                current_pred=current_pred
            )
            benchmark_results.append(result)

        genome_id = getattr(genome, 'genome_id', f'genome_{id(genome)}')

        return GeneralizationReport(
            generation=generation,
            genome_id=genome_id,
            benchmark_results=benchmark_results
        )


# Global task suite instance
_default_task_suite = None

def get_default_task_suite() -> TaskSuite:
    """Get the default task suite (singleton pattern)"""
    global _default_task_suite
    if _default_task_suite is None:
        _default_task_suite = TaskSuite()
    return _default_task_suite

def get_multi_task_evaluator(base_seed: int = 42) -> MultiTaskEvaluator:
    """Get a multi-task evaluator with the default task suite"""
    return MultiTaskEvaluator(get_default_task_suite(), base_seed)
