import numpy as np
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class DiagnosticTask:
    """Represents a generated diagnostic task"""
    name: str
    config: Dict[str, Any]
    difficulty: float
    target_capability: str
    validation_criteria: Dict[str, Any]
    calibration_params: Dict[str, Any]


class DiagnosticTaskGenerator:
    """Generate targeted diagnostic tasks based on failure diagnosis"""

    def __init__(self):
        self.task_templates = self._initialize_task_templates()
        self.difficulty_calibration = {}  # Track task performance for calibration

    def generate_from_failure(self, failure_diagnosis: Dict[str, Any]) -> DiagnosticTask:
        """
        Create task to test specific hypothesis based on failure diagnosis

        Args:
            failure_diagnosis: Diagnosis dict from FailureAnalyzer

        Returns:
            DiagnosticTask configured for the primary failure mode
        """
        # Determine primary failure mode
        if failure_diagnosis.get('architectural', {}).get('severity', 0.0) > 0.5:
            return self._generate_capacity_test(failure_diagnosis)
        elif failure_diagnosis.get('learning', {}).get('severity', 0.0) > 0.5:
            return self._generate_plasticity_test(failure_diagnosis)
        elif failure_diagnosis.get('exploration', {}).get('severity', 0.0) > 0.5:
            return self._generate_exploration_test(failure_diagnosis)
        elif failure_diagnosis.get('credit_assignment', {}).get('severity', 0.0) > 0.5:
            return self._generate_credit_assignment_test(failure_diagnosis)
        elif failure_diagnosis.get('stability', {}).get('severity', 0.0) > 0.5:
            return self._generate_stability_test(failure_diagnosis)
        elif failure_diagnosis.get('overfitting', {}).get('severity', 0.0) > 0.5:
            return self._generate_generalization_test(failure_diagnosis)
        else:
            # Default to basic capacity test
            return self._generate_capacity_test(failure_diagnosis)

    def _generate_capacity_test(self, diagnosis: Optional[Dict[str, Any]] = None) -> DiagnosticTask:
        """Create simple task requiring minimal capacity"""
        severity = diagnosis.get('architectural', {}).get('severity', 0.3) if diagnosis else 0.3

        # Base template for capacity test
        template = self.task_templates['capacity_base'].copy()

        # Parametric difficulty adjustment
        difficulty_multiplier = 0.5 + severity * 0.5  # 0.5 to 1.0

        # Adjust task parameters based on severity
        template['max_steps'] = int(50 * difficulty_multiplier)
        template['complexity'] = difficulty_multiplier
        template['reward_scale'] = 1.0 / difficulty_multiplier  # Easier tasks have higher rewards

        # Add capacity-specific challenges
        if severity > 0.7:
            template['sparse_rewards'] = True
            template['action_space_size'] = 2  # Minimal actions
        else:
            template['sparse_rewards'] = False
            template['action_space_size'] = 4

        # Validation criteria
        validation = {
            'min_expected_score': 0.1 * difficulty_multiplier,
            'max_expected_steps': template['max_steps'] * 1.5,
            'capacity_metrics': ['parameter_efficiency', 'convergence_speed']
        }

        # Calibration parameters
        calibration = {
            'difficulty_range': [0.1, 1.0],
            'performance_target': 0.6,
            'adjustment_factor': 0.1
        }

        return DiagnosticTask(
            name=f'capacity_test_sev_{severity:.2f}',
            config=template,
            difficulty=difficulty_multiplier,
            target_capability='architectural_capacity',
            validation_criteria=validation,
            calibration_params=calibration
        )

    def _generate_plasticity_test(self, diagnosis: Optional[Dict[str, Any]] = None) -> DiagnosticTask:
        """Create task requiring fast adaptation"""
        severity = diagnosis.get('learning', {}).get('severity', 0.3) if diagnosis else 0.3

        # Base template for plasticity test
        template = self.task_templates['plasticity_base'].copy()

        # Parametric difficulty adjustment
        difficulty_multiplier = 0.5 + severity * 0.5

        # Adjust for plasticity requirements
        template['max_steps'] = int(100 * difficulty_multiplier)
        template['reward_changes'] = int(5 * difficulty_multiplier)  # More changes for higher severity
        template['adaptation_window'] = int(20 / difficulty_multiplier)  # Shorter window for harder tasks

        # Add plasticity-specific challenges
        if severity > 0.7:
            template['volatile_rewards'] = True
            template['meta_learning_required'] = True
        else:
            template['volatile_rewards'] = False
            template['meta_learning_required'] = False

        # Validation criteria
        validation = {
            'min_adaptation_rate': 0.05 * difficulty_multiplier,
            'plasticity_metrics': ['learning_speed', 'meta_parameter_adaptation'],
            'stability_threshold': 0.8
        }

        # Calibration parameters
        calibration = {
            'difficulty_range': [0.2, 1.0],
            'performance_target': 0.5,
            'adjustment_factor': 0.15
        }

        return DiagnosticTask(
            name=f'plasticity_test_sev_{severity:.2f}',
            config=template,
            difficulty=difficulty_multiplier,
            target_capability='learning_plasticity',
            validation_criteria=validation,
            calibration_params=calibration
        )

    def _generate_exploration_test(self, diagnosis: Optional[Dict[str, Any]] = None) -> DiagnosticTask:
        """Create task requiring exploration capabilities"""
        severity = diagnosis.get('exploration', {}).get('severity', 0.3) if diagnosis else 0.3

        template = self.task_templates['exploration_base'].copy()
        difficulty_multiplier = 0.5 + severity * 0.5

        template['max_steps'] = int(80 * difficulty_multiplier)
        template['state_space_size'] = int(10 * difficulty_multiplier)
        template['reward_sparsity'] = difficulty_multiplier

        validation = {
            'min_exploration_ratio': 0.3 * difficulty_multiplier,
            'novelty_discovery_rate': 0.1 * difficulty_multiplier
        }

        calibration = {
            'difficulty_range': [0.3, 1.0],
            'performance_target': 0.4,
            'adjustment_factor': 0.2
        }

        return DiagnosticTask(
            name=f'exploration_test_sev_{severity:.2f}',
            config=template,
            difficulty=difficulty_multiplier,
            target_capability='exploration_balance',
            validation_criteria=validation,
            calibration_params=calibration
        )

    def _generate_credit_assignment_test(self, diagnosis: Optional[Dict[str, Any]] = None) -> DiagnosticTask:
        """Create task requiring temporal credit assignment"""
        severity = diagnosis.get('credit_assignment', {}).get('severity', 0.3) if diagnosis else 0.3

        template = self.task_templates['credit_assignment_base'].copy()
        difficulty_multiplier = 0.5 + severity * 0.5

        template['max_steps'] = int(60 * difficulty_multiplier)
        template['delay_steps'] = int(5 * difficulty_multiplier)
        template['reward_delay_variance'] = difficulty_multiplier

        validation = {
            'min_credit_accuracy': 0.4 * difficulty_multiplier,
            'temporal_precision': 0.6
        }

        calibration = {
            'difficulty_range': [0.4, 1.0],
            'performance_target': 0.5,
            'adjustment_factor': 0.1
        }

        return DiagnosticTask(
            name=f'credit_assignment_test_sev_{severity:.2f}',
            config=template,
            difficulty=difficulty_multiplier,
            target_capability='temporal_credit_assignment',
            validation_criteria=validation,
            calibration_params=calibration
        )

    def _generate_stability_test(self, diagnosis: Optional[Dict[str, Any]] = None) -> DiagnosticTask:
        """Create task requiring learning stability"""
        severity = diagnosis.get('stability', {}).get('severity', 0.3) if diagnosis else 0.3

        template = self.task_templates['stability_base'].copy()
        difficulty_multiplier = 0.5 + severity * 0.5

        template['max_steps'] = int(70 * difficulty_multiplier)
        template['noise_level'] = 0.1 * difficulty_multiplier
        template['gradient_clipping'] = 1.0 / difficulty_multiplier

        validation = {
            'max_plasticity_variance': 0.5 / difficulty_multiplier,
            'stability_score_threshold': 0.7
        }

        calibration = {
            'difficulty_range': [0.2, 0.8],
            'performance_target': 0.6,
            'adjustment_factor': 0.05
        }

        return DiagnosticTask(
            name=f'stability_test_sev_{severity:.2f}',
            config=template,
            difficulty=difficulty_multiplier,
            target_capability='learning_stability',
            validation_criteria=validation,
            calibration_params=calibration
        )

    def _generate_generalization_test(self, diagnosis: Optional[Dict[str, Any]] = None) -> DiagnosticTask:
        """Create task requiring generalization capabilities"""
        severity = diagnosis.get('overfitting', {}).get('severity', 0.3) if diagnosis else 0.3

        template = self.task_templates['generalization_base'].copy()
        difficulty_multiplier = 0.5 + severity * 0.5

        template['max_steps'] = int(90 * difficulty_multiplier)
        template['train_env_variety'] = int(3 * difficulty_multiplier)
        template['test_env_variety'] = int(5 * difficulty_multiplier)

        validation = {
            'min_generalization_ratio': 0.5 * difficulty_multiplier,
            'robustness_score': 0.6
        }

        calibration = {
            'difficulty_range': [0.3, 1.0],
            'performance_target': 0.55,
            'adjustment_factor': 0.1
        }

        return DiagnosticTask(
            name=f'generalization_test_sev_{severity:.2f}',
            config=template,
            difficulty=difficulty_multiplier,
            target_capability='generalization_ability',
            validation_criteria=validation,
            calibration_params=calibration
        )

    def _initialize_task_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize base task templates"""
        return {
            'capacity_base': {
                'name': 'capacity_test',
                'reward_scale': 1.0,
                'noise_level': 0.0,
                'max_steps': 50,
                'complexity': 0.5,
                'action_space_size': 4,
                'sparse_rewards': False
            },
            'plasticity_base': {
                'name': 'plasticity_test',
                'reward_scale': 1.0,
                'noise_level': 0.1,
                'max_steps': 100,
                'reward_changes': 3,
                'adaptation_window': 20,
                'volatile_rewards': False,
                'meta_learning_required': False
            },
            'exploration_base': {
                'name': 'exploration_test',
                'reward_scale': 0.5,
                'noise_level': 0.2,
                'max_steps': 80,
                'state_space_size': 10,
                'reward_sparsity': 0.5
            },
            'credit_assignment_base': {
                'name': 'credit_assignment_test',
                'reward_scale': 1.0,
                'noise_level': 0.0,
                'max_steps': 60,
                'delay_steps': 3,
                'reward_delay_variance': 0.5
            },
            'stability_base': {
                'name': 'stability_test',
                'reward_scale': 1.0,
                'noise_level': 0.1,
                'max_steps': 70,
                'gradient_clipping': 1.0
            },
            'generalization_base': {
                'name': 'generalization_test',
                'reward_scale': 1.0,
                'noise_level': 0.0,
                'max_steps': 90,
                'train_env_variety': 3,
                'test_env_variety': 5
            }
        }

    def validate_task(self, task: DiagnosticTask) -> bool:
        """Validate that a generated task meets basic requirements"""
        required_keys = ['name', 'reward_scale', 'max_steps']
        if not all(key in task.config for key in required_keys):
            return False

        # Check parameter ranges
        if not (0.1 <= task.config.get('reward_scale', 0) <= 10.0):
            return False
        if not (10 <= task.config.get('max_steps', 0) <= 1000):
            return False
        if not (0.0 <= task.difficulty <= 1.0):
            return False

        return True

    def calibrate_difficulty(self, task: DiagnosticTask, performance_history: List[float]) -> DiagnosticTask:
        """
        Calibrate task difficulty based on performance history

        Args:
            task: The diagnostic task to calibrate
            performance_history: List of recent performance scores (0-1)

        Returns:
            Calibrated task with adjusted difficulty
        """
        if not performance_history:
            return task

        avg_performance = np.mean(performance_history)
        target = task.calibration_params['performance_target']
        adjustment_factor = task.calibration_params['adjustment_factor']

        # Adjust difficulty based on performance relative to target
        if avg_performance > target + 0.1:
            # Too easy, increase difficulty
            new_difficulty = min(1.0, task.difficulty + adjustment_factor)
        elif avg_performance < target - 0.1:
            # Too hard, decrease difficulty
            new_difficulty = max(0.1, task.difficulty - adjustment_factor)
        else:
            new_difficulty = task.difficulty

        # Update task parameters based on new difficulty
        calibrated_task = self._adjust_task_parameters(task, new_difficulty)
        calibrated_task.difficulty = new_difficulty

        return calibrated_task

    def _adjust_task_parameters(self, task: DiagnosticTask, new_difficulty: float) -> DiagnosticTask:
        """Adjust task parameters for new difficulty level"""
        adjusted_config = task.config.copy()

        # Scale parameters based on difficulty
        difficulty_ratio = new_difficulty / task.difficulty if task.difficulty > 0 else 1.0

        # Adjust common parameters
        if 'max_steps' in adjusted_config:
            adjusted_config['max_steps'] = int(adjusted_config['max_steps'] * difficulty_ratio)

        if 'reward_scale' in adjusted_config:
            adjusted_config['reward_scale'] = adjusted_config['reward_scale'] / difficulty_ratio

        if 'noise_level' in adjusted_config:
            adjusted_config['noise_level'] = min(0.5, adjusted_config['noise_level'] * difficulty_ratio)

        # Task-specific adjustments
        if task.target_capability == 'architectural_capacity':
            adjusted_config['complexity'] = new_difficulty
        elif task.target_capability == 'learning_plasticity':
            adjusted_config['reward_changes'] = int(adjusted_config.get('reward_changes', 3) * difficulty_ratio)
        elif task.target_capability == 'exploration_balance':
            adjusted_config['state_space_size'] = int(adjusted_config.get('state_space_size', 10) * difficulty_ratio)
        elif task.target_capability == 'temporal_credit_assignment':
            adjusted_config['delay_steps'] = int(adjusted_config.get('delay_steps', 3) * difficulty_ratio)

        return DiagnosticTask(
            name=task.name,
            config=adjusted_config,
            difficulty=new_difficulty,
            target_capability=task.target_capability,
            validation_criteria=task.validation_criteria,
            calibration_params=task.calibration_params
        )

    def generate_task_suite(self, diagnosis: Dict[str, Any], num_tasks: int = 3) -> List[DiagnosticTask]:
        """
        Generate a suite of diagnostic tasks for comprehensive evaluation

        Args:
            diagnosis: Failure diagnosis dict
            num_tasks: Number of tasks to generate

        Returns:
            List of diagnostic tasks
        """
        tasks = []

        # Always include primary failure task
        primary_task = self.generate_from_failure(diagnosis)
        tasks.append(primary_task)

        # Add complementary tasks for other high-severity failures
        severity_threshold = 0.4
        high_severity_failures = [
            failure_type for failure_type, data in diagnosis.items()
            if isinstance(data, dict) and data.get('severity', 0) > severity_threshold
        ]

        # Generate tasks for other significant failures
        for failure_type in high_severity_failures[1:num_tasks]:  # Skip primary (already added)
            if failure_type == 'architectural':
                tasks.append(self._generate_capacity_test(diagnosis))
            elif failure_type == 'learning':
                tasks.append(self._generate_plasticity_test(diagnosis))
            elif failure_type == 'exploration':
                tasks.append(self._generate_exploration_test(diagnosis))
            elif failure_type == 'credit_assignment':
                tasks.append(self._generate_credit_assignment_test(diagnosis))
            elif failure_type == 'stability':
                tasks.append(self._generate_stability_test(diagnosis))
            elif failure_type == 'overfitting':
                tasks.append(self._generate_generalization_test(diagnosis))

        # Fill remaining slots with varied diagnostic tasks
        while len(tasks) < num_tasks:
            # Randomly select a task type
            task_types = ['capacity', 'plasticity', 'exploration', 'credit_assignment', 'stability', 'generalization']
            random_type = random.choice(task_types)

            if random_type == 'capacity':
                tasks.append(self._generate_capacity_test())
            elif random_type == 'plasticity':
                tasks.append(self._generate_plasticity_test())
            elif random_type == 'exploration':
                tasks.append(self._generate_exploration_test())
            elif random_type == 'credit_assignment':
                tasks.append(self._generate_credit_assignment_test())
            elif random_type == 'stability':
                tasks.append(self._generate_stability_test())
            elif random_type == 'generalization':
                tasks.append(self._generate_generalization_test())

        return tasks[:num_tasks]  # Ensure we don't exceed requested number
