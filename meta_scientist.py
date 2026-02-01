import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple

class HypothesisEngine:
    """Generate testable hypotheses about learning"""

    def __init__(self):
        self.hypothesis_templates = [
            "Plasticity is too {high/low} for this task",
            "Architecture lacks {capacity/skip_connections/recurrence}",
            "Learning rate needs to be {increased/decreased}",
            "Credit assignment is failing due to {delay/noise/complexity}",
        ]
        self.evidence_db = []

    def generate_hypothesis(self, failure_data: List[Dict[str, Any]], population_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate hypothesis from observed failures"""

        # Pattern matching on failure modes
        patterns = self._detect_patterns(failure_data)

        # Generate hypotheses
        hypotheses = []
        for pattern in patterns:
            h = self._match_template(pattern)
            h['confidence'] = self._estimate_confidence(h, self.evidence_db)
            h['test_design'] = self._design_test(h)
            hypotheses.append(h)

        # Rank by confidence and testability
        return sorted(hypotheses, key=lambda h: h['confidence'], reverse=True)

    def _detect_patterns(self, failure_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pattern detection in failures"""
        patterns = []

        # Check for correlated failures
        if self._check_plasticity_correlation(failure_data):
            patterns.append({'type': 'plasticity', 'direction': 'low'})

        if self._check_architecture_correlation(failure_data):
            patterns.append({'type': 'architecture', 'issue': 'capacity'})

        return patterns

    def _check_plasticity_correlation(self, failure_data: List[Dict[str, Any]]) -> bool:
        """Check if plasticity issues correlate with failures"""
        # Simple check: if many failures have low plasticity severity
        plasticity_failures = 0
        for failure in failure_data:
            diagnosis = failure.get('diagnosis', {})
            if diagnosis.get('learning', {}).get('severity', 0) > 0.5:
                plasticity_failures += 1

        return plasticity_failures > len(failure_data) * 0.3  # 30% threshold

    def _check_architecture_correlation(self, failure_data: List[Dict[str, Any]]) -> bool:
        """Check if architecture issues correlate with failures"""
        # Simple check: if many failures have architectural severity
        arch_failures = 0
        for failure in failure_data:
            diagnosis = failure.get('diagnosis', {})
            if diagnosis.get('architectural', {}).get('severity', 0) > 0.5:
                arch_failures += 1

        return arch_failures > len(failure_data) * 0.3  # 30% threshold

    def _match_template(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Match pattern to hypothesis template"""
        pattern_type = pattern['type']

        if pattern_type == 'plasticity':
            direction = pattern.get('direction', 'low')
            template = f"Plasticity is too {direction} for this task"
        elif pattern_type == 'architecture':
            issue = pattern.get('issue', 'capacity')
            template = f"Architecture lacks {issue}"
        else:
            template = "Unknown pattern detected"

        return {
            'statement': template,
            'pattern': pattern,
            'type': pattern_type
        }

    def _estimate_confidence(self, hypothesis: Dict[str, Any], evidence_db: List[Dict[str, Any]]) -> float:
        """Estimate confidence in hypothesis based on evidence"""
        # Simple confidence based on pattern frequency in evidence_db
        pattern_type = hypothesis['pattern']['type']
        matching_evidence = [e for e in evidence_db if e.get('type') == pattern_type]

        if not matching_evidence:
            return 0.5  # Default confidence

        # Confidence increases with more evidence
        confidence = min(0.9, 0.5 + len(matching_evidence) * 0.1)

        # Add to evidence_db for future use
        self.evidence_db.append(hypothesis['pattern'])

        return confidence

    def _design_test(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Design a test to validate the hypothesis"""
        pattern_type = hypothesis['type']

        if pattern_type == 'plasticity':
            return {
                'test_type': 'parameter_sweep',
                'parameters': ['plastic_lr', 'reward_gain'],
                'ranges': [[0.1, 10.0], [0.1, 5.0]],
                'metric': 'fitness_improvement',
                'duration': 50  # episodes
            }
        elif pattern_type == 'architecture':
            return {
                'test_type': 'architecture_modification',
                'modifications': ['add_layer', 'increase_capacity'],
                'metric': 'fitness_improvement',
                'duration': 100  # episodes
            }
        else:
            return {
                'test_type': 'general_diagnostic',
                'metric': 'fitness_improvement',
                'duration': 25
            }


class ExperimentDesigner:
    """Design and execute controlled experiments"""

    def __init__(self):
        self.random_state = np.random.RandomState(42)

    def design_experiment(self, hypothesis: str) -> Dict[str, Any]:
        """Create experimental protocol"""
        return {
            'hypothesis': hypothesis,
            'control_group': self._create_control(),
            'treatment_group': self._create_treatment(hypothesis),
            'variables_frozen': self._identify_controls(hypothesis),
            'variables_manipulated': self._identify_treatments(hypothesis),
            'success_metric': self._define_metric(hypothesis),
            'sample_size': self._calculate_sample_size(),
            'duration': self._estimate_duration()
        }

    def run_experiment(self, experiment_design: Dict[str, Any]) -> Dict[str, Any]:
        """Execute experiment"""
        # Run control group
        control_results = self._run_group(
            experiment_design['control_group'],
            frozen_vars=experiment_design['variables_frozen']
        )

        # Run treatment group
        treatment_results = self._run_group(
            experiment_design['treatment_group'],
            frozen_vars=experiment_design['variables_frozen']
        )

        # Statistical analysis
        p_value, effect_size = self._analyze_results(
            control_results,
            treatment_results
        )

        return {
            'hypothesis_supported': float(p_value) < 0.05,
            'effect_size': effect_size,
            'confidence': 1.0 - float(p_value)
        }

    def _create_control(self):
        """Create baseline control group configuration"""
        return {
            'learning_rate': 0.001,
            'plasticity': 0.1,
            'architecture': 'standard',
            'task_complexity': 'medium'
        }

    def _create_treatment(self, hypothesis):
        """Create treatment group based on hypothesis"""
        treatment = self._create_control().copy()

        if 'plasticity' in hypothesis.lower():
            if 'high' in hypothesis.lower():
                treatment['plasticity'] = 0.5
            elif 'low' in hypothesis.lower():
                treatment['plasticity'] = 0.01
        elif 'architecture' in hypothesis.lower():
            if 'capacity' in hypothesis.lower():
                treatment['architecture'] = 'high_capacity'
        elif 'learning rate' in hypothesis.lower():
            if 'increased' in hypothesis.lower():
                treatment['learning_rate'] = 0.01
            elif 'decreased' in hypothesis.lower():
                treatment['learning_rate'] = 0.0001

        return treatment

    def _identify_controls(self, hypothesis):
        """Identify variables to keep constant"""
        # Freeze task complexity and environment for most experiments
        return ['task_complexity', 'environment_seed', 'initial_weights']

    def _identify_treatments(self, hypothesis):
        """Identify variables to manipulate"""
        if 'plasticity' in hypothesis.lower():
            return ['plasticity', 'reward_gain']
        elif 'architecture' in hypothesis.lower():
            return ['hidden_layers', 'layer_size']
        elif 'learning rate' in hypothesis.lower():
            return ['learning_rate']
        else:
            return ['plasticity']

    def _define_metric(self, hypothesis):
        """Define success metric for the experiment"""
        if 'plasticity' in hypothesis.lower():
            return 'fitness_improvement_rate'
        elif 'architecture' in hypothesis.lower():
            return 'final_fitness'
        else:
            return 'learning_efficiency'

    def _calculate_sample_size(self):
        """Calculate required sample size for statistical power"""
        # Simple calculation: aim for 80% power, medium effect size
        return 30  # agents per group

    def _estimate_duration(self):
        """Estimate experiment duration in episodes"""
        return 100  # episodes

    def _run_group(self, group_config, frozen_vars):
        """Simulate running a group of agents"""
        sample_size = 30  # from _calculate_sample_size
        results = []

        for _ in range(sample_size):
            # Simulate agent performance
            baseline_performance = self.random_state.normal(50, 10)
            # Apply treatment effects
            if group_config.get('plasticity', 0.1) > 0.3:
                performance = baseline_performance * 1.2  # boost for high plasticity
            elif group_config.get('plasticity', 0.1) < 0.05:
                performance = baseline_performance * 0.8  # penalty for low plasticity
            else:
                performance = baseline_performance

            results.append(performance)

        return np.array(results)

    def _analyze_results(self, control_results: np.ndarray, treatment_results: np.ndarray) -> Tuple[float, float]:
        """Perform statistical analysis"""
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_results, treatment_results)

        # Cohen's d effect size
        mean_diff = np.mean(treatment_results) - np.mean(control_results)
        pooled_std = np.sqrt((np.var(control_results) + np.var(treatment_results)) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

        return float(np.asarray(p_value).item()), float(np.asarray(effect_size).item())
