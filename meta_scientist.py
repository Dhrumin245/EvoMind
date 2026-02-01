import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple, Sequence
from failure_analyzer import FailureAnalyzer
from knowledge_base import KnowledgeBase
from genome import EvolvableGenome

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


class MetaScientist:
    """Orchestrates the complete meta-scientific pipeline"""

    def __init__(self, knowledge_base_path: str = "meta_scientist_kb.db"):
        self.failure_analyzer = FailureAnalyzer()
        self.hypothesis_engine = HypothesisEngine()
        self.experiment_designer = ExperimentDesigner()
        self.knowledge_base = KnowledgeBase(knowledge_base_path)
        self.experiment_history = []

    def analyze_population_failures(self, population: Sequence[Any], task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze failures across a population and generate insights

        Args:
            population: List of genomes to analyze
            task_info: Information about the current task/environment

        Returns:
            Analysis results with diagnoses, hypotheses, and recommendations
        """
        print("Meta-Scientist: Analyzing population failures...")

        # Diagnose failures for each genome
        failure_data = []
        for genome in population:
            diagnosis = self.failure_analyzer.diagnose_failure(genome, task_info)
            failure_data.append({
                'genome_id': genome.genome_id,
                'fitness': genome.fitness,
                'diagnosis': diagnosis
            })

        # Get population statistics
        population_stats = self._compute_population_stats(population)

        # Generate hypotheses from failure patterns
        hypotheses = self.hypothesis_engine.generate_hypothesis(failure_data, population_stats)

        # Get systemic failure patterns
        systemic_fixes = self.failure_analyzer.suggest_systemic_fixes()

        return {
            'failure_data': failure_data,
            'population_stats': population_stats,
            'hypotheses': hypotheses,
            'systemic_fixes': systemic_fixes,
            'total_failures': len([f for f in failure_data if f['diagnosis']['total_severity'] > 0.5])
        }

    def run_automated_experiments(self, hypotheses: List[Dict[str, Any]], population: Sequence[Any],
                                task_info: Dict[str, Any], generation: int) -> List[Dict[str, Any]]:
        """
        Run automated experiments to test hypotheses

        Args:
            hypotheses: List of hypotheses to test
            population: Current population for baseline
            task_info: Task information
            generation: Current generation number

        Returns:
            List of experiment results
        """
        print(f"Meta-Scientist: Running {len(hypotheses)} automated experiments...")

        experiment_results = []

        # Limit to top 3 hypotheses to avoid excessive computation
        for hypothesis in hypotheses[:3]:
            try:
                # Design experiment
                experiment_design = self.experiment_designer.design_experiment(hypothesis['statement'])

                # Run experiment (currently simulated)
                result = self.experiment_designer.run_experiment(experiment_design)

                # Store result
                experiment_record = {
                    'generation': generation,
                    'hypothesis': hypothesis['statement'],
                    'hypothesis_confidence': hypothesis['confidence'],
                    'experiment_design': experiment_design,
                    'result': result,
                    'timestamp': np.datetime64('now'),
                    'task_info': task_info
                }

                experiment_results.append(experiment_record)
                self.experiment_history.append(experiment_record)

                # Store in knowledge base
                self._store_experiment_in_kb(experiment_record)

                print(f"  Experiment completed: {hypothesis['statement'][:50]}... -> {'SUPPORTED' if result['hypothesis_supported'] else 'NOT SUPPORTED'}")

            except Exception as e:
                print(f"  Experiment failed for hypothesis '{hypothesis['statement']}': {e}")
                continue

        return experiment_results

    def learn_from_experiments(self, experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract knowledge and principles from experiment results

        Args:
            experiment_results: Results from recent experiments

        Returns:
            Learning outcomes and new principles
        """
        print("Meta-Scientist: Learning from experiment results...")

        learning_outcomes = {
            'new_principles': [],
            'theory_updates': [],
            'strategy_recommendations': []
        }

        # Extract principles from successful experiments
        successful_experiments = [exp for exp in experiment_results if exp['result']['hypothesis_supported']]

        if successful_experiments:
            principle = self.knowledge_base.extract_principle(successful_experiments)
            if principle:
                learning_outcomes['new_principles'].append(principle)
                print(f"  New principle extracted: {principle.rule[:50]}...")

        # Update hypothesis confidence based on results
        for exp in experiment_results:
            hypothesis = exp['hypothesis']
            supported = exp['result']['hypothesis_supported']

            # Find matching theory and update confidence
            theories = self.knowledge_base.query(hypothesis)
            for theory_result in theories:
                if theory_result['type'] == 'theory':
                    theory = theory_result['content']
                    if supported:
                        # Increase confidence for supported hypotheses
                        theory.confidence = min(1.0, theory.confidence + 0.1)
                        theory.validation_count += 1
                    else:
                        # Add as counter-example
                        self.knowledge_base.add_counter_example(
                            theory.statement,
                            {
                                'experiment': exp,
                                'reason': 'hypothesis_not_supported',
                                'evidence': exp['result']
                            }
                        )
                    self.knowledge_base.save()

        return learning_outcomes

    def generate_recommendations(self, analysis_results: Dict[str, Any], experiment_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate actionable recommendations for the evolution system

        Args:
            analysis_results: Results from failure analysis
            experiment_results: Results from experiments

        Returns:
            Recommendations for system improvement
        """
        recommendations = {
            'immediate_fixes': [],
            'systemic_changes': [],
            'research_directions': []
        }

        # Immediate fixes from failure analysis
        for fix in analysis_results.get('systemic_fixes', []):
            recommendations['immediate_fixes'].append(fix)

        # Systemic changes from successful experiments
        successful_experiments = [exp for exp in experiment_results if exp['result']['hypothesis_supported']]
        for exp in successful_experiments:
            if exp['result']['effect_size'] > 0.5:  # Large effect
                recommendations['systemic_changes'].append({
                    'change_type': 'parameter_adjustment',
                    'description': f"Implement changes suggested by: {exp['hypothesis']}",
                    'expected_impact': exp['result']['effect_size'],
                    'confidence': exp['result']['confidence']
                })

        # Research directions from failed experiments
        failed_experiments = [exp for exp in experiment_results if not exp['result']['hypothesis_supported']]
        if failed_experiments:
            recommendations['research_directions'].append({
                'direction': 'investigate_alternative_hypotheses',
                'description': f"Test alternative explanations for {len(failed_experiments)} rejected hypotheses",
                'priority': 'medium'
            })

        return recommendations

    def get_meta_scientific_report(self, generation: int) -> Dict[str, Any]:
        """
        Generate a comprehensive meta-scientific report

        Args:
            generation: Current generation

        Returns:
            Complete report on meta-scientific activities
        """
        kb_stats = self.knowledge_base.get_statistics()

        return {
            'generation': generation,
            'experiments_conducted': len(self.experiment_history),
            'knowledge_base_stats': kb_stats,
            'recent_hypotheses': len([h for h in self.experiment_history if h['generation'] == generation]),
            'learning_efficiency': self._calculate_learning_efficiency()
        }

    def _compute_population_stats(self, population: Sequence[Any]) -> Dict[str, Any]:
        """Compute statistics about the current population"""
        if not population:
            return {}

        fitnesses = [g.fitness for g in population]
        ages = [getattr(g, 'age', 0) for g in population]

        return {
            'population_size': len(population),
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'mean_age': float(np.mean(ages)),
            'diversity_score': self._calculate_diversity(population)
        }

    def _calculate_diversity(self, population: Sequence[Any]) -> float:
        """Calculate population diversity based on genome parameters"""
        if len(population) < 2:
            return 0.0

        # Simple diversity based on meta-parameters
        meta_params = []
        for genome in population:
            meta = getattr(genome, 'meta', {})
            meta_params.append([
                meta.get('reward_gain', 1.0),
                meta.get('plastic_lr', 1.0),
                meta.get('reward_bias', 0.0)
            ])

        # Calculate average pairwise distance
        distances = []
        for i in range(len(meta_params)):
            for j in range(i+1, len(meta_params)):
                dist = np.linalg.norm(np.array(meta_params[i]) - np.array(meta_params[j]))
                distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def _store_experiment_in_kb(self, experiment_record: Dict[str, Any]):
        """Store experiment result in knowledge base"""
        # Create a theory if hypothesis was supported
        if experiment_record['result']['hypothesis_supported']:
            theory_data = {
                'statement': experiment_record['hypothesis'],
                'evidence': {
                    'experiment_result': experiment_record['result'],
                    'generation': experiment_record['generation'],
                    'task_info': experiment_record['task_info']
                },
                'confidence': experiment_record['result']['confidence'],
                'domain': experiment_record['task_info'].get('domain', 'evolution')
            }

            try:
                self.knowledge_base.add_theory(theory_data)
            except Exception as e:
                print(f"Failed to store theory in KB: {e}")

    def _calculate_learning_efficiency(self) -> float:
        """Calculate how efficiently the meta-scientist is learning"""
        if not self.experiment_history:
            return 0.0

        # Efficiency based on ratio of supported hypotheses
        supported = sum(1 for exp in self.experiment_history if exp['result']['hypothesis_supported'])
        total = len(self.experiment_history)

        return supported / total if total > 0 else 0.0
