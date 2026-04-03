import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple, Sequence, Callable, TYPE_CHECKING
from evaluation.failure_analyzer import FailureAnalyzer
from knowledge.knowledge_base import KnowledgeBase
from core.genome import EvolvableGenome


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
        # failure['diagnosis'] is the _rank_causes return dict; the raw sub-dicts
        # are nested one level deeper under the 'diagnosis' key.
        plasticity_failures = 0
        for failure in failure_data:
            raw = failure.get('diagnosis', {}).get('diagnosis', {})
            if raw.get('learning', {}).get('severity', 0) > 0.5:
                plasticity_failures += 1

        return plasticity_failures > len(failure_data) * 0.3  # 30% threshold

    def _check_architecture_correlation(self, failure_data: List[Dict[str, Any]]) -> bool:
        """Check if architecture issues correlate with failures"""
        arch_failures = 0
        for failure in failure_data:
            raw = failure.get('diagnosis', {}).get('diagnosis', {})
            if raw.get('architectural', {}).get('severity', 0) > 0.5:
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

    def diversify_patterns(self):
        """Diversify hypothesis generation patterns to avoid overfitting"""
        # Add new hypothesis templates to increase variety
        additional_templates = [
            "Task complexity is too {high/low} for current architectures",
            "Selection pressure needs to be {increased/decreased}",
            "Population diversity is {insufficient/sufficient}",
            "Meta-parameters are not being optimized effectively",
        ]

        # Add to existing templates (avoid duplicates)
        for template in additional_templates:
            if template not in self.hypothesis_templates:
                self.hypothesis_templates.append(template)

        # Reset evidence database to encourage exploration
        self.evidence_db = self.evidence_db[-5:]  # Keep only recent evidence


class ExperimentDesigner:
    """Design and execute controlled experiments"""

    def __init__(self, evaluator: Optional[Callable[[EvolvableGenome], float]] = None):
        self.random_state = np.random.RandomState(42)
        self.evaluator: Optional[Callable[[EvolvableGenome], float]] = evaluator

        self.genome_factory: Optional[Callable[[], EvolvableGenome]] = None

    def design_experiment(self, hypothesis: str, test_design: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create experimental protocol.

        Args:
            hypothesis: The hypothesis statement string.
            test_design: Optional spec from HypothesisEngine._design_test. When
                provided, its ``duration``, ``metric``, and ``parameters``/
                ``ranges`` fields take precedence over the defaults derived from
                the hypothesis string.
        """
        # Prefer structured spec values when available, fall back to derived defaults.
        duration = test_design['duration'] if test_design and 'duration' in test_design else self._estimate_duration(hypothesis)
        metric = test_design['metric'] if test_design and 'metric' in test_design else self._define_metric(hypothesis)
        variables_manipulated = (
            test_design['parameters']
            if test_design and 'parameters' in test_design
            else self._identify_treatments(hypothesis)
        )

        return {
            'hypothesis': hypothesis,
            'control_group': self._create_control(),
            'treatment_group': self._create_treatment(hypothesis, test_design=test_design),
            'variables_frozen': self._identify_controls(hypothesis),
            'variables_manipulated': variables_manipulated,
            'parameter_ranges': test_design.get('ranges') if test_design else None,
            'success_metric': metric,
            'sample_size': self._calculate_sample_size(),
            'duration': duration,
            'test_type': test_design.get('test_type') if test_design else None,
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

    def _create_treatment(self, hypothesis, test_design: Optional[Dict[str, Any]] = None):
        """Create treatment group based on hypothesis.

        When ``test_design`` carries ``parameters`` and ``ranges`` (from
        HypothesisEngine._design_test), the midpoint of each range is used as
        the treatment value instead of hardcoded fallbacks.  A name map bridges
        _design_test parameter names to the treatment-dict keys used by
        _run_group / _create_modified_genome.
        """
        treatment = self._create_control().copy()

        # Map from _design_test parameter names -> treatment dict keys
        _param_name_map = {
            'plastic_lr':    'plasticity',
            'reward_gain':   'reward_gain',
            'learning_rate': 'learning_rate',
        }

        # If a structured spec is available, derive values from its ranges
        if test_design and 'parameters' in test_design and 'ranges' in test_design:
            for param, rng in zip(test_design['parameters'], test_design['ranges']):
                treatment_key = _param_name_map.get(str(param), str(param))
                lo, hi = float(rng[0]), float(rng[1])
                treatment[treatment_key] = (lo + hi) / 2.0  # midpoint of the sweep range
            return treatment

        # Fallback: keyword-match the hypothesis string
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

    def _calculate_sample_size(self, effect_size: float = 0.5, alpha: float = 0.05, power: float = 0.80) -> int:
        """Calculate required sample size per group using a two-sample t-test power analysis.

        Defaults target 80% power for a medium effect size (Cohen's d = 0.5) at
        alpha = 0.05.  Formula: n = 2 * ((z_alpha/2 + z_beta) / d)^2
        """
        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - alpha / 2.0)   # critical value for significance
        z_beta  = norm.ppf(power)                # critical value for power
        n = int(np.ceil(2.0 * ((z_alpha + z_beta) / effect_size) ** 2))
        return max(n, 10)  # enforce a sensible minimum

    def _estimate_duration(self, hypothesis: str = '') -> int:
        """Estimate experiment duration in episodes based on hypothesis type.

        Matches the durations defined in HypothesisEngine._design_test so the
        fallback path (no test_design spec) stays consistent with the spec path.
        """
        h = hypothesis.lower()
        if 'plasticity' in h:
            return 50    # plastic changes manifest quickly
        elif 'architecture' in h:
            return 100   # structural changes need more episodes to stabilise
        else:
            return 25    # general / unknown — short diagnostic run

    def _run_group(self, group_config, frozen_vars):
        """Run evaluations on modified genomes for one experiment group.

        ``frozen_vars`` is used to derive a deterministic seed so that both the
        control and treatment groups face identical starting conditions.
        """
        # Derive a reproducible seed from the frozen variables list so both
        # groups start from the same conditions (fixes the unused frozen_vars gap).
        seed_str = ','.join(str(v) for v in (frozen_vars or []))
        group_seed = int(abs(hash(seed_str)) % (2 ** 31))
        rng = np.random.RandomState(group_seed)

        sample_size = self._calculate_sample_size()

        if self.evaluator is None:
            # Simulation fallback when no real evaluator is available.
            results = []
            for _ in range(sample_size):
                baseline_performance = rng.normal(50, 10)
                plasticity = group_config.get('plasticity', 0.1)
                reward_gain = group_config.get('reward_gain', 1.0)
                architecture = group_config.get('architecture', 'standard')
                if plasticity > 0.3:
                    performance = baseline_performance * 1.2 * reward_gain
                elif plasticity < 0.05:
                    performance = baseline_performance * 0.8 * reward_gain
                else:
                    performance = baseline_performance * reward_gain
                # Architecture effect: high_capacity adds ~15% boost
                if architecture == 'high_capacity':
                    performance *= 1.15
                results.append(performance)
            return np.array(results)

        # Real evaluation mode — cap at 10 to keep wall-clock time reasonable.
        real_sample_size = min(10, sample_size)
        results = []

        for _ in range(real_sample_size):
            modified_genome = self._create_modified_genome(group_config)
            if modified_genome is None:
                continue

            try:
                fitness = self.evaluator(modified_genome)
                results.append(float(fitness))
            except Exception as e:
                print(f"Evaluation failed: {e}")
                continue

        return np.array(results) if results else np.array([0.0])

    def _analyze_results(self, control_results: np.ndarray, treatment_results: np.ndarray) -> Tuple[float, float]:
        """Perform statistical analysis (two-sample t-test + Cohen's d)."""
        # Guard against degenerate arrays that would make ttest_ind return NaN.
        # This happens when a group has only one sample or all identical values.
        if len(control_results) < 2 or len(treatment_results) < 2:
            return 1.0, 0.0  # not enough data: not significant, no effect

        if np.std(control_results) == 0 and np.std(treatment_results) == 0:
            # Both groups are constant — no variance to test against.
            return 1.0, 0.0

        # Two-sample t-test
        _t_stat, p_value = stats.ttest_ind(control_results, treatment_results)

        # Cohen's d effect size
        mean_diff = np.mean(treatment_results) - np.mean(control_results)
        pooled_std = np.sqrt((np.var(control_results) + np.var(treatment_results)) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # ttest_ind can still return NaN when one group is constant but the
        # other is not — fall back to non-significant in that case.
        p_val = float(np.asarray(p_value).item())
        if np.isnan(p_val):
            p_val = 1.0

        return p_val, float(np.asarray(effect_size).item())

    def _create_modified_genome(self, group_config) -> Optional[EvolvableGenome]:
        """Create a modified genome based on group configuration."""
        if self.genome_factory is None:
            return None

        base_genome = self.genome_factory()

        # Plasticity parameters — read each field independently from group_config
        # so that reward_gain set by _create_treatment is honoured directly
        # rather than being re-derived from plasticity * 2.0.
        if 'plasticity' in group_config and hasattr(base_genome, 'meta'):
            base_genome.meta['plastic_lr'] = group_config['plasticity']

        if 'reward_gain' in group_config and hasattr(base_genome, 'meta'):
            base_genome.meta['reward_gain'] = group_config['reward_gain']

        # Architecture modification
        if group_config.get('architecture') == 'high_capacity':
            if hasattr(base_genome, 'genes') and base_genome.genes:
                for gene in base_genome.genes:
                    if hasattr(gene, 'output_dim'):
                        gene.output_dim = min(gene.output_dim * 2, 256)

        return base_genome


class MetaScientist:
    """Orchestrates the complete meta-scientific pipeline"""

    def __init__(self, knowledge_base_path: str = "meta_scientist_kb.db"):
        self.failure_analyzer = FailureAnalyzer()
        self.hypothesis_engine = HypothesisEngine()
        self.experiment_designer = ExperimentDesigner()
        self.knowledge_base = KnowledgeBase(knowledge_base_path)
        self.experiment_history = []
        
        # Bidirectional mutation control tracking
        self.intervention_history = []
        self.last_mutation_direction = None  # 'increase', 'decrease', or None
        self.mutation_hysteresis_counter = 0  # Prevent rapid oscillation
        self.chaos_risk_history = []
        
        # NeuroGenesis: Track additional evolutionary health metrics
        self.adaptability_history = []  # Track adaptability scores over generations
        self.saturation_history = []    # Track neural saturation metrics
        self.species_balance_history = []  # Track prey/predator species balance



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
                # Design experiment, passing the structured test spec so that
                # duration, metric, and parameter ranges are used directly
                # instead of being re-derived from the hypothesis string.
                experiment_design = self.experiment_designer.design_experiment(
                    hypothesis['statement'],
                    test_design=hypothesis.get('test_design')
                )

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
                        old_confidence = theory.confidence
                        theory.confidence = min(1.0, theory.confidence + 0.1)
                        theory.validation_count += 1
                        learning_outcomes['theory_updates'].append({
                            'theory': theory.statement,
                            'action': 'confidence_increased',
                            'old_confidence': old_confidence,
                            'new_confidence': theory.confidence,
                            'validation_count': theory.validation_count,
                            'effect_size': exp['result']['effect_size']
                        })
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
                        learning_outcomes['theory_updates'].append({
                            'theory': theory.statement,
                            'action': 'counter_example_added',
                            'confidence': theory.confidence,
                            'effect_size': exp['result']['effect_size']
                        })
                    self.knowledge_base.save()

        # Derive strategy recommendations from experiment patterns
        # Group results by hypothesis category (plasticity / architecture / learning_rate / other)
        category_results: Dict[str, List[Dict[str, Any]]] = {}
        for exp in experiment_results:
            h = exp['hypothesis'].lower()
            if 'plasticity' in h:
                cat = 'plasticity'
            elif 'architecture' in h:
                cat = 'architecture'
            elif 'learning rate' in h:
                cat = 'learning_rate'
            else:
                cat = 'other'
            category_results.setdefault(cat, []).append(exp)

        for cat, exps in category_results.items():
            supported_exps = [e for e in exps if e['result']['hypothesis_supported']]
            support_rate = len(supported_exps) / len(exps) if exps else 0.0
            avg_effect = float(np.mean([e['result']['effect_size'] for e in supported_exps])) if supported_exps else 0.0

            if support_rate >= 0.5 and avg_effect >= 0.3:
                priority = 'high' if avg_effect >= 0.7 else 'medium'
                learning_outcomes['strategy_recommendations'].append({
                    'category': cat,
                    'recommendation': f"Apply {cat.replace('_', ' ')} modifications — "
                                      f"{len(supported_exps)}/{len(exps)} hypotheses supported "
                                      f"(avg effect size: {avg_effect:.2f})",
                    'support_rate': support_rate,
                    'avg_effect_size': avg_effect,
                    'priority': priority,
                    'supporting_hypotheses': [e['hypothesis'] for e in supported_exps]
                })

            elif support_rate < 0.2 and len(exps) >= 2:
                # Consistently failing category — recommend deprioritising it
                learning_outcomes['strategy_recommendations'].append({
                    'category': cat,
                    'recommendation': f"Deprioritise {cat.replace('_', ' ')} exploration — "
                                      f"only {len(supported_exps)}/{len(exps)} hypotheses supported",
                    'support_rate': support_rate,
                    'avg_effect_size': avg_effect,
                    'priority': 'low',
                    'supporting_hypotheses': []
                })

        # Sort recommendations: high priority first
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        learning_outcomes['strategy_recommendations'].sort(
            key=lambda r: priority_order.get(r['priority'], 3)
        )

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
        INCLUDING bidirectional mutation control metrics

        Args:
            generation: Current generation

        Returns:
            Complete report on meta-scientific activities
        """
        kb_stats = self.knowledge_base.get_statistics()

        # Calculate recent chaos risk trend
        recent_chaos_risk = 0.0
        if self.chaos_risk_history:
            recent_entries = [e for e in self.chaos_risk_history 
                            if e.get('generation', 0) > generation - 10]
            if recent_entries:
                recent_chaos_risk = np.mean([e['risk_score'] for e in recent_entries])

        # Count mutation direction changes
        direction_changes = 0
        if len(self.intervention_history) > 1:
            for i in range(1, len(self.intervention_history)):
                prev = self.intervention_history[i-1].get('mutation_direction')
                curr = self.intervention_history[i].get('mutation_direction')
                if prev is not None and curr is not None and prev != curr:
                    direction_changes += 1

        return {
            'generation': generation,
            'experiments_conducted': len(self.experiment_history),
            'knowledge_base_stats': kb_stats,
            'recent_hypotheses': len([h for h in self.experiment_history if h['generation'] == generation]),
            'learning_efficiency': self._calculate_learning_efficiency(),
            # NEW: Bidirectional control metrics
            'chaos_risk_current': recent_chaos_risk,
            'chaos_risk_trend': 'increasing' if len(self.chaos_risk_history) > 5 and 
                                  self.chaos_risk_history[-1]['risk_score'] > 
                                  np.mean([e['risk_score'] for e in self.chaos_risk_history[-5:-1]])
                              else 'stable',
            'mutation_control_direction': self.last_mutation_direction,
            'mutation_direction_changes': direction_changes,
            'intervention_count': len(self.intervention_history)
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

    def _calculate_chaos_risk(self, population: Sequence[Any], 
                             evolution_engine, 
                             species_stats: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate evolutionary chaos risk score (0-1) based on:
        - Species count (too many = fragmentation)
        - Architecture mutation rate (too high = instability)
        - Mutator effectiveness (high effectiveness + high mutation = chaos)
        - Recent fitness variance (turbulence indicator)
        
        Returns:
            float: Chaos risk score 0-1 (0 = stable, 1 = high turbulence risk)
        """
        risk_factors = []
        
        # Factor 1: Species count risk (target: 3-5 species, risk if > 7)
        if species_stats:
            num_species = species_stats.get('num_species', 0)
            if num_species > 7:
                species_risk = min((num_species - 7) / 5.0, 1.0)  # Cap at 1.0
                risk_factors.append(species_risk)
            elif num_species < 2:
                risk_factors.append(0.3)  # Low diversity also risky
        
        # Factor 2: Architecture mutation rate risk
        if hasattr(evolution_engine, 'architecture_mutation_rate'):
            arch_rate = evolution_engine.architecture_mutation_rate
            if arch_rate > 0.08:  # High threshold
                arch_risk = min((arch_rate - 0.08) / 0.12, 1.0)
                risk_factors.append(arch_risk)
        
        # Factor 3: Combined mutation effectiveness risk
        # High effectiveness + high mutation rate = chaos potential
        if hasattr(evolution_engine, 'architecture_mutation_rate'):
            arch_rate = evolution_engine.architecture_mutation_rate
            # Assume mutator effectiveness around 0.7 based on task description
            mutator_effectiveness = 0.725  # From task description
            combined_risk = arch_rate * mutator_effectiveness * 2.0  # Scale factor
            risk_factors.append(min(combined_risk, 1.0))
        
        # Factor 4: Population fitness turbulence (high variance = instability)
        if population and len(population) > 1:
            fitnesses = [getattr(g, 'fitness', 0.0) for g in population]
            if len(fitnesses) > 1 and np.mean(fitnesses) > 0:
                cv = np.std(fitnesses) / (np.mean(fitnesses) + 1e-10)  # Coefficient of variation
                if cv > 0.5:  # High variance
                    turbulence_risk = min((cv - 0.5) / 1.0, 1.0)
                    risk_factors.append(turbulence_risk)
        
        # Calculate overall chaos risk (weighted average of factors)
        if not risk_factors:
            return 0.0
        
        # Use max of risks for conservative estimate, but blend with mean
        max_risk = max(risk_factors)
        mean_risk = np.mean(risk_factors)
        
        # Weight max risk more heavily (conservative approach)
        chaos_risk = 0.6 * max_risk + 0.4 * mean_risk
        
        # Store for trend analysis
        self.chaos_risk_history.append({
            'generation': getattr(self, '_current_generation', 0),
            'risk_score': float(chaos_risk),
            'factors': risk_factors
        })
        
        # Keep only recent history
        if len(self.chaos_risk_history) > 50:
            self.chaos_risk_history = self.chaos_risk_history[-50:]
        
        return float(min(chaos_risk, 1.0))
    
    def _should_allow_mutation_change(self, proposed_direction: str, 
                                     current_chaos_risk: float) -> bool:
        """
        Hysteresis check to prevent rapid oscillation between increase/decrease
        
        Args:
            proposed_direction: 'increase' or 'decrease'
            current_chaos_risk: Current chaos risk score
            
        Returns:
            bool: True if change should be allowed
        """
        # Always allow if no previous direction
        if self.last_mutation_direction is None:
            return True
        
        # If proposing same direction, allow with reduced hysteresis
        if proposed_direction == self.last_mutation_direction:
            self.mutation_hysteresis_counter = max(0, self.mutation_hysteresis_counter - 1)
            return True
        
        # Direction change - check hysteresis
        # Require higher threshold to flip direction
        if proposed_direction == 'decrease' and current_chaos_risk < 0.6:
            # Not enough chaos risk to justify decrease
            return False
        
        if proposed_direction == 'increase' and current_chaos_risk > 0.4:
            # Too much chaos risk to justify increase
            return False
        
        # Check hysteresis counter (need 3 generations of consistent signal)
        self.mutation_hysteresis_counter += 1
        if self.mutation_hysteresis_counter < 3:
            return False
        
        # Reset counter on direction change
        self.mutation_hysteresis_counter = 0
        return True


    def apply_recommendations_to_evolution(self, recommendations: Dict[str, Any],
                                         evolution_engine, curriculum_controller,
                                         generation: int) -> Dict[str, Any]:
        """
        Apply meta-scientific recommendations to actively modify the evolution system

        Args:
            recommendations: Recommendations from generate_recommendations
            evolution_engine: The evolution engine to modify
            curriculum_controller: The curriculum controller to modify
            generation: Current generation

        Returns:
            Summary of changes applied
        """
        applied_changes = {
            'parameter_adjustments': [],
            'curriculum_changes': [],
            'architecture_injections': [],
            'mutation_rate_changes': []
        }

        print("Meta-Scientist: Applying recommendations to evolution system...")

        # Apply systemic changes from successful experiments
        for change in recommendations.get('systemic_changes', []):
            if change['change_type'] == 'parameter_adjustment':
                self._apply_parameter_adjustment(change, evolution_engine, applied_changes)

        # Apply immediate fixes from failure analysis
        for fix in recommendations.get('immediate_fixes', []):
            self._apply_immediate_fix(fix, evolution_engine, curriculum_controller, applied_changes)

        # Spawn experiment populations based on findings
        experiment_populations = self._spawn_experiment_populations(recommendations, generation)
        if experiment_populations:
            applied_changes['experiment_populations'] = experiment_populations

        print(f"Meta-Scientist: Applied {len(applied_changes['parameter_adjustments'])} parameter adjustments, "
              f"{len(applied_changes['curriculum_changes'])} curriculum changes")

        return applied_changes

    def _apply_parameter_adjustment(self, change: Dict[str, Any], evolution_engine, applied_changes: Dict[str, Any]):
        """Apply parameter adjustment based on experimental findings"""
        description = change['description'].lower()

        # Modify mutation probabilities based on findings
        if 'plasticity' in description:
            if 'too low' in description or 'increase' in description:
                # Increase plasticity-related mutation rates
                if hasattr(evolution_engine, 'architecture_mutation_rate'):
                    old_rate = evolution_engine.architecture_mutation_rate
                    evolution_engine.architecture_mutation_rate = min(old_rate * 1.5, 0.1)
                    applied_changes['mutation_rate_changes'].append({
                        'parameter': 'architecture_mutation_rate',
                        'old_value': old_rate,
                        'new_value': evolution_engine.architecture_mutation_rate,
                        'reason': change['description']
                    })
            elif 'too high' in description or 'decrease' in description:
                # Decrease plasticity-related mutation rates
                if hasattr(evolution_engine, 'architecture_mutation_rate'):
                    old_rate = evolution_engine.architecture_mutation_rate
                    evolution_engine.architecture_mutation_rate = max(old_rate * 0.7, 0.01)
                    applied_changes['mutation_rate_changes'].append({
                        'parameter': 'architecture_mutation_rate',
                        'old_value': old_rate,
                        'new_value': evolution_engine.architecture_mutation_rate,
                        'reason': change['description']
                    })

        elif 'architecture' in description and 'capacity' in description:
            # Increase architecture complexity mutations
            if hasattr(evolution_engine, 'architecture_mutation_rate'):
                old_rate = evolution_engine.architecture_mutation_rate
                evolution_engine.architecture_mutation_rate = min(old_rate * 2.0, 0.15)
                applied_changes['mutation_rate_changes'].append({
                    'parameter': 'architecture_mutation_rate',
                    'old_value': old_rate,
                    'new_value': evolution_engine.architecture_mutation_rate,
                    'reason': change['description']
                })

        applied_changes['parameter_adjustments'].append(change)

    def _apply_immediate_fix(self, fix: Dict[str, Any], evolution_engine, curriculum_controller, applied_changes: Dict[str, Any]):
        """Apply immediate fix from failure analysis"""
        fix_type = fix.get('type', '')

        if fix_type == 'plasticity_adjustment':
            # Adjust plasticity parameters
            if hasattr(evolution_engine, 'mutation_strength'):
                old_strength = evolution_engine.mutation_strength
                if fix.get('direction') == 'increase':
                    evolution_engine.mutation_strength = min(old_strength * 1.2, 1.0)
                elif fix.get('direction') == 'decrease':
                    evolution_engine.mutation_strength = max(old_strength * 0.8, 0.01)
                applied_changes['parameter_adjustments'].append({
                    'parameter': 'mutation_strength',
                    'old_value': old_strength,
                    'new_value': evolution_engine.mutation_strength,
                    'reason': f"Immediate fix: {fix.get('description', 'plasticity adjustment')}"
                })

        elif fix_type == 'curriculum_adjustment':
            # Adjust curriculum difficulty
            if hasattr(curriculum_controller, 'reset_to_stage'):
                target_stage = fix.get('target_stage')
                if target_stage:
                    curriculum_controller.reset_to_stage(target_stage)
                    applied_changes['curriculum_changes'].append({
                        'change': f"Reset to stage {target_stage}",
                        'reason': fix.get('description', 'curriculum adjustment')
                    })

    def _spawn_experiment_populations(self, recommendations: Dict[str, Any], generation: int) -> List[Dict[str, Any]]:
        """Spawn specialized experiment populations based on findings"""
        experiment_populations = []

        # If we have research directions, spawn populations to investigate
        research_directions = recommendations.get('research_directions', [])
        if research_directions and generation % 50 == 0:  # Only spawn every 50 generations
            for direction in research_directions[:2]:  # Limit to 2 populations
                if 'alternative_hypotheses' in direction.get('direction', ''):
                    # Spawn a population with modified parameters to test alternatives
                    experiment_pop = {
                        'name': f'experiment_pop_gen_{generation}',
                        'size': 20,  # Small experimental population
                        'specialization': direction.get('direction'),
                        'modified_parameters': {
                            'mutation_rate': 0.01,  # Higher mutation for exploration
                            'architecture_mutation_rate': 0.05,
                            'novelty_weight': 0.5  # Emphasize novelty
                        },
                        'purpose': direction.get('description')
                    }
                    experiment_populations.append(experiment_pop)

        if experiment_populations:
            print(f"Meta-Scientist: Spawning experiment population (count={len(experiment_populations)})")

        return experiment_populations

    def detect_failure_patterns(self, analysis_results: Dict[str, Any], 
                               experiment_results: List[Dict[str, Any]],
                               population: Optional[Sequence[Any]] = None,
                               evolution_engine = None,
                               species_stats: Optional[Dict[str, Any]] = None,
                               generation: int = 0,
                               neural_health: Optional[Dict[str, Any]] = None,
                               prey_species_stats: Optional[Dict[str, Any]] = None,
                               predator_species_stats: Optional[Dict[str, Any]] = None,
                               adaptability_stats: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Detect systemic failure patterns that require intervention
        INCLUDING evolutionary turbulence from excessive diversity
        PLUS NeuroGenesis: adaptability collapse, saturation rise, species imbalance

        Args:
            analysis_results: Results from failure analysis
            experiment_results: Results from experiments
            population: Current population (for chaos risk calculation)
            evolution_engine: Evolution engine (for mutation rates)
            species_stats: Species statistics
            generation: Current generation number
            neural_health: Neural health metrics (dead_layers, saturated_layers)
            prey_species_stats: Prey species statistics
            predator_species_stats: Predator species statistics
            adaptability_stats: Adaptability metrics (avg_adaptability_score, etc.)

        Returns:
            List of detected failure patterns requiring action
        """
        patterns = []

        # Check for stagnation (no improvement over many generations)
        recent_experiments = [exp for exp in experiment_results if exp.get('generation', 0) > 10]
        if len(recent_experiments) > 5:
            supported_recent = sum(1 for exp in recent_experiments if exp['result']['hypothesis_supported'])
            support_rate = supported_recent / len(recent_experiments)

            if support_rate < 0.2:  # Less than 20% of hypotheses supported recently
                patterns.append({
                    'type': 'stagnation',
                    'severity': 'high',
                    'description': f"Low hypothesis support rate ({support_rate:.2f}) indicates stagnation",
                    'recommended_action': 'increase_exploration'
                })

        # Check for overfitting to current hypotheses
        if len(self.experiment_history) > 20:
            recent_hypotheses = [exp['hypothesis'] for exp in self.experiment_history[-20:]]
            unique_recent = len(set(recent_hypotheses))
            if unique_recent < 5:  # Very few unique hypotheses recently
                patterns.append({
                    'type': 'overfitting',
                    'severity': 'medium',
                    'description': f"Only {unique_recent} unique hypotheses in last 20 experiments",
                    'recommended_action': 'diversify_hypotheses'
                })

        # If experiments explicitly support higher-capacity architectures, treat
        # that as an actionable failure pattern instead of only logging it.
        capacity_experiments = [
            exp for exp in experiment_results
            if exp.get('result', {}).get('hypothesis_supported')
            and 'architecture lacks capacity' in str(exp.get('hypothesis', '')).lower()
        ]
        if capacity_experiments:
            mean_effect = float(np.mean([
                exp.get('result', {}).get('effect_size', 0.0) for exp in capacity_experiments
            ]))
            patterns.append({
                'type': 'architecture_capacity',
                'severity': 'high' if mean_effect > 0.05 else 'medium',
                'description': f"Supported capacity experiments indicate under-sized architectures (effect={mean_effect:.3f})",
                'recommended_action': 'increase_capacity',
                'effect_size': mean_effect
            })

        # Check for population-level failures
        total_failures = analysis_results.get('total_failures', 0)
        population_size = analysis_results.get('population_stats', {}).get('population_size', 1)

        if total_failures / population_size > 0.8:  # 80% of population failing
            patterns.append({
                'type': 'systemic_failure',
                'severity': 'critical',
                'description': f"{total_failures}/{population_size} genomes failing - systemic issue",
                'recommended_action': 'curriculum_reset'
            })
        
        # BIDIRECTIONAL CONTROL: Check for evolutionary turbulence (too high diversity)
        if population is not None and evolution_engine is not None:
            chaos_risk = self._calculate_chaos_risk(population, evolution_engine, species_stats)
            
            # High chaos risk indicates evolutionary turbulence
            if chaos_risk > 0.7:  # Critical turbulence threshold
                num_species = species_stats.get('num_species', 'unknown') if species_stats else 'unknown'
                patterns.append({
                    'type': 'evolutionary_turbulence',
                    'severity': 'critical',
                    'description': f"Evolutionary chaos risk {chaos_risk:.2f} - too many species ({num_species}) with high mutation",
                    'recommended_action': 'reduce_mutation',
                    'chaos_risk': chaos_risk
                })
            elif chaos_risk > 0.5:  # Warning threshold
                patterns.append({
                    'type': 'evolutionary_turbulence',
                    'severity': 'high',
                    'description': f"Elevated chaos risk {chaos_risk:.2f} - consider reducing mutation rates",
                    'recommended_action': 'reduce_mutation',
                    'chaos_risk': chaos_risk
                })

        # NEUROGENESIS: Check for adaptability collapse
        if adaptability_stats is not None:
            avg_adaptability = adaptability_stats.get('avg_adaptability_score', 0.0)
            self.adaptability_history.append({
                'generation': generation,
                'avg_adaptability': avg_adaptability
            })
            
            # Keep only recent history
            if len(self.adaptability_history) > 50:
                self.adaptability_history = self.adaptability_history[-50:]
            
            # Check for adaptability collapse (low adaptability sustained over time)
            if len(self.adaptability_history) >= 10:
                recent_adaptability = [h['avg_adaptability'] for h in self.adaptability_history[-10:]]
                mean_recent_adaptability = np.mean(recent_adaptability)
                
                if mean_recent_adaptability < 0.1:  # Critical adaptability collapse
                    patterns.append({
                        'type': 'adaptability_collapse',
                        'severity': 'critical',
                        'description': f"Adaptability collapsed to {mean_recent_adaptability:.3f} over last 10 generations",
                        'recommended_action': 'boost_plasticity_pressure',
                        'adaptability_score': mean_recent_adaptability
                    })
                elif mean_recent_adaptability < 0.3:  # Warning level
                    patterns.append({
                        'type': 'adaptability_collapse',
                        'severity': 'high',
                        'description': f"Low adaptability {mean_recent_adaptability:.3f} - learning may be decorative",
                        'recommended_action': 'boost_plasticity_pressure',
                        'adaptability_score': mean_recent_adaptability
                    })

        # NEUROGENESIS: Check for saturation rise (neural health degradation)
        if neural_health is not None:
            dead_layers = neural_health.get('dead_layers', 0)
            saturated_layers = neural_health.get('saturated_layers', 0)
            genomes_analyzed = neural_health.get('genomes_analyzed', 1)
            
            # Calculate saturation ratio
            total_problematic = dead_layers + saturated_layers
            saturation_ratio = total_problematic / max(genomes_analyzed, 1)
            
            self.saturation_history.append({
                'generation': generation,
                'saturation_ratio': saturation_ratio,
                'dead_layers': dead_layers,
                'saturated_layers': saturated_layers
            })
            
            # Keep only recent history
            if len(self.saturation_history) > 50:
                self.saturation_history = self.saturation_history[-50:]
            
            # Check for saturation rise
            if len(self.saturation_history) >= 5:
                recent_saturation = [h['saturation_ratio'] for h in self.saturation_history[-5:]]
                trend = np.polyfit(range(len(recent_saturation)), recent_saturation, 1)[0]
                
                if saturation_ratio > 0.5:  # Critical: more than 50% of genomes have issues
                    patterns.append({
                        'type': 'saturation_rise',
                        'severity': 'critical',
                        'description': f"Neural saturation critical: {dead_layers} dead, {saturated_layers} saturated layers",
                        'recommended_action': 'prune_architecture',
                        'saturation_ratio': saturation_ratio
                    })
                elif trend > 0.05:  # Rising trend
                    patterns.append({
                        'type': 'saturation_rise',
                        'severity': 'high',
                        'description': f"Saturation rising (trend: {trend:.3f}), {total_problematic} problematic layers",
                        'recommended_action': 'prune_architecture',
                        'saturation_ratio': saturation_ratio
                    })

        # NEUROGENESIS: Check for species imbalance (prey vs predator)
        if prey_species_stats is not None and predator_species_stats is not None:
            prey_species = prey_species_stats.get('num_species', 0)
            predator_species = predator_species_stats.get('num_species', 0)
            
            # Calculate imbalance ratio
            if predator_species > 0:
                imbalance_ratio = prey_species / predator_species
            else:
                imbalance_ratio = float('inf') if prey_species > 0 else 1.0
            
            self.species_balance_history.append({
                'generation': generation,
                'prey_species': prey_species,
                'predator_species': predator_species,
                'imbalance_ratio': imbalance_ratio
            })
            
            # Keep only recent history
            if len(self.species_balance_history) > 50:
                self.species_balance_history = self.species_balance_history[-50:]
            
            # Check for species imbalance
            if imbalance_ratio > 3.0:  # Too many prey species
                patterns.append({
                    'type': 'species_imbalance',
                    'severity': 'high',
                    'description': f"Species imbalance: {prey_species} prey vs {predator_species} predator species",
                    'recommended_action': 'boost_predator_diversity',
                    'imbalance_ratio': imbalance_ratio,
                    'underrepresented': 'predator'
                })
            elif imbalance_ratio < 0.33:  # Too many predator species
                patterns.append({
                    'type': 'species_imbalance',
                    'severity': 'high',
                    'description': f"Species imbalance: {prey_species} prey vs {predator_species} predator species",
                    'recommended_action': 'boost_prey_diversity',
                    'imbalance_ratio': imbalance_ratio,
                    'underrepresented': 'prey'
                })

        return patterns



    def intervene_in_evolution(self, failure_patterns: List[Dict[str, Any]],
                             evolution_engine, curriculum_controller,
                             generation: int,
                             prey_engine = None,
                             predator_engine = None) -> Dict[str, Any]:
        """
        Execute active interventions based on detected failure patterns
        BIDIRECTIONAL: Can both INCREASE mutation (stagnation) and DECREASE (turbulence)
        NEUROGENESIS: Now handles adaptability collapse, saturation rise, species imbalance

        Args:
            failure_patterns: Patterns requiring intervention
            evolution_engine: Evolution engine to modify (legacy, use prey_engine/predator_engine)
            curriculum_controller: Curriculum controller to modify
            generation: Current generation
            prey_engine: Prey evolution engine for species-specific interventions
            predator_engine: Predator evolution engine for species-specific interventions

        Returns:
            Summary of interventions applied
        """
        interventions = {
            'interventions_applied': [],
            'system_resets': [],
            'parameter_emergencies': [],
            'mutation_reductions': [],  # Track mutation decreases
            'plasticity_boosts': [],    # NEUROGENESIS: Track plasticity interventions
            'architecture_prunes': [],  # NEUROGENESIS: Track architecture pruning
            'species_rebalances': []    # NEUROGENESIS: Track species rebalancing
        }

        print("Meta-Scientist: Executing active interventions...")

        for pattern in failure_patterns:
            pattern_type = pattern['type']
            severity = pattern['severity']

            if pattern_type == 'stagnation' and severity == 'high':
                # Check if we should allow increase (hysteresis)
                chaos_risk = pattern.get('chaos_risk', 0.0)
                if self._should_allow_mutation_change('increase', chaos_risk):
                    # Emergency exploration boost
                    if hasattr(evolution_engine, 'mutation_rate'):
                        old_rate = evolution_engine.mutation_rate
                        evolution_engine.mutation_rate = min(0.3, old_rate * 1.2)
                        interventions['parameter_emergencies'].append({
                            'parameter': 'mutation_rate',
                            'old_value': old_rate,
                            'new_value': evolution_engine.mutation_rate,
                            'reason': 'Stagnation emergency - boosting exploration'
                        })
                        self.last_mutation_direction = 'increase'

                    if hasattr(evolution_engine, 'architecture_mutation_rate'):
                        old_arch_rate = evolution_engine.architecture_mutation_rate
                        evolution_engine.architecture_mutation_rate = min(old_arch_rate * 2.0, 0.2)
                        interventions['parameter_emergencies'].append({
                            'parameter': 'architecture_mutation_rate',
                            'old_value': old_arch_rate,
                            'new_value': evolution_engine.architecture_mutation_rate,
                            'reason': 'Stagnation emergency - boosting architecture exploration'
                        })
                        self.last_mutation_direction = 'increase'
                else:
                    print(f"Meta-Scientist: Blocked mutation increase due to hysteresis (chaos risk: {chaos_risk:.2f})")

            elif pattern_type == 'evolutionary_turbulence':
                # BIDIRECTIONAL CONTROL: Reduce mutation to stabilize
                chaos_risk = pattern.get('chaos_risk', 0.5)
                
                if self._should_allow_mutation_change('decrease', chaos_risk):
                    reduction_factor = 0.5 if severity == 'critical' else 0.7
                    
                    if hasattr(evolution_engine, 'mutation_rate'):
                        old_rate = evolution_engine.mutation_rate
                        new_rate = max(old_rate * reduction_factor, 0.1)
                        evolution_engine.mutation_rate = new_rate
                        interventions['mutation_reductions'].append({
                            'parameter': 'mutation_rate',
                            'old_value': old_rate,
                            'new_value': new_rate,
                            'reason': f'Evolutionary turbulence (chaos risk: {chaos_risk:.2f}) - reducing mutation',
                            'severity': severity
                        })
                        self.last_mutation_direction = 'decrease'

                    if hasattr(evolution_engine, 'architecture_mutation_rate'):
                        old_arch_rate = evolution_engine.architecture_mutation_rate
                        # More aggressive reduction for architecture mutations
                        arch_reduction = 0.4 if severity == 'critical' else 0.6
                        new_arch_rate = max(old_arch_rate * arch_reduction, 0.01)
                        evolution_engine.architecture_mutation_rate = new_arch_rate
                        interventions['mutation_reductions'].append({
                            'parameter': 'architecture_mutation_rate',
                            'old_value': old_arch_rate,
                            'new_value': new_arch_rate,
                            'reason': f'Evolutionary turbulence (chaos risk: {chaos_risk:.2f}) - reducing architecture mutation',
                            'severity': severity
                        })
                        self.last_mutation_direction = 'decrease'
                    
                    print(f"Meta-Scientist: REDUCED mutation rates due to turbulence (chaos risk: {chaos_risk:.2f}, severity: {severity})")
                else:
                    print(f"Meta-Scientist: Blocked mutation decrease due to hysteresis (chaos risk: {chaos_risk:.2f})")

            elif pattern_type == 'systemic_failure' and severity == 'critical':
                # Curriculum reset for systemic failures
                if hasattr(curriculum_controller, 'reset_to_beginning'):
                    curriculum_controller.reset_to_beginning()
                    interventions['system_resets'].append({
                        'reset_type': 'curriculum_reset',
                        'reason': 'Critical systemic failure - resetting curriculum'
                    })

            elif pattern_type == 'overfitting':
                # Diversify hypothesis generation
                if hasattr(self.hypothesis_engine, 'diversify_patterns'):
                    self.hypothesis_engine.diversify_patterns()
                    interventions['interventions_applied'].append({
                        'intervention': 'hypothesis_diversification',
                        'reason': 'Overfitting detected - diversifying hypothesis generation'
                    })

            # NEUROGENESIS: Handle adaptability collapse
            elif pattern_type == 'adaptability_collapse':
                adaptability_score = pattern.get('adaptability_score', 0.0)
                print(f"Meta-Scientist: ADAPTABILITY COLLAPSE detected (score: {adaptability_score:.3f})")
                
                # Boost plasticity pressure by increasing learning rule mutation rates
                target_engines = []
                if prey_engine is not None:
                    target_engines.append(('prey', prey_engine))
                if predator_engine is not None:
                    target_engines.append(('predator', predator_engine))
                if not target_engines and evolution_engine is not None:
                    target_engines.append(('default', evolution_engine))
                
                for name, engine in target_engines:
                    # Increase mutation strength for learning rule parameters
                    if hasattr(engine, 'mutation_strength'):
                        old_strength = engine.mutation_strength
                        new_strength = min(old_strength * 2.0, 0.5)  # Cap at 0.5
                        engine.mutation_strength = new_strength
                        interventions['plasticity_boosts'].append({
                            'species': name,
                            'parameter': 'mutation_strength',
                            'old_value': old_strength,
                            'new_value': new_strength,
                            'reason': f'Adaptability collapse (score: {adaptability_score:.3f}) - boosting learning rule mutations'
                        })
                    
                    # Increase architecture mutation to discover better plasticity architectures
                    if hasattr(engine, 'architecture_mutation_rate'):
                        old_arch_rate = engine.architecture_mutation_rate
                        new_arch_rate = min(old_arch_rate * 1.5, 0.15)
                        engine.architecture_mutation_rate = new_arch_rate
                        interventions['plasticity_boosts'].append({
                            'species': name,
                            'parameter': 'architecture_mutation_rate',
                            'old_value': old_arch_rate,
                            'new_value': new_arch_rate,
                            'reason': f'Adaptability collapse - seeking plasticity-friendly architectures'
                        })
                
                # Add hypothesis about plasticity
                if hasattr(self.hypothesis_engine, 'hypothesis_templates'):
                    plasticity_hypothesis = "Plasticity is too low - need to evolve better learning rules"
                    if plasticity_hypothesis not in self.hypothesis_engine.hypothesis_templates:
                        self.hypothesis_engine.hypothesis_templates.append(plasticity_hypothesis)

            elif pattern_type == 'architecture_capacity':
                effect_size = pattern.get('effect_size', 0.0)
                print(f"Meta-Scientist: ARCHITECTURE CAPACITY issue detected (effect: {effect_size:.3f})")

                target_engines = []
                if prey_engine is not None:
                    target_engines.append(('prey', prey_engine))
                if predator_engine is not None:
                    target_engines.append(('predator', predator_engine))
                if not target_engines and evolution_engine is not None:
                    target_engines.append(('default', evolution_engine))

                for name, engine in target_engines:
                    if hasattr(engine, 'architecture_mutation_rate'):
                        old_arch_rate = engine.architecture_mutation_rate
                        new_arch_rate = min(old_arch_rate * 1.35, 0.14)
                        engine.architecture_mutation_rate = new_arch_rate
                        interventions['interventions_applied'].append({
                            'species': name,
                            'parameter': 'architecture_mutation_rate',
                            'old_value': old_arch_rate,
                            'new_value': new_arch_rate,
                            'reason': f'Capacity experiment supported (effect: {effect_size:.3f})'
                        })

                    if hasattr(engine, 'mutation_strength'):
                        old_strength = engine.mutation_strength
                        new_strength = min(old_strength * 1.15, 0.45)
                        engine.mutation_strength = new_strength
                        interventions['interventions_applied'].append({
                            'species': name,
                            'parameter': 'mutation_strength',
                            'old_value': old_strength,
                            'new_value': new_strength,
                            'reason': 'Capacity issue - favor larger functional architectural changes'
                        })

            # NEUROGENESIS: Handle saturation rise
            elif pattern_type == 'saturation_rise':
                saturation_ratio = pattern.get('saturation_ratio', 0.0)
                print(f"Meta-Scientist: SATURATION RISE detected (ratio: {saturation_ratio:.3f})")
                
                target_engines = []
                if prey_engine is not None:
                    target_engines.append(('prey', prey_engine))
                if predator_engine is not None:
                    target_engines.append(('predator', predator_engine))
                if not target_engines and evolution_engine is not None:
                    target_engines.append(('default', evolution_engine))
                
                for name, engine in target_engines:
                    # Reduce architecture complexity mutations (prevent growth of dead neurons)
                    if hasattr(engine, 'architecture_mutation_rate'):
                        old_arch_rate = engine.architecture_mutation_rate
                        new_arch_rate = max(old_arch_rate * 0.5, 0.01)  # Reduce by half
                        engine.architecture_mutation_rate = new_arch_rate
                        interventions['architecture_prunes'].append({
                            'species': name,
                            'parameter': 'architecture_mutation_rate',
                            'old_value': old_arch_rate,
                            'new_value': new_arch_rate,
                            'reason': f'Saturation rise (ratio: {saturation_ratio:.3f}) - reducing architecture growth'
                        })
                    
                    # Increase weight regularization through mutation strength reduction
                    if hasattr(engine, 'mutation_strength'):
                        old_strength = engine.mutation_strength
                        new_strength = max(old_strength * 0.7, 0.01)  # Reduce to stabilize
                        engine.mutation_strength = new_strength
                        interventions['architecture_prunes'].append({
                            'species': name,
                            'parameter': 'mutation_strength',
                            'old_value': old_strength,
                            'new_value': new_strength,
                            'reason': f'Saturation rise - stabilizing weights'
                        })

            # NEUROGENESIS: Handle species imbalance
            elif pattern_type == 'species_imbalance':
                imbalance_ratio = pattern.get('imbalance_ratio', 1.0)
                underrepresented = pattern.get('underrepresented', 'unknown')
                print(f"Meta-Scientist: SPECIES IMBALANCE detected (ratio: {imbalance_ratio:.3f}, underrepresented: {underrepresented})")
                
                # Target the underrepresented species
                target_engine = None
                if underrepresented == 'prey' and prey_engine is not None:
                    target_engine = ('prey', prey_engine)
                elif underrepresented == 'predator' and predator_engine is not None:
                    target_engine = ('predator', predator_engine)
                
                if target_engine:
                    name, engine = target_engine
                    
                    # Boost mutation rates for underrepresented species
                    if hasattr(engine, 'mutation_rate'):
                        old_rate = engine.mutation_rate
                        new_rate = min(old_rate * 2.0, 0.1)  # Double mutation rate
                        engine.mutation_rate = new_rate
                        interventions['species_rebalances'].append({
                            'species': name,
                            'parameter': 'mutation_rate',
                            'old_value': old_rate,
                            'new_value': new_rate,
                            'reason': f'Species imbalance (ratio: {imbalance_ratio:.3f}) - boosting {name} diversity'
                        })
                    
                    # Lower speciation threshold to allow more species formation
                    if hasattr(engine, 'speciation_manager') and engine.speciation_manager is not None:
                        old_threshold = engine.speciation_manager.compatibility_threshold
                        new_threshold = max(old_threshold * 0.7, 0.1)  # Lower threshold = more species
                        engine.speciation_manager.compatibility_threshold = new_threshold
                        interventions['species_rebalances'].append({
                            'species': name,
                            'parameter': 'compatibility_threshold',
                            'old_value': old_threshold,
                            'new_value': new_threshold,
                            'reason': f'Species imbalance - lowering threshold to create more {name} species'
                        })
                    
                    # Increase architecture mutation for the underrepresented species
                    if hasattr(engine, 'architecture_mutation_rate'):
                        old_arch_rate = engine.architecture_mutation_rate
                        new_arch_rate = min(old_arch_rate * 2.0, 0.2)
                        engine.architecture_mutation_rate = new_arch_rate
                        interventions['species_rebalances'].append({
                            'species': name,
                            'parameter': 'architecture_mutation_rate',
                            'old_value': old_arch_rate,
                            'new_value': new_arch_rate,
                            'reason': f'Species imbalance - boosting {name} architecture exploration'
                        })

        # Track intervention in history
        if any(interventions.values()):
            self.intervention_history.append({
                'generation': generation,
                'interventions': interventions,
                'mutation_direction': self.last_mutation_direction
            })

        total_interventions = (
            len(interventions['interventions_applied']) +
            len(interventions['parameter_emergencies']) +
            len(interventions['mutation_reductions']) +
            len(interventions['plasticity_boosts']) +
            len(interventions['architecture_prunes']) +
            len(interventions['species_rebalances']) +
            len(interventions['system_resets'])
        )

        print(f"Meta-Scientist: Applied {total_interventions} total interventions:")
        print(f"  - {len(interventions['interventions_applied'])} general interventions")
        print(f"  - {len(interventions['parameter_emergencies'])} parameter emergencies")
        print(f"  - {len(interventions['mutation_reductions'])} mutation reductions")
        print(f"  - {len(interventions['plasticity_boosts'])} plasticity boosts (NeuroGenesis)")
        print(f"  - {len(interventions['architecture_prunes'])} architecture prunes (NeuroGenesis)")
        print(f"  - {len(interventions['species_rebalances'])} species rebalances (NeuroGenesis)")
        print(f"  - {len(interventions['system_resets'])} system resets")

        return interventions
