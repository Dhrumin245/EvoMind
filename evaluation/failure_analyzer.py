import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from core.genome import EvolvableGenome
import math

class FailureAnalyzer:
    """Decompose learning failures into root causes"""

    def __init__(self):
        self.fix_effectiveness = {}  # Track which fixes work
        self.failure_history = []    # Track failure patterns over time

    def diagnose_failure(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Diagnose learning failures by analyzing genome and task performance

        Args:
            genome: The genome being evaluated
            task: Task information including fitness, behavioral data, etc.

        Returns:
            Dict with diagnosis results and ranked causes
        """
        diagnosis = {
            'architectural': self._check_architecture(genome, task),
            'learning': self._check_learning(genome, task),
            'exploration': self._check_exploration(genome, task),
            'credit_assignment': self._check_credit_assignment(genome, task),
            'stability': self._check_stability(genome, task),
            'overfitting': self._check_overfitting(genome, task),
        }

        # Add to failure history for pattern analysis
        self.failure_history.append({
            'genome_id': genome.genome_id,
            'fitness': genome.fitness,
            'diagnosis': diagnosis,
            'task': task.get('name', 'unknown')
        })

        # Keep history manageable
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-50:]

        return self._rank_causes(diagnosis, genome, task)

    def _check_architecture(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for architectural issues that prevent learning"""
        severity = 0.0
        fixes = []

        # Check network depth
        num_layers = len(genome.genes)
        if num_layers < 2:
            severity += 0.8
            fixes.append({
                'type': 'add_layers',
                'description': 'Network too shallow for complex tasks',
                'priority': 'high'
            })

        # Check for skip connections
        skip_count = sum(1 for gene in genome.genes if getattr(gene, 'skip_connection', False))
        skip_ratio = skip_count / max(num_layers, 1)

        if skip_ratio < 0.1 and num_layers > 3:
            severity += 0.3
            fixes.append({
                'type': 'add_skip_connections',
                'description': 'Add skip connections to improve gradient flow',
                'priority': 'medium'
            })

        # Check capacity (total parameters)
        total_params = sum(gene.input_dim * gene.output_dim for gene in genome.genes)
        task_complexity = task.get('complexity', 1.0)  # Assume task has complexity metric

        if total_params < task_complexity * 1000:
            severity += 0.4
            fixes.append({
                'type': 'increase_capacity',
                'description': 'Network capacity too low for task complexity',
                'priority': 'medium'
            })

        # Check for dead layers (no plasticity)
        dead_layers = 0
        for gene in genome.genes:
            if gene.plasticity is not None:
                plasticity_magnitude = np.mean(np.abs(gene.plasticity))
                if plasticity_magnitude < 0.01:
                    dead_layers += 1

        if dead_layers > num_layers * 0.5:
            severity += 0.6
            fixes.append({
                'type': 'reset_plasticity',
                'description': 'Too many layers have dead plasticity',
                'priority': 'high'
            })

        return {'severity': min(severity, 1.0), 'fixes': fixes}

    def _check_learning(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for learning mechanism issues"""
        severity = 0.0
        fixes = []

        # Check plasticity effectiveness
        plastic_diag = getattr(genome, 'plastic_diagnostics', None)
        if plastic_diag:
            mean_plastic_delta = plastic_diag.get('mean_final_plastic_delta', 0.0)
            if abs(mean_plastic_delta) < 0.01:
                severity += 0.7
                fixes.append({
                    'type': 'increase_plasticity_lr',
                    'description': 'Plasticity learning rate too low',
                    'priority': 'high'
                })

            # Check for unstable plasticity
            if abs(mean_plastic_delta) > 1.0:
                severity += 0.5
                fixes.append({
                    'type': 'stabilize_plasticity',
                    'description': 'Plasticity updates too unstable',
                    'priority': 'medium'
                })

        # Check meta-parameters
        meta = getattr(genome, 'meta', {})
        reward_gain = meta.get('reward_gain', 1.0)
        plastic_lr = meta.get('plastic_lr', 1.0)

        if reward_gain < 0.1:
            severity += 0.4
            fixes.append({
                'type': 'increase_reward_gain',
                'description': 'Reward gain too low for effective learning',
                'priority': 'medium'
            })

        if plastic_lr < 0.1:
            severity += 0.4
            fixes.append({
                'type': 'increase_meta_lr',
                'description': 'Meta learning rate too low',
                'priority': 'medium'
            })

        # Check learning rule network
        if hasattr(genome, 'learning_rule_net') and genome.learning_rule_net is not None:
            # Simple check: if learning rule outputs are too small
            try:
                with torch.no_grad():
                    zero_input = torch.zeros(1, genome.input_size)
                    zero_post = torch.zeros(1, genome.output_size)
                    zero_reward = torch.zeros(1, 1)
                    zero_w = torch.zeros(1, genome.output_size * genome.input_size)
                    zero_t = torch.zeros(1, 1)

                    delta_w = genome.learning_rule_net(zero_input, zero_post, zero_reward, zero_w, zero_t)
                    mean_delta = torch.mean(torch.abs(delta_w)).item()

                    if mean_delta < 0.001:
                        severity += 0.3
                        fixes.append({
                            'type': 'boost_learning_rule',
                            'description': 'Learning rule network outputs too weak',
                            'priority': 'low'
                        })
            except:
                pass  # Skip if torch not available or error

        return {'severity': min(severity, 1.0), 'fixes': fixes}

    def _check_exploration(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for exploration and local optima issues"""
        severity = 0.0
        fixes = []

        # Check fitness stagnation
        learning_curve = getattr(genome, 'learning_curve', {})
        recent_fitnesses = learning_curve.get('episode_final_fitness', [])

        if len(recent_fitnesses) >= 5:
            # Check if fitness has been stagnant
            recent_avg = np.mean(recent_fitnesses[-5:])
            earlier_avg = np.mean(recent_fitnesses[:-5]) if len(recent_fitnesses) > 5 else recent_avg

            if abs(recent_avg - earlier_avg) < 0.01:
                severity += 0.6
                fixes.append({
                    'type': 'increase_exploration',
                    'description': 'Fitness stagnant - likely stuck in local optimum',
                    'priority': 'high'
                })

        # Check behavioral diversity
        behavioral_data = task.get('behavioral_data', {})
        if behavioral_data:
            # Check if behavior is too repetitive
            action_entropy = behavioral_data.get('action_entropy', 1.0)
            if action_entropy < 0.1:
                severity += 0.4
                fixes.append({
                    'type': 'add_behavioral_noise',
                    'description': 'Behavioral repertoire too limited',
                    'priority': 'medium'
                })

        # Check for premature convergence
        age = getattr(genome, 'age', 0)
        fitness = getattr(genome, 'fitness', 0.0)

        if age < 10 and fitness < task.get('min_expected_fitness', 0.0):
            severity += 0.3
            fixes.append({
                'type': 'delay_convergence',
                'description': 'Genome converging too early to suboptimal solution',
                'priority': 'low'
            })

        return {'severity': min(severity, 1.0), 'fixes': fixes}

    def _check_credit_assignment(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for credit assignment and reward processing issues"""
        severity = 0.0
        fixes = []

        # Check reward processing
        meta = getattr(genome, 'meta', {})
        reward_bias = meta.get('reward_bias', 0.0)

        # If reward bias is extreme, credit assignment might be biased
        if abs(reward_bias) > 2.0:
            severity += 0.4
            fixes.append({
                'type': 'balance_reward_bias',
                'description': 'Reward bias too extreme, distorting credit assignment',
                'priority': 'medium'
            })

        # Check if meta-parameters correlate with fitness
        # This would require historical data, but we can check current effectiveness
        adaptability = self._calculate_adaptability(genome, task)
        if adaptability < 0.2:
            severity += 0.5
            fixes.append({
                'type': 'tune_meta_params',
                'description': 'Meta-parameters not effectively guiding learning',
                'priority': 'high'
            })

        # Check for reward signal issues
        task_rewards = task.get('reward_stats', {})
        if task_rewards:
            reward_variance = task_rewards.get('variance', 1.0)
            if reward_variance < 0.01:
                severity += 0.3
                fixes.append({
                    'type': 'increase_reward_variance',
                    'description': 'Reward signal too flat, poor credit assignment',
                    'priority': 'medium'
                })

        return {'severity': min(severity, 1.0), 'fixes': fixes}

    def _check_stability(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for learning stability and gradient flow issues"""
        severity = 0.0
        fixes = []

        # Check plasticity stability
        plastic_diag = getattr(genome, 'plastic_diagnostics', None)
        if plastic_diag:
            stability_scores = plastic_diag.get('stability_scores', [])
            if stability_scores:
                avg_stability = np.mean(stability_scores)
                if avg_stability < 0.3:
                    severity += 0.5
                    fixes.append({
                        'type': 'improve_stability',
                        'description': 'Learning updates too unstable',
                        'priority': 'high'
                    })

        # Check for vanishing/exploding gradients (approximated)
        weight_magnitudes = []
        for gene in genome.genes:
            if gene.weights is not None:
                weight_magnitudes.append(np.mean(np.abs(gene.weights)))

        if weight_magnitudes:
            avg_weight_mag = np.mean(weight_magnitudes)
            if avg_weight_mag < 0.01:
                severity += 0.4
                fixes.append({
                    'type': 'fix_vanishing_weights',
                    'description': 'Weights too small, possible vanishing gradient',
                    'priority': 'medium'
                })
            elif avg_weight_mag > 10.0:
                severity += 0.4
                fixes.append({
                    'type': 'fix_exploding_weights',
                    'description': 'Weights too large, possible exploding gradient',
                    'priority': 'medium'
                })

        # Check activation saturation
        # This would require activation monitoring during evaluation
        activation_data = task.get('activation_stats', {})
        if activation_data:
            saturation_rate = activation_data.get('saturation_rate', 0.0)
            if saturation_rate > 0.8:
                severity += 0.3
                fixes.append({
                    'type': 'fix_activation_saturation',
                    'description': 'Activations saturating, limiting learning capacity',
                    'priority': 'low'
                })

        return {'severity': min(severity, 1.0), 'fixes': fixes}

    def _check_overfitting(self, genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check for overfitting vs generalization balance"""
        severity = 0.0
        fixes = []

        # Check if genome performs well on training but poorly on validation
        train_fitness = task.get('train_fitness', genome.fitness)
        val_fitness = task.get('validation_fitness', genome.fitness)
        generalization_gap = train_fitness - val_fitness

        if generalization_gap > 0.3:
            severity += 0.6
            fixes.append({
                'type': 'add_regularization',
                'description': 'Large generalization gap indicates overfitting',
                'priority': 'high'
            })

        # Check parameter efficiency (performance per parameter)
        total_params = sum(gene.input_dim * gene.output_dim for gene in genome.genes)
        if total_params > 0:
            param_efficiency = genome.fitness / math.log(total_params + 1)
            if param_efficiency < 0.1:
                severity += 0.4
                fixes.append({
                    'type': 'simplify_architecture',
                    'description': 'Poor parameter efficiency, network too complex',
                    'priority': 'medium'
                })

        # Check for specialization vs generalization
        behavioral_diversity = task.get('behavioral_diversity', 1.0)
        if behavioral_diversity < 0.2:
            severity += 0.3
            fixes.append({
                'type': 'increase_behavioral_diversity',
                'description': 'Genome too specialized, lacks generalization',
                'priority': 'low'
            })

        return {'severity': min(severity, 1.0), 'fixes': fixes}

    def _calculate_adaptability(self, genome: EvolvableGenome, task: Dict[str, Any]) -> float:
        """Calculate how well the genome adapts to the task"""
        # Simple adaptability metric based on meta-parameter effectiveness
        meta = getattr(genome, 'meta', {})
        plastic_lr = meta.get('plastic_lr', 1.0)
        reward_gain = meta.get('reward_gain', 1.0)

        # Check if meta-params are in reasonable ranges
        adaptability = 1.0

        if plastic_lr < 0.1 or plastic_lr > 20.0:
            adaptability *= 0.5
        if reward_gain < 0.1 or reward_gain > 10.0:
            adaptability *= 0.7

        # Factor in fitness relative to task difficulty
        task_difficulty = task.get('difficulty', 1.0)
        normalized_fitness = genome.fitness / task_difficulty
        adaptability *= min(normalized_fitness, 1.0)

        return max(0.0, min(1.0, adaptability))

    def _rank_causes(self, diagnosis: Dict[str, Any], genome: EvolvableGenome, task: Dict[str, Any]) -> Dict[str, Any]:
        """Rank failure causes by severity and suggest fixes"""

        # Calculate overall severity
        total_severity = sum(cause['severity'] for cause in diagnosis.values())

        # Rank causes by severity
        ranked_causes = sorted(
            [(name, cause) for name, cause in diagnosis.items()],
            key=lambda x: x[1]['severity'],
            reverse=True
        )

        # Collect all fixes with priorities
        all_fixes = []
        for cause_name, cause_data in diagnosis.items():
            for fix in cause_data['fixes']:
                fix['cause'] = cause_name
                fix['severity'] = cause_data['severity']
                all_fixes.append(fix)

        # Sort fixes by priority and severity
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        ranked_fixes = sorted(
            all_fixes,
            key=lambda x: (priority_order.get(x['priority'], 0), x['severity']),
            reverse=True
        )

        # Track fix effectiveness for future recommendations
        for fix in ranked_fixes[:3]:  # Top 3 fixes
            fix_type = fix['type']
            if fix_type not in self.fix_effectiveness:
                self.fix_effectiveness[fix_type] = {'attempts': 0, 'successes': 0}

        return {
            'total_severity': total_severity,
            'primary_cause': ranked_causes[0][0] if ranked_causes else None,
            'ranked_causes': ranked_causes,
            'recommended_fixes': ranked_fixes[:5],  # Top 5 fixes
            'diagnosis': diagnosis,
            'genome_id': genome.genome_id,
            'task_name': task.get('name', 'unknown')
        }

    def get_failure_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in failure history"""
        if not self.failure_history:
            return {}

        # Analyze common failure causes
        cause_counts = {}
        severity_trends = []

        for entry in self.failure_history[-20:]:  # Last 20 diagnoses
            diagnosis = entry['diagnosis']
            severity_trends.append(sum(c['severity'] for c in diagnosis.values()))

            for cause_name, cause_data in diagnosis.items():
                if cause_name not in cause_counts:
                    cause_counts[cause_name] = 0
                if cause_data['severity'] > 0.5:  # Count significant failures
                    cause_counts[cause_name] += 1

        return {
            'common_causes': sorted(cause_counts.items(), key=lambda x: x[1], reverse=True),
            'severity_trend': np.mean(severity_trends) if severity_trends else 0.0,
            'total_analyzed': len(self.failure_history)
        }

    def suggest_systemic_fixes(self) -> List[Dict[str, Any]]:
        """Suggest fixes that address systemic issues across the population"""
        patterns = self.get_failure_patterns()

        systemic_fixes = []

        # If architectural issues are common
        if patterns.get('common_causes'):
            arch_failures = sum(1 for cause, count in patterns['common_causes']
                              if cause == 'architectural' and count > 5)
            if arch_failures > 0:
                systemic_fixes.append({
                    'type': 'population_architecture_boost',
                    'description': 'Many genomes have architectural issues - consider increasing default depth',
                    'impact': 'population_wide'
                })

            # If learning issues are common
            learning_failures = sum(1 for cause, count in patterns['common_causes']
                                  if cause == 'learning' and count > 5)
            if learning_failures > 0:
                systemic_fixes.append({
                    'type': 'meta_parameter_curriculum',
                    'description': 'Learning failures common - implement meta-parameter curriculum',
                    'impact': 'population_wide'
                })

        return systemic_fixes
