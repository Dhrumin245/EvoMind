import numpy as np
import random
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from core.genome import EvolvableGenome
from core.numeric_safety import safe_linear_slope
from environments.deterministic_env import DeterministicVectorizedArena
from curriculum.curriculum import CurriculumStage, get_stage_config
import json
import time
from pathlib import Path
import threading
import queue



# Probe budget configuration - adaptive based on generation
def get_probe_budget(generation: int) -> int:
    """Adaptive probe budget: fewer probes as evolution progresses"""
    if generation < 50:
        return 5  # Early exploration: more probes
    elif generation < 200:
        return 2  # Mid evolution: moderate probes
    else:
        return 1  # Late evolution: minimal probes

# Legacy constant for backward compatibility
PROBE_BUDGET = 10  # per generation (deprecated, use get_probe_budget)


@dataclass
class ProbeResult:
    """Result of a behavioral probe test"""
    genome_id: str
    probe_name: str
    score: float
    metrics: Dict[str, Any]
    timestamp: float
    generation: int = 0


@dataclass
class ProbeReport:
    """Comprehensive report for a genome's behavioral capabilities"""
    genome_id: str
    generation: int
    timestamp: float
    probe_results: List[ProbeResult]
    summary_scores: Dict[str, float]
    behavioral_profile: Dict[str, Any]

class BehavioralProbe:
    """Isolated skill testing for evolved neural networks"""

    # Asynchronous logging queue
    _log_queue = queue.Queue()
    _log_thread = None
    _stop_event = threading.Event()

    @staticmethod
    def start_async_logging():
        """Start the background logging thread"""
        if BehavioralProbe._log_thread is not None and BehavioralProbe._log_thread.is_alive():
            return  # Already running

        BehavioralProbe._stop_event.clear()
        BehavioralProbe._log_thread = threading.Thread(target=BehavioralProbe._background_logger, daemon=True)
        BehavioralProbe._log_thread.start()

    @staticmethod
    def stop_async_logging():
        """Stop the background logging thread"""
        if BehavioralProbe._log_thread is not None:
            BehavioralProbe._stop_event.set()
            BehavioralProbe._log_queue.put(None)  # Sentinel to wake up thread
            BehavioralProbe._log_thread.join(timeout=5.0)
            BehavioralProbe._log_thread = None

    @staticmethod
    def _background_logger():
        """Background thread that writes probe reports to disk"""
        while not BehavioralProbe._stop_event.is_set():
            try:
                # Get next report from queue with timeout
                report_data = BehavioralProbe._log_queue.get(timeout=1.0)
                if report_data is None:  # Sentinel value
                    break

                # Unpack the report data
                report, filename = report_data

                # Save the report (same logic as before)
                base_dir = Path("artifacts/probes") / f"gen_{report['generation']:04d}"
                base_dir.mkdir(parents=True, exist_ok=True)

                filepath = base_dir / filename

                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)

            except queue.Empty:
                continue  # Timeout, check stop event
            except Exception as e:
                print(f"Error in background logger: {e}")
                continue

    @staticmethod
    def test_memory_capacity(genome: EvolvableGenome, sequence_length: int = 10) -> ProbeResult:
        """Test working memory capacity by presenting and recalling sequences"""
        # Create a simple memory task environment
        env = DeterministicVectorizedArena(
            num_envs=1,
            max_steps=sequence_length * 2 + 10,  # Present + recall + buffer
            seed=random.randint(0, 10000)
        )

        # Generate random sequence to remember
        sequence = np.random.choice([0, 1, 2, 3], size=sequence_length)  # 4 possible actions

        state = env.reset()
        total_reward = 0.0
        correct_recalls = 0

        # Phase 1: Present sequence (genome observes but doesn't act)
        for i, target_action in enumerate(sequence):
            # Get genome's action (should be random during presentation)
            action = genome.act(state[0])
            actions = np.array([action])

            # Step environment with target action (simulated presentation)
            state, reward, done = env.step(np.array([target_action]))
            total_reward += float(reward[0])

        # Phase 2: Test recall (genome should reproduce sequence)
        for i, target_action in enumerate(sequence):
            action = genome.act(state[0])
            actions = np.array([action])

            # Reward for correct recall
            recall_reward = 1.0 if action == target_action else -0.1
            state, reward, done = env.step(actions)
            total_reward += recall_reward

            if action == target_action:
                correct_recalls += 1

        # Calculate memory accuracy
        memory_accuracy = correct_recalls / sequence_length
        memory_score = memory_accuracy * (1.0 + total_reward / (sequence_length * 2))

        metrics = {
            'sequence_length': sequence_length,
            'correct_recalls': correct_recalls,
            'memory_accuracy': memory_accuracy,
            'total_reward': total_reward,
            'presentation_phase_reward': total_reward - (correct_recalls * 1.0 - (sequence_length - correct_recalls) * 0.1),
            'recall_phase_reward': correct_recalls * 1.0 - (sequence_length - correct_recalls) * 0.1
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='memory_capacity',
            score=float(memory_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def test_generalization(genome: EvolvableGenome, train_envs: List[str], test_envs: List[str]) -> ProbeResult:
        """Test zero-shot generalization across different environments"""
        # Simulate different environment configurations
        env_configs = {
            'normal': {'name': 'normal', 'reward_scale': 1.0, 'noise_level': 0.0},
            'noisy': {'name': 'noisy', 'reward_scale': 1.0, 'noise_level': 0.5},
            'sparse': {'name': 'sparse', 'reward_scale': 0.1, 'noise_level': 0.0},
            'dense': {'name': 'dense', 'reward_scale': 10.0, 'noise_level': 0.0},
            'volatile': {'name': 'volatile', 'reward_scale': 1.0, 'noise_level': 0.0, 'reward_volatility': 0.8}
        }

        train_scores = []
        test_scores = []

        # Evaluate on training environments
        for env_name in train_envs:
            if env_name in env_configs:
                score = BehavioralProbe._evaluate_in_environment(genome, env_configs[env_name])
                train_scores.append(score)

        # Evaluate on test environments (novel)
        for env_name in test_envs:
            if env_name in env_configs:
                score = BehavioralProbe._evaluate_in_environment(genome, env_configs[env_name])
                test_scores.append(score)

        # Calculate generalization metrics
        train_mean = np.mean(train_scores) if train_scores else 0.0
        test_mean = np.mean(test_scores) if test_scores else 0.0
        generalization_ratio = test_mean / (train_mean + 1e-6)  # Avoid division by zero

        # Zero-shot learning score: how well performance transfers to novel environments
        zero_shot_score = test_mean - train_mean  # Positive if better on novel tasks

        # Robustness score: consistency across different environments
        all_scores = train_scores + test_scores
        robustness_score = 1.0 - (np.std(all_scores) / (np.mean(all_scores) + 1e-6)) if all_scores else 0.0

        final_score = (generalization_ratio + zero_shot_score + robustness_score) / 3.0

        metrics = {
            'train_environments': train_envs,
            'test_environments': test_envs,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'train_mean': train_mean,
            'test_mean': test_mean,
            'generalization_ratio': generalization_ratio,
            'zero_shot_score': zero_shot_score,
            'robustness_score': robustness_score,
            'environments_tested': len(train_scores) + len(test_scores)
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='generalization',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def test_learning_speed(genome: EvolvableGenome, novel_task: str) -> ProbeResult:
        """Measure adaptation rate to a novel task"""
        # Create novel task environment
        task_configs = {
            'reversed_rewards': {'reward_multiplier': -1.0, 'name': 'reversed'},
            'delayed_rewards': {'delay_steps': 5, 'name': 'delayed'},
            'partial_observability': {'obs_noise': 0.3, 'name': 'partial_obs'},
            'changing_goals': {'goal_changes': True, 'name': 'changing_goals'}
        }

        if novel_task not in task_configs:
            novel_task = 'reversed_rewards'  # Default

        config = task_configs[novel_task]

        # Baseline evaluation (first 3 episodes - reduced from 10)
        baseline_scores = []
        for episode in range(3):
            score = BehavioralProbe._evaluate_episode(genome, config, seed=episode)
            baseline_scores.append(score)

        baseline_mean = float(np.mean(baseline_scores))
        if not np.isfinite(baseline_mean):
            baseline_mean = 0.0

        # Adaptation evaluation (next 5 episodes with plasticity - reduced from 20)
        adaptation_scores = []
        learning_curve = []

        for episode in range(5):

            score = BehavioralProbe._evaluate_episode(genome, config, seed=episode + 10, enable_plasticity=True)
            adaptation_scores.append(score)

            # Calculate learning progress
            if len(adaptation_scores) >= 5:
                recent_mean = np.mean(adaptation_scores[-5:])
                learning_curve.append(recent_mean - baseline_mean)

        # Calculate learning speed metrics
        adaptation_mean = float(np.mean(adaptation_scores))
        if not np.isfinite(adaptation_mean):
            adaptation_mean = baseline_mean
        improvement_rate = float((adaptation_mean - baseline_mean) / (abs(baseline_mean) + 1e-6))
        if not np.isfinite(improvement_rate):
            improvement_rate = 0.0
        improvement_rate = max(0.0, improvement_rate)

        # Speed of adaptation (how quickly performance improves)
        if learning_curve:
            x = np.arange(len(learning_curve), dtype=np.float64)
            learning_acceleration = max(0.0, safe_linear_slope(x, learning_curve, default_value=0.0))
        else:
            learning_acceleration = 0.0

        # Plasticity effectiveness (correlation with improvement)
        if hasattr(genome, 'brain') and genome.brain:
            plastic_diagnostics = genome.brain.get_plastic_diagnostics()
            plasticity_effectiveness = float(plastic_diagnostics.get('mean_plastic_delta', 0.0))
        else:
            plasticity_effectiveness = 0.0
        if not np.isfinite(plasticity_effectiveness):
            plasticity_effectiveness = 0.0

        final_score = improvement_rate + learning_acceleration + plasticity_effectiveness
        if not np.isfinite(final_score):
            final_score = 0.0

        metrics = {
            'novel_task': novel_task,
            'baseline_scores': baseline_scores,
            'adaptation_scores': adaptation_scores,
            'baseline_mean': baseline_mean,
            'adaptation_mean': adaptation_mean,
            'improvement_rate': improvement_rate,
            'learning_acceleration': learning_acceleration,
            'plasticity_effectiveness': plasticity_effectiveness,
            'learning_curve': learning_curve
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='learning_speed',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def test_credit_assignment(genome: EvolvableGenome, delayed_reward_task: str) -> ProbeResult:
        """Test temporal credit assignment in delayed reward scenarios"""
        # Create delayed reward task
        task_configs = {
            'short_delay': {'delay_steps': 3, 'reward_magnitude': 1.0},
            'medium_delay': {'delay_steps': 8, 'reward_magnitude': 1.0},
            'long_delay': {'delay_steps': 15, 'reward_magnitude': 1.0},
            'variable_delay': {'delay_steps': 'random', 'reward_magnitude': 1.0}
        }

        if delayed_reward_task not in task_configs:
            delayed_reward_task = 'medium_delay'

        config = task_configs[delayed_reward_task]

        # Run multiple episodes to assess credit assignment (reduced from 20 to 5)
        episodes = 5
        episode_results = []


        for episode in range(episodes):
            result = BehavioralProbe._evaluate_delayed_reward_episode(genome, config, seed=episode)
            episode_results.append(result)

        # Analyze credit assignment quality
        credit_assignment_scores = []
        temporal_precision_scores = []

        for result in episode_results:
            actions = result['actions']
            rewards = result['rewards']
            delay = result['actual_delay']

            # Find action that led to reward
            reward_idx = None
            for i, r in enumerate(rewards):
                if r > 0.5:  # Significant positive reward
                    reward_idx = i
                    break

            if reward_idx is not None:
                # Check if genome learned to repeat the rewarded action
                rewarded_action = actions[reward_idx - delay] if reward_idx >= delay else actions[0]

                # Count how often this action appears after reward
                post_reward_actions = actions[reward_idx:]
                correct_assignments = sum(1 for a in post_reward_actions if a == rewarded_action)

                credit_score = correct_assignments / len(post_reward_actions) if post_reward_actions else 0.0
                credit_assignment_scores.append(credit_score)

                # Temporal precision: how close to optimal delay
                optimal_delay = delay
                actual_delay_used = reward_idx
                precision = 1.0 - abs(actual_delay_used - optimal_delay) / (optimal_delay + 1)
                temporal_precision_scores.append(max(0, precision))

        # Overall credit assignment quality
        mean_credit_score = np.mean(credit_assignment_scores) if credit_assignment_scores else 0.0
        mean_temporal_precision = np.mean(temporal_precision_scores) if temporal_precision_scores else 0.0

        # Learning stability (consistency across episodes)
        credit_stability = 1.0 - np.std(credit_assignment_scores) if credit_assignment_scores else 0.0

        final_score = (mean_credit_score + mean_temporal_precision + credit_stability) / 3.0

        metrics = {
            'delayed_reward_task': delayed_reward_task,
            'episodes_tested': episodes,
            'mean_credit_assignment_score': mean_credit_score,
            'mean_temporal_precision': mean_temporal_precision,
            'credit_stability': credit_stability,
            'credit_assignment_scores': credit_assignment_scores,
            'temporal_precision_scores': temporal_precision_scores,
            'episode_results': episode_results
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='credit_assignment',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def _evaluate_in_environment(genome: EvolvableGenome, env_config: Dict[str, Any]) -> float:
        """Evaluate genome in a specific environment configuration"""
        env = DeterministicVectorizedArena(
            num_envs=1,
            max_steps=20,  # Reduced from 50 to 20
            seed=random.randint(0, 10000)
        )

        state = env.reset()
        total_reward = 0.0
        steps = 0

        while steps < 20:  # Reduced from 50 to 20

            action = genome.act(state[0])
            actions = np.array([action])

            # Apply environment modifications
            reward_modifier = env_config.get('reward_scale', 1.0)
            noise_level = env_config.get('noise_level', 0.0)

            state, reward, done = env.step(actions)

            # Modify reward based on environment config
            modified_reward = reward[0] * reward_modifier

            # Add noise if specified
            if noise_level > 0:
                modified_reward += np.random.normal(0, noise_level)

            # Apply reward volatility if specified
            if 'reward_volatility' in env_config:
                volatility = env_config['reward_volatility']
                if random.random() < volatility:
                    modified_reward *= -1  # Flip reward sign

            total_reward += modified_reward
            steps += 1

            if done[0]:
                break

        return float(total_reward)

    @staticmethod
    def _evaluate_episode(genome: EvolvableGenome, task_config: Dict[str, Any],
                         seed: int = 0, enable_plasticity: bool = False) -> float:
        """Evaluate single episode with task-specific modifications"""
        env = DeterministicVectorizedArena(
            num_envs=1,
            max_steps=20,  # Reduced from 50 to 20
            seed=seed
        )

        state = env.reset()
        total_reward = 0.0
        steps = 0

        while steps < 20:  # Reduced from 50 to 20

            action = genome.act(state[0])
            actions = np.array([action])

            state, reward, done = env.step(actions)

            # Apply task-specific reward modifications
            modified_reward = reward[0]

            if task_config.get('reward_multiplier', 1.0) != 1.0:
                modified_reward *= task_config['reward_multiplier']

            if 'delay_steps' in task_config and task_config['delay_steps'] > 0:
                # Implement delayed rewards (simplified)
                if steps >= task_config['delay_steps']:
                    delayed_reward = modified_reward
                    total_reward += delayed_reward

            total_reward += modified_reward
            steps += 1

            # Apply plasticity if enabled
            if enable_plasticity and hasattr(genome, 'brain') and genome.brain:
                genome.brain.update_plasticity(modified_reward, done[0])

            if done[0]:
                break

        return float(total_reward)

    @staticmethod
    def _evaluate_delayed_reward_episode(genome: EvolvableGenome, task_config: Dict[str, Any],
                                        seed: int = 0) -> Dict[str, Any]:
        """Evaluate episode for delayed reward credit assignment"""
        env = DeterministicVectorizedArena(
            num_envs=1,
            max_steps=30,  # Reduced from 100 to 30
            seed=seed
        )

        # Determine delay for this episode
        if task_config['delay_steps'] == 'random':
            delay = random.randint(1, 5)  # Reduced from 1-10 to 1-5
        else:
            delay = min(task_config['delay_steps'], 5)  # Cap at 5

        state = env.reset()
        actions_taken = []
        rewards_received = []
        total_reward = 0.0
        steps = 0

        # Choose a "good" action randomly for this episode
        good_action = random.randint(0, 3)  # Assuming 4 possible actions

        while steps < 30:  # Reduced from 100 to 30

            action = genome.act(state[0])
            actions_taken.append(action)
            actions = np.array([action])

            state, reward, done = env.step(actions)

            # Delayed reward mechanism
            if steps == delay and action == good_action:
                # Give reward after delay
                reward_value = task_config.get('reward_magnitude', 1.0)
                total_reward += reward_value
                rewards_received.append(reward_value)
            else:
                rewards_received.append(0.0)

            steps += 1

            if done[0]:
                break

        return {
            'actions': actions_taken,
            'rewards': rewards_received,
            'total_reward': total_reward,
            'delay': delay,
            'actual_delay': delay,
            'good_action': good_action
        }

    @staticmethod
    def run_diagnostic_suite(genome: EvolvableGenome, generation: int = 0) -> ProbeReport:
        """Run complete diagnostic suite on a genome"""
        with torch.no_grad():
            probe_results = []
            probe_errors = []

            def _run_probe(probe_name: str, probe_fn):
                try:
                    result = probe_fn()
                    probe_results.append(result)
                    return result
                except Exception as exc:
                    probe_errors.append({'probe_name': probe_name, 'error': str(exc)})
                    return None

            # Memory capacity test
            memory_result = _run_probe('memory_capacity', lambda: BehavioralProbe.test_memory_capacity(genome))

            # Generalization test
            train_envs = ['normal', 'noisy']
            test_envs = ['sparse', 'dense', 'volatile']
            generalization_result = _run_probe(
                'generalization',
                lambda: BehavioralProbe.test_generalization(genome, train_envs, test_envs)
            )

            # Learning speed test
            learning_result = _run_probe(
                'learning_speed',
                lambda: BehavioralProbe.test_learning_speed(genome, 'reversed_rewards')
            )

            # Credit assignment test
            credit_result = _run_probe(
                'credit_assignment',
                lambda: BehavioralProbe.test_credit_assignment(genome, 'medium_delay')
            )

            # Additional diagnostic tasks - only run every 10 generations to save time
            if generation % 10 == 0:
                try:
                    additional_probes = BehavioralProbe._run_additional_diagnostic_tasks(genome)
                    probe_results.extend(additional_probes)
                except Exception as exc:
                    probe_errors.append({'probe_name': 'additional_diagnostics', 'error': str(exc)})


            # Calculate summary scores
            summary_scores = {}
            if memory_result is not None:
                summary_scores['memory_capacity'] = memory_result.score
            if generalization_result is not None:
                summary_scores['generalization'] = generalization_result.score
            if learning_result is not None:
                summary_scores['learning_speed'] = learning_result.score
            if credit_result is not None:
                summary_scores['credit_assignment'] = credit_result.score

            if probe_results:
                summary_scores['overall_score'] = float(np.mean([r.score for r in probe_results]))
            else:
                summary_scores['overall_score'] = 0.0

            # Create behavioral profile
            behavioral_profile = BehavioralProbe._analyze_behavioral_profile(probe_results) if probe_results else {}
            if probe_errors:
                behavioral_profile = behavioral_profile or {}
                behavioral_profile['partial'] = True
                behavioral_profile['errors'] = probe_errors

            return ProbeReport(
                genome_id=genome.genome_id,
                generation=generation,
                timestamp=time.time(),
                probe_results=probe_results,
                summary_scores=summary_scores,
                behavioral_profile=behavioral_profile
            )

    @staticmethod
    def _run_additional_diagnostic_tasks(genome: EvolvableGenome) -> List[ProbeResult]:
        """Run additional diagnostic tasks for comprehensive evaluation"""
        additional_results = []

        # Task 1: Exploration vs Exploitation balance
        exploration_result = BehavioralProbe._test_exploration_balance(genome)
        additional_results.append(exploration_result)

        # Task 2: Robustness to perturbations
        robustness_result = BehavioralProbe._test_robustness_to_perturbations(genome)
        additional_results.append(robustness_result)

        # Task 3: Multi-step planning capability
        planning_result = BehavioralProbe._test_multi_step_planning(genome)
        additional_results.append(planning_result)

        # Task 4: Adaptation to changing environments
        adaptation_result = BehavioralProbe._test_adaptation_to_change(genome)
        additional_results.append(adaptation_result)

        # Task 5: Social learning capability (simplified)
        social_result = BehavioralProbe._test_social_learning(genome)
        additional_results.append(social_result)

        # Task 6: Meta-learning capability
        meta_result = BehavioralProbe._test_meta_learning(genome)
        additional_results.append(meta_result)

        return additional_results

    @staticmethod
    def _test_exploration_balance(genome: EvolvableGenome) -> ProbeResult:
        """Test balance between exploration and exploitation"""
        env = DeterministicVectorizedArena(num_envs=1, max_steps=30, seed=42)  # Reduced from 100 to 30

        state = env.reset()
        actions_taken = []
        rewards_received = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Assuming 4 actions

        for step in range(30):  # Reduced from 100 to 30

            action = genome.act(state[0])
            actions_taken.append(action)
            action_counts[action] += 1

            actions = np.array([action])
            state, reward, done = env.step(actions)
            rewards_received.append(float(reward[0]))

            if done[0]:
                break

        # Calculate exploration metrics
        total_actions = len(actions_taken)
        action_frequencies = [action_counts[i] / total_actions for i in range(4)]
        exploration_entropy = -sum(p * np.log(p + 1e-10) for p in action_frequencies)

        # Exploitation score: how much time spent on best action
        best_action = max(action_counts.keys(), key=lambda k: action_counts[k])
        exploitation_ratio = action_counts[best_action] / total_actions

        # Balance score: product of exploration and exploitation
        balance_score = exploration_entropy * exploitation_ratio

        metrics = {
            'action_frequencies': action_frequencies,
            'exploration_entropy': exploration_entropy,
            'exploitation_ratio': exploitation_ratio,
            'balance_score': balance_score,
            'total_actions': total_actions
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='exploration_balance',
            score=float(balance_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def _test_robustness_to_perturbations(genome: EvolvableGenome) -> ProbeResult:
        """Test robustness to environmental perturbations"""
        base_scores = []
        perturbed_scores = []

        # Base evaluation (reduced from 5 to 2)
        for seed in range(2):
            score = BehavioralProbe._evaluate_in_environment(genome, {'name': 'normal'})
            base_scores.append(score)

        # Perturbed evaluation (with noise and volatility) (reduced from 5 to 2)
        for seed in range(2):
            score = BehavioralProbe._evaluate_in_environment(genome,
                {'name': 'perturbed', 'noise_level': 0.3, 'reward_volatility': 0.5})
            perturbed_scores.append(score)


        base_mean = np.mean(base_scores)
        perturbed_mean = np.mean(perturbed_scores)
        robustness_score = perturbed_mean / (base_mean + 1e-6)  # Relative performance

        # Stability: lower variance under perturbation
        base_std = np.std(base_scores)
        perturbed_std = np.std(perturbed_scores)
        stability_score = 1.0 - (perturbed_std / (base_std + 1e-6))

        final_score = (robustness_score + stability_score) / 2.0

        metrics = {
            'base_scores': base_scores,
            'perturbed_scores': perturbed_scores,
            'base_mean': base_mean,
            'perturbed_mean': perturbed_mean,
            'robustness_score': robustness_score,
            'stability_score': stability_score
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='robustness_perturbations',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def _test_multi_step_planning(genome: EvolvableGenome) -> ProbeResult:
        """Test capability for multi-step planning"""
        # Create a simple planning task
        env = DeterministicVectorizedArena(num_envs=1, max_steps=15, seed=123)  # Reduced from 20 to 15

        state = env.reset()
        action_sequence = []
        reward_sequence = []
        planning_score = 0.0

        # Look for patterns that suggest planning (e.g., repeating successful actions)
        successful_actions = []

        for step in range(15):  # Reduced from 20 to 15

            action = genome.act(state[0])
            action_sequence.append(action)

            actions = np.array([action])
            state, reward, done = env.step(actions)
            reward_val = float(reward[0])
            reward_sequence.append(reward_val)

            if reward_val > 0.1:  # Positive reward
                successful_actions.append(action)

            if done[0]:
                break

        # Analyze planning indicators
        if successful_actions:
            # Check for repetition of successful actions (planning)
            action_repeats = 0
            for i in range(1, len(action_sequence)):
                if action_sequence[i] == action_sequence[i-1] and action_sequence[i] in successful_actions:
                    action_repeats += 1

            planning_score = action_repeats / len(action_sequence)
        else:
            planning_score = 0.0

        # Look for goal-directed behavior (increasing rewards over time)
        reward_trend = 0.0
        if len(reward_sequence) > 5:
            early_rewards = np.mean(reward_sequence[:5])
            late_rewards = np.mean(reward_sequence[-5:])
            reward_trend = (late_rewards - early_rewards) / (early_rewards + 1e-6)

        final_score = (planning_score + reward_trend) / 2.0

        metrics = {
            'action_sequence': action_sequence,
            'reward_sequence': reward_sequence,
            'successful_actions': successful_actions,
            'planning_score': planning_score,
            'reward_trend': reward_trend
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='multi_step_planning',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def _test_adaptation_to_change(genome: EvolvableGenome) -> ProbeResult:
        """Test adaptation to environmental changes"""
        # Phase 1: Normal environment (reduced from 5 to 2)
        normal_scores = []
        for episode in range(2):
            score = BehavioralProbe._evaluate_episode(genome, {'name': 'normal'}, seed=episode)
            normal_scores.append(score)

        # Phase 2: Changed environment (reversed rewards) (reduced from 10 to 3)
        changed_scores = []
        for episode in range(3):
            score = BehavioralProbe._evaluate_episode(genome,
                {'name': 'changed', 'reward_multiplier': -1.0}, seed=episode + 5, enable_plasticity=True)
            changed_scores.append(score)


        # Calculate adaptation metrics
        normal_mean = np.mean(normal_scores)
        early_change_mean = np.mean(changed_scores[:3])
        late_change_mean = np.mean(changed_scores[-3:])

        # Adaptation speed: how quickly performance recovers
        adaptation_speed = (late_change_mean - early_change_mean) / (abs(early_change_mean) + 1e-6)

        # Overall adaptation: final performance relative to original
        adaptation_quality = late_change_mean / (normal_mean + 1e-6)

        final_score = (adaptation_speed + adaptation_quality) / 2.0

        metrics = {
            'normal_scores': normal_scores,
            'changed_scores': changed_scores,
            'normal_mean': normal_mean,
            'early_change_mean': early_change_mean,
            'late_change_mean': late_change_mean,
            'adaptation_speed': adaptation_speed,
            'adaptation_quality': adaptation_quality
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='adaptation_to_change',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def _test_social_learning(genome: EvolvableGenome) -> ProbeResult:
        """Test social learning capability via demonstration + plasticity.

        Proper three-phase design:
          1. Baseline  – evaluate genome before any demonstration.
          2. Demonstration – step the environment with *expert* actions (not the
             genome's), then feed the resulting reward to the genome's plasticity
             so it can associate observed states with optimal outcomes.
          3. Post-demo – re-evaluate the genome (with plasticity) and measure
             whether performance improved relative to baseline.
        """
        task_config = {'name': 'social_base'}

        # Phase 1: Baseline (no plasticity, fixed seeds)
        baseline_scores = []
        for episode in range(2):
            score = BehavioralProbe._evaluate_episode(
                genome, task_config, seed=episode + 400
            )
            baseline_scores.append(score)
        baseline_mean = float(np.mean(baseline_scores))

        # Phase 2: Demonstration – expert actions drive the env;
        # genome observes states and receives reward signals via plasticity.
        optimal_actions = [1, 1, 0, 0, 1, 1, 0]  # Expert action sequence
        env = DeterministicVectorizedArena(num_envs=1, max_steps=15, seed=456)
        state = env.reset()

        imitation_score = 0.0
        demonstration_reward = 0.0

        for optimal_action in optimal_actions:
            # Record what the genome *would* have chosen (imitation tendency)
            genome_action = genome.act(state[0])
            if genome_action == optimal_action:
                imitation_score += 1.0

            # Step the env with the EXPERT action — this is the demonstration
            state, reward, done = env.step(np.array([optimal_action]))
            reward_val = float(reward[0])
            demonstration_reward += reward_val

            # Feed the demonstration reward into the genome's plasticity so it
            # can update its weights based on the expert's outcomes
            if hasattr(genome, 'brain') and genome.brain:
                genome.brain.update_plasticity(reward_val, done[0])

            if done[0]:
                break

        imitation_accuracy = imitation_score / len(optimal_actions)

        # Phase 3: Post-demonstration – evaluate with plasticity enabled
        post_demo_scores = []
        for episode in range(2):
            score = BehavioralProbe._evaluate_episode(
                genome, task_config, seed=episode + 400, enable_plasticity=True
            )
            post_demo_scores.append(score)
        post_demo_mean = float(np.mean(post_demo_scores))

        # Primary metric: did plasticity-driven learning from the demonstration
        # actually improve performance?
        learning_gain = (post_demo_mean - baseline_mean) / (abs(baseline_mean) + 1e-6)

        # Combined score: imitation tendency + clamped improvement gain
        social_learning_score = (imitation_accuracy + max(0.0, learning_gain)) / 2.0

        metrics = {
            'optimal_actions': optimal_actions,
            'imitation_score': imitation_score,
            'imitation_accuracy': imitation_accuracy,
            'baseline_scores': baseline_scores,
            'baseline_mean': baseline_mean,
            'post_demo_scores': post_demo_scores,
            'post_demo_mean': post_demo_mean,
            'learning_gain': learning_gain,
            'demonstration_reward': demonstration_reward,
            'social_learning_score': social_learning_score,
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='social_learning',
            score=float(social_learning_score),
            metrics=metrics,
            timestamp=time.time()
        )


    @staticmethod
    def _test_meta_learning(genome: EvolvableGenome) -> ProbeResult:
        """Test meta-learning capability"""
        # Test ability to learn learning strategies across different tasks

        tasks = [
            {'name': 'task1', 'reward_multiplier': 1.0},
            {'name': 'task2', 'reward_multiplier': -1.0},
            {'name': 'task3', 'reward_multiplier': 0.5}
        ]

        task_performances = []

        for task in tasks:
            # Quick adaptation to each task (reduced from 5 to 2 episodes)
            task_scores = []
            for episode in range(2):
                score = BehavioralProbe._evaluate_episode(genome, task,
                    seed=episode + 200, enable_plasticity=True)
                task_scores.append(score)

            task_performances.append({
                'task': task['name'],
                'scores': task_scores,
                'improvement': task_scores[-1] - task_scores[0] if len(task_scores) >= 2 else 0.0
            })


        # Meta-learning score: average improvement across tasks
        improvements = [p['improvement'] for p in task_performances]
        meta_learning_score = np.mean(improvements) if improvements else 0.0

        # Consistency: how similar the learning patterns are
        learning_consistency = 1.0 - np.std(improvements) if len(improvements) > 1 else 0.0

        final_score = (meta_learning_score + learning_consistency) / 2.0

        metrics = {
            'task_performances': task_performances,
            'meta_learning_score': meta_learning_score,
            'learning_consistency': learning_consistency,
            'improvements': improvements
        }

        return ProbeResult(
            genome_id=genome.genome_id,
            probe_name='meta_learning',
            score=float(final_score),
            metrics=metrics,
            timestamp=time.time()
        )

    @staticmethod
    def _analyze_behavioral_profile(probe_results: List[ProbeResult]) -> Dict[str, Any]:
        """Analyze overall behavioral profile from probe results"""
        scores = {result.probe_name: result.score for result in probe_results}

        # Categorize capabilities
        cognitive_capabilities = {
            'memory': scores.get('memory_capacity', 0.0),
            'learning': scores.get('learning_speed', 0.0),
            'adaptation': scores.get('credit_assignment', 0.0),
            'generalization': scores.get('generalization', 0.0)
        }

        behavioral_traits = {
            'exploration': scores.get('exploration_balance', 0.0),
            'robustness': scores.get('robustness_perturbations', 0.0),
            'planning': scores.get('multi_step_planning', 0.0),
            'flexibility': scores.get('adaptation_to_change', 0.0)
        }

        advanced_capabilities = {
            'social_learning': scores.get('social_learning', 0.0),
            'meta_learning': scores.get('meta_learning', 0.0)
        }

        # Overall profile assessment
        cognitive_avg = np.mean(list(cognitive_capabilities.values()))
        behavioral_avg = np.mean(list(behavioral_traits.values()))
        advanced_avg = np.mean(list(advanced_capabilities.values()))

        profile = {
            'cognitive_capabilities': cognitive_capabilities,
            'behavioral_traits': behavioral_traits,
            'advanced_capabilities': advanced_capabilities,
            'cognitive_score': cognitive_avg,
            'behavioral_score': behavioral_avg,
            'advanced_score': advanced_avg,
            'overall_profile_score': (cognitive_avg + behavioral_avg + advanced_avg) / 3.0
        }

        # Determine behavioral archetype
        if cognitive_avg > 0.7 and advanced_avg > 0.6:
            profile['archetype'] = 'intelligent_adaptor'
        elif behavioral_avg > 0.7:
            profile['archetype'] = 'robust_explorer'
        elif cognitive_avg > 0.6:
            profile['archetype'] = 'cognitive_specialist'
        else:
            profile['archetype'] = 'generalist'

        return profile

    @staticmethod
    def generate_probe_report(genome: EvolvableGenome, generation: int = 0) -> str:
        """Generate a comprehensive probe report for a genome"""
        report = BehavioralProbe.run_diagnostic_suite(genome, generation)

        # Format as JSON for easy parsing
        report_dict = {
            'genome_id': report.genome_id,
            'generation': report.generation,
            'timestamp': report.timestamp,
            'summary_scores': report.summary_scores,
            'behavioral_profile': report.behavioral_profile,
            'detailed_results': [
                {
                    'probe_name': result.probe_name,
                    'score': result.score,
                    'metrics': result.metrics
                }
                for result in report.probe_results
            ]
        }

        return json.dumps(report_dict, indent=2, default=str)

    @staticmethod
    def save_probe_report(report: ProbeReport, filename: str):
        """Save probe report to file asynchronously"""
        report_dict = {
            'genome_id': report.genome_id,
            'generation': report.generation,
            'timestamp': report.timestamp,
            'summary_scores': report.summary_scores,
            'behavioral_profile': report.behavioral_profile,
            'detailed_results': [
                {
                    'probe_name': result.probe_name,
                    'score': result.score,
                    'metrics': result.metrics
                }
                for result in report.probe_results
            ]
        }

        # Push to async queue instead of writing directly
        BehavioralProbe._log_queue.put((report_dict, filename))

    @staticmethod
    def integrate_with_evaluation_pipeline(genomes: List[EvolvableGenome],
                                         generation: int = 0,
                                         save_reports: bool = True) -> Dict[str, Any]:
        """Integrate behavioral probes into the evaluation pipeline"""
        if generation < 3:
            return {
                'generation': generation,
                'num_genomes_probed': 0,
                'population_summary': {},
                'individual_reports': []
            }
        
        # Skip probes entirely on non-checkpoint generations after gen 100
        if generation > 100 and not save_reports:
            return {
                'generation': generation,
                'num_genomes_probed': 0,
                'population_summary': {},
                'individual_reports': [],
                'skipped': True,
                'reason': 'Non-checkpoint generation after gen 100'
            }
        
        # Use adaptive probe budget based on generation
        effective_budget = get_probe_budget(generation)
        
        probe_reports = []
        selected_genome_ids = set()
        probes_run = 0  # Track number of probes run this generation

        # Limit probe report saving to top K, novelty outliers (middle), failures, and random sample
        if save_reports and genomes:
            K = min(5, len(genomes) // 8)  # Reduced K from 10 to 5

            sorted_genomes = sorted(genomes, key=lambda g: g.fitness, reverse=True)
            selected_indices = set()

            # Top K genomes
            selected_indices.update(range(K))

            # Bottom K genomes (failures)
            selected_indices.update(range(len(genomes) - K, len(genomes)))

            # Middle K genomes (novelty outliers proxy)
            middle_start = len(genomes) // 2 - K // 2
            middle_end = middle_start + K
            selected_indices.update(range(max(0, middle_start), min(len(genomes), middle_end)))

            # Random sample (2% - reduced from 5%)
            random_sample_size = max(1, int(0.02 * len(genomes)))
            random_indices = np.random.choice(len(genomes), size=random_sample_size, replace=False)
            selected_indices.update(random_indices)

            # Map back to original order
            genome_to_index = {genome: i for i, genome in enumerate(genomes)}
            selected_genome_ids = {genomes[i].genome_id for i in selected_indices}

        for genome in genomes:
            # Check probe budget - skip if exhausted
            if probes_run >= effective_budget:
                print(f"Probe budget ({effective_budget}) exhausted for generation {generation}, skipping remaining probes")

                # EXIT THIS STAGE COMPLETELY
                population_summary = BehavioralProbe._aggregate_population_results(probe_reports) if probe_reports else {}
                return {
                    'generation': generation,
                    'num_genomes_probed': len(probe_reports),
                    'population_summary': population_summary,
                    'individual_reports': probe_reports
                }

            try:
                report = BehavioralProbe.run_diagnostic_suite(genome, generation)
                probe_reports.append(report)
                probes_run += 1

                if save_reports and genome.genome_id in selected_genome_ids:
                    filename = f"probe_report_gen_{generation}_{genome.genome_id}.json"
                    BehavioralProbe.save_probe_report(report, filename)

            except Exception as e:
                print(f"Error running probes on genome {genome.genome_id}: {e}")
                continue

            # Memory cleanup after each genome
            if hasattr(genome, 'brain') and genome.brain is not None:
                genome.brain = None
                torch.cuda.empty_cache()

        # Aggregate results across population
        if probe_reports:
            population_summary = BehavioralProbe._aggregate_population_results(probe_reports)
        else:
            population_summary = {}

        return {
            'generation': generation,
            'num_genomes_probed': len(probe_reports),
            'population_summary': population_summary,
            'individual_reports': probe_reports
        }

    @staticmethod
    def _aggregate_population_results(reports: List[ProbeReport]) -> Dict[str, Any]:
        """Aggregate behavioral probe results across a population"""
        if not reports:
            return {}

        # Collect all scores
        probe_names = ['memory_capacity', 'generalization', 'learning_speed', 'credit_assignment',
                      'exploration_balance', 'robustness_perturbations', 'multi_step_planning',
                      'adaptation_to_change', 'social_learning', 'meta_learning']

        population_scores = {name: [] for name in probe_names}

        for report in reports:
            for probe_name in probe_names:
                if probe_name in report.summary_scores:
                    population_scores[probe_name].append(report.summary_scores[probe_name])

        # Calculate population statistics
        summary = {}
        for probe_name, scores in population_scores.items():
            if scores:
                summary[f'{probe_name}_mean'] = float(np.mean(scores))
                summary[f'{probe_name}_std'] = float(np.std(scores))
                summary[f'{probe_name}_max'] = float(np.max(scores))
                summary[f'{probe_name}_min'] = float(np.min(scores))
                summary[f'{probe_name}_p95'] = float(np.percentile(scores, 95))

        # Archetype distribution
        archetypes = {}
        for report in reports:
            archetype = report.behavioral_profile.get('archetype', 'unknown')
            archetypes[archetype] = archetypes.get(archetype, 0) + 1

        summary['archetype_distribution'] = archetypes

        # Overall population behavioral profile
        overall_scores = [report.behavioral_profile['overall_profile_score'] for report in reports]
        summary['population_overall_mean'] = float(np.mean(overall_scores))
        summary['population_overall_std'] = float(np.std(overall_scores))

        return summary
