import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from torch.distributions import Categorical
from torch_brain import TorchBrain
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    num_steps: int = 2048
    num_mini_batches: int = 32
    target_kl: float = 0.01


class PPOTrainer:
    """
    PPO trainer for TorchBrain networks
    Trains the base weights while preserving plasticity for fine-tuning
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, brain: TorchBrain, env_fn, num_steps: int = 1000) -> Dict[str, Any]:
        """
        Train a TorchBrain using PPO

        Args:
            brain: TorchBrain to train
            env_fn: Function that returns a fresh environment
            num_steps: Number of training steps

        Returns:
            Training statistics
        """
        # Create actor and critic networks
        actor_critic = ActorCritic(brain).to(self.device)

        # Optimizer
        optimizer = optim.Adam(actor_critic.parameters(), lr=self.config.learning_rate)

        # Storage for trajectories
        storage = RolloutStorage(
            num_steps=self.config.num_steps,
            num_envs=1,  # Single environment for now
            obs_shape=(brain.input_size,),
            action_shape=(),  # Discrete actions are scalars
            device=self.device
        )

        # Training loop
        total_steps = 0
        episode_rewards = []
        episode_lengths = []

        while total_steps < num_steps:
            # Collect trajectories
            episode_reward, episode_length = self._collect_trajectories(
                actor_critic, env_fn, storage
            )
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            # Compute advantages and returns
            advantages = self._compute_advantages(storage)

            # PPO update
            self._ppo_update(actor_critic, storage, advantages, optimizer)

            total_steps += self.config.num_steps

        return {
            'total_steps': total_steps,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_reward': episode_rewards[-1] if episode_rewards else 0.0
        }

    def _collect_trajectories(self, actor_critic, env_fn, storage) -> Tuple[float, int]:
        """Collect trajectories using current policy"""
        env = env_fn()
        # For multi-agent arena, reset returns (prey_obs, pred_obs)
        # We'll use the first prey agent's observation for single-agent PPO
        prey_obs, pred_obs = env.reset()
        obs = prey_obs[0]  # Use first prey agent's observation
        episode_reward = 0.0
        episode_length = 0

        for step in range(self.config.num_steps):
            # Get action from policy
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action_logits, value = actor_critic(obs_tensor)
                action_probs = F.softmax(action_logits, dim=-1).squeeze(0)
                action = int(torch.multinomial(action_probs, 1).item())
                log_prob = torch.log(action_probs[action].clamp_min(1e-8))

            # Step environment with single action for first prey agent
            # Create action arrays for all agents (only first prey moves)
            prey_actions = np.zeros(len(prey_obs), dtype=int)
            prey_actions[0] = action  # Only first prey acts
            pred_actions = np.zeros(len(pred_obs), dtype=int)  # Predators don't act in this simple setup

            (next_prey_obs, next_pred_obs), rewards, dones, info = env.step(prey_actions, pred_actions)

            # Use reward and done for first prey agent
            reward = float(rewards[0])  # First prey reward
            done = bool(dones[0])  # First prey done
            next_obs = next_prey_obs[0]  # Next observation for first prey

            episode_reward += reward
            episode_length += 1

            # Store transition
            storage.insert(
                obs=obs,
                action=action,
                reward=reward,
                value=value.squeeze().cpu().numpy(),
                log_prob=log_prob.cpu().numpy(),
                done=done
            )

            obs = next_obs

            if done:
                # Reset environment
                prey_obs, pred_obs = env.reset()
                obs = prey_obs[0]
                break

        env.close()
        storage.compute_returns_and_advantages(self.config.gamma, self.config.gae_lambda)

        return episode_reward, episode_length

    def _compute_advantages(self, storage):
        """Compute GAE advantages"""
        advantages = storage.advantages
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.detach()

    def _ppo_update(self, actor_critic, storage, advantages, optimizer):
        """Perform PPO update"""
        # Flatten rollout tensors: (T, N, ...) -> (T*N, ...)
        # This avoids subtle broadcasting bugs (e.g., (B,) vs (B,1)) that can
        # accidentally couple samples and create hard-to-debug autograd behavior.
        obs = storage.obs.reshape(-1, storage.obs_shape[0])
        actions = storage.actions.reshape(-1)
        old_log_probs = storage.log_probs.reshape(-1).detach()
        returns = storage.returns.reshape(-1).detach()
        advantages = advantages.reshape(-1).detach()

        # Create mini-batches
        batch_size = self.config.batch_size
        num_samples = obs.shape[0]

        early_stop = False

        for _ in range(self.config.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(num_samples)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Get batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                action_logits, values = actor_critic(batch_obs)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)

                # Policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values.squeeze(-1), batch_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                # Update
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                nn.utils.clip_grad_norm_(actor_critic.parameters(), self.config.max_grad_norm)
                optimizer.step()

                # Early stopping if KL divergence is too high
                kl_div = (batch_old_log_probs - new_log_probs).mean().detach().item()
                if kl_div > self.config.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break


class ActorCritic(nn.Module):
    """
    Actor-Critic network that wraps a TorchBrain
    Shares the base network but has separate heads for policy and value
    """

    def __init__(self, brain: TorchBrain):
        super().__init__()
        self.brain = brain

        # Get the output dimension of the brain
        with torch.no_grad():
            dummy_input = torch.zeros(1, brain.input_size)
            brain_output = brain(dummy_input)
            self.feature_dim = brain_output.shape[-1]

        # Policy head (actor)
        self.actor_head = nn.Linear(self.feature_dim, brain.output_size)

        # Value head (critic)
        self.critic_head = nn.Linear(self.feature_dim, 1)

        # Initialize heads
        nn.init.xavier_uniform_(self.actor_head.weight)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.xavier_uniform_(self.critic_head.weight)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, x):
        # Reset brain state to prevent gradient accumulation through hidden states
        self.brain.reset_state()

        # Get features from brain
        features = self.brain(x)

        # Get action logits and value
        action_logits = self.actor_head(features)
        value = self.critic_head(features)

        return action_logits, value


class RolloutStorage:
    """
    Storage for rollout data during PPO training
    """

    def __init__(self, num_steps, num_envs, obs_shape, action_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device

        # Storage buffers
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape, dtype=torch.long).to(device)
        self.log_probs = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.bool).to(device)

        self.step = 0

    def insert(self, obs, action, reward, value, log_prob, done):
        """Insert a transition into storage"""
        self.obs[self.step, 0] = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.step] = torch.tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.step] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.step] = torch.tensor(value, dtype=torch.float32, device=self.device)
        self.log_probs[self.step] = torch.tensor(log_prob, dtype=torch.float32, device=self.device)
        self.dones[self.step] = torch.tensor(done, dtype=torch.bool, device=self.device)

        self.step = (self.step + 1) % self.num_steps

    def compute_returns_and_advantages(self, gamma, gae_lambda):
        """Compute returns and advantages using GAE"""
        advantages = torch.zeros_like(self.rewards).to(self.device)
        returns = torch.zeros_like(self.rewards).to(self.device)

        # Compute advantages and returns in reverse
        last_gae_lam = 0
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[step].float()
                next_values = 0
            else:
                next_non_terminal = 1.0 - self.dones[step + 1].float()
                next_values = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            advantages[step] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        returns = advantages + self.values

        # Store computed values
        self.advantages = advantages
        self.returns = returns.detach()
