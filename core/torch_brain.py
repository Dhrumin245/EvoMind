import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging
from collections import deque
from typing import List, Dict, Any, Optional
from core.genome import EvolvableGenome
from diagnostics.plasticity_timing import PlasticityTimingLogger
from diagnostics.reward_recovery import RewardRecoveryLogger
from diagnostics.meta_gene_entropy import MetaGeneEntropyLogger


# -----------------------------------------------------------------------------
# Global caches / knobs
# -----------------------------------------------------------------------------
# Plasticity updates are noisy when rewards are near-zero.
PLASTICITY_REWARD_EPSILON: float = 0.05

# Cache TorchBrain instances by deterministic genome signature to avoid repeated
# torch.compile / graph build overhead across identical genomes.
_BRAIN_CACHE: Dict[str, "TorchBrain"] = {}


def get_cached_brain(genome: EvolvableGenome) -> "TorchBrain":
    """Return a cached TorchBrain keyed by genome.signature."""
    sig = getattr(genome, "signature", None)
    if sig is None:
        # Fallback: no signature property available
        sig = str(id(genome))
    brain = _BRAIN_CACHE.get(sig)
    if brain is None:
        brain = TorchBrain(genome)
        _BRAIN_CACHE[sig] = brain

    # Point the brain at the requesting genome (same signature => same params).
    brain.genome = genome
    brain.meta = genome.meta
    genome._torch_brain = brain
    return brain


class DynamicLinearLayer(nn.Module):
    """Dynamic linear layer that can be reconfigured"""
    
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self, method: str = "xavier_uniform"):
        """Initialize weights with different strategies"""
        if method == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight)
        elif method == "xavier_normal":
            nn.init.xavier_normal_(self.weight)
        elif method == "he_uniform":
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        elif method == "he_normal":
            nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        elif method == "lecun_uniform":
            limit = np.sqrt(3.0 / self.input_dim)
            nn.init.uniform_(self.weight, -limit, limit)
        elif method == "lecun_normal":
            std = 1.0 / np.sqrt(self.input_dim)
            nn.init.normal_(self.weight, 0, std)
        else:
            nn.init.normal_(self.weight, 0, 0.01)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return F.linear(x, self.weight, self.bias)
    
    def copy_weights_from_numpy(self, weights: np.ndarray, bias: Optional[np.ndarray] = None):
        """Copy weights from numpy arrays"""
        with torch.no_grad():
            # Transpose for PyTorch format: (output_dim, input_dim)
            self.weight.copy_(torch.tensor(weights.T, dtype=torch.float32))
            if self.bias is not None and bias is not None:
                self.bias.copy_(torch.tensor(bias, dtype=torch.float32))


class DynamicBatchNorm(nn.Module):
    """Dynamic batch normalization layer"""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Momentum for running stats
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        if self.training:
            # Calculate batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                self.num_batches_tracked += 1

            # Normalize using batch statistics
            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # Normalize using running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

    def copy_params_from_numpy(
        self,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray
    ):
        """Copy parameters from numpy arrays"""
        with torch.no_grad():
            gamma_t = torch.tensor(gamma, dtype=torch.float32)
            beta_t = torch.tensor(beta, dtype=torch.float32)
            rm_t = torch.tensor(running_mean, dtype=torch.float32)
            rv_t = torch.tensor(running_var, dtype=torch.float32)

            if gamma_t.numel() != self.num_features or beta_t.numel() != self.num_features:
                gamma_fixed = torch.ones(self.num_features, dtype=torch.float32)
                beta_fixed = torch.zeros(self.num_features, dtype=torch.float32)
                n = min(self.num_features, int(gamma_t.numel()), int(beta_t.numel()))
                if n > 0:
                    gamma_fixed[:n].copy_(gamma_t.flatten()[:n])
                    beta_fixed[:n].copy_(beta_t.flatten()[:n])
                gamma_t, beta_t = gamma_fixed, beta_fixed

            if rm_t.numel() != self.num_features or rv_t.numel() != self.num_features:
                rm_fixed = torch.zeros(self.num_features, dtype=torch.float32)
                rv_fixed = torch.ones(self.num_features, dtype=torch.float32)
                n = min(self.num_features, int(rm_t.numel()), int(rv_t.numel()))
                if n > 0:
                    rm_fixed[:n].copy_(rm_t.flatten()[:n])
                    rv_fixed[:n].copy_(rv_t.flatten()[:n])
                rm_t, rv_t = rm_fixed, rv_fixed

            self.gamma.copy_(gamma_t)
            self.beta.copy_(beta_t)
            self.running_mean.copy_(rm_t)
            self.running_var.copy_(rv_t)


class DynamicLayerNorm(nn.Module):
    """Dynamic layer normalization layer"""

    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Calculate layer statistics (across feature dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

    def copy_params_from_numpy(self, gamma: np.ndarray, beta: np.ndarray):
        """Copy parameters from numpy arrays"""
        with torch.no_grad():
            gamma_t = torch.tensor(gamma, dtype=torch.float32)
            beta_t = torch.tensor(beta, dtype=torch.float32)

            if gamma_t.numel() != self.num_features or beta_t.numel() != self.num_features:
                gamma_fixed = torch.ones(self.num_features, dtype=torch.float32)
                beta_fixed = torch.zeros(self.num_features, dtype=torch.float32)
                n = min(self.num_features, int(gamma_t.numel()), int(beta_t.numel()))
                if n > 0:
                    gamma_fixed[:n].copy_(gamma_t.flatten()[:n])
                    beta_fixed[:n].copy_(beta_t.flatten()[:n])
                gamma_t, beta_t = gamma_fixed, beta_fixed

            self.gamma.copy_(gamma_t)
            self.beta.copy_(beta_t)


class MLPBlock(nn.Module):
    """Multi-layer perceptron block with evolvable architecture"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)
        self.num_layers = num_layers

        # Build MLP layers
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer
                layers.extend([
                    nn.Linear(current_dim, output_dim),
                    nn.ReLU()
                ])
            else:
                # Hidden layers
                layers.extend([
                    nn.Linear(current_dim, self.hidden_dim),
                    nn.ReLU()
                ])
                current_dim = self.hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ResBlock(nn.Module):
    """Residual block with evolvable architecture"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or max(input_dim, output_dim)

        # Main path
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )

        # Shortcut connection (projection if dimensions don't match)
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main_path(x)
        shortcut = self.shortcut(x)
        return main + shortcut


class GRUBlock(nn.Module):
    """GRU block with evolvable memory size"""

    def __init__(self, input_dim: int, hidden_dim: int, memory_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size

        self.gru = nn.GRU(input_dim, memory_size, batch_first=True)
        self.output_layer = nn.Linear(memory_size, hidden_dim)

        # Initialize hidden state
        self.register_buffer('hidden', torch.zeros(1, 1, memory_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        # Add sequence dimension for GRU
        x_seq = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # Ensure hidden state matches batch size
        batch_size = x_seq.size(0)
        if self.hidden.size(1) != batch_size:
            self.hidden = torch.zeros(1, batch_size, self.memory_size, device=self.hidden.device, dtype=self.hidden.dtype)

        # GRU forward pass
        out, hidden = self.gru(x_seq, self.hidden)
        self.hidden = hidden.detach()

        # Remove sequence dimension and apply output layer
        out = out.squeeze(1)  # (batch_size, memory_size)
        return self.output_layer(out)  # (batch_size, hidden_dim)

    def reset_hidden(self):
        """Reset hidden state"""
        self.hidden.zero_()


class AttentionBlock(nn.Module):
    """Simple attention block with evolvable memory size"""

    def __init__(self, input_dim: int, output_dim: int, memory_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_size = memory_size

        # Key, Query, Value projections
        self.key_proj = nn.Linear(input_dim, memory_size)
        self.query_proj = nn.Linear(input_dim, memory_size)
        self.value_proj = nn.Linear(input_dim, memory_size)

        # Output projection
        self.output_proj = nn.Linear(memory_size, output_dim)

        # Memory buffer for attention
        self.register_buffer('memory', torch.zeros(1, memory_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)

        # Compute attention
        keys = self.key_proj(x)      # (batch_size, memory_size)
        queries = self.query_proj(x) # (batch_size, memory_size)
        values = self.value_proj(x)  # (batch_size, memory_size)

        # Simple dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.memory_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention
        attended = torch.matmul(attention_weights, values)

        # Update memory (simple average)
        self.memory = (0.9 * self.memory + 0.1 * attended.mean(dim=0, keepdim=True)).detach()

        # Output projection
        return self.output_proj(attended)


class PlasticLinear(torch.nn.Module):
    plastic_norms = []  # Class variable to collect norms per generation
    recovery_times = []  # DIAGNOSTIC 3: Recovery times per generation
    meta_gene_entropies = []  # DIAGNOSTIC 5: Meta-gene entropies per generation
    plasticity_weight_variances = []  # DIAGNOSTIC 5: Plasticity weight variances per generation
    learning_rate_entropies = []  # DIAGNOSTIC 5: Learning-rate gene entropies per generation
    plastic_neuron_fractions = []  # DIAGNOSTIC 5: Fraction of plastic neurons per generation

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True, learning_rule_net=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.learning_rule_net = learning_rule_net  # Per-layer learning rule net
        self.timestep = 0  # Track timestep for learning rule

        self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)

        self.register_buffer(
            "plastic_weight",
            torch.zeros(output_dim, input_dim)
        )
        self.plastic_delta_norm = 0.0

        # Per-episode tracking
        self.episode_delta_norms = []
        self.episode_rewards = []

        # DIAGNOSTIC 2: Plastic Weight Activation Timing
        self.plastic_delta_logs = []  # List of ||ΔW|| per timestep

        # Milestone 6: Stability monitoring and adaptive controls
        self.drift_budget = 1.0  # Per-layer drift budget
        self.observed_drift = 0.0  # Running average of observed drift
        self.adaptive_clamp = 0.01  # Adaptive clamp based on observed drift

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with small random values"""
        nn.init.xavier_uniform_(self.base_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def reset(self):
        self.plastic_weight.zero_()

    def reset_episode_tracking(self):
        """Reset per-episode tracking lists"""
        self.episode_delta_norms = []
        self.episode_rewards = []
        self.plastic_delta_logs = []  # Clear to prevent unbounded growth

    def get_episode_data(self) -> Dict[str, List[float]]:
        """Get episode data for plotting"""
        return {
            "delta_norms": self.episode_delta_norms.copy(),
            "rewards": self.episode_rewards.copy()
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.add(self.base_weight, self.plastic_weight)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        self.last_input = x.detach()
        self.last_output = out.detach()
        return out

    def apply_plasticity(self, reward: float, meta: Dict[str, float], state: Optional[torch.Tensor] = None):
        """META-3.2 plastic update with learned function ΔW = f(pre, post, reward, w, t)"""
        # Gate: ignore tiny reward signals (noise-dominated).
        if abs(reward) <= PLASTICITY_REWARD_EPSILON:
            return

        # Track plastic delta before update
        plastic_norm_before = torch.norm(self.plastic_weight).item()

        # META reward modulation - remove tanh to strengthen signal
        r = reward * meta["reward_gain"] + meta["reward_bias"]
        r = torch.tensor(r, device=self.plastic_weight.device)

        # Use per-layer learning rule net to compute ΔW directly
        if self.learning_rule_net is not None:
            # Prepare inputs for the net
            pre = self.last_input.mean(dim=0)  # Average over batch
            post = self.last_output.mean(dim=0)  # Average over batch
            w_flat = (self.base_weight + self.plastic_weight).flatten()  # Current weights flattened
            t = torch.tensor([self.timestep], dtype=torch.float32, device=self.plastic_weight.device)

            # Compute ΔW directly
            delta_w_flat = self.learning_rule_net(pre, post, r, w_flat, t)
            delta_w = delta_w_flat.view(self.output_dim, self.input_dim)

            # Hard-clip plastic updates per step
            delta_w = torch.clamp(delta_w, -0.01, 0.01)

            # Scale by plastic_lr
            delta_w *= meta["plastic_lr"]

            # Normalize by presynaptic activity
            delta_w = delta_w / (torch.norm(pre) + 1e-6)

            # Apply update
            self.plastic_weight += delta_w

            # Plasticity clipping - balanced range to allow learning while preventing runaway plasticity
            self.plastic_weight = torch.clamp(
                self.plastic_weight,
                min=-1.0,
                max=1.0
            )

            # Increment timestep
            self.timestep += 1

            # Track plastic delta after update
            plastic_norm_after = torch.norm(self.plastic_weight).item()
            self.plastic_delta_norm = plastic_norm_after - plastic_norm_before

            # Log per-weight RMS norm to keep values size-invariant
            num_weights = max(1, self.plastic_weight.numel())
            plastic_rms_norm = plastic_norm_after / math.sqrt(float(num_weights))
            PlasticLinear.plastic_norms.append(plastic_rms_norm)

            # Per-episode tracking
            delta_norm = torch.norm(delta_w).item()
            self.episode_delta_norms.append(delta_norm)
            self.episode_rewards.append(r.item())

            # DIAGNOSTIC 2: Log ||ΔW|| per timestep
            self.plastic_delta_logs.append(delta_norm)
        else:
            # Fallback: no learning rule net, skip update
            pass


class PlasticityRegularizer:
    """Prevent runaway plasticity with multiple safeguards"""
    def __init__(self, max_delta=0.5, saturation_threshold=0.9):
        self.max_delta = max_delta
        self.saturation_threshold = saturation_threshold
        self.plasticity_history = deque(maxlen=100)

    def regulate(self, delta_w, current_w):
        # 1. Clip delta magnitude
        delta_w = torch.clamp(delta_w, -self.max_delta, self.max_delta)

        # 2. Check saturation
        saturation = (torch.abs(current_w) > self.saturation_threshold).float().mean()
        if saturation > 0.8:
            delta_w *= 0.1  # Reduce plasticity if saturated

        # 3. Adaptive dampening based on history
        self.plasticity_history.append(torch.abs(delta_w).mean().item())
        if len(self.plasticity_history) > 10:
            recent_trend = np.mean(list(self.plasticity_history)[-10:])
            if recent_trend > self.max_delta * 0.5:
                delta_w *= 0.5  # Dampen if trending high

        return delta_w


class ActivationMonitor:
    """Real-time activation health monitoring"""
    def __init__(self):
        self.saturation_thresholds = {
            'tanh': 0.95,
            'sigmoid': 0.95,
            'relu': 1000.0
        }

    def check_layer(self, activations, activation_fn):
        # Dead neurons (always zero)
        dead_ratio = (torch.abs(activations) < 1e-6).float().mean()

        # Saturated neurons (at limits)
        threshold = self.saturation_thresholds.get(activation_fn, 0.95)
        saturated_ratio = (torch.abs(activations) > threshold).float().mean()

        return {
            'dead_ratio': dead_ratio.item(),
            'saturated_ratio': saturated_ratio.item(),
            'mean_activation': activations.mean().item(),
            'std_activation': activations.std().item()
        }


class NeuralHealthController:
    """Neural Health Controller - turns dead neurons into evolutionary pressure"""

    def __init__(self):
        self.dead_neuron_threshold = 0.1  # 10% dead neurons triggers intervention
        self.saturation_threshold = 0.5   # 50% saturation triggers intervention
        self.health_history = []  # Track health over episodes
        self.mutation_triggers = {
            'skip_connection': False,
            'activation_mutation': False,
            'layer_reinit': False,
            'architecture_change': False
        }

    def assess_network_health(self, brain: "TorchBrain") -> Dict[str, Any]:
        """Assess overall neural health of the network"""
        if not hasattr(brain, 'activation_stats') or not brain.activation_stats:
            return {'healthy': True, 'dead_layers': 0, 'saturated_layers': 0}

        dead_layers = 0
        saturated_layers = 0
        total_dead_ratio = 0.0
        total_saturated_ratio = 0.0

        for stats in brain.activation_stats:
            if stats.get('dead_ratio', 0) > self.dead_neuron_threshold:
                dead_layers += 1
            if stats.get('saturated_ratio', 0) > self.saturation_threshold:
                saturated_layers += 1
            total_dead_ratio += stats.get('dead_ratio', 0)
            total_saturated_ratio += stats.get('saturated_ratio', 0)

        avg_dead_ratio = total_dead_ratio / len(brain.activation_stats) if brain.activation_stats else 0
        avg_saturated_ratio = total_saturated_ratio / len(brain.activation_stats) if brain.activation_stats else 0

        # Determine if network needs intervention
        needs_intervention = (
            dead_layers > 0 or
            saturated_layers > len(brain.activation_stats) * 0.3 or
            avg_dead_ratio > 0.05 or
            avg_saturated_ratio > 0.3
        )

        health_score = 1.0 - min(1.0, (avg_dead_ratio * 2.0 + avg_saturated_ratio))

        health_data = {
            'healthy': not needs_intervention,
            'health_score': health_score,
            'dead_layers': dead_layers,
            'saturated_layers': saturated_layers,
            'avg_dead_ratio': avg_dead_ratio,
            'avg_saturated_ratio': avg_saturated_ratio,
            'needs_intervention': needs_intervention
        }

        self.health_history.append(health_data)

        # Keep only recent history
        if len(self.health_history) > 10:
            self.health_history = self.health_history[-10:]

        return health_data

    def get_fitness_penalty(self, brain: "TorchBrain") -> float:
        """Calculate fitness penalty based on neural health"""
        health_data = self.assess_network_health(brain)

        if health_data['healthy']:
            return 0.0

        # Heavy penalty for dead neurons (multiplicative)
        dead_penalty = 1.0
        if health_data['dead_layers'] > 0:
            dead_penalty = 0.5 ** health_data['dead_layers']  # Exponential decay

        # Penalty for saturation
        saturation_penalty = max(0.0, health_data['avg_saturated_ratio'] - 0.2) * 0.1

        # Overall health score penalty
        health_penalty = (1.0 - health_data['health_score']) * 0.2

        total_penalty = dead_penalty * (1.0 - saturation_penalty - health_penalty)

        return max(0.1, total_penalty)  # Minimum fitness floor

    def trigger_mutations(self, brain: "TorchBrain", genome) -> Dict[str, bool]:
        """Determine which mutations to trigger based on neural health"""
        health_data = self.assess_network_health(brain)

        # Reset triggers
        self.mutation_triggers = {k: False for k in self.mutation_triggers}

        if not health_data['needs_intervention']:
            return self.mutation_triggers

        # Trigger mutations based on specific issues
        if health_data['dead_layers'] > 0:
            self.mutation_triggers['skip_connection'] = True
            self.mutation_triggers['layer_reinit'] = True

        if health_data['saturated_layers'] > len(brain.activation_stats) * 0.5:
            self.mutation_triggers['activation_mutation'] = True

        if health_data['avg_dead_ratio'] > 0.2:
            self.mutation_triggers['architecture_change'] = True

        return self.mutation_triggers

    def get_health_summary(self, brain: "TorchBrain") -> str:
        """Get human-readable health summary"""
        health_data = self.assess_network_health(brain)

        if health_data['healthy']:
            return "Neural health: Good"

        issues = []
        if health_data['dead_layers'] > 0:
            issues.append(f"{health_data['dead_layers']} dead layers")
        if health_data['saturated_layers'] > 0:
            issues.append(f"{health_data['saturated_layers']} saturated layers")

        health_pct = health_data['health_score'] * 100
        return f"Neural health: {health_pct:.1f}% ({', '.join(issues)})"


class CollapseDetector:
    """Detect and prevent network collapse"""
    def __init__(self):
        self.collapse_indicators = []
        self.collapse_history = []  # Track collapse events over time
        self.recovery_metrics = {
            'total_collapses': 0,
            'successful_recoveries': 0,
            'average_recovery_time': 0.0
        }

    def check_collapse(self, brain):
        indicators = []

        # Check gradient flow
        grad_norm = sum(p.grad.norm() for p in brain.parameters() if p.grad is not None)
        if grad_norm < 1e-8:
            indicators.append('gradient_vanishing')

        # Check output diversity
        with torch.no_grad():
            # Sample a few random inputs to check output variance
            sample_inputs = torch.randn(10, brain.input_size, device=next(brain.parameters()).device)
            outputs = brain(sample_inputs)
            output_std = outputs.std(dim=0).mean().item()
            if output_std < 1e-6:
                indicators.append('output_collapse')

        # Check weight explosion
        total_weight_norm = sum(p.norm() for p in brain.parameters())
        if total_weight_norm > 100.0:  # Arbitrary threshold
            indicators.append('weight_explosion')

        # Check for dead layers using activation stats
        if hasattr(brain, 'activation_stats') and brain.activation_stats:
            dead_layers = sum(1 for stats in brain.activation_stats if stats['dead_ratio'] > 0.8)
            if dead_layers > len(brain.activation_stats) * 0.5:
                indicators.append('layer_death')

        # Check for saturation
        if hasattr(brain, 'activation_stats') and brain.activation_stats:
            saturated_layers = sum(1 for stats in brain.activation_stats if stats['saturated_ratio'] > 0.8)
            if saturated_layers > len(brain.activation_stats) * 0.5:
                indicators.append('activation_saturation')

        self.collapse_indicators = indicators
        is_collapsed = len(indicators) > 0

        if is_collapsed:
            self.collapse_history.append({
                'timestep': len(self.collapse_history),
                'indicators': indicators.copy(),
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'output_std': output_std,
                'weight_norm': total_weight_norm.item() if isinstance(total_weight_norm, torch.Tensor) else total_weight_norm
            })
            self.recovery_metrics['total_collapses'] += 1
            brain.logger.warning(f"Network collapse detected: {indicators}")

        return is_collapsed, indicators

    def auto_fix(self, brain, indicators):
        """Automatic intervention"""
        recovery_start = len(self.collapse_history) - 1

        if 'gradient_vanishing' in indicators:
            # Re-initialize problematic layers
            for layer in brain.layers:
                if isinstance(layer, PlasticLinear):
                    layer.reset_parameters()
                    layer.reset()
            brain.logger.info("Re-initialized layers due to gradient vanishing")

        if 'weight_explosion' in indicators:
            # Scale down weights
            for param in brain.parameters():
                param.data *= 0.1
            brain.logger.info("Scaled down weights due to explosion")

        if 'output_collapse' in indicators or 'layer_death' in indicators:
            # Add noise to weights
            for param in brain.parameters():
                noise = torch.randn_like(param) * 0.01
                param.data += noise
            brain.logger.info("Added noise to weights due to output collapse")

        if 'activation_saturation' in indicators:
            # Adjust meta parameters if available
            if hasattr(brain, 'meta'):
                brain.meta['reward_gain'] *= 0.9
                brain.meta['plastic_lr'] *= 0.9
            brain.logger.info("Reduced learning rates due to activation saturation")

        # Check recovery after fix
        is_still_collapsed, _ = self.check_collapse(brain)
        if not is_still_collapsed:
            recovery_time = len(self.collapse_history) - recovery_start
            self.recovery_metrics['successful_recoveries'] += 1
            self.recovery_metrics['average_recovery_time'] = (
                (self.recovery_metrics['average_recovery_time'] * (self.recovery_metrics['successful_recoveries'] - 1) + recovery_time)
                / self.recovery_metrics['successful_recoveries']
            )
            brain.logger.info(f"Recovery successful in {recovery_time} checks")

    def get_recovery_metrics(self):
        """Get recovery metrics"""
        return self.recovery_metrics.copy()


class TorchBrain(nn.Module):
    """
    PyTorch neural network compiler that builds networks from genomes
    Supports dynamic architecture compilation
    """
    
    def __init__(self, genome: Optional[EvolvableGenome] = None):
        super().__init__()
        self.genome = genome
        self.input_size = genome.input_size if genome else 6
        self.output_size = genome.output_size if genome else 4

        # Track inference compilation state
        self._compiled_for_inference: bool = False
        self._compiled_device: Optional[str] = None
        self.device = torch.device('cpu')

        # META parameters from genome
        self.meta = genome.meta if genome else {
            "reward_gain": 1.0,
            "reward_bias": 0.0
        }

        # Network components
        self.layers: nn.ModuleList = nn.ModuleList()
        self.activations: List[str] = []
        self.skip_connections: List[tuple[int, int]] = []  # List of (from_layer, to_layer) tuples
        self.layer_types: List[str] = []

        # Milestone 6: Activation monitoring hooks
        self.activation_stats: List[Dict[str, float]] = []  # Per-layer activation statistics
        self.layer_saturation_fractions: List[float] = []  # Fraction of saturated activations per layer
        self.layer_dead_unit_fractions: List[float] = []  # Fraction of dead units per layer

        # Activation monitor
        self.activation_monitor = ActivationMonitor()
        self.logger = logging.getLogger('TorchBrain')

        # Collapse prevention system
        self.collapse_detector = CollapseDetector()

        # Build network if genome is provided
        if genome is not None:
            self.build_from_genome(genome)
    
    def build_from_genome(self, genome: EvolvableGenome):
        """Build PyTorch network from genome architecture"""
        self.genome = genome
        self.input_size = genome.input_size
        self.output_size = genome.output_size

        # Clear existing components
        self.layers = nn.ModuleList()
        self.activations = []
        self.skip_connections = []
        self.layer_types = []

        # Milestone 5: Check if genome has modular architecture
        if hasattr(genome, 'modules') and genome.modules:
            # Build modular architecture
            self._build_modular_architecture(genome)
            # Fallback: also build linear layers for forward compatibility
            if not self.layers:
                self._build_linear_architecture(genome)
        else:
            # Build legacy linear architecture
            self._build_linear_architecture(genome)

        # Compile forward graph
        self._compile_forward_graph()

    def _build_modular_architecture(self, genome: EvolvableGenome):
        """Build PyTorch network from modular genome architecture"""
        # Create module layers
        self.module_layers: nn.ModuleList = nn.ModuleList()
        self.module_connections = genome.module_connections.copy()
        self.execution_order = genome.get_execution_order()

        for module in genome.modules:
            # Create a sequential module for each genome module
            module_seq = nn.Sequential()

            for gene in module.genes:
                layer = self._create_layer_from_gene(gene)
                module_seq.append(layer)

            self.module_layers.append(module_seq)

    def _build_linear_architecture(self, genome: EvolvableGenome):
        """Build legacy linear architecture from genome genes"""
        # Build each layer from genome genes
        for i, gene in enumerate(genome.genes):
            layer = self._create_layer_from_gene(gene)
            layer_type = gene.layer_type

            self.layers.append(layer)
            self.layer_types.append(layer_type)

            # Store activation function
            self.activations.append(gene.activation)

            # Store skip connection if present
            if gene.skip_connection and gene.skip_target >= 0:
                self.skip_connections.append((gene.skip_target, i))

        # Create batch normalization layers if needed
        self.batch_norms: List[Optional[DynamicBatchNorm]] = []
        self.layer_norms: List[Optional[DynamicLayerNorm]] = []
        for gene in genome.genes:
            if gene.normalization_type == "batchnorm" or gene.batch_norm:
                bn_layer = DynamicBatchNorm(gene.output_dim)
                if gene.bn_gamma is not None and gene.bn_beta is not None and gene.bn_running_mean is not None and gene.bn_running_var is not None:
                    bn_layer.copy_params_from_numpy(
                        gene.bn_gamma,
                        gene.bn_beta,
                        gene.bn_running_mean,
                        gene.bn_running_var
                    )
                self.batch_norms.append(bn_layer)
                self.layer_norms.append(None)
            elif gene.normalization_type == "layernorm":
                ln_layer = DynamicLayerNorm(gene.output_dim)
                if gene.ln_gamma is not None and gene.ln_beta is not None:
                    ln_layer.copy_params_from_numpy(
                        gene.ln_gamma,
                        gene.ln_beta
                    )
                self.batch_norms.append(None)
                self.layer_norms.append(ln_layer)
            else:
                self.batch_norms.append(None)
                self.layer_norms.append(None)

        # Create dropout layers if needed
        self.dropouts: List[Optional[nn.Dropout]] = []
        for gene in genome.genes:
            if gene.dropout_rate > 0:
                self.dropouts.append(nn.Dropout(gene.dropout_rate))
            else:
                self.dropouts.append(None)

    def _create_layer_from_gene(self, gene) -> nn.Module:
        """Create a PyTorch layer from a NeuralGene"""
        layer_type = gene.layer_type

        if layer_type == "linear":
            # Create plastic linear layer
            layer = PlasticLinear(
                input_dim=gene.input_dim,
                output_dim=gene.output_dim,
                use_bias=gene.use_bias,
                learning_rule_net=gene.learning_rule_net
            )

            # Copy weights from genome (stored as out x in)
            if gene.weights is not None:
                layer.base_weight.data.copy_(torch.tensor(gene.weights, dtype=torch.float32))
                if gene.bias is not None and layer.bias is not None:
                    layer.bias.data.copy_(torch.tensor(gene.bias, dtype=torch.float32))

            # Seed plastic weight buffer from genome plasticity gene (Hebbian strength)
            if gene.plasticity is not None:
                layer.plastic_weight.copy_(torch.tensor(gene.plasticity, dtype=torch.float32))

        elif layer_type == "mlp_block":
            # Create MLP block
            layer = MLPBlock(
                input_dim=gene.input_dim,
                output_dim=gene.output_dim,
                hidden_dim=max(gene.input_dim, gene.output_dim),
                num_layers=2
            )

        elif layer_type == "res_block":
            # Create residual block
            layer = ResBlock(
                input_dim=gene.input_dim,
                output_dim=gene.output_dim,
                hidden_dim=max(gene.input_dim, gene.output_dim)
            )

        elif layer_type == "gru":
            # Create GRU block
            layer = GRUBlock(
                input_dim=gene.input_dim,
                hidden_dim=gene.output_dim,
                memory_size=gene.memory_size
            )

        elif layer_type == "attention_block":
            # Create attention block
            layer = AttentionBlock(
                input_dim=gene.input_dim,
                output_dim=gene.output_dim,
                memory_size=gene.memory_size
            )

        else:
            # Fallback to plastic linear
            layer = PlasticLinear(
                input_dim=gene.input_dim,
                output_dim=gene.output_dim,
                use_bias=gene.use_bias,
                learning_rule_net=gene.learning_rule_net
            )

        return layer
    
    def _compile_forward_graph(self):
        """Compile forward pass graph for efficient execution"""
        genome = self.genome
        assert genome is not None, "Genome must be set before compiling forward graph"
        # This creates a cached version of the forward pass
        # We'll use a simple list of operations
        self.forward_ops = []

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            # Store operation: (layer_idx, activation_name, has_batch_norm, has_layer_norm, has_dropout)
            has_bn = self.batch_norms[i] is not None
            has_ln = self.layer_norms[i] is not None
            has_dropout = self.dropouts[i] is not None
            self.forward_ops.append({
                'layer_idx': i,
                'activation': activation,
                'has_bn': has_bn,
                'has_ln': has_ln,
                'has_dropout': has_dropout,
                'dropout_rate': genome.genes[i].dropout_rate if has_dropout else 0.0
            })
    
    def get_activation_fn(self, activation_name: str):
        """Get PyTorch activation function by name"""
        activation_map = {
            'tanh': torch.tanh,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'sigmoid': torch.sigmoid,
            'elu': F.elu,
            'selu': F.selu,
            'gelu': F.gelu,
            'softplus': F.softplus,
            'swish': lambda x: x * torch.sigmoid(x),
            'linear': lambda x: x,
            'sin': torch.sin,
            'cos': torch.cos,
        }
        return activation_map.get(activation_name, torch.tanh)

    def _monitor_activation_stats(self, activations: torch.Tensor, layer_idx: int, activation_name: str):
        """Monitor activation statistics for health diagnostics"""
        stats = self.activation_monitor.check_layer(activations, activation_name)
        self.activation_stats.append(stats)

        # SILENCED: Per-layer warnings spam the terminal and hide generation summaries
        # Dead/saturated neurons are now handled via fitness penalties in compute_fitness_from_metrics()
        # Evolution will naturally select against these pathologies
        # Uncomment for debugging specific genomes:
        # if stats['dead_ratio'] > 0.5:
        #     self.logger.warning(f"Layer {layer_idx}: High dead neuron ratio ({stats['dead_ratio']:.3f})")
        # if stats['saturated_ratio'] > 0.5:
        #     self.logger.warning(f"Layer {layer_idx}: High saturation ratio ({stats['saturated_ratio']:.3f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the compiled network"""
        if not self.layers:
            raise ValueError("TorchBrain has no layers. Call build_from_genome first.")

        # Reset activation monitoring for this forward pass
        self.activation_stats = []
        self.layer_saturation_fractions = []
        self.layer_dead_unit_fractions = []

        # Store activations for skip connections
        activations = [x]  # Store input as activation 0

        for op in self.forward_ops:
            layer_idx = op['layer_idx']
            current_input = activations[-1]

            # Apply skip connections if any
            for skip_from, skip_to in self.skip_connections:
                if skip_to == layer_idx and skip_from + 1 < len(activations):
                    skip_input = activations[skip_from + 1]
                    # Adjust dimensions if needed
                    if skip_input.shape[1] != current_input.shape[1]:
                        # Pad or truncate
                        if skip_input.shape[1] < current_input.shape[1]:
                            pad_width = current_input.shape[1] - skip_input.shape[1]
                            skip_input = F.pad(skip_input, (0, pad_width))
                        else:
                            skip_input = skip_input[:, :current_input.shape[1]]
                    current_input = current_input + skip_input

            # Linear layer
            x = self.layers[layer_idx](current_input)  # type: ignore

            # Batch normalization
            if op['has_bn'] and self.batch_norms[layer_idx] is not None:
                x = self.batch_norms[layer_idx](x)

            # Layer normalization
            if op['has_ln'] and self.layer_norms[layer_idx] is not None:
                x = self.layer_norms[layer_idx](x)

            # Activation
            activation_fn = self.get_activation_fn(op['activation'])
            x = activation_fn(x)  # type: ignore

            # Milestone 6: Monitor activation statistics
            self._monitor_activation_stats(x, layer_idx, op['activation'])

            # Dropout (only during training)
            if op['has_dropout'] and self.training:
                x = F.dropout(x, p=op['dropout_rate'])

            activations.append(x)

        return activations[-1]
    
    def act(self, state: np.ndarray) -> int:
        """Get action for single state"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            output = self.forward(state_tensor)
            return int(torch.argmax(output, dim=1).item())
    
    def act_batch(self, states: np.ndarray) -> np.ndarray:
        """Get actions for batch of states"""
        with torch.no_grad():
            if isinstance(states, np.ndarray):
                states_tensor = torch.tensor(states, dtype=torch.float32)
            else:
                states_tensor = states
            
            output = self.forward(states_tensor)
            return torch.argmax(output, dim=1).cpu().numpy()
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of network architecture"""
        if self.genome is None:
            return {"error": "No genome compiled"}
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_info = []
        for i, gene in enumerate(self.genome.genes):
            layer_info.append({
                'layer': i,
                'type': gene.layer_type,
                'input': gene.input_dim,
                'output': gene.output_dim,
                'activation': gene.activation,
                'bias': gene.use_bias,
                'batch_norm': gene.batch_norm,
                'dropout': gene.dropout_rate,
                'skip_from': gene.skip_target if gene.skip_connection else None,
            })
        
        return {
            'genome_id': self.genome.genome_id,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'total_layers': len(self.layers),
            'total_params': total_params,
            'skip_connections': len(self.skip_connections),
            'layers': layer_info,
        }
    
    def save_model(self, filepath: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'genome_dict': self.genome.to_dict() if self.genome else None,
            'architecture': self.get_architecture_summary(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu'):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=device)
        
        if checkpoint['genome_dict'] is None:
            raise ValueError("Saved model does not contain genome information")
        
        # Reconstruct genome
        genome = EvolvableGenome.from_dict(checkpoint['genome_dict'])
        
        # Create TorchBrain and build network
        model = cls(genome)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        print(f"Architecture: {model.get_architecture_summary()}")
        
        return model
    
    def compile_for_inference(self, device: str = 'cuda'):
        """Compile model for fast inference"""
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'

        if self._compiled_for_inference and self._compiled_device == device:
            return self
        
        self.to(device)
        self.eval()  # Set to evaluation mode
        
        # Use torch.compile if available (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.forward = torch.compile(self.forward, mode="reduce-overhead")
        except:
            pass  # Silently fall back to standard forward

        self._compiled_for_inference = True
        self._compiled_device = device
        
        return self
    
    def update_from_genome(self, genome: EvolvableGenome):
        """Update network weights from genome (without changing architecture)"""
        if len(genome.genes) != len(self.layers):
            raise ValueError("Genome architecture mismatch. Need to rebuild network.")

        # Update weights for each layer
        for i, (gene, layer) in enumerate(zip(genome.genes, self.layers)):
            if gene.weights is not None:
                if isinstance(layer, PlasticLinear):
                    # For PlasticLinear layers, update base_weight and bias directly
                    layer.base_weight.data.copy_(torch.tensor(gene.weights, dtype=torch.float32))
                    if gene.bias is not None and layer.bias is not None:
                        layer.bias.data.copy_(torch.tensor(gene.bias, dtype=torch.float32))
                    if gene.plasticity is not None:
                        layer.plastic_weight.copy_(torch.tensor(gene.plasticity, dtype=torch.float32))
                else:
                    # For other layer types (if any), use copy_weights_from_numpy
                    layer.copy_weights_from_numpy(gene.weights, gene.bias)  # type: ignore

            # Update batch norm if present
            if gene.batch_norm and self.batch_norms[i] is not None and gene.bn_gamma is not None and gene.bn_beta is not None and gene.bn_running_mean is not None and gene.bn_running_var is not None:
                self.batch_norms[i].copy_params_from_numpy(  # type: ignore
                    gene.bn_gamma,
                    gene.bn_beta,
                    gene.bn_running_mean,
                    gene.bn_running_var
                )

        print(f"Weights updated from genome {genome.genome_id}")

    def reset_state(self):
        """Reset state for all stateful layers"""
        for layer in self.layers:
            if isinstance(layer, GRUBlock):
                layer.reset_hidden()
            elif isinstance(layer, AttentionBlock):
                layer.memory.zero_()

    def reset_plasticity(self):
        """Reset plasticity for all plastic layers"""
        for layer in self.layers:
            if isinstance(layer, PlasticLinear):
                layer.reset()

    def reset_episode_tracking(self):
        """Reset episode tracking for all plastic layers"""
        for layer in self.layers:
            if isinstance(layer, PlasticLinear):
                layer.reset_episode_tracking()

    def get_episode_data(self) -> Dict[str, List[float]]:
        """Get episode data for plotting (averaged across plastic layers)"""
        plastic_layers = [layer for layer in self.layers if isinstance(layer, PlasticLinear)]
        if not plastic_layers:
            return {"delta_norms": [], "rewards": []}
 
        # Average across all plastic layers
        delta_norms = []
        rewards = []
        for layer in plastic_layers:
            data = layer.get_episode_data()
            delta_norms.extend(data["delta_norms"])
            rewards.extend(data["rewards"])

        # Take mean across layers for each step
        if delta_norms:
            num_steps = len(delta_norms) // len(plastic_layers)
            delta_norms = [float(np.mean(delta_norms[i::num_steps])) for i in range(num_steps)]
            rewards = [float(np.mean(rewards[i::num_steps])) for i in range(num_steps)]

        return {"delta_norms": delta_norms, "rewards": rewards}

    def update_plasticity(self, reward_signal: float, done: bool, state: Optional[torch.Tensor] = None, pre: Optional[torch.Tensor] = None, post: Optional[torch.Tensor] = None):
        """Update plasticity for all plastic layers with META-3.2 dynamic learning rule"""
        if done:
            return

        # Gate plasticity updates: only learn when reward is informative.
        if abs(float(reward_signal)) <= PLASTICITY_REWARD_EPSILON:
            return

        for layer in self.layers:
            if isinstance(layer, PlasticLinear):
                layer.apply_plasticity(reward_signal, self.meta, state)

    def get_plastic_diagnostics(self) -> Dict[str, Any]:
        """Get lifetime plasticity diagnostics"""
        plastic_layers = [layer for layer in self.layers if isinstance(layer, PlasticLinear)]

        if not plastic_layers:
            return {"plastic_layers": 0, "total_plastic_delta": 0.0, "mean_plastic_delta": 0.0}

        plastic_deltas = [layer.plastic_delta_norm for layer in plastic_layers]

        return {
            "plastic_layers": len(plastic_layers),
            "total_plastic_delta": float(sum(plastic_deltas)),
            "mean_plastic_delta": float(np.mean(plastic_deltas)),
            "max_plastic_delta": float(max(plastic_deltas)),
            "min_plastic_delta": float(min(plastic_deltas)),
            "plastic_deltas": plastic_deltas
        }

    def get_mean_plastic_norm(self) -> float:
        """Get the mean norm of plastic weights across all plastic layers"""
        plastic_layers = [layer for layer in self.layers if isinstance(layer, PlasticLinear)]
        if not plastic_layers:
            return 0.0
        norms = []
        for layer in plastic_layers:
            norm = torch.norm(layer.plastic_weight).item()
            num_weights = max(1, layer.plastic_weight.numel())
            norms.append(norm / math.sqrt(float(num_weights)))
        return float(np.mean(norms))

    def plot_plastic_weight_activation_timing(self, filename="diagnostics/plastic_weight_activation_timing.png"):
        """Plot DIAGNOSTIC 2: Plastic Weight Activation Timing - ||ΔW|| vs timestep per layer"""
        PlasticityTimingLogger.plot_activation_timing(filename)

    @staticmethod
    def calculate_recovery_time(pre_shock_reward: float, post_shock_rewards: List[float]) -> Optional[int]:
        """Calculate recovery time: min t where reward[t] >= 0.8 * reward_pre_shock"""
        threshold = 0.8 * pre_shock_reward
        for t, reward in enumerate(post_shock_rewards):
            if reward >= threshold:
                return t
        return None  # Did not recover

    def plot_learning_speed_compression(self, filename="diagnostics/learning_speed_compression.png"):
        """Plot DIAGNOSTIC 3: Learning Speed Compression - Recovery time vs generation"""
        RewardRecoveryLogger.plot_recovery_times(filename)

    @staticmethod
    def log_meta_gene_diagnostics(population: List['TorchBrain']):
        """Log DIAGNOSTIC 5: Meta-gene selection pressure metrics per generation"""
        if not population:
            return

        # PlasticityNet weight variance
        all_plasticity_weights = []
        for brain in population:
            for layer in brain.layers:
                if isinstance(layer, PlasticLinear):
                    all_plasticity_weights.extend(layer.plastic_weight.detach().flatten().tolist())

        if all_plasticity_weights:
            plasticity_weight_variance = float(np.var(all_plasticity_weights))
        else:
            plasticity_weight_variance = 0.0

        PlasticLinear.plasticity_weight_variances.append(plasticity_weight_variance)

        # Learning-rate gene entropy (using learning_rule_net parameters)
        learning_params = []
        for brain in population:
            if brain.genome and brain.genome.learning_rule_net:
                # Get some summary statistics from the learning rule net
                params = brain.genome.learning_rule_net.get_parameters_as_dict()
                param_values = [params[k] for k in ['mean_delta_w', 'std_delta_w', 'max_delta_w', 'min_delta_w']]
                learning_params.append(param_values)

        if learning_params:
            # Calculate entropy across population for each parameter
            param_arrays = np.array(learning_params).T  # Shape: (4, num_genomes)
            param_entropies = []
            for param_values in param_arrays:
                # Discretize into bins for entropy calculation
                hist, _ = np.histogram(param_values, bins=10, density=True)
                hist = hist[hist > 0]  # Remove zero probabilities
                entropy = -np.sum(hist * np.log(hist))
                param_entropies.append(entropy)
            learning_rate_entropy = float(np.mean(param_entropies))
        else:
            learning_rate_entropy = 0.0

        PlasticLinear.learning_rate_entropies.append(learning_rate_entropy)

        # Fraction of plastic neurons used
        plastic_fractions = []
        for brain in population:
            total_layers = len(brain.layers)
            plastic_layers = sum(1 for layer in brain.layers if isinstance(layer, PlasticLinear))
            fraction = plastic_layers / total_layers if total_layers > 0 else 0.0
            plastic_fractions.append(fraction)

        avg_plastic_fraction = float(np.mean(plastic_fractions)) if plastic_fractions else 0.0
        PlasticLinear.plastic_neuron_fractions.append(avg_plastic_fraction)

        # Meta-gene entropy (reward_gain, reward_bias)
        meta_params = []
        for brain in population:
            if brain.genome and brain.genome.meta:
                params = [brain.genome.meta.get('reward_gain', 1.0),
                         brain.genome.meta.get('reward_bias', 0.0)]
                meta_params.append(params)

        if meta_params:
            param_arrays = np.array(meta_params).T  # Shape: (2, num_genomes)
            param_entropies = []
            for param_values in param_arrays:
                hist, _ = np.histogram(param_values, bins=10, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log(hist))
                param_entropies.append(entropy)
            meta_gene_entropy = float(np.mean(param_entropies))
        else:
            meta_gene_entropy = 0.0

        MetaGeneEntropyLogger.log_generation_meta_entropy(
            meta_gene_entropy,
            plasticity_weight_variance,
            learning_rate_entropy,
            avg_plastic_fraction
        )

    def plot_meta_gene_selection_pressure(self, filename="diagnostics/meta_gene_selection_pressure.png"):
        """Plot DIAGNOSTIC 5: Meta-gene Selection Pressure - Meta-gene entropy vs generation"""
        MetaGeneEntropyLogger.plot_meta_gene_entropy(filename)

    def check_and_fix_collapse(self):
        """Check for network collapse and apply automatic fixes if detected"""
        is_collapsed, indicators = self.collapse_detector.check_collapse(self)
        if is_collapsed:
            self.collapse_detector.auto_fix(self, indicators)
        return is_collapsed, indicators

class ParallelTorchBrain(nn.Module):
    """
    Parallel neural network that can run multiple genomes simultaneously
    Useful for batch evaluation on GPU
    """
    
    def __init__(self, genomes: List[EvolvableGenome]):
        super().__init__()
        self.genomes = genomes
        self.num_genomes = len(genomes)
        
        # Find maximum dimensions for padding
        self.max_input = max(g.input_size for g in genomes)
        self.max_output = max(g.output_size for g in genomes)
        self.max_layers = max(len(g.genes) for g in genomes)
        
        # Create individual TorchBrains (using cached get_brain())
        self.brains: nn.ModuleList = nn.ModuleList([g.get_brain() for g in genomes])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for all genomes
        x: shape (batch_size * num_genomes, input_size)
        Returns: shape (batch_size * num_genomes, output_size)
        """
        # Split input for each genome
        batch_size = x.shape[0] // self.num_genomes
        outputs = []
        
        for i, brain in enumerate(self.brains):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            x_i = x[start_idx:end_idx]
            
            # Pad input if necessary
            if x_i.shape[1] < brain.input_size:  # type: ignore
                pad_width = int(brain.input_size - x_i.shape[1])  # type: ignore
                x_i = F.pad(x_i, (0, pad_width))
            elif x_i.shape[1] > brain.input_size:  # type: ignore
                x_i = x_i[:, :brain.input_size]
            
            # Forward pass
            out_i = brain(x_i)  # type: ignore
            
            # Pad output to max_output if necessary
            if out_i.shape[1] < self.max_output:
                pad_width = int(self.max_output - out_i.shape[1])
                out_i = F.pad(out_i, (0, pad_width))
            
            outputs.append(out_i)
        
        return torch.cat(outputs, dim=0)
    
    def act_batch_parallel(self, states: np.ndarray) -> np.ndarray:
        """
        Get actions for all genomes in parallel
        states: shape (batch_size * num_genomes, input_size)
        Returns: shape (batch_size * num_genomes,)
        """
        with torch.no_grad():
            states_tensor = torch.tensor(states, dtype=torch.float32)
            outputs = self.forward(states_tensor)
            
            # Get actions (argmax)
            actions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Truncate actions to actual output sizes
            for i, brain in enumerate(self.brains):
                batch_size = states.shape[0] // self.num_genomes
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                
                # Actions are already correct dimension due to padding
                pass
            
            return actions
    
    def compile_for_inference(self, device: str = 'cuda'):
        """Compile all brains for inference"""
        for brain in self.brains:
            brain.compile_for_inference(device)  # type: ignore
        self.to(device)
        return self