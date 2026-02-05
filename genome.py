import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import hashlib
import math
import ast
from dataclasses import dataclass, field
from collections import defaultdict, deque


@dataclass
class Module:
    """A modular building block that can be reused across genomes"""
    module_id: str
    genes: List[Any] = field(default_factory=list)  # Will be NeuralGene instances
    input_dim: int = 0
    output_dim: int = 0
    module_type: str = "linear_block"  # "linear_block", "residual_block", "attention_block", etc.

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the module"""
        if not self.genes:
            return x

        activations = [x]
        for gene in self.genes:
            current_input = activations[-1]

            # Handle skip connections within module
            if gene.skip_connection and gene.skip_target >= 0 and gene.skip_target < len(activations):
                skip_input = activations[gene.skip_target]
                if skip_input.shape[1] != gene.input_dim:
                    if skip_input.shape[1] < gene.input_dim:
                        pad_width = gene.input_dim - skip_input.shape[1]
                        skip_input = np.pad(skip_input, ((0, 0), (0, pad_width)), mode='constant')
                    else:
                        skip_input = skip_input[:, :gene.input_dim]
                current_input = gene.skip_gate * current_input + (1 - gene.skip_gate) * skip_input

            output = gene.forward(current_input, training)
            activations.append(output)

        return activations[-1]

    def copy(self) -> 'Module':
        """Create a deep copy"""
        return Module(
            module_id=self.module_id,
            genes=[gene.copy() for gene in self.genes],
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            module_type=self.module_type
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert module to dictionary"""
        return {
            'module_id': self.module_id,
            'genes': [gene.to_dict() for gene in self.genes],
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'module_type': self.module_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Module':
        """Create module from dictionary"""
        return cls(
            module_id=data['module_id'],
            genes=[NeuralGene.from_dict(g) for g in data['genes']],
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            module_type=data.get('module_type', 'linear_block')
        )


@dataclass
class Motif:
    """A discovered architectural pattern that can be reused"""
    motif_id: str
    modules: List[Module] = field(default_factory=list)
    connections: List[Tuple[int, int]] = field(default_factory=list)  # (from_module, to_module)
    fitness_score: float = 0.0
    usage_count: int = 0
    discovery_generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert motif to dictionary"""
        return {
            'motif_id': self.motif_id,
            'modules': [m.to_dict() for m in self.modules],
            'connections': self.connections,
            'fitness_score': self.fitness_score,
            'usage_count': self.usage_count,
            'discovery_generation': self.discovery_generation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Motif':
        """Create motif from dictionary"""
        return cls(
            motif_id=data['motif_id'],
            modules=[Module.from_dict(m) for m in data['modules']],
            connections=data['connections'],
            fitness_score=data.get('fitness_score', 0.0),
            usage_count=data.get('usage_count', 0),
            discovery_generation=data.get('discovery_generation', 0)
        )


class MotifLibrary:
    """Library of discovered motifs for reuse"""
    def __init__(self, max_motifs: int = 50):
        self.motifs: List[Motif] = []
        self.max_motifs = max_motifs
        self.motif_usage_stats: Dict[str, int] = defaultdict(int)

    def add_motif(self, motif: Motif):
        """Add a motif to the library"""
        if len(self.motifs) >= self.max_motifs:
            # Remove least used motif
            least_used = min(self.motifs, key=lambda m: self.motif_usage_stats.get(m.motif_id, 0))
            self.motifs.remove(least_used)
            if least_used.motif_id in self.motif_usage_stats:
                del self.motif_usage_stats[least_used.motif_id]

        self.motifs.append(motif)

    def get_random_motif(self) -> Optional[Motif]:
        """Get a random motif weighted by fitness"""
        if not self.motifs:
            return None

        # Weight by fitness score
        weights = [max(0.1, m.fitness_score) for m in self.motifs]
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(self.motifs)

        r = random.uniform(0, total_weight)
        cumulative = 0
        for motif, weight in zip(self.motifs, weights):
            cumulative += weight
            if r <= cumulative:
                self.motif_usage_stats[motif.motif_id] += 1
                return motif

        return self.motifs[-1]

    def discover_motifs(self, population: List['EvolvableGenome'], generation: int, min_fitness: float = 0.0):
        """Discover new motifs from high-performing genomes"""
        candidates = [g for g in population if g.fitness >= min_fitness]

        for genome in candidates:
            if len(genome.modules) >= 2:  # Need at least 2 modules for a motif
                # Extract subgraph patterns
                motifs = self._extract_motifs_from_genome(genome, generation)
                for motif in motifs:
                    self.add_motif(motif)

    def _extract_motifs_from_genome(self, genome: 'EvolvableGenome', generation: int) -> List[Motif]:
        """Extract architectural motifs from a genome"""
        motifs = []

        # Simple motif extraction: consecutive module pairs
        for i in range(len(genome.modules) - 1):
            module1 = genome.modules[i]
            module2 = genome.modules[i + 1]

            # Check if there's a connection between them
            connection_exists = any(
                conn[0] == i and conn[1] == i + 1
                for conn in genome.module_connections
            )

            if connection_exists:
                motif = Motif(
                    motif_id=f"motif_{generation}_{len(motifs)}",
                    modules=[module1.copy(), module2.copy()],
                    connections=[(0, 1)],
                    fitness_score=genome.fitness,
                    discovery_generation=generation
                )
                motifs.append(motif)

        return motifs

    def to_dict(self) -> Dict[str, Any]:
        """Convert library to dictionary"""
        return {
            'motifs': [m.to_dict() for m in self.motifs],
            'motif_usage_stats': dict(self.motif_usage_stats),
            'max_motifs': self.max_motifs
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MotifLibrary':
        """Create library from dictionary"""
        library = cls(max_motifs=data.get('max_motifs', 50))
        library.motifs = [Motif.from_dict(m) for m in data['motifs']]
        library.motif_usage_stats = defaultdict(int, data.get('motif_usage_stats', {}))
        return library


@dataclass
class GenomeMetadata:
    """Metadata for genome lineage and evolution history"""
    parent_ids: List[str] = field(default_factory=list)
    birth_generation: int = 0
    origin_population: str = "unknown"
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    last_eval_metrics: Optional[Dict[str, Any]] = None


class LearningRuleNet(nn.Module):
    """
    Neural network that directly computes ΔW = f(pre, post, reward, w, t)
    based on pre-synaptic, post-synaptic activity, reward, current weights, and timestep.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Input size: pre_mean (input_dim) + post_mean (output_dim) + reward (1) + w_flat (output_dim * input_dim) + t (1)
        total_input = input_dim + output_dim + 1 + (output_dim * input_dim) + 1
        self.fc1 = nn.Linear(total_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim * input_dim)  # ΔW flattened

        # Initialize with small weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, pre_mean: torch.Tensor, post_mean: torch.Tensor, reward: torch.Tensor, w_flat: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute ΔW directly.

        Args:
            pre_mean: Mean pre-synaptic activity (input_dim,)
            post_mean: Mean post-synaptic activity (output_dim,)
            reward: Reward signal (scalar)
            w_flat: Flattened current weights (output_dim * input_dim,)
            t: Timestep (scalar)

        Returns:
            ΔW flattened (output_dim * input_dim,)
        """
        # Concatenate inputs
        x = torch.cat([pre_mean, post_mean, reward.view(-1), w_flat, t.view(-1)], dim=-1)

        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        delta_w_flat = self.output(x)

        return delta_w_flat

    def get_parameters_as_dict(self) -> Dict[str, float]:
        """Get current parameters as static dict (for backward compatibility)"""
        # This is a fallback for cases where we need static values
        # We'll use the network's output with zero inputs
        with torch.no_grad():
            zero_pre_mean = torch.zeros(self.input_dim)
            zero_post_mean = torch.zeros(self.output_dim)
            zero_reward = torch.zeros(1)
            zero_w_flat = torch.zeros(self.output_dim * self.input_dim)
            zero_t = torch.zeros(1)
            delta_w_flat = self.forward(zero_pre_mean, zero_post_mean, zero_reward, zero_w_flat, zero_t)
            # Return some summary statistics
            return {
                'mean_delta_w': delta_w_flat.mean().item(),
                'std_delta_w': delta_w_flat.std().item(),
                'max_delta_w': delta_w_flat.max().item(),
                'min_delta_w': delta_w_flat.min().item()
            }

    def _coerce_param_array(self, value: Any, expected_shape: Tuple[int, ...], key: str) -> np.ndarray:
        """Coerce JSON-loaded values into a numeric ndarray with expected shape.

        Handles legacy checkpoints where numpy arrays were serialized via `default=str`,
        resulting in numpy-style string representations with whitespace/newlines.
        """
        if value is None:
            raise ValueError(f"Missing value for {key}")

        # Already an ndarray
        if isinstance(value, np.ndarray):
            arr = value
        # JSON list
        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=np.float32)
        # Legacy: numpy array string repr (whitespace-separated, no commas)
        elif isinstance(value, str):
            s = value.strip()
            # Fast path: parse as whitespace-separated numeric stream.
            numeric = np.fromstring(s.replace('[', ' ').replace(']', ' '), sep=' ', dtype=np.float32)
            if numeric.size == 0:
                # Fallback: sometimes it may be a Python list string with commas.
                try:
                    parsed = ast.literal_eval(s)
                except Exception as e:
                    raise ValueError(f"Could not parse legacy weight string for {key}") from e
                arr = np.array(parsed, dtype=np.float32)
            else:
                arr = numeric
        else:
            # numpy scalar, torch tensor, etc.
            try:
                arr = np.array(value, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Unsupported type for {key}: {type(value)}") from e

        expected_elems = int(np.prod(expected_shape))
        if arr.shape != expected_shape:
            flat = arr.reshape(-1)
            if flat.size != expected_elems:
                raise ValueError(
                    f"Shape mismatch for {key}: got {arr.shape} (flat {flat.size}), expected {expected_shape} (flat {expected_elems})"
                )
            arr = flat.reshape(expected_shape)

        return arr.astype(np.float32, copy=False)

    def copy_weights_from_numpy(self, weights_dict: Dict[str, Any]):
        """Copy weights from numpy arrays / lists / legacy strings (for genome loading)."""
        with torch.no_grad():
            if 'fc1_weight' in weights_dict:
                w = self._coerce_param_array(weights_dict['fc1_weight'], tuple(self.fc1.weight.shape), 'fc1_weight')
                self.fc1.weight.copy_(torch.tensor(w, dtype=torch.float32))
            if 'fc1_bias' in weights_dict:
                b = self._coerce_param_array(weights_dict['fc1_bias'], tuple(self.fc1.bias.shape), 'fc1_bias')
                self.fc1.bias.copy_(torch.tensor(b, dtype=torch.float32))
            if 'fc2_weight' in weights_dict:
                w = self._coerce_param_array(weights_dict['fc2_weight'], tuple(self.fc2.weight.shape), 'fc2_weight')
                self.fc2.weight.copy_(torch.tensor(w, dtype=torch.float32))
            if 'fc2_bias' in weights_dict:
                b = self._coerce_param_array(weights_dict['fc2_bias'], tuple(self.fc2.bias.shape), 'fc2_bias')
                self.fc2.bias.copy_(torch.tensor(b, dtype=torch.float32))
            if 'output_weight' in weights_dict:
                w = self._coerce_param_array(weights_dict['output_weight'], tuple(self.output.weight.shape), 'output_weight')
                self.output.weight.copy_(torch.tensor(w, dtype=torch.float32))
            if 'output_bias' in weights_dict:
                b = self._coerce_param_array(weights_dict['output_bias'], tuple(self.output.bias.shape), 'output_bias')
                self.output.bias.copy_(torch.tensor(b, dtype=torch.float32))

    def to_numpy_dict(self) -> Dict[str, np.ndarray]:
        """Convert network parameters to numpy dict (for genome saving)"""
        return {
            'fc1_weight': self.fc1.weight.detach().cpu().numpy(),
            'fc1_bias': self.fc1.bias.detach().cpu().numpy(),
            'fc2_weight': self.fc2.weight.detach().cpu().numpy(),
            'fc2_bias': self.fc2.bias.detach().cpu().numpy(),
            'output_weight': self.output.weight.detach().cpu().numpy(),
            'output_bias': self.output.bias.detach().cpu().numpy(),
        }

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Mutate the network parameters"""
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_strength
                param[mask] += noise[mask]

    def copy(self) -> 'LearningRuleNet':
        """Create a deep copy"""
        new_net = LearningRuleNet(self.input_dim, self.output_dim, self.hidden_dim)
        new_net.load_state_dict(self.state_dict())
        return new_net


class ActivationFunction:
    """Container for different activation functions with evolvable types"""
    
    ACTIVATIONS = {
        'tanh': np.tanh,
        'relu': lambda x: np.maximum(0, x),
        'leaky_relu': lambda x: np.where(x > 0, x, 0.01 * x),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
        'elu': lambda x: np.where(x > 0, x, np.exp(x) - 1),
        'selu': lambda x: 1.0507 * np.where(x > 0, x, 1.67326 * (np.exp(x) - 1)),
        'gelu': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
        'softplus': lambda x: np.log1p(np.exp(x)),
        'swish': lambda x: x * (1 / (1 + np.exp(-x))),
        'linear': lambda x: x,
        'sin': np.sin,
        'cos': np.cos,
    }
     
    TORCH_ACTIVATIONS = {
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
    
    @staticmethod
    def get_numpy_fn(name: str):
        return ActivationFunction.ACTIVATIONS.get(name, np.tanh)
    
    @staticmethod
    def get_torch_fn(name: str):
        return ActivationFunction.TORCH_ACTIVATIONS.get(name, torch.tanh)
    
    @staticmethod
    def get_random_activation() -> str:
        return random.choice(list(ActivationFunction.ACTIVATIONS.keys()))


class NeuralGene:
    """Represents a single gene in the neural network genome"""

    def __init__(
        self,
        gene_id: str,
        input_dim: int,
        output_dim: int,
        layer_type: str = "linear",
        activation: str = "tanh",
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        normalization_type: str = "none",  # "none", "layernorm", "batchnorm"
        dropout_schedule: Optional[Dict[str, float]] = None,  # {"initial": 0.0, "final": 0.2, "decay_steps": 1000}
        memory_size: int = 0,  # For recurrent layers (GRU state size)
        batch_norm: bool = False,  # Deprecated, use normalization_type
        skip_connection: bool = False,
        skip_target: int = -1,  # -1 means no skip, 0-indexed layer to skip to
        skip_gate: float = 0.5,  # Learned gate for skip connection blending
        plasticity: Optional[np.ndarray] = None,
        learning_rule_net: Optional['LearningRuleNet'] = None,
    ):
        self.gene_id = gene_id
        self.layer_type = layer_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.normalization_type = normalization_type
        self.dropout_schedule = dropout_schedule
        self.memory_size = memory_size
        self.batch_norm = batch_norm  # Deprecated, use normalization_type
        self.skip_connection = skip_connection
        self.skip_target = skip_target
        self.skip_gate = skip_gate
        self.plasticity = plasticity
        self.learning_rule_net = learning_rule_net

        # Weight matrices stored as (output_dim, input_dim) to align with Torch
        self.weights: np.ndarray = np.zeros((self.output_dim, self.input_dim), dtype=np.float32)
        self.bias: Optional[np.ndarray] = np.zeros(self.output_dim, dtype=np.float32) if self.use_bias else None

        # For normalization (batch norm or layer norm)
        if self.normalization_type == "batchnorm" or self.batch_norm:
            self.bn_gamma: Optional[np.ndarray] = np.ones(self.output_dim, dtype=np.float32)
            self.bn_beta: Optional[np.ndarray] = np.zeros(self.output_dim, dtype=np.float32)
            self.bn_running_mean: Optional[np.ndarray] = np.zeros(self.output_dim, dtype=np.float32)
            self.bn_running_var: Optional[np.ndarray] = np.ones(self.output_dim, dtype=np.float32)
        else:
            self.bn_gamma = None
            self.bn_beta = None
            self.bn_running_mean = None
            self.bn_running_var = None

        # For layer norm (if needed)
        if self.normalization_type == "layernorm":
            self.ln_gamma: Optional[np.ndarray] = np.ones(self.output_dim, dtype=np.float32)
            self.ln_beta: Optional[np.ndarray] = np.zeros(self.output_dim, dtype=np.float32)
        else:
            self.ln_gamma = None
            self.ln_beta = None

        # For recurrent layers (GRU, attention)
        self.hidden_state: Optional[np.ndarray] = None
        if self.memory_size > 0 and self.layer_type in ["gru", "attention_block"]:
            self.hidden_state = np.zeros((self.memory_size,), dtype=np.float32)

        self.initialize_weights()

        # Initialize learning_rule_net if not provided and plasticity is enabled
        if self.learning_rule_net is None and self.plasticity is not None:
            self.learning_rule_net = LearningRuleNet(input_dim=self.input_dim, output_dim=self.output_dim, hidden_dim=16)
    
    def initialize_weights(
        self,
        method: str = "xavier_uniform",
        scale: float = 1.0,
        seed: Optional[int] = None
    ):
        """Initialize weights with different strategies"""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        if self.input_dim is None or self.output_dim is None:
            return
        
        # Initialize weights
        if method == "xavier_uniform":
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            self.weights = rng.uniform(-limit, limit, (self.output_dim, self.input_dim)).astype(np.float32)
        elif method == "xavier_normal":
            std = np.sqrt(2.0 / (self.input_dim + self.output_dim))
            self.weights = rng.randn(self.output_dim, self.input_dim).astype(np.float32) * std
        elif method == "he_uniform":
            limit = np.sqrt(6.0 / self.input_dim)
            self.weights = rng.uniform(-limit, limit, (self.output_dim, self.input_dim)).astype(np.float32)
        elif method == "he_normal":
            std = np.sqrt(2.0 / self.input_dim)
            self.weights = rng.randn(self.output_dim, self.input_dim).astype(np.float32) * std
        elif method == "lecun_uniform":
            limit = np.sqrt(3.0 / self.input_dim)
            self.weights = rng.uniform(-limit, limit, (self.output_dim, self.input_dim)).astype(np.float32)
        elif method == "lecun_normal":
            std = 1.0 / np.sqrt(self.input_dim)
            self.weights = rng.randn(self.output_dim, self.input_dim).astype(np.float32) * std
        else:  # random
            self.weights = rng.randn(self.output_dim, self.input_dim).astype(np.float32) * 0.01
        
        # Apply scale
        self.weights *= scale
        
        # Initialize bias
        if self.use_bias:
            self.bias = np.zeros(self.output_dim, dtype=np.float32)
        
        # Initialize batch norm parameters if needed
        if self.batch_norm:
            self.bn_gamma = np.ones(self.output_dim, dtype=np.float32)
            self.bn_beta = np.zeros(self.output_dim, dtype=np.float32)
            self.bn_running_mean = np.zeros(self.output_dim, dtype=np.float32)
            self.bn_running_var = np.ones(self.output_dim, dtype=np.float32)

        # Initialize plasticity if not provided
        if self.plasticity is None:
            self.plasticity = np.random.uniform(
                low=-0.1, high=0.1, size=self.weights.shape
            ).astype(np.float32)
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass for this gene/layer"""
        if self.weights is None:
            raise ValueError(f"Gene {self.gene_id} weights not initialized")
        
        # Linear transformation (weights stored as (out, in))
        z = np.dot(x, self.weights.T)
        
        # Add bias
        if self.use_bias:
            z += self.bias
        
        # Batch normalization
        if self.batch_norm and self.bn_gamma is not None and self.bn_beta is not None:
            if training and self.bn_running_mean is not None and self.bn_running_var is not None:
                mean = np.mean(z, axis=0)
                var = np.var(z, axis=0)
                self.bn_running_mean = 0.9 * self.bn_running_mean + 0.1 * mean
                self.bn_running_var = 0.9 * self.bn_running_var + 0.1 * var
                z = (z - mean) / np.sqrt(var + 1e-8)
            else:
                if self.bn_running_mean is not None and self.bn_running_var is not None:
                    z = (z - self.bn_running_mean) / np.sqrt(self.bn_running_var + 1e-8)
            z = self.bn_gamma * z + self.bn_beta
        
        # Activation
        activation_fn = ActivationFunction.get_numpy_fn(self.activation)
        output = activation_fn(z)
        
        # Dropout (only during training)
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, output.shape)
            output = output * mask / (1 - self.dropout_rate)
        
        return output
    
    def mutate(
        self,
        weight_mutation_rate: float = 0.1,
        weight_mutation_strength: float = 0.1,
        architecture_mutation: bool = False,
        plasticity_mutation_rate: float = 0.05,
        plasticity_mutation_strength: float = 0.1,
        max_neurons: int = 128,
        min_neurons: int = 4
    ) -> bool:
        """
        Mutate this gene
        Returns: True if architecture was mutated
        """
        mutated_architecture = False

        # Weight mutation
        weight_mask = np.random.random(self.weights.shape) < weight_mutation_rate
        self.weights[weight_mask] += np.random.randn(*weight_mask.shape)[weight_mask] * weight_mutation_strength

        if self.bias is not None:
            bias_mask = np.random.random(self.bias.shape) < weight_mutation_rate
            self.bias[bias_mask] += np.random.randn(*bias_mask.shape)[bias_mask] * weight_mutation_strength

        # Plasticity mutation
        if random.random() < plasticity_mutation_rate:
            if self.plasticity is not None:
                self.plasticity = (self.plasticity + np.random.normal(
                    0, plasticity_mutation_strength, self.plasticity.shape
                )).astype(np.float32)

        # Architecture mutations
        if architecture_mutation:
            self._add_neuron()
            self._sync_plasticity()
            mutated_architecture = True

        return mutated_architecture
    
    def _add_neuron(self):
        """Add a neuron to this layer"""
        old_output_dim = self.output_dim
        self.output_dim += 1

        # Expand weights (add one output neuron → add one row)
        if self.weights is not None:
            new_row = np.random.randn(1, self.input_dim).astype(np.float32) * 0.01
            self.weights = np.vstack([self.weights, new_row])

        # Expand bias
        if self.bias is not None:
            self.bias = np.append(self.bias, 0.0)

        if self.plasticity is not None and self.plasticity.size > 0 and len(self.plasticity.shape) >= 2:
            new_shape = self.weights.shape  # (out_dim, in_dim)

            new_plasticity = np.zeros(new_shape, dtype=self.plasticity.dtype)

            min_rows = min(self.plasticity.shape[0], new_shape[0])
            min_cols = min(self.plasticity.shape[1], new_shape[1])

            new_plasticity[:min_rows, :min_cols] = \
                self.plasticity[:min_rows, :min_cols]

            self.plasticity = new_plasticity

        # Expand batch norm parameters
        if self.batch_norm:
            if self.bn_gamma is not None:
                self.bn_gamma = np.append(self.bn_gamma, 1.0)
            if self.bn_beta is not None:
                self.bn_beta = np.append(self.bn_beta, 0.0)
            if self.bn_running_mean is not None:
                self.bn_running_mean = np.append(self.bn_running_mean, 0.0)
            if self.bn_running_var is not None:
                self.bn_running_var = np.append(self.bn_running_var, 1.0)
    
    def _remove_neuron(self):
        """Remove a random neuron from this layer"""
        if self.output_dim <= 1:
            return

        # Select neuron to remove
        neuron_idx = random.randint(0, self.output_dim - 1)
        self.output_dim -= 1

        # Remove from weights
        if self.weights is not None:
            self.weights = np.delete(self.weights, neuron_idx, axis=0)

        # Remove from bias
        if self.use_bias and self.bias is not None:
            self.bias = np.delete(self.bias, neuron_idx)

        # Remove from plasticity
        if self.plasticity is not None:
            self.plasticity = np.delete(self.plasticity, neuron_idx, axis=0)

        # Remove from batch norm
        if self.batch_norm:
            if self.bn_gamma is not None:
                self.bn_gamma = np.delete(self.bn_gamma, neuron_idx)
            if self.bn_beta is not None:
                self.bn_beta = np.delete(self.bn_beta, neuron_idx)
            if self.bn_running_mean is not None:
                self.bn_running_mean = np.delete(self.bn_running_mean, neuron_idx)
        if self.bn_running_var is not None:
            self.bn_running_var = np.delete(self.bn_running_var, neuron_idx)

    def _sync_plasticity(self):
        if self.plasticity is None:
            return

        w_shape = self.weights.shape
        if self.plasticity.shape != w_shape:
            new_p = np.zeros(w_shape, dtype=self.plasticity.dtype)
            if len(w_shape) >= 2 and len(self.plasticity.shape) >= 2:
                r = min(w_shape[0], self.plasticity.shape[0])
                c = min(w_shape[1], self.plasticity.shape[1])
                new_p[:r, :c] = self.plasticity[:r, :c]
            self.plasticity = new_p

    def to_dict(self) -> Dict[str, Any]:
        """Convert gene to dictionary"""
        return {
            'gene_id': self.gene_id,
            'layer_type': self.layer_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'dropout_rate': self.dropout_rate,
            'normalization_type': self.normalization_type,
            'dropout_schedule': self.dropout_schedule,
            'memory_size': self.memory_size,
            'batch_norm': self.batch_norm,  # Backward compatibility
            'skip_connection': self.skip_connection,
            'skip_target': self.skip_target,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'bias': self.bias.tolist() if self.bias is not None else None,
            'plasticity': self.plasticity.tolist() if self.plasticity is not None else None,
            'bn_gamma': self.bn_gamma.tolist() if self.bn_gamma is not None else None,
            'bn_beta': self.bn_beta.tolist() if self.bn_beta is not None else None,
            'bn_running_mean': self.bn_running_mean.tolist() if self.bn_running_mean is not None else None,
            'bn_running_var': self.bn_running_var.tolist() if self.bn_running_var is not None else None,
            'ln_gamma': self.ln_gamma.tolist() if self.ln_gamma is not None else None,
            'ln_beta': self.ln_beta.tolist() if self.ln_beta is not None else None,
            'hidden_state': self.hidden_state.tolist() if self.hidden_state is not None else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeuralGene':
        """Create gene from dictionary"""
        gene = cls(
            gene_id=data['gene_id'],
            layer_type=data.get('layer_type', 'linear'),
            input_dim=data['input_dim'],
            output_dim=data['output_dim'],
            activation=data['activation'],
            use_bias=data['use_bias'],
            dropout_rate=data['dropout_rate'],
            normalization_type=data.get('normalization_type', 'none'),
            dropout_schedule=data.get('dropout_schedule'),
            memory_size=data.get('memory_size', 0),
            batch_norm=data.get('batch_norm', False),  # Backward compatibility
            skip_connection=data['skip_connection'],
            skip_target=data['skip_target'],
        )

        # Override normalization_type if batch_norm was set in old format
        if data.get('batch_norm', False) and data.get('normalization_type', 'none') == 'none':
            gene.normalization_type = 'batchnorm'

        if data['weights'] is not None:
            gene.weights = np.array(data['weights'], dtype=np.float32)
        if data['bias'] is not None:
            gene.bias = np.array(data['bias'], dtype=np.float32)
        if 'plasticity' in data and data['plasticity'] is not None:
            gene.plasticity = np.array(data['plasticity'], dtype=np.float32)

        # Batch norm parameters
        if data.get('bn_gamma') is not None:
            gene.bn_gamma = np.array(data['bn_gamma'], dtype=np.float32)
        if data.get('bn_beta') is not None:
            gene.bn_beta = np.array(data['bn_beta'], dtype=np.float32)
        if data.get('bn_running_mean') is not None:
            gene.bn_running_mean = np.array(data['bn_running_mean'], dtype=np.float32)
        if data.get('bn_running_var') is not None:
            gene.bn_running_var = np.array(data['bn_running_var'], dtype=np.float32)

        # Layer norm parameters
        if data.get('ln_gamma') is not None:
            gene.ln_gamma = np.array(data['ln_gamma'], dtype=np.float32)
        if data.get('ln_beta') is not None:
            gene.ln_beta = np.array(data['ln_beta'], dtype=np.float32)

        # Hidden state for recurrent layers
        if data.get('hidden_state') is not None:
            gene.hidden_state = np.array(data['hidden_state'], dtype=np.float32)

        return gene
    
    def copy(self) -> 'NeuralGene':
        """Create a deep copy of this gene"""
        copy_data = self.to_dict()
        return self.from_dict(copy_data)


class ConnectionGene:
    """Individual synapse with its own learning rule"""
    
    def __init__(self, from_neuron: int, to_neuron: int):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = np.random.randn() * 0.01
        self.learning_rule_params = {
            'A': np.random.uniform(-0.1, 0.1),
            'B': np.random.uniform(-0.1, 0.1),
            'C': np.random.uniform(-0.1, 0.1),
        }
        self.enabled = True  # For sparse connection representation
    
    def update_plasticity(self, pre_activity: float, post_activity: float, reward: float, timestep: int):
        """Per-connection plasticity update"""
        # Simple learning rule: Δw = A * pre + B * post + C * reward
        delta_w = (self.learning_rule_params['A'] * pre_activity + 
                   self.learning_rule_params['B'] * post_activity + 
                   self.learning_rule_params['C'] * reward)
        self.weight += delta_w
    
    def mutate(self, weight_mutation_rate: float = 0.1, param_mutation_rate: float = 0.1, mutation_strength: float = 0.01):
        """Connection-level mutation operators"""
        # Mutate weight
        if np.random.random() < weight_mutation_rate:
            self.weight += np.random.randn() * mutation_strength
        
        # Mutate learning rule parameters
        for key in self.learning_rule_params:
            if np.random.random() < param_mutation_rate:
                self.learning_rule_params[key] += np.random.uniform(-mutation_strength, mutation_strength)
        
        # Toggle enabled (sparse representation)
        if np.random.random() < 0.01:  # Low probability to toggle
            self.enabled = not self.enabled
    
    def copy(self) -> 'ConnectionGene':
        """Create a deep copy"""
        new_conn = ConnectionGene(self.from_neuron, self.to_neuron)
        new_conn.weight = self.weight
        new_conn.learning_rule_params = self.learning_rule_params.copy()
        new_conn.enabled = self.enabled
        return new_conn
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'from_neuron': self.from_neuron,
            'to_neuron': self.to_neuron,
            'weight': self.weight,
            'learning_rule_params': self.learning_rule_params,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectionGene':
        """Create from dictionary"""
        conn = cls(data['from_neuron'], data['to_neuron'])
        conn.weight = data['weight']
        conn.learning_rule_params = data['learning_rule_params']
        conn.enabled = data.get('enabled', True)
        return conn


class EvolvableGenome:
    """
    Modular genome that encodes evolvable neural network architectures using reusable modules
    Supports:
    - Module reuse and duplication
    - Motif discovery and insertion
    - DAG execution order for complex connectivity
    - Automatic motif mining from high-performing genomes
    """

    def __init__(
        self,
        genome_id: Optional[str] = None,
        input_size: int = 6,
        output_size: int = 4,
        min_modules: int = 1,
        max_modules: int = 8,
        min_layers: Optional[int] = None,  # Backward-compatible alias for min_modules
        max_layers: Optional[int] = None,  # Backward-compatible alias for max_modules
        min_neurons: int = 4,
        max_neurons: int = 128,
        init_modules: int = 2,
        init_neurons: int = 16,
        seed: Optional[int] = None,
        weights=None,
        plasticity=None,
        meta=None,
        motif_library: Optional['MotifLibrary'] = None
    ):
        # Backward compatibility: map legacy layer args to module args
        if min_layers is not None:
            min_modules = min_layers
        if max_layers is not None:
            max_modules = max_layers

        self.genome_id = genome_id or f"gen_{random.randint(0, 9999):04d}"
        self.input_size = input_size
        self.output_size = output_size
        self.min_modules = min_modules
        self.max_modules = max_modules
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons

        # Legacy attributes for backward compatibility
        self.min_layers = min_modules
        self.max_layers = max_modules

        # Modular architecture: modules and connections
        self.modules: List[Module] = []
        self.module_connections: List[Tuple[int, int]] = []  # (from_module_idx, to_module_idx)

        # Legacy support: maintain genes for backward compatibility
        self.genes: List[NeuralGene] = []

        self.fitness = 0.0
        self.norm_fitness = 0.0
        self.age = 0  # Generation age

        # META-3.3: Learning curve tracking for self-directed evolution
        self.learning_curve = {
            'episode_initial_fitness': [],
            'episode_final_fitness': [],
            'episode_learning_speed': [],  # Rate of fitness improvement
            'episode_plasticity_effectiveness': [],  # Correlation between plasticity and improvement
            'episode_learning_rule_stability': [],  # How stable learning rule parameters were
            'episode_meta_adaptability': [],  # How well meta-parameters adapted
            'episode_count': 0
        }

        # Type annotations for new attributes
        self.meta: Dict[str, float] = {}
        self.learning_rule_net: Optional[LearningRuleNet] = None
        self.learning_rule: Optional[Dict[str, float]] = None
        self.plastic_diagnostics: Optional[Dict[str, Any]] = None
        self.metadata: GenomeMetadata = GenomeMetadata()
        self._gpu_compiled: bool = False
        self._gpu_layers: List[Dict[str, Any]] = []
        self._torch_brain: Optional[Any] = None
        self._signature_cache: Optional[str] = None
        self.layers: List[Dict[str, Any]] = []

        # Motif library for reuse
        self.motif_library = motif_library or MotifLibrary()

        # Initialize architecture
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._initialize_modular_architecture(init_modules, init_neurons)

        # Create self.layers by flattening modules into genes for compatibility
        self._flatten_modules_to_genes()

        # Initialize META genes
        if meta is None:
            self.meta = {
                "reward_gain": np.random.uniform(1.0, 10.0),  # Stronger reward modulation
                "reward_bias": np.random.uniform(-1.0, 1.0),  # Wider bias range
                "plastic_lr": np.random.uniform(1.0, 20.0)    # Much higher learning rates
            }
        else:
            self.meta = meta

        # Initialize LearningRuleNet instead of static dict
        self.learning_rule_net = LearningRuleNet(input_dim=self.input_size, output_dim=self.output_size, hidden_dim=16)

    def _initialize_modular_architecture(self, num_modules: int, neurons_per_module: int):
        """Initialize modular architecture with reusable building blocks"""
        num_modules = max(self.min_modules, min(num_modules, self.max_modules))
        neurons_per_module = max(self.min_neurons, min(neurons_per_module, self.max_neurons))

        # Create initial modules
        prev_dim = self.input_size
        for i in range(num_modules):
            # Last module outputs to final layer
            if i == num_modules - 1:
                output_dim = self.output_size
                module_type = "output_block"
            else:
                output_dim = neurons_per_module
                module_type = random.choice(["linear_block", "residual_block", "attention_block"])

            # Create module with 1-3 genes
            num_genes = random.randint(1, 3)
            genes = []
            current_dim = prev_dim

            for j in range(num_genes):
                if j == num_genes - 1:
                    # Last gene in module
                    gene_output = output_dim
                    activation = "linear" if i == num_modules - 1 else random.choice(list(ActivationFunction.ACTIVATIONS.keys()))
                else:
                    gene_output = neurons_per_module
                    activation = random.choice(list(ActivationFunction.ACTIVATIONS.keys()))

                gene = NeuralGene(
                    gene_id=f"module_{i}_gene_{j}",
                    input_dim=current_dim,
                    output_dim=gene_output,
                    activation=activation,
                    use_bias=True,
                    dropout_rate=random.uniform(0.0, 0.2),
                    normalization_type=random.choice(["none", "layernorm"]),
                    plasticity=np.random.uniform(-0.1, 0.1, (gene_output, current_dim)).astype(np.float32)
                )
                gene.initialize_weights(method="he_normal", scale=0.1)
                genes.append(gene)
                current_dim = gene_output

            module = Module(
                module_id=f"module_{i}",
                genes=genes,
                input_dim=prev_dim,
                output_dim=output_dim,
                module_type=module_type
            )
            self.modules.append(module)
            prev_dim = output_dim

        # Create connections (simple chain for now)
        for i in range(len(self.modules) - 1):
            self.module_connections.append((i, i + 1))

    def _flatten_modules_to_genes(self):
        """Flatten modular architecture into linear genes for backward compatibility"""
        self.genes = []
        for module in self.modules:
            self.genes.extend(module.genes)

    def mutate_modules(self) -> bool:
        """Mutate the modular architecture"""
        mutated = False

        # Module-level mutations
        if random.random() < 0.1:  # 10% chance for module mutations
            mutation_type = random.choice(['add_module', 'remove_module', 'duplicate_module', 'insert_motif'])

            if mutation_type == 'add_module' and len(self.modules) < self.max_modules:
                self._add_module()
                mutated = True
            elif mutation_type == 'remove_module' and len(self.modules) > self.min_modules:
                self._remove_module()
                mutated = True
            elif mutation_type == 'duplicate_module':
                self._duplicate_module()
                mutated = True
            elif mutation_type == 'insert_motif':
                self._insert_motif()
                mutated = True

        # Mutate individual modules
        for module in self.modules:
            if random.random() < 0.3:  # 30% chance per module
                self._mutate_module(module)
                mutated = True

        if mutated:
            self._flatten_modules_to_genes()
            self.invalidate_caches()

        return mutated

    def _add_module(self):
        """Add a new module at random position"""
        if len(self.modules) >= self.max_modules:
            return

        pos = random.randint(0, len(self.modules))
        input_dim = self.input_size if pos == 0 else self.modules[pos - 1].output_dim
        output_dim = self.output_size if pos == len(self.modules) else self.modules[pos].input_dim

        # Create new module
        genes = []
        current_dim = input_dim
        num_genes = random.randint(1, 2)

        for j in range(num_genes):
            gene_output = output_dim if j == num_genes - 1 else random.randint(self.min_neurons, self.max_neurons)
            gene = NeuralGene(
                gene_id=f"new_module_{len(self.modules)}_gene_{j}",
                input_dim=current_dim,
                output_dim=gene_output,
                activation=random.choice(list(ActivationFunction.ACTIVATIONS.keys())),
                use_bias=True,
                plasticity=np.random.uniform(-0.1, 0.1, (gene_output, current_dim)).astype(np.float32)
            )
            gene.initialize_weights(method="he_normal", scale=0.1)
            genes.append(gene)
            current_dim = gene_output

        new_module = Module(
            module_id=f"new_module_{len(self.modules)}",
            genes=genes,
            input_dim=input_dim,
            output_dim=output_dim,
            module_type=random.choice(["linear_block", "residual_block"])
        )

        self.modules.insert(pos, new_module)

        # Update connections
        self.module_connections = []
        for i in range(len(self.modules) - 1):
            self.module_connections.append((i, i + 1))

    def _remove_module(self):
        """Remove a random module"""
        if len(self.modules) <= self.min_modules:
            return

        pos = random.randint(0, len(self.modules) - 1)
        self.modules.pop(pos)

        # Update connections
        self.module_connections = []
        for i in range(len(self.modules) - 1):
            self.module_connections.append((i, i + 1))

    def _duplicate_module(self):
        """Duplicate a random module"""
        if not self.modules:
            return

        source_idx = random.randint(0, len(self.modules) - 1)
        source_module = self.modules[source_idx]

        # Create duplicate with new ID
        duplicate = source_module.copy()
        duplicate.module_id = f"duplicate_{source_module.module_id}"

        # Insert after source
        pos = source_idx + 1
        self.modules.insert(pos, duplicate)

        # Update connections
        self.module_connections = []
        for i in range(len(self.modules) - 1):
            self.module_connections.append((i, i + 1))

    def _insert_motif(self):
        """Insert a motif from the library"""
        motif = self.motif_library.get_random_motif()
        if motif is None:
            return

        # Insert motif modules
        insert_pos = random.randint(0, len(self.modules))
        for i, module in enumerate(motif.modules):
            new_module = module.copy()
            new_module.module_id = f"motif_{motif.motif_id}_module_{i}"
            self.modules.insert(insert_pos + i, new_module)

        # Add motif connections
        offset = insert_pos
        for from_idx, to_idx in motif.connections:
            self.module_connections.append((offset + from_idx, offset + to_idx))

        # Reconnect the chain
        self._rebuild_connections()

    def _mutate_module(self, module: Module):
        """Mutate a single module"""
        # Add/remove genes within module
        if random.random() < 0.2 and len(module.genes) < 4:  # Add gene
            pos = random.randint(0, len(module.genes))
            input_dim = module.input_dim if pos == 0 else module.genes[pos - 1].output_dim
            output_dim = module.output_dim if pos == len(module.genes) else module.genes[pos].input_dim

            gene = NeuralGene(
                gene_id=f"{module.module_id}_new_gene_{len(module.genes)}",
                input_dim=input_dim,
                output_dim=output_dim,
                activation=random.choice(list(ActivationFunction.ACTIVATIONS.keys())),
                use_bias=True,
                plasticity=np.random.uniform(-0.1, 0.1, (output_dim, input_dim)).astype(np.float32)
            )
            gene.initialize_weights(method="he_normal", scale=0.1)
            module.genes.insert(pos, gene)

        elif random.random() < 0.1 and len(module.genes) > 1:  # Remove gene
            pos = random.randint(0, len(module.genes) - 1)
            module.genes.pop(pos)

        # Mutate genes within module
        for gene in module.genes:
            gene.mutate(weight_mutation_rate=0.1, architecture_mutation=False)

    def _rebuild_connections(self):
        """Rebuild module connections to ensure DAG"""
        # Simple chain for now - can be extended for more complex topologies
        self.module_connections = []
        for i in range(len(self.modules) - 1):
            self.module_connections.append((i, i + 1))

    def repair_dead_layers(self, dead_layer_indices: List[int]):
        """
        Architectural repair mechanisms: Death → mutation, not just penalty

        Args:
            dead_layer_indices: List of gene indices that are dead
        """
        if not dead_layer_indices:
            return

        # Apply different repair strategies based on dead layer patterns
        for gene_idx in dead_layer_indices:
            if gene_idx >= len(self.genes):
                continue

            gene = self.genes[gene_idx]

            # Strategy 1: Mutate activation function (most common fix)
            if random.random() < 0.6:
                old_activation = gene.activation
                gene.activation = ActivationFunction.get_random_activation()
                # Avoid getting stuck in the same activation
                while gene.activation == old_activation and random.random() < 0.8:
                    gene.activation = ActivationFunction.get_random_activation()

            # Strategy 2: Insert skip connection to bypass dead layer
            elif random.random() < 0.3 and gene_idx > 0:
                # Add skip connection from earlier layer
                skip_target = random.randint(0, gene_idx - 1)
                gene.skip_connection = True
                gene.skip_target = skip_target
                gene.skip_gate = random.uniform(0.3, 0.7)  # Learned gate

            # Strategy 3: Split layer (add intermediate representation)
            elif random.random() < 0.2 and len(self.genes) < self.max_layers:
                self._split_layer(gene_idx)

            # Strategy 4: Prune and reconnect (remove dead subgraph)
            elif random.random() < 0.1:
                self._prune_dead_subgraph(gene_idx)

        # Update dimensions and caches after repairs
        self._update_gene_dimensions()
        self.invalidate_caches()

    def _split_layer(self, gene_idx: int):
        """Split a dead layer into two smaller layers"""
        if gene_idx >= len(self.genes) or len(self.genes) >= self.max_layers:
            return

        gene = self.genes[gene_idx]

        # Create intermediate dimension
        intermediate_dim = max(self.min_neurons,
                              min(self.max_neurons,
                                  (gene.input_dim + gene.output_dim) // 2))

        # Create two new genes to replace the dead one
        gene1 = NeuralGene(
            gene_id=f"{gene.gene_id}_split1",
            input_dim=gene.input_dim,
            output_dim=intermediate_dim,
            activation=ActivationFunction.get_random_activation(),
            use_bias=True,
            plasticity=np.random.uniform(-0.1, 0.1, (intermediate_dim, gene.input_dim)).astype(np.float32)
        )
        gene1.initialize_weights(method="he_normal", scale=0.1)

        gene2 = NeuralGene(
            gene_id=f"{gene.gene_id}_split2",
            input_dim=intermediate_dim,
            output_dim=gene.output_dim,
            activation=gene.activation,  # Keep original activation for output layer
            use_bias=True,
            plasticity=np.random.uniform(-0.1, 0.1, (gene.output_dim, intermediate_dim)).astype(np.float32)
        )
        gene2.initialize_weights(method="he_normal", scale=0.1)

        # Replace the dead gene with the two new genes
        self.genes[gene_idx:gene_idx+1] = [gene1, gene2]

        # Update gene IDs
        for i, g in enumerate(self.genes):
            g.gene_id = f"layer_{i}"

    def _prune_dead_subgraph(self, start_gene_idx: int):
        """Prune dead subgraph starting from given gene"""
        if start_gene_idx >= len(self.genes) - 1:  # Don't prune output layer
            return

        # Find connected dead genes (simplified: just remove this gene and reconnect)
        if len(self.genes) <= self.min_layers:
            return

        # Remove the dead gene
        removed_gene = self.genes.pop(start_gene_idx)

        # Update skip targets for remaining genes
        for gene in self.genes:
            if gene.skip_connection:
                if gene.skip_target == start_gene_idx:
                    gene.skip_connection = False
                    gene.skip_target = -1
                elif gene.skip_target > start_gene_idx:
                    gene.skip_target -= 1

        # Update gene IDs
        for i, gene in enumerate(self.genes):
            gene.gene_id = f"layer_{i}"

    def repair_saturated_layers(self, saturated_layer_indices: List[int]):
        """
        Repair saturated layers (opposite of dead layers)

        Args:
            saturated_layer_indices: List of gene indices that are saturated
        """
        for gene_idx in saturated_layer_indices:
            if gene_idx >= len(self.genes):
                continue

            gene = self.genes[gene_idx]

            # Strategy 1: Change to more stable activation
            if random.random() < 0.5:
                stable_activations = ['tanh', 'sigmoid', 'elu', 'selu']
                if gene.activation not in stable_activations:
                    gene.activation = random.choice(stable_activations)

            # Strategy 2: Add regularization (dropout)
            elif random.random() < 0.3:
                gene.dropout_rate = min(0.5, gene.dropout_rate + 0.1)

            # Strategy 3: Add batch normalization
            elif random.random() < 0.2 and not gene.batch_norm:
                gene.batch_norm = True
                gene.bn_gamma = np.ones(gene.output_dim, dtype=np.float32)
                gene.bn_beta = np.zeros(gene.output_dim, dtype=np.float32)
                gene.bn_running_mean = np.zeros(gene.output_dim, dtype=np.float32)
                gene.bn_running_var = np.ones(gene.output_dim, dtype=np.float32)

    def get_execution_order(self) -> List[int]:
        """Get topological execution order for modules (handles DAG)"""
        # For now, simple chain execution
        return list(range(len(self.modules)))

    def forward_modular(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through modular architecture"""
        if not self.modules:
            return x

        # Get execution order
        order = self.get_execution_order()

        # Track module outputs
        module_outputs = {}

        for module_idx in order:
            module = self.modules[module_idx]

            # Get inputs from predecessors
            if module_idx == 0:
                # First module gets network input
                module_input = x
            else:
                # Combine inputs from predecessors
                predecessors = [idx for idx, conn in enumerate(self.module_connections) if conn[1] == module_idx]
                if predecessors:
                    # For simplicity, concatenate inputs (can be extended)
                    inputs = [module_outputs[conn[0]] for conn in self.module_connections if conn[1] == module_idx]
                    module_input = np.concatenate(inputs, axis=-1) if len(inputs) > 1 else inputs[0]
                else:
                    module_input = x  # Fallback

            # Forward through module
            output = module.forward(module_input, training)
            module_outputs[module_idx] = output

        # Return final output
        return module_outputs[len(self.modules) - 1]

    def record_mutation(self, event: Dict[str, Any]):
        """Record a mutation event in the genome's history"""
        self.metadata.mutation_history.append(event)

    def set_parents(self, parent_ids: List[str], generation: int):
        """Set parent IDs and birth generation for lineage tracking"""
        self.metadata.parent_ids = parent_ids
        self.metadata.birth_generation = generation

    def record_episode_learning_curve(self,
                                    initial_fitness: float,
                                    final_fitness: float,
                                    plasticity_changes: List[float],
                                    learning_rule_params: Optional[Dict[str, float]] = None,
                                    meta_params: Optional[Dict[str, float]] = None):
        """
        Record learning curve data for an episode to enable self-directed evolution.

        Args:
            initial_fitness: Fitness at start of episode
            final_fitness: Fitness at end of episode
            plasticity_changes: List of plasticity delta magnitudes during episode
            learning_rule_params: Learning rule parameters used during episode
            meta_params: Meta-parameters used during episode
        """
        # Record basic learning metrics
        self.learning_curve['episode_initial_fitness'].append(initial_fitness)
        self.learning_curve['episode_final_fitness'].append(final_fitness)

        # Calculate learning speed (improvement rate)
        learning_speed = (final_fitness - initial_fitness) / max(len(plasticity_changes), 1)
        self.learning_curve['episode_learning_speed'].append(learning_speed)

        # Calculate plasticity effectiveness (correlation between plasticity and improvement)
        if plasticity_changes:
            # Simple measure: how much plasticity correlated with fitness improvement
            plasticity_magnitude = np.mean(np.abs(plasticity_changes))
            fitness_improvement = max(0, final_fitness - initial_fitness)
            effectiveness = fitness_improvement / (plasticity_magnitude + 1e-6)  # Avoid division by zero
            self.learning_curve['episode_plasticity_effectiveness'].append(float(effectiveness))
        else:
            self.learning_curve['episode_plasticity_effectiveness'].append(0.0)

        # Calculate learning rule stability (how consistent parameters were)
        if learning_rule_params and len(self.learning_curve['episode_learning_rule_stability']) > 0:
            # Compare with previous episode's parameters
            prev_params = self._get_previous_learning_rule_params()
            if prev_params:
                stability = self._calculate_parameter_stability(learning_rule_params, prev_params)
                self.learning_curve['episode_learning_rule_stability'].append(stability)
            else:
                self.learning_curve['episode_learning_rule_stability'].append(1.0)  # First episode
        else:
            self.learning_curve['episode_learning_rule_stability'].append(1.0)

        # Calculate meta-parameter adaptability
        if meta_params:
            adaptability = self._calculate_meta_adaptability(meta_params, final_fitness - initial_fitness)
            self.learning_curve['episode_meta_adaptability'].append(adaptability)
        else:
            self.learning_curve['episode_meta_adaptability'].append(0.0)

        self.learning_curve['episode_count'] += 1

        # Store current parameters for next comparison
        self._last_learning_rule_params = learning_rule_params.copy() if learning_rule_params else None

    def analyze_self_improvement(self) -> Dict[str, Any]:
        """
        Analyze the genome's learning curves to answer self-directed evolution questions.

        Returns:
            Dict with analysis of learning patterns and self-improvement metrics
        """
        if self.learning_curve['episode_count'] < 2:
            return {
                'learning_faster': None,  # Not enough data
                'plasticity_helpful': None,
                'learning_rule_stable': None,
                'meta_adaptive': None,
                'overall_improvement_trend': 0.0
            }

        # Did I learn faster this episode?
        recent_speeds = self.learning_curve['episode_learning_speed'][-5:]  # Last 5 episodes
        learning_faster = np.mean(recent_speeds) > np.mean(self.learning_curve['episode_learning_speed'][:-5])

        # Did plasticity help or hurt?
        recent_effectiveness = self.learning_curve['episode_plasticity_effectiveness'][-3:]  # Last 3 episodes
        plasticity_helpful = np.mean(recent_effectiveness) > 0.1  # Threshold for "helpful"

        # Did my learning rule stabilize or explode?
        recent_stability = self.learning_curve['episode_learning_rule_stability'][-3:]
        learning_rule_stable = np.mean(recent_stability) > 0.7  # Threshold for stability

        # How well did meta-parameters adapt?
        recent_adaptability = self.learning_curve['episode_meta_adaptability'][-3:]
        meta_adaptive = np.mean(recent_adaptability) > 0.0

        # Overall improvement trend (slope of learning speeds)
        if len(self.learning_curve['episode_learning_speed']) >= 5:
            speeds = np.array(self.learning_curve['episode_learning_speed'])
            trend = np.polyfit(range(len(speeds)), speeds, 1)[0]  # Linear trend
        else:
            trend = 0.0

        return {
            'learning_faster': learning_faster,
            'plasticity_helpful': plasticity_helpful,
            'learning_rule_stable': learning_rule_stable,
            'meta_adaptive': meta_adaptive,
            'overall_improvement_trend': float(trend),
            'avg_learning_speed': float(np.mean(recent_speeds)),
            'avg_plasticity_effectiveness': float(np.mean(recent_effectiveness)),
            'avg_learning_rule_stability': float(np.mean(recent_stability)),
            'avg_meta_adaptability': float(np.mean(recent_adaptability))
        }

    def _get_previous_learning_rule_params(self) -> Optional[Dict[str, float]]:
        """Get learning rule parameters from previous episode"""
        return getattr(self, '_last_learning_rule_params', None)

    def _calculate_parameter_stability(self, current: Dict[str, float], previous: Dict[str, float]) -> float:
        """Calculate how stable learning rule parameters are between episodes"""
        if not previous:
            return 1.0

        stability_scores = []
        for key in current.keys():
            if key in previous:
                curr_val = current[key]
                prev_val = previous[key]
                # Relative stability (1.0 = identical, 0.0 = very different)
                if abs(prev_val) > 1e-6:
                    stability = 1.0 - min(abs(curr_val - prev_val) / abs(prev_val), 1.0)
                else:
                    stability = 1.0 - min(abs(curr_val - prev_val), 1.0)
                stability_scores.append(stability)

        return float(np.mean(stability_scores)) if stability_scores else 1.0

    def _calculate_meta_adaptability(self, meta_params: Dict[str, float], fitness_change: float) -> float:
        """Calculate how well meta-parameters adapted to the learning scenario"""
        # Simple adaptability measure: correlation between meta-param magnitudes and fitness change
        meta_magnitude = np.mean([abs(v) for v in meta_params.values()])

        # Positive fitness change with appropriate meta magnitude = good adaptability
        if fitness_change > 0:
            adaptability = min(meta_magnitude / 10.0, 1.0)  # Scale to 0-1
        else:
            # Negative fitness change: too much meta magnitude might be bad
            adaptability = max(0, 1.0 - meta_magnitude / 20.0)

        return float(adaptability)

    @property
    def brain(self):
        """Get TorchBrain instance if available (cached)"""
        if self._torch_brain is None:
            try:
                from torch_brain import get_cached_brain
                self._torch_brain = get_cached_brain(self)
            except ImportError:
                self._torch_brain = None
        return self._torch_brain
    
    @brain.setter
    def brain(self, value):
        """Set TorchBrain instance (for caching)"""
        self._torch_brain = value
    
    def get_brain(self):
        """
        Get or create cached TorchBrain instance.
        MANDATORY: Use this instead of TorchBrain(genome) during evaluation.
        """
        if self._torch_brain is None:
            from torch_brain import get_cached_brain
            self._torch_brain = get_cached_brain(self)
        return self._torch_brain

    def invalidate_caches(self):
        """Invalidate derived caches after any genome mutation/structural change."""
        self._gpu_compiled = False
        self._gpu_layers = []
        self._torch_brain = None
        self._signature_cache = None

    @property
    def signature(self) -> str:
        """Stable hash of architecture + parameters for caching."""
        if self._signature_cache is not None:
            return self._signature_cache

        assert self.learning_rule_net is not None, "Learning rule net not initialized"

        hasher = hashlib.sha256()
        hasher.update(str(self.input_size).encode("utf-8"))
        hasher.update(str(self.output_size).encode("utf-8"))

        # Genes: structure + parameter bytes
        for gene in self.genes:
            hasher.update(str(gene.layer_type).encode("utf-8"))
            hasher.update(str(gene.input_dim).encode("utf-8"))
            hasher.update(str(gene.output_dim).encode("utf-8"))
            hasher.update(str(gene.activation).encode("utf-8"))
            hasher.update(b"1" if gene.use_bias else b"0")
            hasher.update(str(float(gene.dropout_rate)).encode("utf-8"))
            hasher.update(b"1" if gene.batch_norm else b"0")
            hasher.update(b"1" if gene.skip_connection else b"0")
            hasher.update(str(int(gene.skip_target)).encode("utf-8"))

            if gene.weights is not None:
                w = np.asarray(gene.weights, dtype=np.float32)
                hasher.update(w.shape.__repr__().encode("utf-8"))
                hasher.update(w.tobytes())
            if gene.bias is not None:
                b = np.asarray(gene.bias, dtype=np.float32)
                hasher.update(b.shape.__repr__().encode("utf-8"))
                hasher.update(b.tobytes())
            if gene.plasticity is not None:
                p = np.asarray(gene.plasticity, dtype=np.float32)
                hasher.update(p.shape.__repr__().encode("utf-8"))
                hasher.update(p.tobytes())

        # Meta + learning rule net parameters
        hasher.update(json.dumps(self.meta, sort_keys=True).encode("utf-8"))
        learning_rule_params = self.learning_rule_net.to_numpy_dict()
        # Convert numpy arrays to lists for JSON serialization
        serializable_params = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in learning_rule_params.items()}
        hasher.update(json.dumps(serializable_params, sort_keys=True).encode("utf-8"))

        self._signature_cache = hasher.hexdigest()
        return self._signature_cache
    
    def _initialize_architecture(self, num_layers: int, neurons_per_layer: int):
        """Initialize random architecture with expanded NAS capabilities"""
        num_layers = max(self.min_layers, min(num_layers, self.max_layers))
        neurons_per_layer = max(self.min_neurons, min(neurons_per_layer, self.max_neurons))

        # Available layer types for NAS expansion
        layer_types = ["linear", "mlp_block", "res_block", "gru", "attention_block"]
        normalization_types = ["none", "layernorm", "batchnorm"]

        # Build layers
        prev_dim = self.input_size
        for i in range(num_layers):
            # Last layer has output_size neurons and must be linear
            if i == num_layers - 1:
                layer_output = self.output_size
                layer_type = "linear"
                activation = "linear"  # Output layer typically linear for action values
                normalization_type = "none"
                memory_size = 0
            else:
                layer_output = neurons_per_layer
                layer_type = random.choice(layer_types)
                activation = ActivationFunction.get_random_activation()
                normalization_type = random.choice(normalization_types)
                memory_size = random.randint(8, 64) if layer_type in ["gru", "attention_block"] else 0

            gene = NeuralGene(
                gene_id=f"layer_{i}",
                layer_type=layer_type,
                input_dim=prev_dim,
                output_dim=layer_output,
                activation=activation,
                use_bias=True,
                dropout_rate=random.uniform(0.0, 0.3),
                normalization_type=normalization_type,
                dropout_schedule={"initial": 0.0, "final": random.uniform(0.1, 0.4), "decay_steps": 1000},
                memory_size=memory_size,
                batch_norm=(normalization_type == "batchnorm"),  # Backward compatibility
                skip_connection=False,
                skip_target=-1
            )

            # Initialize weights
            gene.initialize_weights(method="he_normal", scale=0.1)

            # Keep output layer non-plastic to avoid policy drift
            if i == num_layers - 1:
                gene.plasticity = None

            self.genes.append(gene)
            prev_dim = layer_output

        # Randomly add skip connections (30% chance per layer)
        for i, gene in enumerate(self.genes):
            if i > 0 and random.random() < 0.3:
                target = random.randint(0, i - 1)
                gene.skip_connection = True
                gene.skip_target = target
    
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through the network.

        NOTE: This is an inference-only Numpy implementation that does NOT apply plasticity updates.
        Long-term, CPU path should be inference-only; learning must live in TorchBrain.
        Use TorchBrain (via self.brain) for any learning/plasticity functionality.
        """
        if len(self.genes) == 0:
            raise ValueError("Genome has no layers")
        
        # Store activations for skip connections
        activations = [x]  # Store input as activation 0
        
        for i, gene in enumerate(self.genes):
            current_input = activations[-1]
            
            # Handle skip connections
            if gene.skip_connection and gene.skip_target >= 0 and gene.skip_target < len(activations):
                skip_input = activations[gene.skip_target]
                # Ensure dimensions match (pad or truncate if needed)
                if skip_input.shape[1] != gene.input_dim:
                    # Simple padding/truncation for now
                    if skip_input.shape[1] < gene.input_dim:
                        pad_width = gene.input_dim - skip_input.shape[1]
                        skip_input = np.pad(skip_input, ((0, 0), (0, pad_width)), mode='constant')
                    else:
                        skip_input = skip_input[:, :gene.input_dim]
                # Learned skip gate for emergent signal routing
                current_input = gene.skip_gate * current_input + (1 - gene.skip_gate) * skip_input
            
            # Forward pass through gene
            output = gene.forward(current_input, training)
            activations.append(output)
        
        return activations[-1]
    
    def act(self, state: np.ndarray) -> int:
        """Get action for a single state"""
        # Use TorchBrain if available (for plasticity)
        brain = self.brain  # Access property once to cache it
        if brain is not None:
            return brain.act(state)

        # Numpy fallback
        assert len(state) == self.input_size, f"State size mismatch: {len(state)} != {self.input_size}"

        # Add batch dimension
        state_batch = state.reshape(1, -1)

        # Forward pass
        output = self.forward(state_batch, training=False)

        # Choose action (argmax)
        return int(np.argmax(output[0]))

    def act_batch(self, states: np.ndarray) -> np.ndarray:
        """Batch inference on CPU"""
        # Use TorchBrain if available (for plasticity)
        brain = self.brain  # Access property once to cache it
        if brain is not None:
            return brain.act_batch(states)

        # Numpy fallback
        # Forward pass
        outputs = self.forward(states, training=False)

        # Argmax for each batch element
        return np.argmax(outputs, axis=1)
    
    def act_batch_gpu(self, states: np.ndarray, device: str = 'cuda') -> np.ndarray:
        """Batch inference on GPU/CPU"""
        # Determine actual device (fallback to CPU if CUDA not available)
        actual_device = device if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        # Compile if not already
        if not self._gpu_compiled:
            self.compile_gpu(actual_device)

        # Convert to tensor
        if isinstance(states, np.ndarray):
            states_tensor = torch.tensor(states, device=actual_device, dtype=torch.float32)
        else:
            states_tensor = states
        
        # Forward pass on GPU
        x = states_tensor
        activations = [x]
        
        for layer_dict in self._gpu_layers:
            # Handle skip connections
            if 'skip_target' in layer_dict and layer_dict['skip_target'] >= 0 and layer_dict['skip_target'] < len(activations):
                skip_input = activations[layer_dict['skip_target']]
                # Ensure dimensions match
                if skip_input.shape[1] != x.shape[1]:
                    # Simple padding/truncation
                    if skip_input.shape[1] < x.shape[1]:
                        pad_width = x.shape[1] - skip_input.shape[1]
                        skip_input = F.pad(skip_input, (0, pad_width))
                    else:
                        skip_input = skip_input[:, :x.shape[1]]
                x = x + skip_input
            
            # Linear layer
            x = F.linear(x, layer_dict['weight'], layer_dict['bias'])
            
            # Batch normalization
            if 'bn_gamma' in layer_dict:
                if layer_dict.get('training', False):
                    x = F.batch_norm(x, layer_dict['running_mean'], layer_dict['running_var'], 
                                   layer_dict['bn_gamma'], layer_dict['bn_beta'], training=True)
                else:
                    x = F.batch_norm(x, layer_dict['running_mean'], layer_dict['running_var'],
                                   layer_dict['bn_gamma'], layer_dict['bn_beta'], training=False)
            
            # Activation
            activation_fn = ActivationFunction.get_torch_fn(layer_dict['activation'])
            x = activation_fn(x)
            
            # Dropout (only during training)
            if 'dropout_rate' in layer_dict and layer_dict.get('training', False):
                x = F.dropout(x, p=layer_dict['dropout_rate'])
            
            activations.append(x)
        
        # Get actions (argmax)
        return torch.argmax(x, dim=1).cpu().numpy()
    
    def compile_gpu(self, device: str = 'cuda'):
        """Compile genome to GPU/CPU format"""
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'

        self._gpu_layers = []

        for gene in self.genes:
            layer_dict = {
                'weight': torch.tensor(gene.weights, device=device, dtype=torch.float32),  # Stored as (out_features, in_features)
                'activation': gene.activation,
                'training': False,
            }

            if gene.use_bias and gene.bias is not None:
                layer_dict['bias'] = torch.tensor(gene.bias, device=device, dtype=torch.float32)
            else:
                layer_dict['bias'] = None

            if gene.batch_norm:
                layer_dict['bn_gamma'] = torch.tensor(gene.bn_gamma, device=device, dtype=torch.float32)
                layer_dict['bn_beta'] = torch.tensor(gene.bn_beta, device=device, dtype=torch.float32)
                layer_dict['running_mean'] = torch.tensor(gene.bn_running_mean, device=device, dtype=torch.float32)
                layer_dict['running_var'] = torch.tensor(gene.bn_running_var, device=device, dtype=torch.float32)

            if gene.dropout_rate > 0:
                layer_dict['dropout_rate'] = gene.dropout_rate

            if gene.skip_connection:
                layer_dict['skip_target'] = gene.skip_target

            self._gpu_layers.append(layer_dict)

        self._gpu_compiled = True
    
    def mutate(
        self,
        weight_mutation_rate: float = 0.1,
        weight_mutation_strength: float = 0.1,
        architecture_mutation_rate: float = 0.05,
        layer_mutation_rate: float = 0.02,
        plasticity_mutation_strength: float = 0.1,
        max_layers: int = 8,
        min_layers: int = 1
    ):
        """
        Mutate the genome with various mutation types
        """
        mutated = False

        # Record mutation event
        mutation_event = {
            'generation': self.age,
            'timestamp': time.time(),
            'type': 'mutation',
            'details': {}
        }

        # Anneal plasticity mutation rate: start high to encourage exploration, decay with age
        plasticity_mutation_rate = self._plasticity_mutation_rate()

        # Milestone 5: Module-level mutations (architecture creativity)
        if self.mutate_modules():
            mutated = True
            mutation_event['details']['modular'] = True

        # Weight mutations
        architecture_mutation = random.random() < architecture_mutation_rate
        for gene in self.genes:
            if gene.mutate(
                weight_mutation_rate,
                weight_mutation_strength,
                architecture_mutation,
                plasticity_mutation_rate=plasticity_mutation_rate,
                plasticity_mutation_strength=plasticity_mutation_strength,
            ):
                mutated = True

        # Learning rule net mutations
        if random.random() < 0.1:  # 10% chance to mutate learning rule net
            assert self.learning_rule_net is not None, "Learning rule net not initialized"
            self.learning_rule_net.mutate(mutation_rate=0.1, mutation_strength=0.1)
            mutation_event['details']['learning_rule_net'] = True

        # Structural mutations (add/remove layers)
        if random.random() < layer_mutation_rate:
            mutation_type = random.choice(['add_layer', 'remove_layer', 'swap_layers'])

            if mutation_type == 'add_layer' and len(self.genes) < max_layers:
                self._add_layer()
                mutated = True
                mutation_event['details']['structural'] = 'add_layer'
            elif mutation_type == 'remove_layer' and len(self.genes) > min_layers:
                self._remove_layer()
                mutated = True
                mutation_event['details']['structural'] = 'remove_layer'
            elif mutation_type == 'swap_layers' and len(self.genes) > 1:
                self._swap_layers()
                mutated = True
                mutation_event['details']['structural'] = 'swap_layers'

        # Connection mutations (skip connections)
        if random.random() < architecture_mutation_rate:
            self._mutate_connections()
            mutated = True
            mutation_event['details']['connections'] = True

        # Record mutation if any changes occurred
        if mutated:
            self.record_mutation(mutation_event)
            self._update_gene_dimensions()
            self.invalidate_caches()

        return mutated

    def _plasticity_mutation_rate(self) -> float:
        """High early plasticity mutation (0.2) annealed toward 0.05 as genome ages"""
        # Exponential decay with age keeps early exploration high while stabilizing later
        return 0.05 + 0.15 * np.exp(-self.age / 20.0)

    def _add_layer(self):
        """Add a new layer at a random position"""
        if len(self.genes) >= self.max_layers:
            return

        # Choose random position (not at the end for output layer)
        pos = random.randint(0, len(self.genes) - 1)

        # Determine dimensions
        if pos == 0:
            input_dim = self.input_size
            prev_output = self.input_size
        else:
            input_dim = self.genes[pos - 1].output_dim
            prev_output = self.genes[pos - 1].output_dim

        if pos < len(self.genes):
            output_dim = self.genes[pos].input_dim
        else:
            output_dim = self.output_size

        # Create new gene
        new_gene = NeuralGene(
            gene_id=f"layer_new_{len(self.genes)}",
            layer_type="linear",
            input_dim=input_dim,
            output_dim=output_dim,
            activation=ActivationFunction.get_random_activation(),
            use_bias=True,
            dropout_rate=0.0,
            batch_norm=False,
            skip_connection=False,
            skip_target=-1
        )

        new_gene.initialize_weights(method="he_normal", scale=0.1)

        # Update skip_targets for existing genes
        for gene in self.genes:
            if gene.skip_connection and gene.skip_target >= pos:
                gene.skip_target += 1

        # Insert at position
        self.genes.insert(pos, new_gene)

        # Update gene IDs
        for i, gene in enumerate(self.genes):
            gene.gene_id = f"layer_{i}"
    
    def _remove_layer(self):
        """Remove a random layer"""
        if len(self.genes) <= self.min_layers:
            return

        # Don't remove output layer (last one)
        pos = random.randint(0, len(self.genes) - 2)

        # Update skip_targets for remaining genes
        for gene in self.genes:
            if gene.skip_connection:
                if gene.skip_target == pos:
                    # Skip target is the layer being removed, disable skip
                    gene.skip_connection = False
                    gene.skip_target = -1
                elif gene.skip_target > pos:
                    # Skip target is after the removed layer, decrement
                    gene.skip_target -= 1

        del self.genes[pos]

        # Update gene IDs
        for i, gene in enumerate(self.genes):
            gene.gene_id = f"layer_{i}"
    
    def _swap_layers(self):
        """Swap two random layers"""
        if len(self.genes) < 2:
            return
        
        i, j = random.sample(range(len(self.genes)), 2)
        self.genes[i], self.genes[j] = self.genes[j], self.genes[i]
        
        # Update gene IDs
        for idx, gene in enumerate(self.genes):
            gene.gene_id = f"layer_{idx}"
    
    def _mutate_connections(self):
        """Mutate skip connections"""
        for i, gene in enumerate(self.genes):
            if random.random() < 0.3:  # 30% chance per layer
                if gene.skip_connection:
                    # Remove skip connection
                    gene.skip_connection = False
                    gene.skip_target = -1
                else:
                    # Add skip connection
                    if i > 0:
                        target = random.randint(0, i - 1)
                        gene.skip_connection = True
                        gene.skip_target = target
    
    def _update_gene_dimensions(self):
        """Update input/output dimensions after structural mutations"""
        prev_dim = self.input_size

        for i, gene in enumerate(self.genes):
            # Update input dimension
            if gene.input_dim != prev_dim:
                # Need to resize weights
                old_weights = gene.weights
                old_input_dim = gene.input_dim
                new_input_dim = prev_dim

                # Resize weights (stored as (out, in))
                new_weights = np.zeros((gene.output_dim, new_input_dim), dtype=np.float32)
                if old_weights is not None:
                    copy_dim = min(old_input_dim, new_input_dim)
                    new_weights[:, :copy_dim] = old_weights[:, :copy_dim]
                gene.weights = new_weights
                gene.input_dim = new_input_dim

                # Resize plasticity
                if gene.plasticity is not None:
                    old_plasticity = gene.plasticity
                    new_plasticity = np.zeros((gene.output_dim, new_input_dim), dtype=np.float32)
                    if old_plasticity is not None:
                        copy_dim = min(old_input_dim, new_input_dim)
                        new_plasticity[:, :copy_dim] = old_plasticity[:, :copy_dim]
                    gene.plasticity = new_plasticity

            prev_dim = gene.output_dim

        # Ensure last layer outputs correct size
        if len(self.genes) > 0:
            last_gene = self.genes[-1]
            if last_gene.output_dim != self.output_size:
                # Update output dimension
                last_gene.output_dim = self.output_size

                # Resize weights properly (stored as (out, in))
                old_weights = last_gene.weights
                if old_weights is not None:
                    old_out, old_in = old_weights.shape
                    new_weights = np.zeros((self.output_size, last_gene.input_dim), dtype=np.float32)

                    # Copy overlapping region
                    copy_out = min(old_out, self.output_size)
                    copy_in = min(old_in, last_gene.input_dim)
                    new_weights[:copy_out, :copy_in] = old_weights[:copy_out, :copy_in]

                    # Reinitialize extra rows (new outputs)
                    if self.output_size > old_out:
                        new_weights[old_out:self.output_size, :last_gene.input_dim] = np.random.randn(self.output_size - old_out, last_gene.input_dim) * 0.1

                    last_gene.weights = new_weights

                # Resize plasticity
                if last_gene.plasticity is not None:
                    old_plasticity = last_gene.plasticity
                    if len(old_plasticity.shape) >= 2:
                        old_out, old_in = old_plasticity.shape[0], old_plasticity.shape[1]
                    else:
                        old_out, old_in = 0, 0
                    new_plasticity = np.zeros((self.output_size, last_gene.input_dim), dtype=np.float32)

                    # Copy overlapping region
                    if old_out > 0 and old_in > 0:
                        copy_out = min(old_out, self.output_size)
                        copy_in = min(old_in, last_gene.input_dim)
                        new_plasticity[:copy_out, :copy_in] = old_plasticity[:copy_out, :copy_in]

                    # Reinitialize extra rows (new outputs)
                    if self.output_size > old_out:
                        new_plasticity[old_out:self.output_size, :last_gene.input_dim] = np.random.uniform(low=-0.1, high=0.1, size=(self.output_size - old_out, last_gene.input_dim))

                    last_gene.plasticity = new_plasticity

                # Update bias
                if last_gene.use_bias:
                    last_gene.bias = np.zeros(self.output_size, dtype=np.float32)

                # Update batch norm parameters if they exist
                if last_gene.batch_norm:
                    last_gene.bn_gamma = np.ones(self.output_size, dtype=np.float32)
                    last_gene.bn_beta = np.zeros(self.output_size, dtype=np.float32)
                    last_gene.bn_running_mean = np.zeros(self.output_size, dtype=np.float32)
                    last_gene.bn_running_var = np.ones(self.output_size, dtype=np.float32)

        # Realign plasticity tensors with current weight shapes after structural changes
        for gene in self.genes:
            gene._sync_plasticity()

        # Safety check
        self._assert_consistency()

    def _assert_consistency(self):
        """Assert that all dimensions are consistent across genes and plasticity matrices, fixing inconsistencies"""
        prev_dim = self.input_size
        for i, gene in enumerate(self.genes):
            # Check input dimension consistency
            if gene.input_dim != prev_dim:
                raise ValueError(f"Gene {i} input_dim {gene.input_dim} != expected {prev_dim}")

            # Check plasticity shape consistency
            if gene.plasticity is not None:
                expected_shape = (gene.output_dim, gene.input_dim)
                if gene.plasticity.shape != expected_shape:
                    old_shape = gene.plasticity.shape
                    new_plasticity = np.zeros(expected_shape, dtype=np.float32)
                    copy_rows = min(old_shape[0] if len(old_shape) > 0 else 0, expected_shape[0])
                    copy_cols = min(old_shape[1] if len(old_shape) > 1 else 0, expected_shape[1])
                    if len(old_shape) > 1 and len(expected_shape) > 1:
                        new_plasticity[:copy_rows, :copy_cols] = gene.plasticity[:copy_rows, :copy_cols]
                    gene.plasticity = new_plasticity

            # Check weights shape consistency
            if gene.weights is not None:
                expected_shape = (gene.output_dim, gene.input_dim)
                if gene.weights.shape != expected_shape:
                    old_shape = gene.weights.shape
                    new_weights = np.zeros(expected_shape, dtype=np.float32)
                    copy_rows = min(old_shape[0], expected_shape[0])
                    copy_cols = min(old_shape[1], expected_shape[1])
                    new_weights[:copy_rows, :copy_cols] = gene.weights[:copy_rows, :copy_cols]
                    gene.weights = new_weights

            # Check bias shape consistency
            if gene.bias is not None:
                expected_shape = (gene.output_dim,)
                if gene.bias.shape != expected_shape:
                    old_shape = gene.bias.shape
                    new_bias = np.zeros(expected_shape, dtype=np.float32)
                    copy_len = min(len(old_shape), expected_shape[0])
                    new_bias[:copy_len] = gene.bias[:copy_len]
                    gene.bias = new_bias

            # Check batch norm parameters
            if gene.batch_norm:
                if gene.bn_gamma is not None and gene.bn_gamma.shape != (gene.output_dim,):
                    old_shape = gene.bn_gamma.shape
                    new_bn_gamma = np.ones((gene.output_dim,), dtype=np.float32)
                    copy_len = min(len(old_shape), gene.output_dim)
                    new_bn_gamma[:copy_len] = gene.bn_gamma[:copy_len]
                    gene.bn_gamma = new_bn_gamma

                if gene.bn_beta is not None and gene.bn_beta.shape != (gene.output_dim,):
                    old_shape = gene.bn_beta.shape
                    new_bn_beta = np.zeros((gene.output_dim,), dtype=np.float32)
                    copy_len = min(len(old_shape), gene.output_dim)
                    new_bn_beta[:copy_len] = gene.bn_beta[:copy_len]
                    gene.bn_beta = new_bn_beta

                if gene.bn_running_mean is not None and gene.bn_running_mean.shape != (gene.output_dim,):
                    old_shape = gene.bn_running_mean.shape
                    new_bn_running_mean = np.zeros((gene.output_dim,), dtype=np.float32)
                    copy_len = min(len(old_shape), gene.output_dim)
                    new_bn_running_mean[:copy_len] = gene.bn_running_mean[:copy_len]
                    gene.bn_running_mean = new_bn_running_mean

                if gene.bn_running_var is not None and gene.bn_running_var.shape != (gene.output_dim,):
                    old_shape = gene.bn_running_var.shape
                    new_bn_running_var = np.ones((gene.output_dim,), dtype=np.float32)
                    copy_len = min(len(old_shape), gene.output_dim)
                    new_bn_running_var[:copy_len] = gene.bn_running_var[:copy_len]
                    gene.bn_running_var = new_bn_running_var

            prev_dim = gene.output_dim

        # Check last layer output dimension
        if len(self.genes) > 0 and self.genes[-1].output_dim != self.output_size:
            raise ValueError(f"Last gene output_dim {self.genes[-1].output_dim} != output_size {self.output_size}")

    @staticmethod
    def crossover(parent1: 'EvolvableGenome', parent2: 'EvolvableGenome') -> 'EvolvableGenome':
        """
        Crossover two genomes
        Uses alignment-based crossover for variable-length genomes
        """
        child_id = f"gen_{random.randint(0, 9999):04d}"
        
        # Simple approach: inherit architecture from one parent, weights from both
        # Choose which parent to inherit architecture from
        arch_parent = parent1 if random.random() < 0.5 else parent2
        
        # Create child with same architecture
        child = EvolvableGenome(
            genome_id=child_id,
            input_size=arch_parent.input_size,
            output_size=arch_parent.output_size,
            min_modules=arch_parent.min_modules,
            max_modules=arch_parent.max_modules,
            min_neurons=arch_parent.min_neurons,
            max_neurons=arch_parent.max_neurons,
        )
        
        # Clear default genes
        child.genes = []
        
        # For each layer in arch_parent, crossover with matching layer from other parent if possible
        for i, parent_gene in enumerate(arch_parent.genes):
            # Try to find matching layer in other parent
            other_parent = parent2 if arch_parent == parent1 else parent1
            
            # Look for layer with similar position and dimensions
            matching_gene = None
            if i < len(other_parent.genes):
                other_gene = other_parent.genes[i]
                if (other_gene.input_dim == parent_gene.input_dim and 
                    other_gene.output_dim == parent_gene.output_dim):
                    matching_gene = other_gene
            
            # Create child gene
            child_gene = parent_gene.copy()
            
            # Crossover weights if matching gene found
            if matching_gene is not None:
                # Uniform crossover for weights
                mask = np.random.random(parent_gene.weights.shape) < 0.5
                child_gene.weights = np.where(mask, parent_gene.weights, matching_gene.weights)
                
                # Crossover bias
                if parent_gene.use_bias and matching_gene.use_bias and parent_gene.bias is not None and matching_gene.bias is not None:
                    bias_mask = np.random.random(parent_gene.bias.shape) < 0.5
                    child_gene.bias = np.where(bias_mask, parent_gene.bias, matching_gene.bias)
                
                # Crossover batch norm params
                if (parent_gene.batch_norm and matching_gene.batch_norm and 
                    parent_gene.bn_gamma is not None and matching_gene.bn_gamma is not None and
                    parent_gene.bn_beta is not None and matching_gene.bn_beta is not None):
                    child_gene.bn_gamma = (parent_gene.bn_gamma + matching_gene.bn_gamma) / 2
                    child_gene.bn_beta = (parent_gene.bn_beta + matching_gene.bn_beta) / 2

                # Crossover plasticity
                if parent_gene.plasticity is not None and matching_gene.plasticity is not None:
                    plasticity_mask = np.random.random(parent_gene.plasticity.shape) < 0.5
                    child_gene.plasticity = np.where(plasticity_mask, parent_gene.plasticity, matching_gene.plasticity)
            
            child.genes.append(child_gene)

        # Blend meta-parameters (e.g., reward modulation) for recombination
        child.meta = {}
        for key in set(parent1.meta.keys()) | set(parent2.meta.keys()):
            v1 = parent1.meta.get(key)
            v2 = parent2.meta.get(key)
            if v1 is None:
                child.meta[key] = v2 if v2 is not None else 0.0
            elif v2 is None:
                child.meta[key] = v1
            else:
                # Mix by averaging; could switch to uniform per-key if needed
                child.meta[key] = float(0.5 * (v1 + v2))

        # Crossover learning rule nets
        assert parent1.learning_rule_net is not None, "Parent1 learning rule net not initialized"
        child.learning_rule_net = parent1.learning_rule_net.copy()
        # Blend parameters from both parents
        assert parent2.learning_rule_net is not None, "Parent2 learning rule net not initialized"
        parent2_params = parent2.learning_rule_net.to_numpy_dict()
        child.learning_rule_net.copy_weights_from_numpy(parent2_params)

        return child
    
    def reset_fitness(self):
        """Reset fitness values"""
        self.fitness = 0.0
        self.norm_fitness = 0.0
    
    def copy(self) -> 'EvolvableGenome':
        """Create a deep copy of the genome"""
        copy_genome = EvolvableGenome(
            genome_id=f"{self.genome_id}_copy",
            input_size=self.input_size,
            output_size=self.output_size,
            min_modules=self.min_modules,
            max_modules=self.max_modules,
            min_neurons=self.min_neurons,
            max_neurons=self.max_neurons,
        )

        copy_genome.genes = [gene.copy() for gene in self.genes]
        copy_genome.fitness = self.fitness
        copy_genome.norm_fitness = self.norm_fitness
        copy_genome.age = self.age
        copy_genome._gpu_compiled = False
        copy_genome.meta = {k: v for k, v in self.meta.items()}
        copy_genome.learning_rule = self.learning_rule.copy() if self.learning_rule else None
        assert self.learning_rule_net is not None, "Learning rule net not initialized"
        copy_genome.learning_rule_net = self.learning_rule_net.copy()
        copy_genome.invalidate_caches()

        return copy_genome
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary"""
        assert self.learning_rule_net is not None, "Learning rule net not initialized"
        return {
            'genome_id': self.genome_id,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'min_layers': self.min_layers,
            'max_layers': self.max_layers,
            'min_neurons': self.min_neurons,
            'max_neurons': self.max_neurons,
            'fitness': self.fitness,
            'norm_fitness': self.norm_fitness,
            'age': self.age,
            'genes': [gene.to_dict() for gene in self.genes],
            'meta': self.meta,
            'learning_rule': self.learning_rule,
            'learning_rule_net': self.learning_rule_net.to_numpy_dict(),
            'metadata': {
                'parent_ids': self.metadata.parent_ids,
                'birth_generation': self.metadata.birth_generation,
                'origin_population': self.metadata.origin_population,
                'mutation_history': self.metadata.mutation_history,
                'last_eval_metrics': self.metadata.last_eval_metrics
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolvableGenome':
        """Create genome from dictionary"""
        genome = cls(
            genome_id=data['genome_id'],
            input_size=data['input_size'],
            output_size=data['output_size'],
            min_layers=data['min_layers'],
            max_layers=data['max_layers'],
            min_neurons=data['min_neurons'],
            max_neurons=data['max_neurons'],
        )

        genome.genes = [NeuralGene.from_dict(gene_data) for gene_data in data['genes']]
        genome.fitness = data['fitness']
        genome.norm_fitness = data['norm_fitness']
        genome.age = data['age']
        genome.meta = data.get('meta', {
            "reward_gain": np.random.uniform(0.1, 2.0),
            "reward_bias": np.random.uniform(-0.5, 0.5),
            "plastic_lr": np.random.uniform(0.01, 1.0)
        })
        genome.learning_rule = data.get('learning_rule')

        # Load learning_rule_net if present, otherwise create new one
        if 'learning_rule_net' in data:
            # For backward compatibility, if old format, create with input_dim only
            if 'learning_rule_net' in data and 'fc1_weight' in data['learning_rule_net']:
                # Old format: assume input_dim = genome.input_size, output_dim = genome.output_size
                genome.learning_rule_net = LearningRuleNet(input_dim=genome.input_size, output_dim=genome.output_size, hidden_dim=16)
                genome.learning_rule_net.copy_weights_from_numpy(data['learning_rule_net'])
            else:
                # New format or missing: create default
                genome.learning_rule_net = LearningRuleNet(input_dim=genome.input_size, output_dim=genome.output_size, hidden_dim=16)
        else:
            # Backward compatibility: create new LearningRuleNet
            genome.learning_rule_net = LearningRuleNet(input_dim=genome.input_size, output_dim=genome.output_size, hidden_dim=16)

        # Load metadata if present
        if 'metadata' in data:
            metadata_data = data['metadata']
            genome.metadata.parent_ids = metadata_data.get('parent_ids', [])
            genome.metadata.birth_generation = metadata_data.get('birth_generation', 0)
            genome.metadata.origin_population = metadata_data.get('origin_population', 'unknown')
            genome.metadata.mutation_history = metadata_data.get('mutation_history', [])
            genome.metadata.last_eval_metrics = metadata_data.get('last_eval_metrics')

        return genome
    
    def save(self, filename: str):
        """Save genome to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'EvolvableGenome':
        """Load genome from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get summary of genome architecture"""
        total_params = 0
        layer_info = []
        
        for gene in self.genes:
            layer_params = gene.input_dim * gene.output_dim
            if gene.use_bias:
                layer_params += gene.output_dim
            if gene.batch_norm:
                layer_params += gene.output_dim * 4  # gamma, beta, running_mean, running_var
            
            total_params += layer_params
            
            layer_info.append({
                'type': gene.layer_type,
                'input': gene.input_dim,
                'output': gene.output_dim,
                'activation': gene.activation,
                'params': layer_params,
                'bias': gene.use_bias,
                'batch_norm': gene.batch_norm,
                'dropout': gene.dropout_rate,
                'skip': gene.skip_connection,
                'skip_target': gene.skip_target,
            })
        
        return {
            'genome_id': self.genome_id,
            'total_layers': len(self.genes),
            'total_params': total_params,
            'layers': layer_info,
            'fitness': self.fitness,
            'age': self.age,
        }
    
    def __str__(self) -> str:
        """String representation of genome"""
        summary = self.get_architecture_summary()
        lines = [
            f"Genome: {summary['genome_id']}",
            f"Fitness: {summary['fitness']:.2f}, Age: {summary['age']}",
            f"Layers: {summary['total_layers']}, Params: {summary['total_params']}",
            "Architecture:"
        ]
        
        for i, layer in enumerate(summary['layers']):
            skip_info = f" (skip from {layer['skip_target']})" if layer['skip'] else ""
            lines.append(
                f"  L{i}: {layer['input']} → {layer['output']} "
                f"[{layer['activation']}] "
                f"bias:{layer['bias']} bn:{layer['batch_norm']} "
                f"dropout:{layer['dropout']:.2f}{skip_info}"
            )
        
        return "\n".join(lines)


# Backward compatibility alias
Genome = EvolvableGenome