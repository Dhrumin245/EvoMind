"""
ConnectionGene — per-synapse NEAT-style evolution unit.

This is a reserved building block for a future per-connection NEAT evolution
path. It is NOT currently wired into the active modular co-evolution loop
(which uses NeuralGene / EvolvableGenome). It lives here rather than in
genome.py to keep the data/serialization layer clean.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


class ConnectionGene:
    """Individual synapse with its own evolvable learning rule.

    Each connection encodes a 3-parameter Hebbian rule:
        Δw = A * x_pre + B * x_post + C * reward

    Weights and rule parameters are independently mutable and can be
    toggled enabled/disabled for sparse connectivity evolution.
    """

    def __init__(self, from_neuron: int, to_neuron: int) -> None:
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight: float = float(np.random.randn() * 0.01)
        self.learning_rule_params: Dict[str, float] = {
            'A': float(np.random.uniform(-0.1, 0.1)),
            'B': float(np.random.uniform(-0.1, 0.1)),
            'C': float(np.random.uniform(-0.1, 0.1)),
        }
        self.enabled: bool = True  # Sparse representation: disabled synapses contribute 0

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def update_plasticity(
        self,
        pre_activity: float,
        post_activity: float,
        reward: float,
        timestep: int,  # reserved for time-dependent rules
    ) -> None:
        """Apply per-connection learning rule for one timestep."""
        delta_w = (
            self.learning_rule_params['A'] * pre_activity
            + self.learning_rule_params['B'] * post_activity
            + self.learning_rule_params['C'] * reward
        )
        self.weight += delta_w
        if not np.isfinite(self.weight):
            self.weight = float(
                np.nan_to_num(self.weight, nan=0.0, posinf=1.0, neginf=-1.0)
            )

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def mutate(
        self,
        weight_mutation_rate: float = 0.1,
        param_mutation_rate: float = 0.1,
        mutation_strength: float = 0.01,
    ) -> None:
        """Mutate connection weight, learning rule params, and enabled flag."""
        # Weight perturbation
        if np.random.random() < weight_mutation_rate:
            self.weight += float(np.random.randn() * mutation_strength)
            if not np.isfinite(self.weight):
                self.weight = float(
                    np.nan_to_num(self.weight, nan=0.0, posinf=1.0, neginf=-1.0)
                )

        # Learning rule parameter perturbation
        for key in self.learning_rule_params:
            if np.random.random() < param_mutation_rate:
                self.learning_rule_params[key] += float(
                    np.random.uniform(-mutation_strength, mutation_strength)
                )

        # Sparse toggle (1% chance)
        if np.random.random() < 0.01:
            self.enabled = not self.enabled

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def copy(self) -> 'ConnectionGene':
        """Deep copy."""
        new_conn = ConnectionGene(self.from_neuron, self.to_neuron)
        new_conn.weight = self.weight
        new_conn.learning_rule_params = self.learning_rule_params.copy()
        new_conn.enabled = self.enabled
        return new_conn

    def to_dict(self) -> Dict[str, Any]:
        return {
            'from_neuron': self.from_neuron,
            'to_neuron': self.to_neuron,
            'weight': self.weight,
            'learning_rule_params': self.learning_rule_params,
            'enabled': self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConnectionGene':
        conn = cls(data['from_neuron'], data['to_neuron'])
        conn.weight = data['weight']
        conn.learning_rule_params = data['learning_rule_params']
        conn.enabled = data.get('enabled', True)
        return conn

    def __repr__(self) -> str:
        state = "on" if self.enabled else "off"
        return (
            f"ConnectionGene({self.from_neuron}->{self.to_neuron}, "
            f"w={self.weight:.4f}, {state})"
        )
