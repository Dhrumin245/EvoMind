import random
import numpy as np
from core.genome import Genome
from core.genome import NeuralGene

class PredatorGenome(Genome):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "predator"
        # Predator-specific neural architecture (evolvable)
        self.architecture = [10, 16, 12, 8, 4]  # Different from prey

    @classmethod
    def random_initialization(cls, input_size=8, output_size=4):
        """Create a random predator genome"""
        return cls(
            input_size=input_size,
            output_size=output_size,
            init_modules=4,
            init_neurons=16
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'PredatorGenome':
        """Create predator genome from dictionary"""
        genome = super().from_dict(data)
        # Convert to PredatorGenome instance
        predator_genome = cls.__new__(cls)
        predator_genome.__dict__.update(genome.__dict__)
        predator_genome.role = "predator"
        predator_genome.architecture = [10, 16, 12, 8, 4]
        return predator_genome

    def mutate_architecture(self):
        # Override with predator-specific architecture mutations
        # Predator focus: coordination and strategy - prefer complex networks, stable activations

        # Predator-specific mutations: bias toward complex, coordinated architectures
        if random.random() < 0.3:  # 30% chance for predator-specific mutations
            mutation_type = random.choice([
                'complex_activation', 'increase_neurons', 'add_layer',
                'enhance_learning_rule', 'add_attention'
            ])

            if mutation_type == 'complex_activation':
                # Predators can afford complex activations: elu, selu, gelu
                complex_activations = ['elu', 'selu', 'gelu', 'swish']
                for gene in self.genes[:-1]:  # Don't change output layer
                    if gene.activation not in complex_activations and random.random() < 0.6:
                        gene.activation = random.choice(complex_activations)

            elif mutation_type == 'increase_neurons':
                # Predators prefer larger networks for strategy
                for gene in self.genes[:-1]:  # Don't increase output layer
                    if gene.output_dim < 32 and random.random() < 0.5:
                        old_dim = gene.output_dim
                        gene.output_dim = min(64, old_dim + random.randint(2, 8))
                        # Resize weights accordingly
                        if gene.weights is not None:
                            new_weights = np.zeros((gene.output_dim, gene.input_dim), dtype=gene.weights.dtype)
                            new_weights[:old_dim] = gene.weights
                            gene.weights = new_weights
                        if gene.bias is not None:
                            new_bias = np.zeros(gene.output_dim, dtype=gene.bias.dtype)
                            new_bias[:old_dim] = gene.bias
                            gene.bias = new_bias

            elif mutation_type == 'add_layer' and len(self.genes) < 8:
                # Add layers for deeper processing
                if random.random() < 0.3:
                    pos = random.randint(0, len(self.genes) - 1)
                    input_dim = self.input_size if pos == 0 else self.genes[pos - 1].output_dim
                    output_dim = self.output_size if pos == len(self.genes) else self.genes[pos].input_dim

                    new_gene = NeuralGene(
                        gene_id=f"predator_layer_new_{len(self.genes)}",
                        input_dim=input_dim,
                        output_dim=output_dim,
                        activation=random.choice(['elu', 'selu', 'gelu']),
                        use_bias=True,
                        plasticity=np.random.uniform(-0.1, 0.1, (output_dim, input_dim)).astype(np.float32)
                    )
                    new_gene.initialize_weights(method="he_normal", scale=0.1)

                    self.genes.insert(pos, new_gene)
                    # Update gene IDs and skip targets
                    for idx, g in enumerate(self.genes):
                        g.gene_id = f"layer_{idx}"
                        if g.skip_connection and g.skip_target >= pos:
                            g.skip_target += 1

            elif mutation_type == 'enhance_learning_rule':
                # Predators prefer sophisticated learning rules
                if self.learning_rule_net and random.random() < 0.4:
                    # Bias toward larger hidden dims for complexity
                    if self.learning_rule_net.hidden_dim < 32:
                        self.learning_rule_net.mutate_architecture(mutation_rate=0.6)

            elif mutation_type == 'add_attention':
                # Add attention-like mechanisms (simplified as strong skip connections)
                for i, gene in enumerate(self.genes):
                    if not gene.skip_connection and i > 1 and random.random() < 0.3:
                        # Create attention-like skip from much earlier layer
                        gene.skip_connection = True
                        gene.skip_target = random.randint(0, max(0, i-2))
                        gene.skip_gate = random.uniform(0.5, 0.9)  # Strong attention

        # Update dimensions after mutations
        self._update_gene_dimensions()
        self.invalidate_caches()
        return True

class PredatorPackBrain:
    def __init__(self, genome):
        self.genome = genome

    def act(self, pred_states):
        # pred_states: [total_predators, features]
        # For now, treat each predator independently
        actions = self.genome.brain.act_batch(pred_states)

        # Post-process for pack coordination (placeholder)
        return self._coordinate_pack_actions(actions)

    def _coordinate_pack_actions(self, actions):
        # Add predator-specific coordination logic
        # Example: assign roles, coordinate attacks
        return actions
