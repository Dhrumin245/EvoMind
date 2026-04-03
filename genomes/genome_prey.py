import random
import numpy as np
from core.genome import Genome
from core.genome import NeuralGene

class PreyGenome(Genome):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role = "prey"
        # Prey-specific neural architecture (evolvable)
        self.architecture = [8, 12, 6, 4]  # Different from predator

    @classmethod
    def random_initialization(cls, input_size=8, output_size=4):
        """Create a random prey genome"""
        return cls(
            input_size=input_size,
            output_size=output_size,
            init_modules=3,
            init_neurons=12
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'PreyGenome':
        """Create prey genome from dictionary"""
        genome = super().from_dict(data)
        # Convert to PreyGenome instance
        prey_genome = cls.__new__(cls)
        prey_genome.__dict__.update(genome.__dict__)
        prey_genome.role = "prey"
        prey_genome.architecture = [8, 12, 6, 4]
        return prey_genome

    def mutate_architecture(self):
        # Override with prey-specific architecture mutations
        # Prey focus: evasion and speed - prefer smaller networks, faster activations

        # Prey-specific mutations: preserve agility, but do not keep shrinking
        # capacity when the training loop is already reporting under-capacity.
        if random.random() < 0.3:  # 30% chance for prey-specific mutations
            mutation_type = random.choice([
                'simplify_activation', 'tune_capacity', 'add_skip_connection',
                'add_layer', 'mutate_learning_rule'
            ])

            if mutation_type == 'simplify_activation':
                # Prey prefer fast activations: tanh, leaky_relu, linear
                fast_activations = ['tanh', 'leaky_relu', 'linear']
                for gene in self.genes[:-1]:  # Don't change output layer
                    if gene.activation not in fast_activations:
                        gene.activation = random.choice(fast_activations)

            elif mutation_type == 'tune_capacity':
                # Keep prey fairly lean, but allow moderate capacity growth so
                # useful policies are not bottlenecked by undersized layers.
                for gene in self.genes[:-1]:  # Don't reduce output layer
                    if gene.output_dim < 24 and random.random() < 0.5:
                        old_dim = gene.output_dim
                        gene.output_dim = min(24, old_dim + random.randint(2, 6))
                        if gene.weights is not None:
                            new_weights = np.zeros((gene.output_dim, gene.input_dim), dtype=gene.weights.dtype)
                            new_weights[:old_dim] = gene.weights
                            gene.weights = new_weights
                        if gene.bias is not None:
                            new_bias = np.zeros(gene.output_dim, dtype=gene.bias.dtype)
                            new_bias[:old_dim] = gene.bias
                            gene.bias = new_bias
                    elif gene.output_dim > 10 and random.random() < 0.2:
                        old_dim = gene.output_dim
                        gene.output_dim = max(8, old_dim - random.randint(1, 2))
                        if gene.weights is not None:
                            gene.weights = gene.weights[:gene.output_dim]
                        if gene.bias is not None:
                            gene.bias = gene.bias[:gene.output_dim]

            elif mutation_type == 'add_skip_connection':
                # Add skip connections for faster information flow
                for i, gene in enumerate(self.genes):
                    if not gene.skip_connection and i > 0 and random.random() < 0.4:
                        gene.skip_connection = True
                        gene.skip_target = random.randint(0, i-1)
                        gene.skip_gate = random.uniform(0.1, 0.5)  # Subtle skip

            elif mutation_type == 'add_layer' and len(self.genes) < 6:
                if random.random() < 0.2:
                    pos = random.randint(0, len(self.genes) - 1)
                    input_dim = self.input_size if pos == 0 else self.genes[pos - 1].output_dim
                    output_dim = self.output_size if pos == len(self.genes) else self.genes[pos].input_dim

                    new_gene = NeuralGene(
                        gene_id=f"prey_layer_new_{len(self.genes)}",
                        input_dim=input_dim,
                        output_dim=output_dim,
                        activation=random.choice(['tanh', 'elu', 'leaky_relu']),
                        use_bias=True,
                        plasticity=np.random.uniform(-0.1, 0.1, (output_dim, input_dim)).astype(np.float32)
                    )
                    new_gene.initialize_weights(method="he_normal", scale=0.1)

                    self.genes.insert(pos, new_gene)
                    for idx, g in enumerate(self.genes):
                        g.gene_id = f"layer_{idx}"
                        if g.skip_connection and g.skip_target >= pos:
                            g.skip_target += 1

            elif mutation_type == 'mutate_learning_rule':
                # Prey prefer stable, fast learning
                if self.learning_rule_net and random.random() < 0.3:
                    if self.learning_rule_net.hidden_dim < 24:
                        self.learning_rule_net.mutate_architecture(mutation_rate=0.7)

        # Update dimensions after mutations
        self._update_gene_dimensions()
        self.invalidate_caches()
        return True
