from genome import Genome

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
        # Example: mutate neuron counts, add/remove layers, change activations
        pass
