import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from genome import EvolvableGenome, NeuralGene, Genome
from population import Population
import math


class EvolutionMetrics:
    """Track evolution metrics and statistics"""

    def __init__(self):
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.architectures_history = []
        self.plastic_usage_history = []
        self.learning_rule_history = []
        self.mutation_stats = {
            'weight_mutations': 0,
            'structural_mutations': 0,
            'layer_additions': 0,
            'layer_removals': 0,
            'neuron_changes': 0,
            'activation_changes': 0,
            'connection_changes': 0
        }
    
    def update(self, population: List[EvolvableGenome], generation: int):
        """Update metrics for current generation"""
        self.generation = generation
        fitnesses = [g.fitness for g in population]

        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(np.mean(fitnesses))

        # Track plastic usage for META-2 plot
        plastic_usages = []
        for genome in population:
            plastic_diag = getattr(genome, 'plastic_diagnostics', None)
            if plastic_diag: 
                plastic_usages.append(plastic_diag.get('mean_final_plastic_delta', 0.0))

        if plastic_usages:
            mean_plastic_usage = float(np.mean(plastic_usages))
            max_plastic_usage = float(np.max(plastic_usages))
        else:
            mean_plastic_usage = 0.0
            max_plastic_usage = 0.0

        self.plastic_usage_history.append({
            'mean': mean_plastic_usage,
            'max': max_plastic_usage
        })

        # Track architecture statistics
        arch_stats = self._get_architecture_stats(population)
        self.architectures_history.append(arch_stats)

        # Track learning rule distributions
        learning_rule_stats = self._get_learning_rule_stats(population)
        self.learning_rule_history.append(learning_rule_stats)
    
    def _get_architecture_stats(self, population: List[EvolvableGenome]) -> Dict[str, Any]:
        """Get architecture statistics for population"""
        total_layers = []
        total_params = []
        layer_counts = []
        activations_used = []
        skip_connections = []
        
        for genome in population:
            # Extract architecture info directly from genes
            total_layers.append(len(genome.genes))
            
            # Count total parameters
            total_params_count = sum(gene.input_dim * gene.output_dim for gene in genome.genes)
            total_params.append(total_params_count)
            
            # Count layer types
            layer_counts.append(len(genome.genes))
            
            # Collect activation functions
            for gene in genome.genes:
                activations_used.append(gene.activation)
            
            # Count skip connections
            skip_count = sum(1 for gene in genome.genes if hasattr(gene, 'skip_connection') and gene.skip_connection)
            skip_connections.append(skip_count)
        
        return {
            'avg_layers': np.mean(total_layers),
            'std_layers': np.std(total_layers),
            'avg_params': np.mean(total_params),
            'std_params': np.std(total_params),
            'avg_skip': np.mean(skip_connections),
            'activation_distribution': self._count_activations(activations_used)
        }
    
    def _count_activations(self, activations: List[str]) -> Dict[str, int]:
        """Count occurrence of each activation function"""
        counts = {}
        for act in activations:
            counts[act] = counts.get(act, 0) + 1
        return counts

    def _get_learning_rule_stats(self, population: List[EvolvableGenome]) -> Dict[str, Any]:
        """Get learning rule parameter distributions for population"""
        A_vals = []
        B_vals = []
        C_vals = []
        D_vals = []
        E_vals = []

        for genome in population:
            if hasattr(genome, 'learning_rule') and genome.learning_rule:
                A_vals.append(genome.learning_rule.get('A', 0.0))
                B_vals.append(genome.learning_rule.get('B', 0.0))
                C_vals.append(genome.learning_rule.get('C', 0.0))
                D_vals.append(genome.learning_rule.get('D', 0.0))
                E_vals.append(genome.learning_rule.get('E', 0.0))

        if not A_vals:
            return {
                'A': {'mean': 0.0, 'std': 0.0, 'values': []},
                'B': {'mean': 0.0, 'std': 0.0, 'values': []},
                'C': {'mean': 0.0, 'std': 0.0, 'values': []},
                'D': {'mean': 0.0, 'std': 0.0, 'values': []},
                'E': {'mean': 0.0, 'std': 0.0, 'values': []}
            }

        return {
            'A': {
                'mean': float(np.mean(A_vals)),
                'std': float(np.std(A_vals)),
                'values': A_vals
            },
            'B': {
                'mean': float(np.mean(B_vals)),
                'std': float(np.std(B_vals)),
                'values': B_vals
            },
            'C': {
                'mean': float(np.mean(C_vals)),
                'std': float(np.std(C_vals)),
                'values': C_vals
            },
            'D': {
                'mean': float(np.mean(D_vals)),
                'std': float(np.std(D_vals)),
                'values': D_vals
            },
            'E': {
                'mean': float(np.mean(E_vals)),
                'std': float(np.std(E_vals)),
                'values': E_vals
            }
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evolution summary"""
        if not self.best_fitness_history:
            return {}

        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness_history[-1],
            'avg_fitness': self.avg_fitness_history[-1],
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'plastic_usage_history': self.plastic_usage_history,
            'learning_rule_history': self.learning_rule_history,
            'current_architecture': self.architectures_history[-1] if self.architectures_history else {},
            'mutation_stats': self.mutation_stats
        }


class TournamentSelection:
    """Tournament selection with configurable strategies"""
    
    def __init__(self, tournament_size: int = 5, selection_pressure: float = 1.0):
        self.tournament_size = tournament_size
        self.selection_pressure = selection_pressure
    
    def select(self, genomes: List[EvolvableGenome], use_norm_fitness: bool = True) -> EvolvableGenome:
        """Select a genome using tournament selection"""
        if len(genomes) < self.tournament_size:
            contenders = genomes
        else:
            # Select random contenders without replacement
            contenders = random.sample(genomes, self.tournament_size)
        
        # Select based on fitness (normalized or raw)
        if use_norm_fitness:
            return max(contenders, key=lambda g: g.norm_fitness)
        else:
            return max(contenders, key=lambda g: g.fitness)


class RankBasedSelection:
    """Rank-based selection with exponential probabilities"""
    
    def __init__(self, selection_pressure: float = 1.5):
        self.selection_pressure = selection_pressure
    
    def select(self, genomes: List[EvolvableGenome]) -> EvolvableGenome:
        """Select a genome using rank-based selection"""
        # Sort genomes by fitness
        sorted_genomes = sorted(genomes, key=lambda g: g.fitness, reverse=True)
        
        # Exponential ranking probabilities
        ranks = np.arange(1, len(sorted_genomes) + 1)
        probabilities = np.exp(-self.selection_pressure * ranks / len(sorted_genomes))
        probabilities = probabilities / probabilities.sum()
        
        # Select based on rank probabilities
        selected_idx = np.random.choice(len(sorted_genomes), p=probabilities)
        return sorted_genomes[selected_idx]


class ArchitectureAwareCrossover:
    """Crossover that handles variable-length genomes with different architectures"""
    
    def __init__(self, 
                 alignment_method: str = 'similarity',
                 weight_crossover_rate: float = 0.5):
        """
        Args:
            alignment_method: 'similarity' (align by layer similarity) or 
                            'position' (align by position)
            weight_crossover_rate: Probability of taking weight from parent2
        """
        self.alignment_method = alignment_method
        self.weight_crossover_rate = weight_crossover_rate
    
    def crossover(self, parent1: EvolvableGenome, parent2: EvolvableGenome) -> EvolvableGenome:
        """
        Crossover two genomes with potentially different architectures
        
        Strategy:
        1. Align layers from both parents
        2. For aligned layers: crossover weights
        3. For unaligned layers: inherit from one parent
        4. Handle skip connections
        """
        # Create child with architecture inherited from parent1 (base architecture)
        child = parent1.copy()
        child.genome_id = f"child_{random.randint(0, 9999):04d}"
        child.fitness = 0.0
        child.norm_fitness = 0.0
        
        # Align layers
        aligned_pairs = self._align_layers(parent1, parent2)
        
        # Crossover aligned layers
        for idx1, idx2 in aligned_pairs:
            if idx1 < len(child.genes) and idx2 < len(parent2.genes):
                self._crossover_layer(child.genes[idx1], parent2.genes[idx2])
        
        # Randomly inherit some unaligned layers from parent2
        self._inherit_unaligned_layers(child, parent2, aligned_pairs)
        
        # Update gene dimensions to ensure consistency
        child._update_gene_dimensions()
        
        return child
    
    def _align_layers(self, parent1: EvolvableGenome, parent2: EvolvableGenome) -> List[Tuple[int, int]]:
        """Align layers between two parents based on similarity"""
        aligned_pairs = []
        
        if self.alignment_method == 'position':
            # Simple position-based alignment
            min_layers = min(len(parent1.genes), len(parent2.genes))
            aligned_pairs = [(i, i) for i in range(min_layers)]
        
        elif self.alignment_method == 'similarity':
            # Similarity-based alignment using layer characteristics
            for i, gene1 in enumerate(parent1.genes):
                best_match = -1
                best_similarity = -1
                
                for j, gene2 in enumerate(parent2.genes):
                    # Calculate layer similarity
                    similarity = self._layer_similarity(gene1, gene2)
                    
                    if similarity > best_similarity and j not in [p[1] for p in aligned_pairs]:
                        best_similarity = similarity
                        best_match = j
                
                if best_match != -1:
                    aligned_pairs.append((i, best_match))
        
        return aligned_pairs
    
    def _layer_similarity(self, gene1: NeuralGene, gene2: NeuralGene) -> float:
        """Calculate similarity between two layers"""
        similarity = 0.0
        
        # Dimension similarity (normalized)
        dim_similarity = 1.0 - abs(gene1.input_dim - gene2.input_dim) / max(gene1.input_dim, gene2.input_dim)
        dim_similarity += 1.0 - abs(gene1.output_dim - gene2.output_dim) / max(gene1.output_dim, gene2.output_dim)
        dim_similarity /= 2.0
        
        similarity += dim_similarity * 0.4
        
        # Activation similarity
        if gene1.activation == gene2.activation:
            similarity += 0.3
        
        # Architecture similarity
        if gene1.use_bias == gene2.use_bias:
            similarity += 0.1
        if gene1.batch_norm == gene2.batch_norm:
            similarity += 0.1
        if gene1.skip_connection == gene2.skip_connection:
            similarity += 0.1
        
        return similarity
    
    def _crossover_layer(self, child_gene: NeuralGene, parent2_gene: NeuralGene):
        """Crossover weights and biases between two aligned layers"""
        # Skip if dimensions don't match
        if (child_gene.input_dim != parent2_gene.input_dim or 
            child_gene.output_dim != parent2_gene.output_dim):
            return
        
        # Crossover weights
        if child_gene.weights is not None and parent2_gene.weights is not None:
            mask = np.random.random(child_gene.weights.shape) < self.weight_crossover_rate
            child_gene.weights = np.where(mask, parent2_gene.weights, child_gene.weights)
        
        # Crossover bias
        if (child_gene.use_bias and parent2_gene.use_bias and 
            child_gene.bias is not None and parent2_gene.bias is not None):
            mask = np.random.random(child_gene.bias.shape) < self.weight_crossover_rate
            child_gene.bias = np.where(mask, parent2_gene.bias, child_gene.bias)
        
        # Crossover batch norm parameters
        if child_gene.batch_norm and parent2_gene.batch_norm:
            # Average gamma and beta
            if child_gene.bn_gamma is not None and parent2_gene.bn_gamma is not None:
                child_gene.bn_gamma = (child_gene.bn_gamma + parent2_gene.bn_gamma) / 2
            if child_gene.bn_beta is not None and parent2_gene.bn_beta is not None:
                child_gene.bn_beta = (child_gene.bn_beta + parent2_gene.bn_beta) / 2
            
            # For running stats, inherit from parent with better fitness or average
            if child_gene.bn_running_mean is not None and parent2_gene.bn_running_mean is not None:
                child_gene.bn_running_mean = (child_gene.bn_running_mean + parent2_gene.bn_running_mean) / 2
            if child_gene.bn_running_var is not None and parent2_gene.bn_running_var is not None:
                child_gene.bn_running_var = (child_gene.bn_running_var + parent2_gene.bn_running_var) / 2
        
        # Randomly inherit activation (30% chance)
        if random.random() < 0.3:
            child_gene.activation = parent2_gene.activation
    
    def _inherit_unaligned_layers(self, 
                                 child: EvolvableGenome, 
                                 parent2: EvolvableGenome, 
                                 aligned_pairs: List[Tuple[int, int]]):
        """Randomly inherit unaligned layers from parent2"""
        # Get indices of parent2 layers not aligned
        parent2_aligned_indices = {idx2 for _, idx2 in aligned_pairs}
        parent2_unaligned_indices = [i for i in range(len(parent2.genes)) 
                                     if i not in parent2_aligned_indices]
        
        if not parent2_unaligned_indices:
            return
        
        # Randomly select some unaligned layers to inherit
        num_to_inherit = random.randint(0, min(2, len(parent2_unaligned_indices)))
        if num_to_inherit == 0:
            return
        
        selected_indices = random.sample(parent2_unaligned_indices, num_to_inherit)
        
        for idx in selected_indices:
            # Insert at random position in child
            insert_pos = random.randint(0, len(child.genes))
            gene_copy = parent2.genes[idx].copy()
            
            # Update gene_id to avoid conflicts
            gene_copy.gene_id = f"inherited_{idx}"
            
            child.genes.insert(insert_pos, gene_copy)


class AdaptiveMutation:
    """Adaptive mutation with dynamic rates based on population diversity"""
    
    def __init__(self,
                 base_weight_rate: float = 0.1,
                 base_weight_strength: float = 0.1,
                 base_arch_rate: float = 0.05,
                 base_layer_rate: float = 0.02,
                 min_rate: float = 0.001,
                 max_rate: float = 0.3):
        """
        Args:
            base_weight_rate: Base rate for weight mutations
            base_weight_strength: Base strength for weight mutations
            base_arch_rate: Base rate for architecture mutations
            base_layer_rate: Base rate for layer mutations
            min_rate: Minimum mutation rate
            max_rate: Maximum mutation rate
        """
        self.base_weight_rate = base_weight_rate
        self.base_weight_strength = base_weight_strength
        self.base_arch_rate = base_arch_rate
        self.base_layer_rate = base_layer_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        # Track mutation success rates
        self.success_rates = {
            'weight': 0.5,
            'arch': 0.5,
            'layer': 0.5
        }
    
    def get_adaptive_rates(self, 
                          population: List[EvolvableGenome],
                          generation: int) -> Dict[str, float]:
        """
        Calculate adaptive mutation rates based on population diversity
        """
        # Calculate population diversity
        diversity = self._calculate_diversity(population)
        
        # Adjust rates based on diversity
        # Low diversity → increase mutation rates
        # High diversity → decrease mutation rates
        
        diversity_factor = 1.0 - diversity  # Inverse relationship
        
        weight_rate = np.clip(
            self.base_weight_rate * (1.0 + diversity_factor),
            self.min_rate, self.max_rate
        )
        
        arch_rate = np.clip(
            self.base_arch_rate * (1.0 + diversity_factor * 2),
            self.min_rate, self.max_rate
        )
        
        layer_rate = np.clip(
            self.base_layer_rate * (1.0 + diversity_factor * 2),
            self.min_rate, self.max_rate
        )
        
        # Adjust based on stagnation
        if generation > 50:
            # Check if fitness is stagnating
            stagnation_factor = self._calculate_stagnation(population, generation)
            if stagnation_factor > 0.8:
                # Increase mutation rates to escape local optima
                weight_rate = min(weight_rate * 1.5, self.max_rate)
                arch_rate = min(arch_rate * 2.0, self.max_rate)
                layer_rate = min(layer_rate * 2.0, self.max_rate)
        
        return {
            'weight_rate': weight_rate,
            'weight_strength': self.base_weight_strength,
            'arch_rate': arch_rate,
            'layer_rate': layer_rate
        }
    
    def _calculate_diversity(self, population: List[EvolvableGenome]) -> float:
        """Calculate population diversity (0-1)"""
        if len(population) < 2:
            return 0.0
        
        # Measure diversity in multiple dimensions
        
        # 1. Fitness diversity
        fitnesses = [g.fitness for g in population]
        fitness_std = np.std(fitnesses) if len(fitnesses) > 1 else 0.0
        fitness_diversity = min(fitness_std / (np.mean(fitnesses) + 1e-10), 1.0)
        
        # 2. Architecture diversity
        layer_counts = [len(g.genes) for g in population]
        layer_diversity = np.std(layer_counts) / max(np.mean(layer_counts), 1.0)
        
        # 3. Parameter diversity (simplified)
        param_counts = [sum(gene.input_dim * gene.output_dim for gene in g.genes) 
                       for g in population]
        param_diversity = np.std(param_counts) / max(np.mean(param_counts), 1.0)
        
        # Combine diversities
        total_diversity = (fitness_diversity + layer_diversity + param_diversity) / 3.0
        
        return float(min(total_diversity, 1.0))
    
    def _calculate_stagnation(self, 
                            population: List[EvolvableGenome], 
                            generation: int) -> float:
        """Calculate stagnation factor (0-1)"""
        # Simplified: check if best fitness hasn't improved in last N generations
        # This would need access to fitness history
        return 0.0  # Placeholder
    
    def update_success_rates(self, 
                           parent_fitness: float, 
                           child_fitness: float,
                           mutation_type: str):
        """Update mutation success rates based on fitness improvement"""
        improvement = child_fitness - parent_fitness
        
        if improvement > 0:
            # Successful mutation
            self.success_rates[mutation_type] = (
                0.9 * self.success_rates[mutation_type] + 0.1
            )
        else:
            # Unsuccessful mutation
            self.success_rates[mutation_type] = (
                0.9 * self.success_rates[mutation_type]
            )


class EvolutionEngine:
    def __init__(
        self,
        population_size,
        tournament_size,
        elite_count,
        mutation_rate,
        mutation_strength,
        architecture_mutation_rate,
        genome_cls,
    ):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.architecture_mutation_rate = architecture_mutation_rate
        self.genome_cls = genome_cls
        self.selector = TournamentSelection(tournament_size)
        self.metrics = EvolutionMetrics()
        self.best_fitness_history = []
        self.stagnation_counter = 0
        self.generation = 0
        self.mutator = None

    def _calculate_adaptability_score(self, genome: EvolvableGenome, plastic_usage: float) -> float:
        """
        Calculate how effectively a genome uses its meta-parameters and plasticity.
        This rewards genomes that adapt well, not just those that use lots of plasticity.

        Args:
            genome: The genome to evaluate
            plastic_usage: Raw plastic usage from diagnostics

        Returns:
            Adaptability score (0-1) where higher is better adaptation
        """
        if not hasattr(genome, 'plastic_diagnostics'):
            return 0.0
        
        plastic_diag = getattr(genome, 'plastic_diagnostics', None)
        if not plastic_diag:
            return 0.0

        # Get meta-parameters
        meta_gain = getattr(genome, 'meta', {}).get('reward_gain', 1.0)
        meta_bias = getattr(genome, 'meta', {}).get('reward_bias', 0.0)
        plastic_lr = getattr(genome, 'meta', {}).get('plastic_lr', 1.0)

        # Get plasticity diagnostics
        mean_delta = plastic_diag.get('mean_final_plastic_delta', 0.0)
        max_delta = plastic_diag.get('max_plastic_delta', 0.0)
        stability = 1.0 - abs(mean_delta)  # Lower absolute plasticity = more stable

        # META-3.2: Calculate meta-parameter effectiveness
        # Reward genomes that have well-tuned meta-parameters for their plasticity usage

        # 1. Plasticity efficiency: reward moderate, controlled plasticity
        if plastic_usage > 0:
            # Optimal plasticity is moderate - not too little, not too much
            plasticity_efficiency = 1.0 - abs(plastic_usage - 0.3) / 0.3  # Peak at 0.3
            plasticity_efficiency = max(0.0, plasticity_efficiency)
        else:
            plasticity_efficiency = 0.0

        # 2. Meta-parameter coherence: reward meta-params that match plasticity behavior
        meta_coherence = 0.0
        if plastic_usage > 0.1:  # Only evaluate if there's meaningful plasticity
            # High plastic_lr should correlate with high plasticity usage
            lr_effectiveness = min(plastic_lr / 10.0, 1.0) * (plastic_usage / 0.5)
            # Appropriate meta_gain should correlate with fitness improvement
            gain_effectiveness = min(abs(meta_gain) / 5.0, 1.0)
            meta_coherence = (lr_effectiveness + gain_effectiveness) / 2.0

        # 3. Stability bonus: reward genomes that maintain stability while adapting
        stability_bonus = stability * 0.3

        # Combine scores
        adaptability_score = (
            plasticity_efficiency * 0.4 +      # 40% - efficient plasticity usage
            meta_coherence * 0.4 +             # 40% - well-tuned meta-parameters
            stability_bonus * 0.2              # 20% - stability while adapting
        )

        return float(max(0.0, min(1.0, adaptability_score)))

    @staticmethod
    def calculate_effective_fitness(raw_fitness: float, meta_gain: float, adaptability_score: float,
                                   threshold: float = 0.5, instability_penalty: float = 1.0,
                                   genome: Optional['EvolvableGenome'] = None) -> float:
        """
        Calculate effective fitness incorporating adaptability score, stability penalties,
        and parameter complexity penalties.

        Args:
            raw_fitness: Raw fitness from evaluation
            meta_gain: META gene reward gain
            adaptability_score: Score measuring effective adaptation (0-1)
            threshold: Plastic usage threshold for penalty
            instability_penalty: Penalty applied when plastic usage exceeds threshold
            genome: Genome to calculate parameter penalty for

        Returns:
            Effective fitness value
        """
        # META-3.2: Base fitness plus adaptability bonus
        effective_fitness = raw_fitness + meta_gain * adaptability_score

        # Stability penalty for excessive plasticity (passed through adaptability_score)
        if adaptability_score < 0.2:  # Very poor adaptation
            effective_fitness -= instability_penalty * 0.5

        # Parameter complexity penalty: fitness - λ * log(params)
        # Penalize huge models or slow inference
        if genome is not None:
            total_params = sum(
                gene.input_dim * gene.output_dim +
                (gene.output_dim if gene.use_bias else 0) +
                (gene.output_dim * 4 if gene.batch_norm else 0)  # BN params
                for gene in genome.genes
            )

            # λ = 0.1, penalize log of total parameters
            lambda_penalty = 0.1
            if total_params > 0:
                param_penalty = lambda_penalty * math.log(total_params)
                effective_fitness -= param_penalty

        return effective_fitness

    def create_next_generation(self, population, generation: int):
        """
        CO-EVOLUTION SAFE
        - Never collapses population
        - Never returns empty
        - Deterministic behavior
        """

        # ----------------------------
        # EXTRACT GENOMES AND NAME
        # ----------------------------
        if isinstance(population, list):
            genomes = population
            pop_name = "population"
        else:
            genomes = population.genomes
            pop_name = population.name

        # ----------------------------
        # SAFETY CHECK
        # ----------------------------
        if not genomes:
            print("⚠ Population empty — reinitializing")
            return Population(
                size=self.population_size,
                name=pop_name
            )

        # Calculate effective fitness before normalization
        for genome in genomes:
            if hasattr(genome, 'plastic_diagnostics') and genome.plastic_diagnostics:
                plastic_usage = genome.plastic_diagnostics.get('mean_final_plastic_delta', 0.0)
                meta_gain = genome.meta.get('reward_gain', 0.0) if hasattr(genome, 'meta') else 0.0

                # META-3.2: Calculate meta-parameter effectiveness
                # Reward genomes that use plasticity effectively, not just those that use it a lot
                base_fitness = genome.fitness
                adaptability_score = self._calculate_adaptability_score(genome, plastic_usage)

                effective_fitness = self.calculate_effective_fitness(
                    base_fitness, meta_gain, adaptability_score
                )
                genome.fitness = effective_fitness

                # META-3.2 learning rule regularization (now using learning_rule_net)
                if hasattr(genome, 'learning_rule_net'):
                    # Use the network's static parameters for regularization
                    static_params = genome.learning_rule_net.get_parameters_as_dict()
                    rule_penalty = sum(abs(v) for v in static_params.values())
                    genome.fitness -= 0.01 * rule_penalty

        normalize_fitness(genomes)

        new_genomes = []

        # ----------------------------
        # ELITISM
        # ----------------------------
        elites = sorted(genomes, key=lambda g: g.fitness, reverse=True)
        elites = elites[: min(self.elite_count, len(elites))]

        for elite in elites:
            clone = elite.copy()
            clone.fitness = 0.0
            clone.norm_fitness = 0.0
            new_genomes.append(clone)

        # ----------------------------
        # OFFSPRING
        # ----------------------------
        while len(new_genomes) < self.population_size:

            parent1 = self.selector.select(genomes)
            parent2 = self.selector.select(genomes)

            if parent1.genome_id == parent2.genome_id:
                parent2 = self.selector.select(genomes)

            try:
                child = self.genome_cls.crossover(parent1, parent2)
            except Exception:
                child = parent1.copy()

            mutated = child.mutate(
                weight_mutation_rate=self.mutation_rate,
                weight_mutation_strength=self.mutation_strength,
                architecture_mutation_rate=self.architecture_mutation_rate
            )

            if mutated is None or isinstance(mutated, bool):
                mutated = child.copy()

            mutated.fitness = 0.0
            mutated.norm_fitness = 0.0
            new_genomes.append(mutated)

        # ----------------------------
        # FINAL POPULATION
        # ----------------------------
        new_population = Population(size=0, name=pop_name)
        new_population.genomes = new_genomes
        return new_population

    
    def _select_elites(self, 
                      population: List[EvolvableGenome], 
                      elite_count: int) -> List[EvolvableGenome]:
        """Select elite genomes"""
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
        
        # Return copies of elites
        elites = []
        for i in range(min(elite_count, len(sorted_pop))):
            elite_copy = sorted_pop[i].copy()
            elite_copy.age += 1  # Increment age
            elites.append(elite_copy)
        
        return elites
    
    def _check_stagnation(self, population: List[EvolvableGenome]):
        """Check for stagnation and adjust parameters if needed"""
        current_best = max(g.fitness for g in population)
        
        if not self.best_fitness_history:
            self.best_fitness_history.append(current_best)
            return
        
        last_best = self.best_fitness_history[-1]
        
        if current_best <= last_best + 1e-6:  # No improvement
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.best_fitness_history.append(current_best)
        
        # If stagnating for too long, increase mutation rates
        if self.stagnation_counter > 50 and self.mutator:
            # Dynamically adjust base rates
            self.mutator.base_arch_rate = min(
                self.mutator.base_arch_rate * 1.5, 
                self.mutator.max_rate
            )
            self.mutator.base_layer_rate = min(
                self.mutator.base_layer_rate * 1.5,
                self.mutator.max_rate
            )
            print(f"Stagnation detected (gen {self.generation}), increasing mutation rates")
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution progress"""
        return self.metrics.get_summary()


class SimpleCrossover:
    """Simple crossover for backward compatibility"""
    
    def crossover(self, parent1: EvolvableGenome, parent2: EvolvableGenome) -> EvolvableGenome:
        """Simple uniform crossover"""
        # Inherit architecture from parent1
        child = parent1.copy()
        
        # Only crossover weights if architectures match
        if len(parent1.genes) == len(parent2.genes):
            for i, (gene1, gene2) in enumerate(zip(parent1.genes, parent2.genes)):
                if (gene1.input_dim == gene2.input_dim and 
                    gene1.output_dim == gene2.output_dim):
                    # Uniform crossover for weights
                    mask = np.random.random(gene1.weights.shape) < 0.5
                    child.genes[i].weights = np.where(mask, gene1.weights, gene2.weights)
                    
                    if (gene1.use_bias and gene2.use_bias and 
                        gene1.bias is not None and gene2.bias is not None):
                        bias_mask = np.random.random(gene1.bias.shape) < 0.5
                        child.genes[i].bias = np.where(bias_mask, gene1.bias, gene2.bias)
        
        return child


# Legacy functions for backward compatibility
def tournament_selection(genomes: List[EvolvableGenome], tournament_size: int = 5) -> EvolvableGenome:
    """Legacy tournament selection function"""
    selector = TournamentSelection(tournament_size)
    return selector.select(genomes)


def get_elites(genomes: List[EvolvableGenome], elite_count: int = 5) -> List[EvolvableGenome]:
    """Legacy function to get elites"""
    sorted_genomes = sorted(genomes, key=lambda g: g.fitness, reverse=True)
    elites = []
    for g in sorted_genomes[:elite_count]:
        elite_copy = g.copy()
        elite_copy.age += 1
        elites.append(elite_copy)
    return elites

def normalize_fitness(genomes: List[EvolvableGenome]):
    """Normalize fitness values (z-score normalization)"""
    if not genomes:
        return
    
    fitnesses = np.array([g.fitness for g in genomes], dtype=np.float32)
    
    mean = np.mean(fitnesses)
    std = np.std(fitnesses)
    
    if std < 1e-10:
        for g in genomes:
            g.norm_fitness = 0.0
        return
    
    for g, fitness in zip(genomes, fitnesses):
        g.norm_fitness = (fitness - mean) / std


def rank_based_selection(genomes: List[EvolvableGenome], selection_pressure: float = 1.5) -> EvolvableGenome:
    """Legacy rank-based selection function"""
    selector = RankBasedSelection(selection_pressure)
    return selector.select(genomes)
    