import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from core.genome import EvolvableGenome, NeuralGene, Genome
from core.population import Population
import math
from dataclasses import dataclass, field


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

        # Milestone 5: Discover motifs from high-performing genomes
        if generation % 5 == 0:  # Discover motifs every 5 generations
            min_fitness_top25 = float(np.percentile(fitnesses, 75))  # Top 25% of population
            for genome in population:
                if hasattr(genome, 'motif_library') and genome.motif_library is not None:
                    genome.motif_library.discover_motifs(population, generation, min_fitness=min_fitness_top25)

        # Milestone 5: Discover motifs from high-performing genomes
        if population and hasattr(population[0], 'motif_library') and population[0].motif_library is not None:
            mean_fitness = float(np.mean([g.fitness for g in population]))
            population[0].motif_library.discover_motifs(population, generation, min_fitness=mean_fitness)
    
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
        speciation_enabled: bool = True,
        novelty_archive_enabled: bool = True,
        # Speciation knobs
        compatibility_threshold: float = 3.0,
        speciation_architecture_weight: float = 0.3,
        speciation_behavior_weight: float = 0.4,
        speciation_param_weight: float = 0.3,
        min_species_size: int = 5,
        max_species_stagnation: int = 15,
        # Novelty knobs
        novelty_threshold: float = 0.1,
        max_archive_size: int = 100,
        immigration_rate: float = 0.1,
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

        # Speciation components
        self.speciation_enabled = speciation_enabled
        distance_calculator = (
            GenomeDistance(
                architecture_weight=speciation_architecture_weight,
                behavior_weight=speciation_behavior_weight,
                param_weight=speciation_param_weight,
            )
            if speciation_enabled
            else None
        )
        self.distance_calculator = distance_calculator
        self.speciation_manager = (
            SpeciationManager(
                distance_calculator=distance_calculator,
                compatibility_threshold=compatibility_threshold,
                min_species_size=min_species_size,
                max_stagnation=max_species_stagnation,
            )
            if distance_calculator is not None
            else None
        )

        # Novelty archive components
        self.novelty_archive_enabled = novelty_archive_enabled
        self.novelty_archive = NoveltyArchive(
            max_size=max_archive_size,
            novelty_threshold=novelty_threshold
        ) if novelty_archive_enabled else None
        self.immigration_rate = immigration_rate

    def compute_species_stats(self, genomes: List['EvolvableGenome'], generation: int) -> Dict[str, Any]:
        """Compute and return current speciation stats for a population.

        This is safe to call for logging; it does not alter genome fitness.
        """
        if not self.speciation_enabled or self.speciation_manager is None:
            return {'num_species': 0, 'avg_species_size': 0.0, 'total_members': 0, 'species_sizes': []}

        # Deterministic assignment given fixed genome order/fields.
        self.speciation_manager.speciate_population(genomes, generation)
        return self.speciation_manager.get_species_stats()

    def _build_novelty_embedding(self, genome: 'EvolvableGenome', generation: int) -> 'BehaviorEmbedding':
        """Build a stable, fixed-size embedding for novelty scoring.

        Note: this is a lightweight proxy embedding (architecture + meta + fitness),
        suitable for novelty tracking/logging even before richer behavior features exist.
        """
        meta = getattr(genome, 'meta', {}) or {}
        reward_gain = float(meta.get('reward_gain', 0.0))
        reward_bias = float(meta.get('reward_bias', 0.0))
        plastic_lr = float(meta.get('plastic_lr', 0.0))
        num_layers = float(len(getattr(genome, 'genes', []) or []))
        total_params = 0.0
        for gene in getattr(genome, 'genes', []) or []:
            try:
                total_params += float(gene.input_dim * gene.output_dim)
            except Exception:
                continue

        embedding = np.array(
            [
                float(getattr(genome, 'fitness', 0.0)),
                reward_gain,
                reward_bias,
                plastic_lr,
                num_layers,
                total_params,
            ],
            dtype=np.float32,
        )

        return BehaviorEmbedding(
            genome_id=str(getattr(genome, 'genome_id', 'unknown')),
            fitness=float(getattr(genome, 'fitness', 0.0)),
            embedding=embedding,
            generation=int(generation),
        )

    def compute_novelty_stats(
        self,
        genomes: List['EvolvableGenome'],
        generation: int,
        add_top_k_to_archive: int = 5,
    ) -> Dict[str, Any]:
        """Compute novelty scores for a population and update archive with top-K.

        Returns summary stats suitable for logging.
        """
        if not self.novelty_archive_enabled or self.novelty_archive is None:
            return {
                'mean': 0.0,
                'max': 0.0,
                'p95': 0.0,
                'archive': {'size': 0, 'avg_fitness': 0.0, 'generations_covered': 0},
            }

        if not genomes:
            return {
                'mean': 0.0,
                'max': 0.0,
                'p95': 0.0,
                'archive': self.novelty_archive.get_archive_stats(),
            }

        embeddings = [self._build_novelty_embedding(g, generation) for g in genomes]
        raw_scores = [self.novelty_archive.get_novelty_score(e) for e in embeddings]
        scores = [float(s) if np.isfinite(s) else 0.0 for s in raw_scores]

        # Update archive with top-K by novelty score.
        if add_top_k_to_archive and add_top_k_to_archive > 0:
            top = sorted(zip(scores, embeddings), key=lambda t: t[0], reverse=True)[:add_top_k_to_archive]
            for _, emb in top:
                self.novelty_archive.add_behavior(emb)

        arr = np.asarray(scores, dtype=np.float32)
        return {
            'mean': float(np.mean(arr)) if arr.size else 0.0,
            'max': float(np.max(arr)) if arr.size else 0.0,
            'p95': float(np.percentile(arr, 95)) if arr.size else 0.0,
            'archive': self.novelty_archive.get_archive_stats(),
        }

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
                # Set parent IDs and birth generation for lineage tracking
                child.set_parents([parent1.genome_id, parent2.genome_id], generation)
            except Exception:
                child = parent1.copy()
                # For fallback, set single parent
                child.set_parents([parent1.genome_id], generation)

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


@dataclass
class BehaviorEmbedding:
    """Represents a genome's behavior for novelty calculation"""
    genome_id: str
    fitness: float
    embedding: np.ndarray  # Behavior vector (e.g., action frequencies, state visitations)
    generation: int
    genome: Optional['EvolvableGenome'] = None  # Store the actual genome for injection

    def distance_to(self, other: 'BehaviorEmbedding') -> float:
        """Calculate behavioral distance to another embedding"""
        return float(np.linalg.norm(self.embedding - other.embedding))


class GenomeDistance:
    """Calculates distance between genomes for speciation and novelty"""

    def __init__(self, architecture_weight: float = 0.3, behavior_weight: float = 0.4, param_weight: float = 0.3):
        self.architecture_weight = architecture_weight
        self.behavior_weight = behavior_weight
        self.param_weight = param_weight

    def calculate_distance(self, genome1: EvolvableGenome, genome2: EvolvableGenome) -> float:
        """Calculate overall distance between two genomes with topology-aware and plasticity-aware speciation"""
        arch_dist = self._architecture_distance(genome1, genome2)
        behavior_dist = self._behavior_distance(genome1, genome2)
        param_dist = self._parameter_distance(genome1, genome2)

        # Add topology-aware distance (structural connectivity patterns)
        topology_dist = self._topology_distance(genome1, genome2)

        # Add plasticity-aware distance (learning rule and meta-parameter differences)
        plasticity_dist = self._plasticity_distance(genome1, genome2)

        total_distance = (
            self.architecture_weight * arch_dist +
            self.behavior_weight * behavior_dist +
            self.param_weight * param_dist +
            0.1 * topology_dist +  # Weight for topology awareness
            0.15 * plasticity_dist  # Weight for plasticity awareness
        )

        return float(total_distance)

    def _architecture_distance(self, genome1: EvolvableGenome, genome2: EvolvableGenome) -> float:
        """Calculate architectural distance"""
        # Layer count difference
        layer_diff = abs(len(genome1.genes) - len(genome2.genes))
        max_layers = max(len(genome1.genes), len(genome2.genes))

        # Activation function differences
        activation_diffs = 0
        min_layers = min(len(genome1.genes), len(genome2.genes))
        for i in range(min_layers):
            if genome1.genes[i].activation != genome2.genes[i].activation:
                activation_diffs += 1

        # Dimension differences
        dim_diffs = 0
        for i in range(min_layers):
            gene1, gene2 = genome1.genes[i], genome2.genes[i]
            dim_diffs += abs(gene1.input_dim - gene2.input_dim) + abs(gene1.output_dim - gene2.output_dim)

        # Normalize and combine
        arch_distance = (
            (layer_diff / max(max_layers, 1)) * 0.4 +
            (activation_diffs / max(min_layers, 1)) * 0.3 +
            min(dim_diffs / 100.0, 1.0) * 0.3  # Cap dimension differences
        )

        return float(arch_distance)

    def _behavior_distance(self, genome1: EvolvableGenome, genome2: EvolvableGenome) -> float:
        """Calculate behavioral distance based on fitness and meta-parameters"""
        # Use fitness difference as proxy for behavioral similarity
        fitness_diff = abs(genome1.fitness - genome2.fitness)
        max_fitness = max(abs(genome1.fitness), abs(genome2.fitness), 1.0)

        # Meta-parameter differences
        meta1 = genome1.meta
        meta2 = genome2.meta
        meta_diff = 0.0
        for key in set(meta1.keys()) | set(meta2.keys()):
            val1 = meta1.get(key, 0.0)
            val2 = meta2.get(key, 0.0)
            meta_diff += abs(val1 - val2)

        behavior_distance = (
            (fitness_diff / max_fitness) * 0.6 +
            min(meta_diff / 10.0, 1.0) * 0.4  # Normalize meta differences
        )

        return float(behavior_distance)

    def _parameter_distance(self, genome1: EvolvableGenome, genome2: EvolvableGenome) -> float:
        """Calculate parameter-level distance"""
        # Compare weight magnitudes and learning rule parameters
        param_dist = 0.0

        # Weight magnitude differences
        weights1 = []
        weights2 = []
        for gene in genome1.genes:
            if gene.weights is not None:
                weights1.extend(np.abs(gene.weights).flatten())
        for gene in genome2.genes:
            if gene.weights is not None:
                weights2.extend(np.abs(gene.weights).flatten())

        if weights1 and weights2:
            mean_weight1 = np.mean(weights1)
            mean_weight2 = np.mean(weights2)
            weight_diff = abs(mean_weight1 - mean_weight2) / max(mean_weight1, mean_weight2, 1e-6)
            param_dist += weight_diff * 0.5

        # Learning rule differences (if available)
        lr1 = getattr(genome1, 'learning_rule_net', None)
        lr2 = getattr(genome2, 'learning_rule_net', None)
        if lr1 is not None and lr2 is not None and hasattr(lr1, 'get_parameters_as_dict') and hasattr(lr2, 'get_parameters_as_dict'):
            params1 = lr1.get_parameters_as_dict()
            params2 = lr2.get_parameters_as_dict()

            rule_diff = 0.0
            for key in set(params1.keys()) | set(params2.keys()):
                val1 = params1.get(key, 0.0)
                val2 = params2.get(key, 0.0)
                rule_diff += abs(val1 - val2)

            param_dist += min(rule_diff / 10.0, 1.0) * 0.5

        return float(param_dist)

    def _topology_distance(self, genome1: EvolvableGenome, genome2: EvolvableGenome) -> float:
        """Calculate topology-aware distance (structural connectivity patterns)"""
        # Compare skip connections and module connections
        topology_dist = 0.0

        # Compare skip connection patterns
        skip1 = [getattr(g, 'skip_connection', False) for g in genome1.genes]
        skip2 = [getattr(g, 'skip_connection', False) for g in genome2.genes]

        min_len = min(len(skip1), len(skip2))
        if min_len > 0:
            skip_diff = sum(s1 != s2 for s1, s2 in zip(skip1[:min_len], skip2[:min_len]))
            topology_dist += skip_diff / min_len * 0.5

        # Compare module connections if they exist
        if hasattr(genome1, 'module_connections') and hasattr(genome2, 'module_connections'):
            conn1 = getattr(genome1, 'module_connections', [])
            conn2 = getattr(genome2, 'module_connections', [])

            if conn1 and conn2:
                # Simple connection count difference
                conn_diff = abs(len(conn1) - len(conn2))
                max_conn = max(len(conn1), len(conn2))
                topology_dist += min(conn_diff / max(max_conn, 1), 1.0) * 0.5

        return float(topology_dist)

    def _plasticity_distance(self, genome1: EvolvableGenome, genome2: EvolvableGenome) -> float:
        """Calculate plasticity-aware distance (learning rule and meta-parameter differences)"""
        plasticity_dist = 0.0

        # Compare meta-parameters
        meta1 = getattr(genome1, 'meta', {})
        meta2 = getattr(genome2, 'meta', {})

        meta_keys = set(meta1.keys()) | set(meta2.keys())
        meta_diffs = []

        for key in meta_keys:
            val1 = meta1.get(key, 0.0)
            val2 = meta2.get(key, 0.0)
            if abs(val1 + val2) > 1e-6:  # Avoid division by zero
                diff = abs(val1 - val2) / (abs(val1 + val2) / 2.0)  # Relative difference
                meta_diffs.append(diff)

        if meta_diffs:
            plasticity_dist += np.mean(meta_diffs) * 0.6

        # Compare learning rule networks if they exist
        lr1 = getattr(genome1, 'learning_rule_net', None)
        lr2 = getattr(genome2, 'learning_rule_net', None)

        if lr1 is not None and lr2 is not None and hasattr(lr1, 'get_parameters_as_dict') and hasattr(lr2, 'get_parameters_as_dict'):
            try:
                params1 = lr1.get_parameters_as_dict()
                params2 = lr2.get_parameters_as_dict()

                rule_diffs = []
                for key in set(params1.keys()) | set(params2.keys()):
                    val1 = params1.get(key, 0.0)
                    val2 = params2.get(key, 0.0)
                    if abs(val1 + val2) > 1e-6:
                        diff = abs(val1 - val2) / (abs(val1 + val2) / 2.0)
                        rule_diffs.append(diff)

                if rule_diffs:
                    plasticity_dist += np.mean(rule_diffs) * 0.4
            except Exception:
                # If parameter extraction fails, skip learning rule comparison
                pass

        return float(min(plasticity_dist, 1.0))


@dataclass
class Species:
    """Represents a species of similar genomes"""
    species_id: str
    representative: EvolvableGenome
    members: List[EvolvableGenome] = field(default_factory=list)
    created_generation: int = 0
    last_improved: int = 0
    best_fitness: float = 0.0
    stagnation_counter: int = 0

    def update_representative(self, new_rep: EvolvableGenome):
        """Update species representative"""
        self.representative = new_rep.copy()

    def add_member(self, genome: EvolvableGenome):
        """Add genome to species"""
        self.members.append(genome)

    def clear_members(self):
        """Clear member list for next generation"""
        self.members = []

    def get_fitness_stats(self) -> Dict[str, float]:
        """Get fitness statistics for the species"""
        if not self.members:
            return {'mean': 0.0, 'max': 0.0, 'size': 0}

        fitnesses = [g.fitness for g in self.members]
        return {
            'mean': float(np.mean(fitnesses)),
            'max': float(np.max(fitnesses)),
            'size': len(self.members)
        }


class SpeciationManager:
    """Manages speciation of genomes into species"""

    def __init__(self,
                 distance_calculator: GenomeDistance,
                 compatibility_threshold: float = 3.0,
                 min_species_size: int = 5,
                 max_stagnation: int = 15):
        self.distance_calculator = distance_calculator
        self.compatibility_threshold = compatibility_threshold
        self.min_species_size = min_species_size
        self.max_stagnation = max_stagnation
        self.species: List[Species] = []
        self.next_species_id = 0

    def speciate_population(self, population: List[EvolvableGenome], generation: int) -> List[Species]:
        """Assign genomes to species and return updated species list"""
        # Clear previous members
        for species in self.species:
            species.clear_members()

        # Assign each genome to a species
        for genome in population:
            assigned = False
            for species in self.species:
                distance = self.distance_calculator.calculate_distance(genome, species.representative)
                if distance < self.compatibility_threshold:
                    species.add_member(genome)
                    assigned = True
                    break

            # Create new species if not assigned
            if not assigned:
                new_species = Species(
                    species_id=f"species_{self.next_species_id}",
                    representative=genome.copy(),
                    created_generation=generation
                )
                new_species.add_member(genome)
                self.species.append(new_species)
                self.next_species_id += 1

        # Remove stagnant species
        self._remove_stagnant_species(generation)

        # Update species representatives and stats
        for species in self.species:
            if species.members:
                # Choose best member as representative
                best_member = max(species.members, key=lambda g: g.fitness)
                species.update_representative(best_member)

                # Update fitness stats
                stats = species.get_fitness_stats()
                if stats['max'] > species.best_fitness:
                    species.best_fitness = stats['max']
                    species.last_improved = generation
                    species.stagnation_counter = 0
                else:
                    species.stagnation_counter += 1

        return self.species

    def _remove_stagnant_species(self, generation: int):
        """Remove species that haven't improved for too long"""
        surviving_species = []
        for species in self.species:
            if species.stagnation_counter <= self.max_stagnation or len(species.members) >= self.min_species_size:
                surviving_species.append(species)
            else:
                print(f"Removing stagnant species {species.species_id} (stagnation: {species.stagnation_counter})")

        self.species = surviving_species

    def get_offspring_quotas(self, population_size: int) -> Dict[str, int]:
        """Calculate offspring quotas for each species based on fitness"""
        if not self.species:
            return {}

        total_fitness = sum(species.get_fitness_stats()['mean'] * len(species.members) for species in self.species)
        if total_fitness == 0:
            # Equal allocation if no fitness
            quota_per_species = population_size // len(self.species)
            quotas = {species.species_id: quota_per_species for species in self.species}
            # Distribute remainder
            remainder = population_size - sum(quotas.values())
            for i in range(remainder):
                quotas[self.species[i % len(self.species)].species_id] += 1
        else:
            # Fitness-proportional allocation
            quotas = {}
            total_allocated = 0
            for species in self.species:
                species_fitness = species.get_fitness_stats()['mean'] * len(species.members)
                quota = int((species_fitness / total_fitness) * population_size)
                quotas[species.species_id] = max(1, quota)  # At least 1 offspring
                total_allocated += quotas[species.species_id]

            # Adjust to match population size
            while total_allocated < population_size:
                for species in self.species:
                    if total_allocated >= population_size:
                        break
                    quotas[species.species_id] += 1
                    total_allocated += 1

        return quotas

    def get_species_stats(self) -> Dict[str, Any]:
        """Get statistics about current species"""
        if not self.species:
            return {'num_species': 0, 'avg_species_size': 0, 'total_members': 0}

        species_sizes = [len(s.members) for s in self.species]
        return {
            'num_species': len(self.species),
            'avg_species_size': float(np.mean(species_sizes)),
            'total_members': sum(species_sizes),
            'species_sizes': species_sizes
        }


class NoveltyArchive:
    """Archive of novel behaviors for diversity maintenance"""

    def __init__(self, max_size: int = 100, novelty_threshold: float = 0.1):
        self.archive: List[BehaviorEmbedding] = []
        self.max_size = max_size
        self.novelty_threshold = novelty_threshold

    def add_behavior(self, embedding: BehaviorEmbedding) -> bool:
        """Add behavior to archive if novel enough. Returns True if added."""
        if len(self.archive) < self.max_size:
            # Always add if archive not full
            self.archive.append(embedding)
            return True

        # Calculate novelty score (average distance to k nearest neighbors)
        novelty_score = self._calculate_novelty(embedding, k=15)

        if novelty_score > self.novelty_threshold:
            # Replace oldest or least novel item
            if len(self.archive) >= self.max_size:
                # Replace the oldest item
                self.archive.pop(0)
            self.archive.append(embedding)
            return True

        return False

    def _calculate_novelty(self, embedding: BehaviorEmbedding, k: int = 15) -> float:
        """Calculate novelty score as average distance to k nearest neighbors"""
        if not self.archive:
            return float('inf')

        distances = []
        for archived in self.archive:
            dist = embedding.distance_to(archived)
            distances.append(dist)

        # Sort distances and take average of k nearest
        distances.sort()
        k = min(k, len(distances))
        avg_distance = np.mean(distances[:k])

        return float(avg_distance)

    def get_novelty_score(self, embedding: BehaviorEmbedding) -> float:
        """Get novelty score for a behavior embedding"""
        return self._calculate_novelty(embedding)

    def get_random_elites(self, num_elites: int) -> List[BehaviorEmbedding]:
        """Get random elite behaviors from archive for immigration"""
        if len(self.archive) <= num_elites:
            return self.archive.copy()

        return random.sample(self.archive, num_elites)

    def get_archive_stats(self) -> Dict[str, Any]:
        """Get statistics about the archive"""
        if not self.archive:
            return {'size': 0, 'avg_novelty': 0.0, 'generations_covered': 0}

        generations = [emb.generation for emb in self.archive]
        fitnesses = [emb.fitness for emb in self.archive]

        return {
            'size': len(self.archive),
            'avg_fitness': float(np.mean(fitnesses)),
            'min_generation': min(generations),
            'max_generation': max(generations),
            'generations_covered': max(generations) - min(generations) + 1
        }

    def sample_diverse(self, num_samples: int) -> List[EvolvableGenome]:
        """Sample diverse genomes from archive for injection"""
        if not self.archive:
            return []

        if len(self.archive) <= num_samples:
            # Return all available genomes
            return [emb.genome for emb in self.archive if emb.genome is not None]

        # Sample diverse behaviors using farthest-first traversal
        selected = []
        remaining = self.archive.copy()

        # Start with random seed
        first = random.choice(remaining)
        selected.append(first)
        remaining.remove(first)

        # Add farthest points iteratively
        for _ in range(min(num_samples - 1, len(remaining))):
            farthest = None
            max_min_dist = -1

            for candidate in remaining:
                # Find minimum distance to any selected point
                min_dist = min(candidate.distance_to(selected_emb) for selected_emb in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest = candidate

            if farthest:
                selected.append(farthest)
                remaining.remove(farthest)

        return [emb.genome for emb in selected if emb.genome is not None]


class ArchitectPopulation:
    """Population that evolves architectures"""

    def __init__(self, population_size: int = 20):
        self.architecture_templates = []
        self.population_size = population_size
        self.generation = 0
        self.meta_fitness_history = []

    def evolve_architectures(self, performance_data: Dict[str, Any]):
        """
        Meta-evolve architecture patterns based on main population performance

        Args:
            performance_data: Dict containing fitness stats, architecture diversity,
                            motif effectiveness, and other meta-metrics
        """
        # Extract key performance metrics
        avg_fitness = performance_data.get('avg_fitness', 0.0)
        architecture_diversity = performance_data.get('architecture_diversity', 0.0)
        motif_effectiveness = performance_data.get('motif_effectiveness', 0.0)

        # Calculate meta-fitness for current templates
        current_meta_fitness = self._calculate_meta_fitness(
            avg_fitness, architecture_diversity, motif_effectiveness
        )
        self.meta_fitness_history.append(current_meta_fitness)

        # Discover effective motifs from high-performing architectures
        discovered_motifs = self._discover_motifs(performance_data)

        # Evolve architecture templates
        self._evolve_templates(discovered_motifs, current_meta_fitness)

        # Share successful templates with main populations
        self._share_templates()

        self.generation += 1

    def _calculate_meta_fitness(self, avg_fitness: float,
                               architecture_diversity: float,
                               motif_effectiveness: float) -> float:
        """Calculate meta-fitness for architecture evolution"""
        # Reward diversity, motif effectiveness, and main population fitness
        meta_fitness = (
            avg_fitness * 0.4 +
            architecture_diversity * 0.3 +
            motif_effectiveness * 0.3
        )
        return meta_fitness

    def _discover_motifs(self, performance_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Discover effective architectural motifs from performance data"""
        motifs = []

        # Extract successful architecture patterns
        successful_architectures = performance_data.get('successful_architectures', [])

        for arch in successful_architectures:
            # Analyze layer patterns, activation functions, skip connections
            motif = {
                'layer_pattern': [gene.output_dim for gene in arch.genes],
                'activation_pattern': [gene.activation for gene in arch.genes],
                'skip_connections': sum(1 for gene in arch.genes if getattr(gene, 'skip_connection', False)),
                'fitness': arch.fitness,
                'meta_fitness': self._calculate_motif_meta_fitness(arch)
            }
            motifs.append(motif)

        return motifs

    def _calculate_motif_meta_fitness(self, architecture: EvolvableGenome) -> float:
        """Calculate meta-fitness for a specific architectural motif"""
        # Reward motifs that balance complexity and performance
        complexity_penalty = len(architecture.genes) * 0.1
        diversity_bonus = len(set(gene.activation for gene in architecture.genes)) * 0.2

        return architecture.fitness - complexity_penalty + diversity_bonus

    def _evolve_templates(self, discovered_motifs: List[Dict[str, Any]], current_meta_fitness: float):
        """Evolve architecture templates using discovered motifs"""
        # Selection: keep best templates
        if self.architecture_templates:
            self.architecture_templates.sort(key=lambda t: t.get('meta_fitness', 0), reverse=True)
            self.architecture_templates = self.architecture_templates[:self.population_size // 2]

        # Add discovered motifs as new templates
        for motif in discovered_motifs[:5]:  # Add top 5 motifs
            template = {
                'pattern': motif,
                'meta_fitness': motif['meta_fitness'],
                'generation_discovered': self.generation,
                'usage_count': 0
            }
            self.architecture_templates.append(template)

        # Mutation: slightly modify existing templates
        for template in self.architecture_templates:
            if random.random() < 0.1:  # 10% mutation rate
                self._mutate_template(template)

        # Ensure population size
        while len(self.architecture_templates) < self.population_size:
            # Create new template by combining existing ones
            if self.architecture_templates:
                parent1 = random.choice(self.architecture_templates)
                parent2 = random.choice(self.architecture_templates)
                child_template = self._crossover_templates(parent1, parent2)
                self.architecture_templates.append(child_template)

    def _mutate_template(self, template: Dict[str, Any]):
        """Mutate an architecture template"""
        pattern = template['pattern']

        # Randomly modify layer dimensions
        if 'layer_pattern' in pattern and random.random() < 0.5:
            idx = random.randint(0, len(pattern['layer_pattern']) - 1)
            pattern['layer_pattern'][idx] = max(1, pattern['layer_pattern'][idx] + random.randint(-2, 2))

        # Randomly change activation functions
        if 'activation_pattern' in pattern and random.random() < 0.3:
            idx = random.randint(0, len(pattern['activation_pattern']) - 1)
            activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
            pattern['activation_pattern'][idx] = random.choice(activations)

    def _crossover_templates(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two architecture templates"""
        child_pattern = {}

        # Combine layer patterns
        p1_layers = parent1['pattern'].get('layer_pattern', [])
        p2_layers = parent2['pattern'].get('layer_pattern', [])
        min_len = min(len(p1_layers), len(p2_layers))

        if min_len > 0:
            crossover_point = random.randint(1, min_len)
            child_layers = p1_layers[:crossover_point] + p2_layers[crossover_point:]
        else:
            child_layers = p1_layers or p2_layers

        child_pattern['layer_pattern'] = child_layers

        # Combine activation patterns similarly
        p1_acts = parent1['pattern'].get('activation_pattern', [])
        p2_acts = parent2['pattern'].get('activation_pattern', [])
        min_act_len = min(len(p1_acts), len(p2_acts))

        if min_act_len > 0:
            act_crossover = random.randint(1, min_act_len)
            child_acts = p1_acts[:act_crossover] + p2_acts[act_crossover:]
        else:
            child_acts = p1_acts or p2_acts

        child_pattern['activation_pattern'] = child_acts

        # Average skip connections
        child_pattern['skip_connections'] = (
            parent1['pattern'].get('skip_connections', 0) +
            parent2['pattern'].get('skip_connections', 0)
        ) // 2

        return {
            'pattern': child_pattern,
            'meta_fitness': (parent1.get('meta_fitness', 0) + parent2.get('meta_fitness', 0)) / 2,
            'generation_discovered': self.generation,
            'usage_count': 0
        }

    def _share_templates(self):
        """Share successful architecture templates with main populations"""
        # Return top templates for main population to use
        if self.architecture_templates:
            self.architecture_templates.sort(key=lambda t: t.get('meta_fitness', 0), reverse=True)
            return self.architecture_templates[:5]  # Share top 5 templates
        return []

    def get_best_template(self) -> Optional[Dict[str, Any]]:
        """Get the best architecture template"""
        if not self.architecture_templates:
            return None
        return max(self.architecture_templates, key=lambda t: t.get('meta_fitness', 0))


class MutatorPopulation:
    """Population that evolves mutation strategies"""

    def __init__(self, population_size: int = 15):
        self.mutation_strategies = []
        self.population_size = population_size
        self.generation = 0
        self.meta_fitness_history = []

    def evolve_mutators(self, mutation_effectiveness: Dict[str, Any]):
        """
        Evolve mutation strategies based on their effectiveness

        Args:
            mutation_effectiveness: Dict containing mutation success rates,
                                  fitness improvements, and diversity metrics
        """
        # Calculate meta-fitness for current strategies
        current_meta_fitness = self._calculate_meta_fitness(mutation_effectiveness)
        self.meta_fitness_history.append(current_meta_fitness)

        # Evolve mutation strategies
        self._evolve_strategies(mutation_effectiveness)

        # Update strategy effectiveness tracking
        self._update_strategy_tracking(mutation_effectiveness)

        self.generation += 1

    def _calculate_meta_fitness(self, mutation_effectiveness: Dict[str, Any]) -> float:
        """Calculate meta-fitness for mutation strategy evolution"""
        # Reward strategies that improve fitness while maintaining diversity
        fitness_improvement = mutation_effectiveness.get('avg_fitness_improvement', 0.0)
        diversity_maintenance = mutation_effectiveness.get('diversity_preservation', 0.0)
        exploration_rate = mutation_effectiveness.get('exploration_success', 0.0)

        meta_fitness = (
            fitness_improvement * 0.4 +
            diversity_maintenance * 0.3 +
            exploration_rate * 0.3
        )
        return meta_fitness

    def _evolve_strategies(self, mutation_effectiveness: Dict[str, Any]):
        """Evolve mutation strategies using effectiveness data"""
        # Selection: keep best strategies
        if self.mutation_strategies:
            self.mutation_strategies.sort(key=lambda s: s.get('meta_fitness', 0), reverse=True)
            self.mutation_strategies = self.mutation_strategies[:self.population_size // 2]

        # Generate new strategies based on effectiveness patterns
        successful_patterns = self._analyze_successful_patterns(mutation_effectiveness)

        for pattern in successful_patterns[:3]:  # Add top 3 patterns
            strategy = {
                'parameters': pattern,
                'meta_fitness': pattern.get('effectiveness', 0.0),
                'generation_created': self.generation,
                'usage_count': 0
            }
            self.mutation_strategies.append(strategy)

        # Mutate existing strategies
        for strategy in self.mutation_strategies:
            if random.random() < 0.15:  # 15% mutation rate
                self._mutate_strategy(strategy)

        # Ensure population size
        while len(self.mutation_strategies) < self.population_size:
            # Create new strategy by combining existing ones
            if self.mutation_strategies:
                parent1 = random.choice(self.mutation_strategies)
                parent2 = random.choice(self.mutation_strategies)
                child_strategy = self._crossover_strategies(parent1, parent2)
                self.mutation_strategies.append(child_strategy)

    def _analyze_successful_patterns(self, mutation_effectiveness: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze what made mutations successful"""
        patterns = []

        # Extract successful mutation parameters
        success_rates = mutation_effectiveness.get('mutation_success_rates', {})

        for mutation_type, rate in success_rates.items():
            if rate > 0.6:  # Consider successful if >60% success rate
                pattern = {
                    'type': mutation_type,
                    'rate': rate,
                    'effectiveness': rate,
                    'fitness_impact': mutation_effectiveness.get(f'{mutation_type}_fitness_impact', 0.0)
                }
                patterns.append(pattern)

        return patterns

    def _mutate_strategy(self, strategy: Dict[str, Any]):
        """Mutate a mutation strategy"""
        params = strategy['parameters']

        # Adjust mutation rates
        if 'rate' in params:
            # Add some noise to the rate
            noise = random.uniform(-0.1, 0.1)
            params['rate'] = np.clip(params['rate'] + noise, 0.001, 0.5)

        # Modify strategy parameters
        if random.random() < 0.3:
            # Add new parameter or modify existing
            param_keys = ['strength', 'scope', 'frequency']
            key = random.choice(param_keys)
            if key not in params:
                params[key] = random.uniform(0.1, 1.0)
            else:
                params[key] = np.clip(params[key] + random.uniform(-0.2, 0.2), 0.1, 2.0)

    def _crossover_strategies(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two mutation strategies"""
        child_params = {}

        # Combine parameters from both parents
        all_keys = set(parent1['parameters'].keys()) | set(parent2['parameters'].keys())

        for key in all_keys:
            if key in parent1['parameters'] and key in parent2['parameters']:
                # Average numerical parameters
                if isinstance(parent1['parameters'][key], (int, float)):
                    child_params[key] = (parent1['parameters'][key] + parent2['parameters'][key]) / 2
                else:
                    # Randomly choose for non-numerical
                    child_params[key] = random.choice([parent1['parameters'][key], parent2['parameters'][key]])
            elif key in parent1['parameters']:
                child_params[key] = parent1['parameters'][key]
            else:
                child_params[key] = parent2['parameters'][key]

        return {
            'parameters': child_params,
            'meta_fitness': (parent1.get('meta_fitness', 0) + parent2.get('meta_fitness', 0)) / 2,
            'generation_created': self.generation,
            'usage_count': 0
        }

    def _update_strategy_tracking(self, mutation_effectiveness: Dict[str, Any]):
        """Update tracking of strategy effectiveness"""
        # Update usage counts and effectiveness for existing strategies
        for strategy in self.mutation_strategies:
            strategy_type = strategy['parameters'].get('type', 'unknown')
            if strategy_type in mutation_effectiveness.get('mutation_success_rates', {}):
                # Increment usage and update fitness based on recent performance
                strategy['usage_count'] += 1
                recent_effectiveness = mutation_effectiveness['mutation_success_rates'][strategy_type]
                # Blend old and new effectiveness
                old_effectiveness = strategy.get('current_effectiveness', 0.5)
                strategy['current_effectiveness'] = 0.8 * old_effectiveness + 0.2 * recent_effectiveness

    def get_best_strategy(self, mutation_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get the best mutation strategy, optionally for a specific type"""
        if not self.mutation_strategies:
            return None

        if mutation_type:
            # Filter by type
            matching_strategies = [s for s in self.mutation_strategies
                                 if s['parameters'].get('type') == mutation_type]
            if matching_strategies:
                return max(matching_strategies, key=lambda s: s.get('current_effectiveness', 0))
            else:
                return None

        return max(self.mutation_strategies, key=lambda s: s.get('current_effectiveness', 0))

    def get_adaptive_rates(self) -> Dict[str, float]:
        """Get adaptive mutation rates based on evolved strategies"""
        rates = {}

        if not self.mutation_strategies:
            return {'weight_rate': 0.1, 'arch_rate': 0.05, 'layer_rate': 0.02}

        # Aggregate rates from best strategies
        for strategy in self.mutation_strategies[:5]:  # Use top 5 strategies
            params = strategy['parameters']
            effectiveness = strategy.get('current_effectiveness', 0.5)

            if 'rate' in params:
                strategy_type = params.get('type', 'weight')
                if strategy_type not in rates:
                    rates[strategy_type] = []
                rates[strategy_type].append(params['rate'] * effectiveness)

        # Average rates by type
        final_rates = {}
        for strategy_type, rate_list in rates.items():
            if rate_list:
                avg_rate = np.mean(rate_list)
                if strategy_type == 'weight':
                    final_rates['weight_rate'] = np.clip(avg_rate, 0.01, 0.3)
                elif strategy_type == 'arch':
                    final_rates['arch_rate'] = np.clip(avg_rate, 0.001, 0.1)
                elif strategy_type == 'layer':
                    final_rates['layer_rate'] = np.clip(avg_rate, 0.001, 0.05)

        # Fill in defaults for missing types
        final_rates.setdefault('weight_rate', 0.1)
        final_rates.setdefault('arch_rate', 0.05)
        final_rates.setdefault('layer_rate', 0.02)

        return final_rates


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


class NoveltyInjector:
    """Automatic diversity maintenance through novelty injection"""

    def __init__(self, novelty_archive: NoveltyArchive, diversity_threshold: float = 0.1):
        """
        Args:
            novelty_archive: Reference to the novelty archive
            diversity_threshold: Threshold below which diversity is considered collapsed
        """
        self.novelty_archive = novelty_archive
        self.diversity_threshold = diversity_threshold
        self.injection_history = []
        self.last_diversity = 1.0
        self.injection_effectiveness = {
            'total_injections': 0,
            'successful_injections': 0,
            'fitness_improvements': [],
            'diversity_improvements': []
        }

    def inject_from_archive(self, population: List[EvolvableGenome], injection_rate: float = 0.05) -> List[EvolvableGenome]:
        """Replace worst performers with novel archive members"""
        if not self.novelty_archive or len(self.novelty_archive.archive) == 0:
            return population

        num_inject = int(len(population) * injection_rate)
        if num_inject == 0:
            return population

        # Sort by fitness (worst first)
        population.sort(key=lambda g: g.fitness)

        # Get novel genomes from archive
        novel_genomes = self.novelty_archive.sample_diverse(num_inject)

        if not novel_genomes:
            return population

        # Replace worst performers
        for i in range(min(num_inject, len(novel_genomes))):
            # Create a fresh copy of the novel genome
            novel_copy = novel_genomes[i].copy()
            novel_copy.genome_id = f"novel_injected_{random.randint(0, 9999):04d}"
            novel_copy.fitness = 0.0
            novel_copy.norm_fitness = 0.0

            # Replace the worst genome
            population[i] = novel_copy

        # Track injection
        self.injection_history.append({
            'generation': len(self.injection_history),
            'num_injected': min(num_inject, len(novel_genomes)),
            'archive_size': len(self.novelty_archive.archive)
        })

        self.injection_effectiveness['total_injections'] += 1

        return population

    def should_inject(self, population: List[EvolvableGenome]) -> bool:
        """Determine if injection should be triggered based on diversity collapse"""
        current_diversity = self._calculate_population_diversity(population)

        # Check for diversity collapse
        diversity_collapsed = current_diversity < self.diversity_threshold

        # Check for stagnation (fitness not improving)
        fitness_stagnant = self._check_fitness_stagnation(population)

        # Update tracking
        self.last_diversity = current_diversity

        return diversity_collapsed or fitness_stagnant

    def _calculate_population_diversity(self, population: List[EvolvableGenome]) -> float:
        """Calculate current population diversity (0-1)"""
        if len(population) < 2:
            return 0.0

        # Multi-dimensional diversity calculation
        fitnesses = [g.fitness for g in population]
        architectures = [len(g.genes) for g in population]
        params = [sum(gene.input_dim * gene.output_dim for gene in g.genes) for g in population]

        # Normalize and combine diversity metrics
        fitness_div = np.std(fitnesses) / (np.mean(fitnesses) + 1e-10) if fitnesses else 0.0
        arch_div = np.std(architectures) / (np.mean(architectures) + 1e-10) if architectures else 0.0
        param_div = np.std(params) / (np.mean(params) + 1e-10) if params else 0.0

        total_diversity = (fitness_div + arch_div + param_div) / 3.0
        return float(min(total_diversity, 1.0))

    def _check_fitness_stagnation(self, population: List[EvolvableGenome]) -> bool:
        """Check if population fitness is stagnating"""
        # Simple check: if best fitness hasn't changed significantly in recent history
        # This could be enhanced with more sophisticated stagnation detection
        return False  # Placeholder - would need fitness history

    def update_injection_effectiveness(self,
                                     pre_injection_fitness: float,
                                     post_injection_fitness: float,
                                     pre_injection_diversity: float,
                                     post_injection_diversity: float):
        """Update tracking of injection effectiveness"""
        fitness_improvement = post_injection_fitness - pre_injection_fitness
        diversity_improvement = post_injection_diversity - pre_injection_diversity

        self.injection_effectiveness['fitness_improvements'].append(fitness_improvement)
        self.injection_effectiveness['diversity_improvements'].append(diversity_improvement)

        # Consider injection successful if it improved either fitness or diversity
        if fitness_improvement > 0 or diversity_improvement > 0:
            self.injection_effectiveness['successful_injections'] += 1

    def get_injection_stats(self) -> Dict[str, Any]:
        """Get statistics about injection effectiveness"""
        total = self.injection_effectiveness['total_injections']
        successful = self.injection_effectiveness['successful_injections']

        fitness_improvements = self.injection_effectiveness['fitness_improvements']
        diversity_improvements = self.injection_effectiveness['diversity_improvements']

        return {
            'total_injections': total,
            'successful_injections': successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_fitness_improvement': np.mean(fitness_improvements) if fitness_improvements else 0.0,
            'avg_diversity_improvement': np.mean(diversity_improvements) if diversity_improvements else 0.0,
            'injection_history': self.injection_history[-10:]  # Last 10 injections
        }


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
    