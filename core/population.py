from core.genome import Genome
import statistics
import numpy as np
import random
from typing import List, Dict, Optional, Tuple

class Population:
    # Define possible population roles
    ROLES = {'prey', 'predator', 'architect', 'mutator', 'generic'}
    
    def __init__(self, size=100, name='default', role='generic', generation=0):
        assert size >= 0, "Population size must be non-negative"
        assert role in self.ROLES, f"Invalid role. Must be one of: {self.ROLES}"
        
        self.name = name
        self.role = role  # 'prey', 'predator', 'architect', 'mutator', 'generic'
        self.generation = generation  # Current generation number
        self.history = []  # Track population statistics over time
        
        # Create genomes with role context
        self.genomes = []
        for i in range(size):
            genome = Genome()
            # Set initial metadata for lineage tracking
            setattr(genome, 'birth_generation', generation)
            setattr(genome, 'origin_population', name)
            setattr(genome, 'role', role)
            self.genomes.append(genome)
    
    def __len__(self):
        return len(self.genomes)
    
    def __iter__(self):
        return iter(self.genomes)
    
    def __getitem__(self, index):
        return self.genomes[index]
    
    def reset_fitness(self):
        for genome in self.genomes:
            genome.reset_fitness()
    
    def fitness_values(self):
        return [g.fitness for g in self.genomes]
    
    def get_best_genome(self):
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)
    
    def average_fitness(self):
        if not self.genomes:
            return 0.0
        return statistics.mean(self.fitness_values())
    
    def fitness_std(self):
        if len(self.genomes) < 2:
            return 0.0
        return statistics.pstdev(self.fitness_values())
    
    def validate(self):
        assert len(self.genomes) > 0, "Population has no genomes"
        for g in self.genomes:
            # Validate that genome has genes and fitness
            assert hasattr(g, 'genes') and len(g.genes) > 0, "Genome has no genes"
            assert hasattr(g, 'fitness'), "Genome has no fitness"
    
    def sort_by_fitness(self):
        """Sort genomes by fitness in descending order"""
        self.genomes.sort(key=lambda g: g.fitness, reverse=True)
    
    def get_fitness_stats(self):
        """Get fitness statistics as dict"""
        if not self.genomes:
            return {}
        
        fitnesses = self.fitness_values()
        return {
            'mean': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'min': np.min(fitnesses),
            'max': np.max(fitnesses),
            'median': np.median(fitnesses)
        }
    
    # ========== NEW FUNCTIONALITY ==========
    
    def set_role(self, role: str):
        """Set the role of this population and all its genomes"""
        assert role in self.ROLES, f"Invalid role. Must be one of: {self.ROLES}"
        self.role = role
        for genome in self.genomes:
            genome.role = role
    
    def get_lineages(self) -> List[List[str]]:
        """Get parent IDs for all genomes in the population"""
        lineages = []
        for genome in self.genomes:
            if hasattr(genome, 'parent_ids'):
                lineages.append(genome.parent_ids)
            else:
                lineages.append([])  # Empty lineage for genomes without parents
        return lineages
    
    def get_genealogy_tree(self, max_depth: int = 3) -> Dict:
        """
        Get a simplified genealogy tree for analysis.
        Returns a dictionary mapping genome IDs to their ancestry.
        """
        genealogy = {}
        for genome in self.genomes:
            if hasattr(genome, 'genome_id'):
                genealogy[genome.genome_id] = {
                    'parents': genome.parent_ids if hasattr(genome, 'parent_ids') else [],
                    'birth_generation': genome.birth_generation if hasattr(genome, 'birth_generation') else 0,
                    'fitness': genome.fitness,
                    'role': genome.role if hasattr(genome, 'role') else self.role
                }
        return genealogy
    
    def add_genomes(self, new_genomes: List[Genome], update_generation: bool = True):
        """Add new genomes to the population"""
        for genome in new_genomes:
            setattr(genome, 'role', self.role)
            if update_generation:
                setattr(genome, 'birth_generation', self.generation)
        self.genomes.extend(new_genomes)
    
    def remove_genomes(self, indices: List[int]):
        """Remove genomes at specified indices"""
        # Sort in reverse to avoid index shifting issues
        for index in sorted(indices, reverse=True):
            if 0 <= index < len(self.genomes):
                self.genomes.pop(index)
    
    def update_generation(self, new_generation: int):
        """Update the generation number and record history"""
        # Record current stats before updating
        stats = self.get_fitness_stats()
        stats.update({
            'generation': self.generation,
            'population_size': len(self),
            'role': self.role
        })
        self.history.append(stats)
        
        # Update generation
        self.generation = new_generation
    
    def coevolve_with(self, other_population: 'Population', 
                     competition_rate: float = 0.1) -> Tuple[float, float]:
        """
        Basic co-evolution mechanism between two populations.
        Returns tuple of (self_score, other_score) based on competition.
        """
        if not self.genomes or not other_population.genomes:
            return 0.0, 0.0
        
        # Simple competition: compare average fitness
        self_avg = self.average_fitness()
        other_avg = other_population.average_fitness()
        
        # Apply competition adjustment
        if self_avg > other_avg:
            advantage = (self_avg - other_avg) * competition_rate
            return advantage, -advantage
        else:
            disadvantage = (other_avg - self_avg) * competition_rate
            return -disadvantage, disadvantage
    
    def adapt_mutation_rates(self, performance_threshold: float = 0.5):
        """
        Adapt mutation rates based on population performance.
        Lower performance = higher mutation rate.
        """
        if not self.genomes:
            return
        
        avg_fitness = self.average_fitness()
        max_fitness = max(self.fitness_values())
        
        # Calculate performance metric (0 to 1)
        performance = avg_fitness / max_fitness if max_fitness > 0 else 0
        
        # Adjust mutation rates
        for genome in self.genomes:
            if hasattr(genome, 'mutation_rate'):
                # If performance is low, increase mutation rate
                if performance < performance_threshold:
                    genome.mutation_rate = min(0.5, genome.mutation_rate * 1.2)
                else:
                    genome.mutation_rate = max(0.01, genome.mutation_rate * 0.9)
    
    def get_elite(self, elite_size: int = 10) -> List[Genome]:
        """Get the top-performing genomes (elite)"""
        if not self.genomes:
            return []
        
        sorted_genomes = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        return sorted_genomes[:min(elite_size, len(sorted_genomes))]
    
    def get_diversity_score(self) -> float:
        """
        Calculate population diversity based on genome weights.
        Higher score = more diverse population.
        """
        if len(self.genomes) < 2:
            return 0.0

        sample = random.sample(self.genomes, min(20, len(self.genomes)))

        # Calculate average pairwise Euclidean distance between sampled genomes
        distances = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                # Flatten all weights from genes into a single vector
                weights_i = np.concatenate([gene.weights.flatten() for gene in sample[i].genes])
                weights_j = np.concatenate([gene.weights.flatten() for gene in sample[j].genes])
                # Align lengths to compare heterogeneous architectures
                min_len = min(weights_i.shape[0], weights_j.shape[0])
                if min_len == 0:
                    continue
                distance = np.linalg.norm(weights_i[:min_len] - weights_j[:min_len])
                distances.append(distance)

        return float(np.mean(distances)) if distances else 0.0
    
    def to_dict(self) -> Dict:
        """Serialize population information for saving/loading"""
        return {
            'name': self.name,
            'role': self.role,
            'generation': self.generation,
            'size': len(self),
            'stats': self.get_fitness_stats()
        }