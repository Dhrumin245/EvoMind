from typing import List, Optional, Dict, Any
from genomes.genome_prey import PreyGenome
from genomes.genome_predator import PredatorGenome
from api.trainer import EvoTrainer
import numpy as np

class AgentInterface:
    """
    Agentic Interface for querying evolved genomes
    
    Provides access to best genomes for inference via brain.act_batch(observation)
    """
    
    def __init__(self, trainer: EvoTrainer):
        self.trainer = trainer
    
    def get_best_genome(self, genome_type: str, generation: Optional[int] = None) -> Optional[Any]:
        """
        Get best genome of specified type
        
        Args:
            genome_type: "prey" or "predator"
            generation: Specific generation (None = latest/best)
        
        Returns:
            Genome instance or None
        """
        state = self.trainer.state
        if not state:
            return None
        
        # Get population for type
        if genome_type == "prey":
            population = state.prey_population
            hall_of_fame = getattr(state, 'prey_hall_of_fame', [])
        elif genome_type == "predator":
            population = state.predator_population
            hall_of_fame = getattr(state, 'predator_hall_of_fame', [])
        else:
            raise ValueError(f"Invalid genome_type: {genome_type}")
        
        # Check hall of fame first (best ever)
        if hall_of_fame:
            best_hof = max(hall_of_fame, key=lambda g: getattr(g, 'fitness', 0))
            if generation is None or getattr(best_hof, 'birth_generation', 0) <= generation:
                return best_hof
        
        # Current population best
        if population:
            return max(population, key=lambda g: getattr(g, 'fitness', 0))
        
        return None
    
    def query(self, observation: List[float], genome_type: str, 
              generation: Optional[int] = None) -> Dict[str, Any]:
        """
        Query agent: observation → action
        
        Args:
            observation: Environment state vector
            genome_type: "prey" or "predator"  
            generation: Specific generation (None = best available)
        
        Returns:
            Dict with action, metadata
        """
        genome = self.get_best_genome(genome_type, generation)
        if not genome:
            return {
                "error": f"No {genome_type} genome available",
                "genome_id": None,
                "fitness": 0.0,
                "action": [0.0] * 4  # Default no-op action
            }
        
        try:
            # Get brain and compute action
            brain = genome.get_brain()
            if brain is None:
                raise ValueError("Genome has no brain")

            obs_array = np.array(observation, dtype=np.float32)
            action = brain.act_batch(obs_array[None, :])[0]  # Batch dim
            
            # Ensure action is reasonable length
            action = action[:10].tolist()  # Cap at 10
            
            return {
                "action": action,
                "genome_id": getattr(genome, 'genome_id', 'unknown'),
                "fitness": float(getattr(genome, 'fitness', 0.0)),
                "genome_type": genome_type,
                "generation": getattr(genome, 'birth_generation', 0),
                "confidence": min(1.0, float(getattr(genome, 'fitness', 0.0) / 10.0))
            }
            
        except Exception as e:
            return {
                "error": f"Inference failed: {str(e)}",
                "genome_id": getattr(genome, 'genome_id', 'unknown'),
                "action": [0.0] * 4
            }
    
    def list_available_genomes(self, genome_type: str) -> List[Dict[str, Any]]:
        """List available genomes with metadata"""
        state = self.trainer.state
        if not state:
            return []
        
        if genome_type == "prey":
            population = state.prey_population
            hall_of_fame = getattr(state, 'prey_hall_of_fame', [])
        elif genome_type == "predator":
            population = state.predator_population
            hall_of_fame = getattr(state, 'predator_hall_of_fame', [])
        else:
            return []
        
        genomes_info = []
        
        # Hall of fame (best ever)
        for genome in hall_of_fame[:5]:  # Top 5
            genomes_info.append({
                "genome_id": getattr(genome, 'genome_id', 'unknown'),
                "fitness": float(getattr(genome, 'fitness', 0.0)),
                "generation": getattr(genome, 'birth_generation', 0),
                "source": "hall_of_fame",
                "architecture": str(len(getattr(genome, 'genes', []))) + "-layer"
            })
        
        # Current population top 5
        top_current = sorted(population, key=lambda g: getattr(g, 'fitness', 0), reverse=True)[:5]
        for genome in top_current:
            genomes_info.append({
                "genome_id": getattr(genome, 'genome_id', 'unknown'),
                "fitness": float(getattr(genome, 'fitness', 0.0)),
                "generation": getattr(genome, 'birth_generation', 0),
                "source": "current_population",
                "architecture": str(len(getattr(genome, 'genes', []))) + "-layer"
            })
        
        return sorted(genomes_info, key=lambda x: x['fitness'], reverse=True)

