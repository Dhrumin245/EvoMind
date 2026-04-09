from typing import List, Optional, Dict, Any, Iterable, Tuple

import numpy as np

from api.trainer import EvoTrainer


class AgentInterface:
    """
    Agentic interface for querying evolved genomes.

    Provides access to best genomes for inference via brain.act_batch(observation).
    """

    def __init__(self, trainer: EvoTrainer):
        self.trainer = trainer

    @staticmethod
    def _normalize_max_action_length(max_action_length: Optional[int]) -> int:
        if max_action_length is None:
            return 10
        return max(1, min(int(max_action_length), 10))

    @staticmethod
    def _architecture_label(genome: Any) -> str:
        architecture = getattr(genome, "architecture", None)
        if isinstance(architecture, list) and architecture:
            return "x".join(str(int(dim)) for dim in architecture)
        return f"{len(getattr(genome, 'genes', []))}-layer"

    @staticmethod
    def _genome_summary(genome: Any, genome_type: str, source: str) -> Dict[str, Any]:
        return {
            "genome_id": str(getattr(genome, "genome_id", "unknown")),
            "genome_type": genome_type,
            "fitness": float(getattr(genome, "fitness", 0.0)),
            "generation": int(getattr(genome, "birth_generation", 0) or 0),
            "source": source,
            "gene_count": int(len(getattr(genome, "genes", []))),
            "input_size": int(getattr(genome, "input_size", 0) or 0),
            "output_size": int(getattr(genome, "output_size", 0) or 0),
            "architecture": AgentInterface._architecture_label(genome),
        }

    def _iter_genomes(self, genome_type: str) -> Iterable[Tuple[Any, str]]:
        state = self.trainer.state
        if not state:
            return []

        if genome_type == "prey":
            hall_of_fame = getattr(state, "prey_hall_of_fame", [])
            population = state.prey_population
        elif genome_type == "predator":
            hall_of_fame = getattr(state, "predator_hall_of_fame", [])
            population = state.predator_population
        else:
            raise ValueError(f"Invalid genome_type: {genome_type}")

        entries: List[Tuple[Any, str]] = []
        entries.extend((genome, "hall_of_fame") for genome in hall_of_fame)
        entries.extend((genome, "current_population") for genome in population)
        return entries

    def get_best_genome(self, genome_type: str, generation: Optional[int] = None) -> Optional[Any]:
        """
        Get best genome of specified type.

        Args:
            genome_type: "prey" or "predator"
            generation: Specific generation (None = latest/best)

        Returns:
            Genome instance or None
        """
        state = self.trainer.state
        if not state:
            return None

        if genome_type == "prey":
            population = state.prey_population
            hall_of_fame = getattr(state, "prey_hall_of_fame", [])
        elif genome_type == "predator":
            population = state.predator_population
            hall_of_fame = getattr(state, "predator_hall_of_fame", [])
        else:
            raise ValueError(f"Invalid genome_type: {genome_type}")

        if hall_of_fame:
            best_hof = max(hall_of_fame, key=lambda g: getattr(g, "fitness", 0))
            if generation is None or getattr(best_hof, "birth_generation", 0) <= generation:
                return best_hof

        if population:
            return max(population, key=lambda g: getattr(g, "fitness", 0))

        return None

    def query(
        self,
        observation: List[float],
        genome_type: str,
        generation: Optional[int] = None,
        max_action_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query agent: observation -> action.
        """
        genome = self.get_best_genome(genome_type, generation)
        if not genome:
            return {
                "error": f"No {genome_type} genome available",
                "genome_id": None,
                "fitness": 0.0,
                "action": [0.0] * 4,
            }

        try:
            brain = genome.get_brain()
            if brain is None:
                raise ValueError("Genome has no brain")

            obs_array = np.array(observation, dtype=np.float32)
            action_length = self._normalize_max_action_length(max_action_length)
            action = brain.act_batch(obs_array[None, :])[0][:action_length].tolist()

            return {
                "action": action,
                "genome_id": getattr(genome, "genome_id", "unknown"),
                "fitness": float(getattr(genome, "fitness", 0.0)),
                "genome_type": genome_type,
                "generation": int(getattr(genome, "birth_generation", 0) or 0),
                "confidence": min(1.0, float(getattr(genome, "fitness", 0.0) / 10.0)),
            }

        except Exception as e:
            return {
                "error": f"Inference failed: {str(e)}",
                "genome_id": getattr(genome, "genome_id", "unknown"),
                "action": [0.0] * 4,
            }

    def query_batch(
        self,
        observations: List[List[float]],
        genome_type: str,
        generation: Optional[int] = None,
        max_action_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        genome = self.get_best_genome(genome_type, generation)
        if not genome:
            return {
                "error": f"No {genome_type} genome available",
                "genome_id": None,
                "fitness": 0.0,
                "actions": [],
            }

        try:
            brain = genome.get_brain()
            if brain is None:
                raise ValueError("Genome has no brain")

            obs_array = np.array(observations, dtype=np.float32)
            if obs_array.ndim != 2:
                raise ValueError("observations must be a 2D list")

            action_length = self._normalize_max_action_length(max_action_length)
            actions = brain.act_batch(obs_array)[:, :action_length].tolist()

            return {
                "actions": actions,
                "genome_id": getattr(genome, "genome_id", "unknown"),
                "fitness": float(getattr(genome, "fitness", 0.0)),
                "genome_type": genome_type,
                "generation": int(getattr(genome, "birth_generation", 0) or 0),
                "confidence": min(1.0, float(getattr(genome, "fitness", 0.0) / 10.0)),
                "batch_size": len(actions),
            }
        except Exception as e:
            return {
                "error": f"Batch inference failed: {str(e)}",
                "genome_id": getattr(genome, "genome_id", "unknown"),
                "actions": [],
            }

    def list_available_genomes(
        self,
        genome_type: Optional[str] = None,
        limit_per_type: int = 10,
    ) -> List[Dict[str, Any]]:
        """List available genomes with metadata."""
        state = self.trainer.state
        if not state:
            return []

        genome_types = [genome_type] if genome_type else ["prey", "predator"]
        summaries: List[Dict[str, Any]] = []
        seen_ids = set()
        per_type_limit = max(1, int(limit_per_type))

        for current_type in genome_types:
            typed_entries = sorted(
                self._iter_genomes(current_type),
                key=lambda item: getattr(item[0], "fitness", 0),
                reverse=True,
            )
            typed_count = 0
            for genome, source in typed_entries:
                genome_id = str(getattr(genome, "genome_id", "unknown"))
                unique_key = (current_type, genome_id)
                if unique_key in seen_ids:
                    continue
                summaries.append(self._genome_summary(genome, current_type, source))
                seen_ids.add(unique_key)
                typed_count += 1
                if typed_count >= per_type_limit:
                    break

        return sorted(summaries, key=lambda item: item["fitness"], reverse=True)

    def get_genome_by_id(
        self,
        genome_id: str,
        genome_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        genome_types = [genome_type] if genome_type else ["prey", "predator"]

        for current_type in genome_types:
            for genome, source in self._iter_genomes(current_type):
                if str(getattr(genome, "genome_id", "unknown")) == genome_id:
                    return {
                        "genome": genome,
                        "genome_type": current_type,
                        "source": source,
                        "summary": self._genome_summary(genome, current_type, source),
                    }

        return None
