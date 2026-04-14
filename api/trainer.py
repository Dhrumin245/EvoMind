import asyncio
import os
import time
import json
import csv
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from api.schemas import TrainStatus
from api.storage import data_dir

# Core project imports
from main import (
    TrainingState,
    EvolutionConfig,
    load_coevolution_state,
    save_coevolution_state,
    train_coevolution_async,
    AsyncDeterministicEvaluator,
    CurriculumController,
    CurriculumStage,
    ArchitectPopulation,
    MutatorPopulation,
    PreyGenome,
    PredatorGenome,
)
from evolution.evolution import EvolutionEngine

logger = logging.getLogger(__name__)

class EvoTrainer:
    """
    Training Controller for Evomind API
    
    Manages the full TrainingState lifecycle with async control:
    - start() → background training task
    - stop() → graceful shutdown
    - status() → live metrics
    - resume(checkpoint) → load and continue
    """
    
    def __init__(self, base_dir: str = "data", config_path: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.state_base_path = self.base_dir / "coevolution_state.json"
        self.config_path = Path(config_path) if config_path is not None else (self.base_dir / "config.json")
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.state: Optional[TrainingState] = None
        self.training_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.start_time = 0.0
        
        # Engines (lazy init)
        self.prey_engine: Optional[EvolutionEngine] = None
        self.predator_engine: Optional[EvolutionEngine] = None
        self.evaluator: Optional[AsyncDeterministicEvaluator] = None
        self.curriculum_controller: Optional[CurriculumController] = None
        self.architect_population: Optional[ArchitectPopulation] = None
        self.mutator_population: Optional[MutatorPopulation] = None
        
        # Status tracking
        self.last_status = {
            "status": "stopped",
            "generation": 0,
            "uptime_seconds": 0.0,
            "last_update": ""
        }

    def _split_state_exists(self) -> bool:
        return (self.base_dir / "config.json").exists() and (self.base_dir / "expirement_state.json").exists()

    def _metrics_file_candidates(self) -> List[Path]:
        shared_data_dir = data_dir()
        # Keep backward compatibility with historical typo-based filenames.
        return [
            self.base_dir / "metrices.csv",
            self.base_dir / "metrics.csv",
            shared_data_dir / "metrices.csv",
            shared_data_dir / "metrics.csv",
        ]

    def get_metrics_file(self) -> Optional[Path]:
        for candidate in self._metrics_file_candidates():
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def get_metrics_rows(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        since_generation: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load full metric rows from CSV, preserving all columns."""
        metrics_file = self.get_metrics_file()
        if metrics_file is None:
            return []

        rows: List[Dict[str, Any]] = []
        with metrics_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue
                generation_raw = row.get("generation")
                if since_generation is not None and generation_raw is not None:
                    try:
                        if int(float(generation_raw)) < since_generation:
                            continue
                    except (TypeError, ValueError):
                        # Keep malformed rows instead of silently dropping data.
                        pass

                rows.append(row)

        safe_offset = max(0, int(offset))
        if safe_offset >= len(rows):
            return []

        sliced = rows[safe_offset:]
        if limit is not None:
            safe_limit = max(0, int(limit))
            sliced = sliced[:safe_limit]

        return sliced

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def get_insights(self, last_n: int = 10) -> Dict[str, Any]:
        """Return trend insights for the last N generations."""
        window = max(1, int(last_n))

        # Prefer in-memory generation stats for live accuracy.
        if self.state and self.state.generation_stats:
            recent_stats = self.state.generation_stats[-window:]
            fitness_trend: List[Dict[str, Any]] = []
            diversity_trend: List[Dict[str, Any]] = []
            learning_trend: List[Dict[str, Any]] = []

            for stats in recent_stats:
                generation = self._as_int(stats.get("generation"), self._as_int(stats.get("gen"), 0))
                stage = str(stats.get("stage", "unknown"))

                fitness_trend.append({
                    "generation": generation,
                    "stage": stage,
                    "prey_best": self._as_float(stats.get("best_prey_fitness")),
                    "prey_average": self._as_float(stats.get("mean_prey_fitness")),
                    "predator_best": self._as_float(stats.get("best_predator_fitness")),
                    "predator_average": self._as_float(stats.get("mean_predator_fitness")),
                })

                prey_species = stats.get("prey_species", {})
                predator_species = stats.get("predator_species", {})
                diversity_trend.append({
                    "generation": generation,
                    "stage": stage,
                    "prey_species": self._as_int(prey_species.get("num_species") if isinstance(prey_species, dict) else 0),
                    "predator_species": self._as_int(predator_species.get("num_species") if isinstance(predator_species, dict) else 0),
                })

                learning_trend.append({
                    "generation": generation,
                    "stage": stage,
                    "adaptability": self._as_float(stats.get("avg_adaptability_score")),
                    "meta_effectiveness": self._as_float(stats.get("avg_meta_effectiveness")),
                    "performance_change": self._as_float(stats.get("avg_reward_delta")),
                    "instability": self._as_float(stats.get("avg_instability")),
                })

            return {
                "window": window,
                "source": "memory",
                "fitness_trend": fitness_trend,
                "diversity_trend": diversity_trend,
                "learning_trend": learning_trend,
            }

        # Fallback to metrics CSV if in-memory stats are not available.
        rows = self.get_metrics_rows()
        if not rows:
            return {
                "window": window,
                "source": "none",
                "fitness_trend": [],
                "diversity_trend": [],
                "learning_trend": [],
            }

        recent_rows = rows[-window:]
        fitness_trend = []
        diversity_trend = []
        learning_trend = []

        for row in recent_rows:
            generation = self._as_int(row.get("generation"), 0)
            stage = str(row.get("stage", "unknown"))

            fitness_trend.append({
                "generation": generation,
                "stage": stage,
                "prey_best": self._as_float(row.get("best_prey_fitness")),
                "prey_average": self._as_float(row.get("mean_prey_fitness")),
                "predator_best": self._as_float(row.get("best_predator_fitness")),
                "predator_average": self._as_float(row.get("mean_predator_fitness")),
            })

            diversity_trend.append({
                "generation": generation,
                "stage": stage,
                "prey_species": self._as_int(row.get("prey_species.num_species")),
                "predator_species": self._as_int(row.get("predator_species.num_species")),
            })

            learning_trend.append({
                "generation": generation,
                "stage": stage,
                "adaptability": self._as_float(row.get("avg_adaptability_score")),
                "meta_effectiveness": self._as_float(row.get("avg_meta_effectiveness")),
                "performance_change": self._as_float(row.get("avg_reward_delta")),
                "instability": self._as_float(row.get("avg_instability")),
            })

        return {
            "window": window,
            "source": "metrics_csv",
            "fitness_trend": fitness_trend,
            "diversity_trend": diversity_trend,
            "learning_trend": learning_trend,
        }

    def get_metrics_payload(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        since_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return paginated raw metric rows, preferring live in-memory stats."""
        source = "none"
        rows: List[Dict[str, Any]] = []

        if self.state and self.state.generation_stats:
            source = "memory"
            rows = []
            for stat in self.state.generation_stats:
                if not isinstance(stat, dict):
                    continue
                generation = self._as_int(stat.get("generation"), self._as_int(stat.get("gen"), 0))
                if since_generation is not None and generation < since_generation:
                    continue
                rows.append(stat)
        else:
            csv_rows = self.get_metrics_rows(limit=None, offset=0, since_generation=since_generation)
            if csv_rows:
                source = "metrics_csv"
                rows = csv_rows

        total = len(rows)
        safe_offset = max(0, int(offset))
        if safe_offset >= total:
            paged_rows: List[Dict[str, Any]] = []
        else:
            paged_rows = rows[safe_offset:]

        if limit is not None:
            safe_limit = max(0, int(limit))
            paged_rows = paged_rows[:safe_limit]

        return {
            "source": source,
            "total": total,
            "count": len(paged_rows),
            "limit": limit,
            "offset": safe_offset,
            "since_generation": since_generation,
            "items": paged_rows,
        }

    def list_checkpoints(self, limit: Optional[int] = 20) -> List[Dict[str, Any]]:
        """List checkpoint marker files and their associated artifacts."""
        if not self.checkpoint_dir.exists():
            return []

        checkpoint_items: List[Dict[str, Any]] = []
        marker_files = sorted(
            (
                path for path in self.checkpoint_dir.glob("checkpoint_gen_*.json")
                if not path.name.endswith("_config.json")
                and not path.name.endswith("_expirement_state.json")
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for marker_path in marker_files:
            try:
                marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
            except Exception:
                marker_payload = {}

            checkpoint_items.append({
                "checkpoint_path": str(marker_path),
                "generation": self._as_int(marker_payload.get("generation"), 0),
                "saved_at_utc": str(marker_payload.get("saved_at_utc", "")),
                "config_path": str(marker_payload.get("config", "")) or None,
                "experiment_path": str(marker_payload.get("experiment", "")) or None,
                "metrics_path": str(marker_payload.get("metrics", "")) or None,
                "marker_exists": marker_path.exists(),
            })

        if limit is None:
            return checkpoint_items

        safe_limit = max(0, int(limit))
        return checkpoint_items[:safe_limit]

    @staticmethod
    def _is_within_directory(candidate: Path, directory: Path) -> bool:
        try:
            candidate.relative_to(directory)
            return True
        except ValueError:
            return False

    def resolve_checkpoint_path(self, checkpoint_path: str) -> Path:
        raw_value = str(checkpoint_path).strip()
        if not raw_value:
            raise ValueError("checkpoint_path is required")

        requested_path = Path(raw_value)
        candidate_path = (
            requested_path
            if requested_path.is_absolute()
            else (self.checkpoint_dir / requested_path)
        )

        checkpoint_root = self.checkpoint_dir.resolve(strict=False)
        resolved_candidate = candidate_path.resolve(strict=False)
        if not self._is_within_directory(resolved_candidate, checkpoint_root):
            raise ValueError("Checkpoint path must stay within the job checkpoint directory")

        if not candidate_path.exists():
            raise ValueError("Checkpoint path not found")

        if not candidate_path.is_file():
            raise ValueError("Checkpoint path must point to a checkpoint marker file")

        return candidate_path
    
    async def initialize(self):
        """Initialize or load training state"""
        try:
            if self._split_state_exists():
                try:
                    # Load existing split state produced by save_coevolution_state
                    self.state = load_coevolution_state(str(self.state_base_path))
                    logger.info(f"Loaded existing state at generation {self.state.generation}")
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning(f"Failed to load existing state ({exc}); starting fresh")
                    self.state = None

            if self.state is None:
                # Fresh initialization with populated prey/predator populations
                config = EvolutionConfig()
                self.state = TrainingState(config)
                self.state.prey_population = [
                    PreyGenome.random_initialization() for _ in range(config.population_size)
                ]
                self.state.predator_population = [
                    PredatorGenome.random_initialization() for _ in range(config.predator_population_size)
                ]
                logger.info(
                    "Initialized fresh training state with populations "
                    f"({len(self.state.prey_population)} prey, {len(self.state.predator_population)} predators)"
                )

            state = self.state
            if state is None:
                raise RuntimeError("Training state was not initialized")

            if self.evaluator is None:
                requested_device = os.getenv("EVOMIND_DEVICE", "auto").strip().lower()
                use_gpu = False
                if requested_device in ("auto", "cuda"):
                    try:
                        import torch
                        use_gpu = bool(torch.cuda.is_available())
                    except Exception:
                        use_gpu = False

                self.evaluator = AsyncDeterministicEvaluator(
                    base_seed=state.config.base_seed,
                    num_workers=state.config.num_workers,
                    use_gpu=use_gpu,
                    envs_per_genome=state.config.envs_per_genome,
                    max_steps=state.config.max_steps,
                )

            if self.curriculum_controller is None:
                self.curriculum_controller = CurriculumController()

            if self.architect_population is None:
                self.architect_population = ArchitectPopulation(population_size=20)

            if self.mutator_population is None:
                self.mutator_population = MutatorPopulation(population_size=15)
            
            if self.prey_engine is None:
                self.prey_engine = EvolutionEngine(
                    population_size=state.config.population_size,
                    tournament_size=state.config.tournament_size,
                    elite_count=state.config.elite_count,
                    mutation_rate=state.config.mutation_rate,
                    mutation_strength=state.config.mutation_strength,
                    architecture_mutation_rate=state.config.architecture_mutation_rate,
                    genome_cls=PreyGenome,
                    architect_population=self.architect_population,
                    mutator_population=self.mutator_population,
                )

            if self.predator_engine is None:
                self.predator_engine = EvolutionEngine(
                    population_size=state.config.predator_population_size,
                    tournament_size=state.config.tournament_size,
                    elite_count=state.config.elite_count,
                    mutation_rate=state.config.mutation_rate,
                    mutation_strength=state.config.mutation_strength,
                    architecture_mutation_rate=state.config.architecture_mutation_rate,
                    genome_cls=PredatorGenome,
                    architect_population=self.architect_population,
                    mutator_population=self.mutator_population,
                )
            
            self.update_status()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            return False
    
    def update_status(self):
        """Update internal status snapshot"""
        if self.state:
            latest_stats = self.state.generation_stats[-1] if self.state.generation_stats else {}
            prey_species = latest_stats.get("prey_species", {}) if isinstance(latest_stats, dict) else {}
            predator_species = latest_stats.get("predator_species", {}) if isinstance(latest_stats, dict) else {}
            neural_health = latest_stats.get("neural_health", {}) if isinstance(latest_stats, dict) else {}

            stage_name = (
                latest_stats.get("stage")
                if isinstance(latest_stats, dict) and latest_stats.get("stage") is not None
                else getattr(self.state, "current_stage", "unknown")
            )

            self.last_status.update({
                "status": "running" if self.is_running else "stopped",
                "generation": self.state.generation,
                "stage": stage_name,
                "best_prey_fitness": (
                    max([g.fitness for g in self.state.prey_population]) 
                    if self.state.prey_population else 0.0
                ),
                "best_predator_fitness": (
                    max([g.fitness for g in self.state.predator_population]) 
                    if self.state.predator_population else 0.0
                ),
                "mean_prey_fitness": (
                    sum(g.fitness for g in self.state.prey_population) / len(self.state.prey_population)
                    if self.state.prey_population else 0.0
                ),
                "mean_predator_fitness": (
                    sum(g.fitness for g in self.state.predator_population) / len(self.state.predator_population)
                    if self.state.predator_population else 0.0
                ),
                "curriculum_stage": getattr(self.state, 'current_stage', 'unknown'),
                "total_generations_trained": self.state.generation,
                "uptime_seconds": time.time() - self.start_time if self.start_time else 0.0,
                "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fitness": {
                    "prey": {
                        "best": (
                            max([g.fitness for g in self.state.prey_population])
                            if self.state.prey_population else 0.0
                        ),
                        "average": (
                            sum(g.fitness for g in self.state.prey_population) / len(self.state.prey_population)
                            if self.state.prey_population else 0.0
                        ),
                    },
                    "predator": {
                        "best": (
                            max([g.fitness for g in self.state.predator_population])
                            if self.state.predator_population else 0.0
                        ),
                        "average": (
                            sum(g.fitness for g in self.state.predator_population) / len(self.state.predator_population)
                            if self.state.predator_population else 0.0
                        ),
                    },
                },
                "learning": {
                    "adaptability": float(latest_stats.get("avg_adaptability_score", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                    "meta_effectiveness": float(latest_stats.get("avg_meta_effectiveness", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                    "performance_change": float(latest_stats.get("avg_reward_delta", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                    "instability": float(latest_stats.get("avg_instability", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                },
                "behavior": {
                    "success_rate": float(latest_stats.get("avg_success_rate", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                    "stability": float(latest_stats.get("avg_stability", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                    "novelty": float(latest_stats.get("avg_novelty", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                },
                "diversity": {
                    "prey_species": int(prey_species.get("num_species", 0)) if isinstance(prey_species, dict) else 0,
                    "predator_species": int(predator_species.get("num_species", 0)) if isinstance(predator_species, dict) else 0,
                },
                "neural_health": {
                    "dead_connections": int(neural_health.get("dead_layers", 0)) if isinstance(neural_health, dict) else 0,
                    "saturation": int(neural_health.get("saturated_layers", 0)) if isinstance(neural_health, dict) else 0,
                },
                "system": {
                    "evaluation_time_sec": float(latest_stats.get("eval_time", 0.0)) if isinstance(latest_stats, dict) else 0.0,
                    "status": "running" if self.is_running else "stopped",
                    "uptime_seconds": time.time() - self.start_time if self.start_time else 0.0,
                    "last_update": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            })
    
    async def start(self) -> Dict[str, Any]:
        """Start background training"""
        if self.is_running:
            return {"status": "already_running"}

        if self.state is None:
            success = await self.initialize()
            if not success:
                return {"status": "initialization_failed"}
        
        self.start_time = time.time()
        self.is_running = True
        
        # Create background training task
        self.training_task = asyncio.create_task(self._training_loop())
        
        logger.info("Training started in background")
        self.update_status()
        return self.last_status
    
    async def _training_loop(self):
        """Main training loop - runs train_coevolution_async per generation"""
        try:
            state = self.state
            if (
                state is None
                or self.evaluator is None
                or self.curriculum_controller is None
                or self.prey_engine is None
                or self.predator_engine is None
                or self.architect_population is None
                or self.mutator_population is None
            ):
                raise RuntimeError("Trainer dependencies are not initialized")

            evaluator = self.evaluator
            prey_engine = self.prey_engine
            predator_engine = self.predator_engine
            architect_population = self.architect_population
            mutator_population = self.mutator_population

            while self.is_running and state.generation < state.config.generations:
                current_stage = self.curriculum_controller.get_current_config()
                stage = CurriculumStage[current_stage["name"].upper()]

                # Run one generation
                def _run_generation() -> Dict[str, Any]:
                    return asyncio.run(
                        train_coevolution_async(
                            state.generation,
                            state,
                            evaluator,
                            stage,
                            prey_engine,
                            predator_engine,
                            architect_population,
                            mutator_population,
                        )
                    )

                stats = await asyncio.to_thread(_run_generation)

                # Persist metrics history and hall-of-fame on evaluated population
                state.generation_stats.append(stats)
                state.update_hall_of_fame()

                # Evolve for next generation
                prey_population = prey_engine.create_next_generation(
                    state.prey_population,
                    state.generation,
                    pop_name="prey",
                )
                predator_population = predator_engine.create_next_generation(
                    state.predator_population,
                    state.generation,
                    pop_name="predator",
                )
                state.prey_population = prey_population.genomes
                state.predator_population = predator_population.genomes

                state.generation += 1

                # Keep canonical state current for automatic resume.
                await asyncio.to_thread(save_coevolution_state, state, str(self.state_base_path))
                if evaluator is not None and hasattr(evaluator, "save_seeds"):
                    await asyncio.to_thread(evaluator.save_seeds)
                
                # Save checkpoint every 5 generations
                if state.generation % 5 == 0:
                    await self.save_checkpoint()
                
                self.update_status()
                
        except asyncio.CancelledError:
            logger.info("Training loop cancelled")
        except Exception as e:
            logger.error(f"Training loop error: {e}")
            self.is_running = False
        finally:
            self.is_running = False
            self.update_status()
    
    async def stop(self) -> Dict[str, Any]:
        """Gracefully stop training"""
        self.is_running = False
        
        if self.training_task:
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass
        
        await self.save_checkpoint()
        self.update_status()
        return self.last_status
    
    async def resume(self, checkpoint_path: str) -> Dict[str, Any]:
        """Resume from specific checkpoint"""
        try:
            resolved_checkpoint_path = self.resolve_checkpoint_path(checkpoint_path)
            # Load state from checkpoint
            self.state = load_coevolution_state(str(resolved_checkpoint_path))
            
            # Restart training task
            self.start_time = time.time()
            self.is_running = True
            self.training_task = asyncio.create_task(self._training_loop())
            
            self.update_status()
            return {
                **self.last_status,
                "checkpoint_path": str(resolved_checkpoint_path),
            }
            
        except Exception as e:
            logger.error(f"Resume failed: {e}")
            return {"status": "resume_failed", "error": str(e)}
    
    async def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save current state as checkpoint"""
        if not self.state:
            raise ValueError("No training state to save")
        
        path_obj: Path
        if path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path_obj = self.checkpoint_dir / f"checkpoint_gen_{self.state.generation}_{timestamp}.json"
        else:
            path_obj = Path(path)

        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Always update canonical split state for API auto-resume.
        await asyncio.to_thread(save_coevolution_state, self.state, str(self.state_base_path))
        # Save generation snapshot in checkpoint namespace.
        await asyncio.to_thread(save_coevolution_state, self.state, str(path_obj))

        # Create marker JSON so a visible file exists at checkpoint_path.
        marker_payload = {
            "generation": self.state.generation,
            "checkpoint_base": str(path_obj),
            "config": str(path_obj.with_name(f"{path_obj.stem}_config.json")),
            "experiment": str(path_obj.with_name(f"{path_obj.stem}_expirement_state.json")),
            "metrics": str(path_obj.with_name(f"{path_obj.stem}_metrices.csv")),
            "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        await asyncio.to_thread(
            path_obj.write_text,
            json.dumps(marker_payload, indent=2),
            "utf-8",
        )

        if self.evaluator is not None and hasattr(self.evaluator, "save_seeds"):
            await asyncio.to_thread(self.evaluator.save_seeds)

        logger.info(f"Checkpoint saved: {path_obj}")
        return str(path_obj)
    
    def status(self) -> TrainStatus:
        """Get current training status"""
        self.update_status()
        return TrainStatus(**self.last_status)
    
    def get_best_genome(self, genome_type: str) -> Optional[Any]:
        """Get best genome of specified type"""
        if not self.state:
            return None
        
        if genome_type == "prey":
            if self.state.prey_population:
                return max(self.state.prey_population, key=lambda g: g.fitness)
        elif genome_type == "predator":
            if self.state.predator_population:
                return max(self.state.predator_population, key=lambda g: g.fitness)
        
        return None

