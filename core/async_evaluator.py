"""Asynchronous evaluator with deterministic seeds"""
import asyncio
import concurrent.futures
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, cast
from environments.deterministic_env import DeterministicSeedManager
import time
import os
import multiprocessing
import threading
from dataclasses import dataclass
from enum import Enum

class EvaluationMode(Enum):
    SINGLE_AGENT = "single_agent"
    CO_EVOLUTION = "co_evolution"
    SELF_PLAY = "self_play"

@dataclass
class EvaluationConfig:
    """Configuration for evaluation to ensure determinism"""
    base_seed: int = 42
    num_workers: int = 4
    use_gpu: bool = False
    envs_per_genome: int = 8
    max_steps: int = 1000
    mode: EvaluationMode = EvaluationMode.SINGLE_AGENT
    # Multi-agent specific
    num_prey: int = 10
    num_predators: int = 3
    batch_size: int = 8
    
    def __post_init__(self):
        # Ensure valid configuration
        assert self.num_workers > 0, "num_workers must be positive"
        assert self.envs_per_genome > 0, "envs_per_genome must be positive"

class AsyncDeterministicEvaluator:
    """
    Asynchronous evaluator with full determinism
    """
    
    def __init__(self,
                 config: Optional[EvaluationConfig] = None,
                 **kwargs):
        """
        Args:
            config: Evaluation configuration dataclass
            **kwargs: Override config values
        """
        if config is None:
            config = EvaluationConfig(**kwargs)
        
        self.config = config
        self.envs_per_genome = config.envs_per_genome
        self.max_steps = config.max_steps
        self.use_gpu = config.use_gpu
        self.use_gpu = False  # Force CPU for async
        self.mode = config.mode
        
        # Seed manager - NO TIME-BASED OFFSETS
        self.seed_manager = DeterministicSeedManager(config.base_seed)
        
        # Determine number of workers
        self.num_workers = config.num_workers
        
        # GPU lock for thread safety
        # self.gpu_lock = threading.Lock() if self.use_gpu else None
        
        # Thread pool for CPU work
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix='eval_cpu_worker_'
        )
        
        # Process pool for GPU work (safer for CUDA)
        self.process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(4, self.num_workers),  # Fewer processes for GPU
            mp_context=multiprocessing.get_context('spawn')
        )
        
        # Track deterministic seeds
        self.seed_registry = {}
        
        # Diagnostic control flag (enable only every N generations)
        self.enable_diagnostics = False
        
        print(f"AsyncDeterministicEvaluator initialized")
        print(f"  Mode: {self.mode.value}")
        print(f"  Seed: {config.base_seed}, Workers: {self.num_workers}")
        print(f"  GPU: {self.use_gpu}, Envs per genome: {self.envs_per_genome}")
        print(f"  Deterministic: Yes (no time-based offsets)")
    
    def get_deterministic_seed(self, 
                              identifier: str,
                              generation: int,
                              genome_idx: int,
                              opponent_idx: Optional[int] = None) -> int:
        """
        Get deterministic seed WITHOUT time-based offsets
        """
        seed_key = f"g{generation}_genome{genome_idx}"
        if opponent_idx is not None:
            seed_key += f"_opponent{opponent_idx}"
        if self.mode != EvaluationMode.SINGLE_AGENT:
            seed_key += f"_{self.mode.value}"
        
        seed = self.seed_manager.get_seed(seed_key, offset=0)  # NO TIME OFFSET
        self.seed_registry[identifier] = seed
        return seed
    
    async def evaluate_genome_async(self, 
                                   genome, 
                                   genome_idx: int,
                                   generation: int,
                                   stage_config: Optional[Dict[str, Any]] = None,
                                   opponent_genome: Optional[Dict[str, Any]] = None,
                                   opponent_idx: Optional[int] = None) -> float:
        """
        Evaluate a single genome asynchronously with determinism
        """
        # Get deterministic seed WITHOUT time component
        eval_seed = self.get_deterministic_seed(
            identifier=f"genome_{genome_idx}",
            generation=generation,
            genome_idx=genome_idx,
            opponent_idx=opponent_idx
        )
        
        # Choose execution method based on GPU usage
        if self.use_gpu:
            # Use process pool for GPU safety
            loop = asyncio.get_event_loop()
            fitness = await loop.run_in_executor(
                self.process_pool,
                self._evaluate_genome_process_safe,
                genome,
                eval_seed,
                generation,
                genome_idx,
                stage_config,
                opponent_genome,
                opponent_idx
            )
        else:
            # Use thread pool for CPU-only
            loop = asyncio.get_event_loop()
            fitness = await loop.run_in_executor(
                self.thread_pool,
                self._evaluate_genome_cpu,
                genome,
                eval_seed,
                generation,
                genome_idx,
                stage_config,
                opponent_genome,
                opponent_idx
            )
        
        return fitness
    
    def _evaluate_genome_process_safe(self,
                                     genome,
                                     seed: int,
                                     generation: int,
                                     genome_idx: int,
                                     stage_config: Optional[Dict[str, Any]] = None,
                                     opponent_genome: Optional[Dict[str, Any]] = None,
                                     opponent_idx: Optional[int] = None) -> float:
        """
        Process-safe evaluation (each process has its own GPU context)
        """
        if self.use_gpu:
            raise RuntimeError("GPU not allowed in async evaluator")
        # Each process imports its own modules
        import numpy as np

        if self.mode == EvaluationMode.SINGLE_AGENT:
            fitness, _ = self._evaluate_single_agent(genome, seed, stage_config)
            return fitness
        elif self.mode == EvaluationMode.CO_EVOLUTION:
            if opponent_genome is None:
                raise ValueError("opponent_genome required for co-evaluation")
            fitness, metrics = self._evaluate_co_evolution(
                genome, opponent_genome, seed, generation, genome_idx, opponent_idx
            )
            return fitness
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _evaluate_genome_cpu(self,
                            genome,
                            seed: int,
                            generation: int,
                            genome_idx: int,
                            stage_config: Optional[Dict[str, Any]] = None,
                            opponent_genome: Optional[Dict[str, Any]] = None,
                            opponent_idx: Optional[int] = None) -> float:
        """
        CPU-only evaluation (thread-safe)
        """
        if self.mode == EvaluationMode.SINGLE_AGENT:
            fitness, _ = self._evaluate_single_agent(genome, seed, stage_config)
            return fitness
        elif self.mode == EvaluationMode.CO_EVOLUTION:
            if opponent_genome is None:
                raise ValueError("opponent_genome required for co-evaluation")
            fitness, _ = self._evaluate_co_evolution(
                genome, opponent_genome, seed, generation, genome_idx, opponent_idx
            )
            return fitness
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _evaluate_single_agent(self,
                              genome,
                              seed: int,
                              stage_config: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Single-agent evaluation with multi-seed fitness for META generalization
        Returns: (fitness, plastic_diagnostics)
        """
        print("DEBUG: evaluating genome with multi-seed fitness")
        print(
            genome.meta["reward_gain"],
            genome.meta["reward_bias"]
        )

        from environments.deterministic_env import DeterministicVectorizedArena

        # Create ONE arena and reuse across all seeds
        # Use 5x environments to run all seeds in parallel
        env = DeterministicVectorizedArena(
            num_envs=self.envs_per_genome * 5,  # Run 5 seeds in parallel
            max_steps=self.max_steps,
            seed=seed,
            stage_config=stage_config,
            enable_diagnostics=self.enable_diagnostics
        )

        rewards = []
        all_plastic_diagnostics = []

        for seed_offset in range(5):
            # Reset plasticity before each rollout for META generalization
            if hasattr(genome, "brain"):
                genome.brain.reset_plasticity()

            # Reset environment for this seed
            states = env.reset()
            
            # Extract states for this seed offset
            start_idx = seed_offset * self.envs_per_genome
            end_idx = (seed_offset + 1) * self.envs_per_genome
            seed_states = states[start_idx:end_idx]
            
            total_reward = 0.0
            step = 0
            episode_plastic_deltas = []

            while step < self.max_steps:
                # Get actions
                actions = genome.act_batch(seed_states)  # Always use CPU for process safety

                # We need to handle the full batch - create padded actions
                batch_actions = np.zeros((self.envs_per_genome * 5,) + actions.shape[1:], dtype=actions.dtype)
                batch_actions[start_idx:end_idx] = actions

                # Step environment
                states, step_rewards, dones = env.step(batch_actions)
                
                # Extract rewards for this seed
                seed_states = states[start_idx:end_idx]
                seed_rewards = step_rewards[start_idx:end_idx]
                seed_dones = dones[start_idx:end_idx]

                # Update plasticity with reward signal
                if hasattr(genome, "brain"):
                    r = float(np.mean(seed_rewards))
                    if abs(r) > 0.05:
                        genome.brain.update_plasticity(r, done=False)

                    # Collect plastic diagnostics only when explicitly enabled.
                    if self.enable_diagnostics:
                        diagnostics = genome.brain.get_plastic_diagnostics()
                        episode_plastic_deltas.append(diagnostics["total_plastic_delta"])

                total_reward += np.mean(seed_rewards)
                step += 1

                if np.all(seed_dones):
                    break

            rewards.append(float(total_reward))

            # Store episode diagnostics
            if episode_plastic_deltas:
                all_plastic_diagnostics.append({
                    "episode_reward": float(total_reward),
                    "plastic_deltas": episode_plastic_deltas,
                    "final_plastic_delta": episode_plastic_deltas[-1] if episode_plastic_deltas else 0.0
                })

        env.close()

        # Aggregate diagnostics across episodes
        plastic_diagnostics = {
            "episodes": all_plastic_diagnostics,
            "mean_final_plastic_delta": float(np.mean([ep["final_plastic_delta"] for ep in all_plastic_diagnostics])) if all_plastic_diagnostics else 0.0,
            "std_final_plastic_delta": float(np.std([ep["final_plastic_delta"] for ep in all_plastic_diagnostics])) if all_plastic_diagnostics else 0.0
        }

        # Store plastic diagnostics in genome for evolution
        genome.plastic_diagnostics = plastic_diagnostics

        return float(np.mean(rewards)), plastic_diagnostics
    
    def _evaluate_co_evolution(self,
                              genome,
                              opponent_genome,
                              seed: int,
                              generation: int,
                              genome_idx: int,
                              opponent_idx: Optional[int] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Multi-agent co-evolution evaluation
        """
        # Import multi-agent modules inside the process/thread
        from environments.arena_multi import MultiAgentArena
        from genome_prey import PreyGenome
        from genome_predator import PredatorGenome
        
        # Determine roles
        if isinstance(genome, PreyGenome) and isinstance(opponent_genome, PredatorGenome):
            prey_genome = genome
            predator_genome = opponent_genome
            role = "prey"
        elif isinstance(genome, PredatorGenome) and isinstance(opponent_genome, PreyGenome):
            predator_genome = genome
            prey_genome = opponent_genome
            role = "predator"
        else:
            raise ValueError(f"Invalid genome types for co-evolution: "
                           f"{type(genome)} vs {type(opponent_genome)}")
        
        # Create multi-agent arena
        arena = MultiAgentArena(
            batch_size=self.config.batch_size,
            num_prey_per_env=self.config.num_prey,
            num_predators_per_env=self.config.num_predators,
            config={'max_steps': self.max_steps},
            seed=seed
        )
        
        prey_state, pred_state = arena.reset()
        total_reward = 0.0
        steps_run = 0
        last_info: Dict[str, Any] = {}

        # Ensure brains are cached
        prey_brain = getattr(prey_genome, "brain", None)
        predator_brain = getattr(predator_genome, "brain", None)

        # Reset episode tracking for plasticity diagnostics
        if role == "prey":
            if prey_brain is not None and hasattr(prey_brain, "reset_episode_tracking"):
                prey_brain.reset_episode_tracking()
        else:
            if predator_brain is not None and hasattr(predator_brain, "reset_episode_tracking"):
                predator_brain.reset_episode_tracking()

        for step in range(self.max_steps):
            steps_run = step + 1

            # Get actions based on role
            if role == "prey":
                prey_actions = prey_genome.act_batch(prey_state)

                # Create dummy predator actions (opponent will be evaluated separately)
                # Or use opponent genome if available
                pred_actions = predator_genome.act_batch(pred_state)
            else:  # predator
                prey_actions = prey_genome.act_batch(prey_state)
                pred_actions = predator_genome.act_batch(pred_state)

            # Step the arena
            (prey_state, pred_state), r_prey, r_pred, info = arena.step(
                prey_actions, pred_actions
            )
            last_info = info

            # Update plasticity with reward signal
            if role == "prey":
                brain = getattr(prey_genome, "brain", None)
                if brain is not None and hasattr(brain, "apply_plasticity"):
                    brain.apply_plasticity(float(r_prey.mean().item()), brain.meta)
                total_reward += r_prey.mean().item()
            else:
                brain = getattr(predator_genome, "brain", None)
                if brain is not None and hasattr(brain, "apply_plasticity"):
                    brain.apply_plasticity(float(r_pred.mean().item()), brain.meta)
                total_reward += r_pred.mean().item()

            if np.any(info['env_done']):
                break

        metrics: Dict[str, Any] = {
            "role": role,
            "seed": seed,
            "generation": generation,
            "genome_idx": genome_idx,
            "opponent_idx": opponent_idx,
            "steps_run": steps_run,
            "episode_return": float(total_reward),
            "env_done": bool(np.any(last_info.get("env_done", False))) if isinstance(last_info, dict) else False,
        }
        return float(total_reward), metrics
    
    async def evaluate_coevolution_pair_async(self,
                                             prey_genome,
                                             predator_genome,
                                             generation: int,
                                             pair_idx: int,
                                             stage_config: Optional[Dict[str, Any]] = None) -> Tuple[float, float]:
        """
        Evaluate a prey-predator pair asynchronously
        Returns: (prey_fitness, predator_fitness)
        """
        # Get deterministic seed for the pair
        pair_seed = self.get_deterministic_seed(
            identifier=f"pair_{pair_idx}",
            generation=generation,
            genome_idx=pair_idx,
            opponent_idx=None
        )
        
        # Import inside async context
        from environments.arena_multi import MultiAgentArena
        
        # Create arena
        arena = MultiAgentArena(
            batch_size=self.config.batch_size,
            num_prey_per_env=self.config.num_prey,
            num_predators_per_env=self.config.num_predators,
            config={'max_steps': self.max_steps},
            seed=pair_seed
        )
        
        prey_state, pred_state = arena.reset()
        prey_total_reward = 0.0
        predator_total_reward = 0.0
        
        for step in range(self.max_steps):
            # Get actions
            prey_actions = prey_genome.act_batch(prey_state)
            pred_actions = predator_genome.act_batch(pred_state)

            # Step environment
            (prey_state, pred_state), r_prey, r_pred, info = arena.step(
                prey_actions, pred_actions
            )

            prey_total_reward += r_prey.mean().item()
            predator_total_reward += r_pred.mean().item()

            if np.any(info['env_done']):
                break
        return float(prey_total_reward), float(predator_total_reward)
    
    async def evaluate_batch_async(self,
                                  genomes: List,
                                  generation: int,
                                  stage_configs: Optional[List[Optional[Dict[str, Any]]]] = None,
                                  opponent_genomes: Optional[List] = None) -> List[float]:
        """
        Evaluate multiple genomes asynchronously
        """
        stage_configs_list: List[Optional[Dict[str, Any]]] = stage_configs if stage_configs is not None else [None] * len(genomes)

        if opponent_genomes is None and self.mode == EvaluationMode.CO_EVOLUTION:
            raise ValueError("opponent_genomes required for co-evaluation mode")

        # Create evaluation tasks
        tasks = []
        for i, (genome, config) in enumerate(zip(genomes, stage_configs_list)):
            opponent = opponent_genomes[i] if opponent_genomes else None
            opponent_idx = i if opponent_genomes else None

            task = asyncio.ensure_future(
                self.evaluate_genome_async(
                    genome, i, generation, config, opponent, opponent_idx
                )
            )
            tasks.append(task)

        # Run all evaluations in parallel
        fitnesses = await asyncio.gather(*tasks)
        return fitnesses
    
    async def evaluate_population_async(self,
                                       population,
                                       generation: int,
                                       stage_config: Optional[Dict[str, Any]] = None,
                                       opponent_population: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate entire population asynchronously
        Supports both single-agent and co-evolution
        """
        print("DEBUG: starting async evaluation")
        genomes = population.genomes if hasattr(population, 'genomes') else population
        
        # Prepare opponent genomes for co-evolution
        opponent_genomes = None
        if self.mode == EvaluationMode.CO_EVOLUTION and opponent_population:
            opponent_genomes = opponent_population.genomes if hasattr(opponent_population, 'genomes') else opponent_population
        
        # Evaluate all genomes
        fitnesses = await self.evaluate_batch_async(
            genomes,
            generation,
            cast(Optional[List[Optional[Dict[str, Any]]]], [stage_config] * len(genomes) if stage_config else None),
            opponent_genomes
        )
        
        # Assign fitness if population has genomes
        if hasattr(population, 'genomes'):
            for i, (genome, fitness) in enumerate(zip(population.genomes, fitnesses)):
                genome.fitness = fitness
                if i % 10 == 0:
                    print(f"  Evaluated {i}/{len(population.genomes)} prey genomes")
        
        # Calculate statistics
        fitness_array = np.array(fitnesses, dtype=np.float32)
        
        return {
            'fitnesses': fitnesses,
            'mean': float(np.mean(fitness_array)),
            'std': float(np.std(fitness_array)),
            'max': float(np.max(fitness_array)),
            'min': float(np.min(fitness_array)),
            'median': float(np.median(fitness_array))
        }
    
    def save_seeds(self, filename: str = "seed_registry.json"):
        """Save seed registry for reproducibility"""
        self.seed_manager.save_seeds(filename)
    
    def load_seeds(self, filename: str = "seed_registry.json"):
        """Load seed registry"""
        self.seed_manager.load_seeds(filename)
    
    def get_seed_info(self) -> Dict:
        """Get seed information"""
        return self.seed_manager.get_seed_registry()

    def summarize_seed_coverage(self, generation: int, max_examples: int = 5) -> Dict[str, Any]:
        """Summarize deterministic seed coverage for a generation."""
        registry = self.seed_manager.get_seed_registry()
        prefix = f"g{generation}_"
        seeds = [seed for key, seed in registry.items() if key.startswith(prefix)]
        if not seeds:
            return {
                "generation": generation,
                "total": 0,
                "unique": 0,
                "min": None,
                "max": None,
                "examples": [],
            }

        unique_seeds = sorted(set(seeds))
        return {
            "generation": generation,
            "total": len(seeds),
            "unique": len(unique_seeds),
            "min": min(seeds),
            "max": max(seeds),
            "examples": unique_seeds[:max_examples],
        }

    def log_seed_coverage(self, generation: int, max_examples: int = 5) -> None:
        """Log seed coverage summary for a generation at INFO level."""
        summary = self.summarize_seed_coverage(generation, max_examples=max_examples)
        logger = logging.getLogger("seed_coverage")
        if summary["total"] == 0:
            logger.info("Seed coverage gen %d: none recorded", generation)
            return

        examples = ", ".join(str(s) for s in summary["examples"])
        if summary["unique"] > len(summary["examples"]):
            examples = f"{examples}, ..." if examples else "..."
        logger.info(
            "Seed coverage gen %d: %d evals, %d unique (min=%d, max=%d) examples=[%s]",
            generation,
            summary["total"],
            summary["unique"],
            summary["min"],
            summary["max"],
            examples,
        )
    
    def close(self):
        """Cleanup resources"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class HybridEvaluator:
    """
    Hybrid evaluator that combines async and sync evaluation strategies
    """
    
    def __init__(self, 
                 config: Optional[EvaluationConfig] = None,
                 **kwargs):
        """
        Args:
            config: Evaluation configuration dataclass
            **kwargs: Override config values
        """
        if config is None:
            config = EvaluationConfig(**kwargs)
        
        self.config = config
        self.batch_size = config.batch_size
        self.envs_per_batch = config.envs_per_genome * config.batch_size
        self.envs_per_genome = config.envs_per_genome
        self.use_gpu = config.use_gpu
        self.mode = config.mode
        
        # Seed manager - NO TIME-BASED OFFSETS
        self.seed_manager = DeterministicSeedManager(config.base_seed)
        
        # Diagnostic control flag (enable only every N generations)
        self.enable_diagnostics = False
        
        print(f"HybridEvaluator initialized")
        print(f"  Mode: {self.mode.value}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Envs per genome: {self.envs_per_genome}")
        print(f"  Deterministic: Yes (no time-based offsets)")
    
    def get_deterministic_batch_seed(self,
                                    batch_idx: int,
                                    generation: int) -> int:
        """
        Get deterministic seed for a batch WITHOUT time-based offsets
        """
        seed_key = f"batch_{batch_idx}_gen{generation}"
        if self.mode != EvaluationMode.SINGLE_AGENT:
            seed_key += f"_{self.mode.value}"
        
        return self.seed_manager.get_seed(seed_key, offset=0)  # NO TIME OFFSET
    
    def evaluate_batch(self,
                      genomes: List,
                      generation: int,
                      stage_config: Optional[Dict[str, Any]] = None,
                      opponent_genomes: Optional[List] = None) -> List[float]:
        """
        Evaluate batch of genomes using hybrid strategy
        """
        if self.mode == EvaluationMode.CO_EVOLUTION:
            assert opponent_genomes is not None, "opponent_genomes required for co-evolution"
            prey_fitnesses, _ = self._evaluate_coevolution_batch(
                genomes, opponent_genomes, generation, stage_config
            )
            return prey_fitnesses
        else:
            return self._evaluate_single_agent_batch(
                genomes, generation, stage_config
            )
    
    def _evaluate_single_agent_batch(self,
                                    genomes: List,
                                    generation: int,
                                    stage_config: Optional[Dict[str, Any]] = None) -> List[float]:
        """
        Evaluate batch of single-agent genomes
        """
        if len(genomes) > self.batch_size:
            # Split into smaller batches
            all_fitnesses = []
            for batch_idx in range(0, len(genomes), self.batch_size):
                batch = genomes[batch_idx:batch_idx + self.batch_size]
                fitnesses = self._evaluate_single_batch(batch, batch_idx // self.batch_size, generation, stage_config)
                all_fitnesses.extend(fitnesses)
            
            return all_fitnesses
        else:
            return self._evaluate_single_batch(genomes, 0, generation, stage_config)
    
    def _evaluate_single_batch(self,
                              genomes: List,
                              batch_idx: int,
                              generation: int,
                              stage_config: Optional[Dict] = None) -> List[float]:
        """
        Evaluate a single batch of genomes (single-agent)
        """
        num_genomes = len(genomes)

        # Reset plasticity for all genomes before rollout
        for genome in genomes:
            if hasattr(genome, "brain"):
                genome.brain.reset_plasticity()
            print(
                genome.meta["reward_gain"],
                genome.meta["reward_bias"]
            )

        # Create deterministic environment
        from environments.deterministic_env import DeterministicVectorizedArena

        # Calculate actual number of environments needed
        num_envs = num_genomes * self.envs_per_genome

        # Get deterministic seed WITHOUT time component
        batch_seed = self.get_deterministic_batch_seed(batch_idx, generation)

        env = DeterministicVectorizedArena(
            num_envs=num_envs,
            max_steps=self.config.max_steps,
            seed=batch_seed,
            stage_config=stage_config,
            enable_diagnostics=self.enable_diagnostics
        )

        states = env.reset()
        
        # Track rewards for each genome
        rewards = np.zeros(num_genomes, dtype=np.float32)
        
        # Run evaluation
        for step in range(self.config.max_steps):
            # Get actions for all genomes
            batch_actions = []

            for i, genome in enumerate(genomes):
                start_idx = i * self.envs_per_genome
                end_idx = (i + 1) * self.envs_per_genome
                genome_states = states[start_idx:end_idx]

                # Always use CPU for batch evaluation safety
                actions = genome.act_batch(genome_states)
                batch_actions.append(actions)

            # Concatenate actions
            all_actions = np.concatenate(batch_actions)

            # Step environment
            states, step_rewards, dones = env.step(all_actions)

            # Update plasticity for each genome with reward signal
            for i, genome in enumerate(genomes):
                start_idx = i * self.envs_per_genome
                end_idx = (i + 1) * self.envs_per_genome
                genome_reward = np.mean(step_rewards[start_idx:end_idx])
                if hasattr(genome, "brain"):
                    genome.brain.apply_plasticity(float(genome_reward), genome.brain.meta)

            # Accumulate rewards for each genome
            for i in range(num_genomes):
                start_idx = i * self.envs_per_genome
                end_idx = (i + 1) * self.envs_per_genome
                rewards[i] += np.mean(step_rewards[start_idx:end_idx])

            if np.all(dones):
                break
        
        env.close()
        return rewards.tolist()
    
    def _evaluate_coevolution_batch(self,
                                   prey_genomes: List,
                                   predator_genomes: List,
                                   generation: int,
                                   stage_config: Optional[Dict] = None) -> Tuple[List[float], List[float]]:
        """
        Evaluate co-evolution batch - REUSE ONE ARENA FOR ALL PAIRS
        Returns: (prey_fitnesses, predator_fitnesses)
        """
        if len(prey_genomes) != len(predator_genomes):
            raise ValueError(f"Mismatched populations: "
                           f"{len(prey_genomes)} prey vs {len(predator_genomes)} predators")
        
        # Import multi-agent arena
        from environments.arena_multi import MultiAgentArena
        
        # Get deterministic seed for the batch
        batch_seed = self.seed_manager.get_seed(
            f"coevolution_batch_gen{generation}",
            offset=0  # NO TIME OFFSET
        )
        
        # Create ONE arena and reuse for all pairs
        arena = MultiAgentArena(
            batch_size=len(prey_genomes),  # One environment per pair
            num_prey_per_env=self.config.num_prey,
            num_predators_per_env=self.config.num_predators,
            config={'max_steps': self.config.max_steps},
            seed=batch_seed
        )
        
        prey_state, pred_state = arena.reset()
        prey_rewards = np.zeros(len(prey_genomes), dtype=np.float32)
        predator_rewards = np.zeros(len(prey_genomes), dtype=np.float32)

        # Reset episode tracking for all genomes
        for genome in prey_genomes + predator_genomes:
            brain = getattr(genome, "brain", None)
            if brain is not None and hasattr(brain, "reset_episode_tracking"):
                brain.reset_episode_tracking()

        for step in range(self.config.max_steps):
            # Get actions from all pairs at once
            prey_actions_list = []
            pred_actions_list = []
            
            for i, (prey_genome, pred_genome) in enumerate(zip(prey_genomes, predator_genomes)):
                prey_actions = prey_genome.act_batch(prey_state[i:i+1])
                pred_actions = pred_genome.act_batch(pred_state[i:i+1])
                prey_actions_list.append(prey_actions)
                pred_actions_list.append(pred_actions)
            
            prey_actions_batch = np.concatenate(prey_actions_list, axis=0)
            pred_actions_batch = np.concatenate(pred_actions_list, axis=0)

            (prey_state, pred_state), r_prey, r_pred, info = arena.step(
                prey_actions_batch, pred_actions_batch
            )

            # Update plasticity for all genomes
            for i, (prey_genome, pred_genome) in enumerate(zip(prey_genomes, predator_genomes)):
                # Update plasticity with reward signal
                brain = getattr(prey_genome, "brain", None)
                if brain is not None and hasattr(brain, "update_plasticity"):
                    brain.update_plasticity(reward_signal=float(r_prey[i].mean().item()))
                brain = getattr(pred_genome, "brain", None)
                if brain is not None and hasattr(brain, "update_plasticity"):
                    brain.update_plasticity(reward_signal=float(r_pred[i].mean().item()))

                prey_rewards[i] += r_prey[i].mean().item()
                predator_rewards[i] += r_pred[i].mean().item()

            if np.any(info['env_done']):
                break
        
        arena.close()
        
        # Store metrics in genomes for later retrieval
        for i, (prey_genome, pred_genome) in enumerate(zip(prey_genomes, predator_genomes)):
            if hasattr(prey_genome, 'last_eval_metrics'):
                prey_genome.last_eval_metrics = {
                    'fitness': prey_rewards[i],
                    'energy_cost': 0.0,  # TODO: compute from arena info
                    'learning_speed': 0.0,  # TODO: compute from plasticity diagnostics
                    'stability': 0.0,  # TODO: compute from reward variance
                    'task_success': prey_rewards[i] > 0,
                    'episode_return': prey_rewards[i],
                    'complexity_penalty': 0.0,
                    'novelty': 0.0,
                    'seed': batch_seed,
                    'stage': 'coevolution',
                    'opponent_id': getattr(pred_genome, 'genome_id', None)
                }
            if hasattr(pred_genome, 'last_eval_metrics'):
                pred_genome.last_eval_metrics = {
                    'fitness': predator_rewards[i],
                    'energy_cost': 0.0,  # TODO: compute from arena info
                    'learning_speed': 0.0,  # TODO: compute from plasticity diagnostics
                    'stability': 0.0,  # TODO: compute from reward variance
                    'task_success': predator_rewards[i] > 0,
                    'episode_return': predator_rewards[i],
                    'complexity_penalty': 0.0,
                    'novelty': 0.0,
                    'seed': batch_seed,
                    'stage': 'coevolution',
                    'opponent_id': getattr(prey_genome, 'genome_id', None)
                }

        return prey_rewards.tolist(), predator_rewards.tolist()
