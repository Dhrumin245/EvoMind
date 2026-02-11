"""
Asynchronous vectorized environment with parallel stepping
"""
import numpy as np
import asyncio
import concurrent.futures
from typing import List, Tuple, Optional
import threading
from queue import Queue
import time


class AsyncVectorizedArena:
    """
    Asynchronous vectorized environment that steps environments in parallel.
    Uses asyncio and thread pools for concurrent execution.
    """
    
    def __init__(self, 
                 num_envs: int = 100, 
                 max_steps: int = 1000,
                 seed: Optional[int] = None,
                 num_workers: Optional[int] = None):
        """
        Args:
            num_envs: Number of parallel environments
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
            num_workers: Number of worker threads/processes (None = auto)
        """
        self.num_envs = num_envs
        self.max_steps = max_steps
        
        # Set random seed
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        # Determine number of workers
        self.num_workers = num_workers or min(num_envs, 4)  # Reasonable default
        
        # Split environments among workers
        self.envs_per_worker = num_envs // self.num_workers
        self.worker_allocations = []
        
        start_idx = 0
        for i in range(self.num_workers):
            end_idx = start_idx + self.envs_per_worker
            if i == self.num_workers - 1:  # Last worker gets remaining
                end_idx = num_envs
            self.worker_allocations.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Initialize environments
        self.reset()
        
        # Thread pool for async stepping
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix='env_worker_'
        )
        
        # For async operations
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        print(f"AsyncVectorizedArena initialized: {num_envs} envs, {self.num_workers} workers")
    
    def _init_single_env(self, env_idx: int):
        """Initialize a single environment's state"""
        # Generate deterministic seed for this environment
        if self.seed is not None:
            rng = np.random.RandomState(self.seed + env_idx)
        else:
            rng = np.random.RandomState()
        
        agent_x = 400.0  # SCREEN_WIDTH / 2
        agent_y = 300.0  # SCREEN_HEIGHT / 2
        agent_dir = 0.0
        agent_speed = 2.0
        
        food_x = rng.uniform(50, 750)  # SCREEN_WIDTH - 50
        food_y = rng.uniform(50, 550)  # SCREEN_HEIGHT - 50
        
        return {
            'agent_x': agent_x,
            'agent_y': agent_y,
            'agent_dir': agent_dir,
            'agent_speed': agent_speed,
            'food_x': food_x,
            'food_y': food_y,
            'steps': 0,
            'done': False,
            'seed': self.seed + env_idx if self.seed is not None else None
        }
    
    def reset(self) -> np.ndarray:
        """Reset all environments"""
        self.envs = [self._init_single_env(i) for i in range(self.num_envs)]
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get current state of all environments"""
        states = []
        for env in self.envs:
            if env['done']:
                # Return zero state for done environments
                states.append(np.zeros(6, dtype=np.float32))
                continue
            
            # Calculate state components
            dx = env['food_x'] - env['agent_x']
            dy = env['food_y'] - env['agent_y']
            
            dist_food = np.sqrt(dx*dx + dy*dy) / np.sqrt(800**2 + 600**2)
            
            angle_to_food = np.arctan2(dy, dx)
            angle_food = ((angle_to_food - env['agent_dir'] + np.pi) % (2*np.pi) - np.pi) / np.pi
            
            # Wall distances
            left = env['agent_x']
            right = 800 - env['agent_x']
            top = env['agent_y']
            bottom = 600 - env['agent_y']
            wall_dist = min(left, right, top, bottom) / 400.0  # SCREEN_WIDTH/2
            
            # Speed normalized
            speed_norm = env['agent_speed'] / 2.0
            
            states.append([
                float(dist_food),
                float(angle_food),
                float(wall_dist),
                0.0,  # reserved
                float(speed_norm),
                1.0   # bias
            ])
        
        return np.array(states, dtype=np.float32)
    
    async def step_async(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step all environments asynchronously
        
        Args:
            actions: Array of shape (num_envs,) with action indices
            
        Returns:
            tuple of (states, rewards, dones)
        """
        # Split work among workers
        tasks = []
        for worker_idx, (start, end) in enumerate(self.worker_allocations):
            worker_actions = actions[start:end]
            worker_envs = self.envs[start:end]
            task = asyncio.ensure_future(
                self._step_batch_async(worker_envs, worker_actions, worker_idx)
            )
            tasks.append(task)
        
        # Wait for all workers to complete
        worker_results = await asyncio.gather(*tasks)
        
        # Combine results
        all_states = []
        all_rewards = []
        all_dones = []

        for states, rewards, dones, updated_envs in worker_results:
            all_states.append(states)
            all_rewards.append(rewards)
            all_dones.append(dones)

        # Update main envs list (workers modified slices)
        self.envs = sum([result[3] for result in worker_results], [])
        
        return (
            np.concatenate(all_states, axis=0),
            np.concatenate(all_rewards),
            np.concatenate(all_dones)
        )
    
    async def _step_batch_async(self, envs_batch: List[dict], actions_batch: np.ndarray, worker_id: int):
        """Step a batch of environments asynchronously"""
        # Run CPU-intensive work in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._step_batch_sync,
            envs_batch.copy(),  # Copy to avoid race conditions
            actions_batch,
            worker_id
        )
        return result
    
    def _step_batch_sync(self, envs_batch: List[dict], actions_batch: np.ndarray, worker_id: int):
        """Synchronous stepping for a batch of environments"""
        states_batch = []
        rewards_batch = []
        dones_batch = []
        updated_envs = []
        
        for env, action in zip(envs_batch, actions_batch):
            if env['done']:
                # Environment is already done
                states_batch.append(np.zeros(6, dtype=np.float32))
                rewards_batch.append(0.0)
                dones_batch.append(True)
                updated_envs.append(env)
                continue
            
            # Create local RNG for this environment
            if env['seed'] is not None:
                rng = np.random.RandomState(env['seed'] + env['steps'])
            else:
                rng = np.random.RandomState()
            
            # Apply action
            reward = -0.01  # step penalty
            
            if action == 1:
                env['agent_dir'] -= 0.2
            elif action == 2:
                env['agent_dir'] += 0.2
            elif action == 0:
                env['agent_x'] += np.cos(env['agent_dir']) * env['agent_speed']
                env['agent_y'] += np.sin(env['agent_dir']) * env['agent_speed']
            
            # Wall collision
            wall_hit = False
            if env['agent_x'] < 0 or env['agent_x'] > 800:
                env['agent_x'] = np.clip(env['agent_x'], 0, 800)
                env['agent_dir'] = np.pi - env['agent_dir']
                wall_hit = True
            
            if env['agent_y'] < 0 or env['agent_y'] > 600:
                env['agent_y'] = np.clip(env['agent_y'], 0, 600)
                env['agent_dir'] = -env['agent_dir']
                wall_hit = True
            
            if wall_hit:
                reward -= 0.5
                env['agent_speed'] = 0.5
            else:
                env['agent_speed'] = min(2.0, env['agent_speed'] + 0.1)
            
            # Food collision
            dx = env['agent_x'] - env['food_x']
            dy = env['agent_y'] - env['food_y']
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 18.0:
                reward += 10.0
                # Respawn food with deterministic RNG
                env['food_x'] = rng.uniform(50, 750)
                env['food_y'] = rng.uniform(50, 550)
            
            # Update steps
            env['steps'] += 1
            if env['steps'] >= self.max_steps:
                env['done'] = True
            
            # Calculate state
            state = self._get_env_state(env)
            
            states_batch.append(state)
            rewards_batch.append(reward)
            dones_batch.append(env['done'])
            updated_envs.append(env)
        
        return (
            np.array(states_batch, dtype=np.float32),
            np.array(rewards_batch, dtype=np.float32),
            np.array(dones_batch, dtype=bool),
            updated_envs
        )
    
    def _get_env_state(self, env: dict) -> np.ndarray:
        """Get state for a single environment"""
        if env['done']:
            return np.zeros(6, dtype=np.float32)
        
        dx = env['food_x'] - env['agent_x']
        dy = env['food_y'] - env['agent_y']
        
        dist_food = np.sqrt(dx*dx + dy*dy) / np.sqrt(800**2 + 600**2)
        
        angle_to_food = np.arctan2(dy, dx)
        angle_food = ((angle_to_food - env['agent_dir'] + np.pi) % (2*np.pi) - np.pi) / np.pi
        
        # Wall distances
        left = env['agent_x']
        right = 800 - env['agent_x']
        top = env['agent_y']
        bottom = 600 - env['agent_y']
        wall_dist = min(left, right, top, bottom) / 400.0
        
        speed_norm = env['agent_speed'] / 2.0
        
        return np.array([
            float(dist_food),
            float(angle_food),
            float(wall_dist),
            0.0,
            float(speed_norm),
            1.0
        ], dtype=np.float32)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Synchronous wrapper for async step
        """
        return self.loop.run_until_complete(self.step_async(actions))
    
    def close(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self.loop.close()


class AsyncBatchEvaluator:
    """
    Evaluates multiple genomes asynchronously using multiple environments
    """
    
    def __init__(self, 
                 num_genomes: int,
                 envs_per_genome: int = 64,
                 seed: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 use_gpu: bool = False):
        """
        Args:
            num_genomes: Number of genomes to evaluate in parallel
            envs_per_genome: Number of environments per genome
            seed: Base random seed
            num_workers: Number of parallel workers
            use_gpu: Whether to use GPU for neural network inference
        """
        self.num_genomes = num_genomes
        self.envs_per_genome = envs_per_genome
        self.seed = seed
        self.use_gpu = use_gpu
        
        # Create async environments for each genome
        self.envs = []
        for i in range(num_genomes):
            env_seed = seed + i * 1000 if seed is not None else None
            env = AsyncVectorizedArena(
                num_envs=envs_per_genome,
                seed=env_seed,
                num_workers=num_workers
            )
            self.envs.append(env)
        
        # For parallel execution
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(num_genomes, 8)
        )
    
    async def evaluate_genome_async(self, 
                                   genome, 
                                   env_idx: int, 
                                   max_steps: int = 1000) -> float:
        """
        Evaluate a single genome asynchronously
        """
        env = self.envs[env_idx]
        states = env.reset()
        
        total_reward = 0.0
        step = 0
        
        while step < max_steps:
            # Get actions for all environments
            if self.use_gpu and hasattr(genome, 'act_batch_gpu'):
                actions = genome.act_batch_gpu(states)
            else:
                actions = genome.act_batch(states)
            
            # Async step
            states, rewards, dones = await env.step_async(actions)
            
            total_reward += np.mean(rewards)
            step += 1
            
            # Check if all environments are done
            if np.all(dones):
                break
        
        return float(total_reward)
    
    async def evaluate_batch_async(self, genomes: List) -> List[float]:
        """
        Evaluate multiple genomes asynchronously
        """
        tasks = []
        for i, genome in enumerate(genomes):
            task = asyncio.ensure_future(
                self.evaluate_genome_async(genome, i)
            )
            tasks.append(task)
        
        # Run all evaluations in parallel
        if tasks:
            await asyncio.wait(tasks, timeout=30)
        results = [task.result() for task in tasks]
        return results
    
    def evaluate_batch_sync(self, genomes: List) -> List[float]:
        """
        Synchronous wrapper for batch evaluation
        """
        # Create event loop if not exists
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.evaluate_batch_async(genomes))
    
    def close(self):
        """Cleanup resources"""
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)