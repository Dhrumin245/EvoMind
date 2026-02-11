"""
⚠️⚠️⚠️ DEPRECATED FILE - DO NOT USE ⚠️⚠️⚠️

This file contains SLOW implementations that use Python loops.

✅ CORRECT: Use DeterministicVectorizedArena instead
   Import with: from deterministic_env import DeterministicVectorizedArena

❌ WRONG: VectorEnv, AsyncVectorEnv, ArenaEnvCore
   These are significantly slower than NumPy vectorization.

All classes in this file will emit DeprecationWarnings.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from environments.arena_core_env import ArenaEnvCore
import warnings


class VectorEnv:
    """
    ⚠️ DEPRECATED: VectorEnv is SLOWER than DeterministicVectorizedArena.
    
    DO NOT USE THIS CLASS. Use DeterministicVectorizedArena instead.
    VectorEnv uses Python loops and is significantly slower than NumPy vectorization.
    
    Use: from deterministic_env import DeterministicVectorizedArena
    """
    def __init__(self, num_envs: int, stage: Any, config: Optional[Dict[str, Any]] = None):
        """
        ⚠️ DEPRECATED: Use DeterministicVectorizedArena instead.
        
        Args:
            num_envs: Number of parallel environments
            stage: Curriculum stage
            config: Optional environment configuration overrides
        """
        warnings.warn(
            "VectorEnv is DEPRECATED and SLOW. Use DeterministicVectorizedArena instead. "
            "Import with: from deterministic_env import DeterministicVectorizedArena",
            DeprecationWarning,
            stacklevel=2
        )
        assert num_envs > 0, "Number of environments must be positive"
        self.num_envs = num_envs
        
        # Store config for resetting
        self.stage = stage
        self.config = config or {}
        
        # Initialize all environments
        self.envs = []
        for _ in range(num_envs):
            if config:
                env = ArenaEnvCore(stage=stage, config=config)
            else:
                env = ArenaEnvCore(stage=stage)
            self.envs.append(env)
        
        # Track which environments are done for auto-reset
        self.dones = np.zeros(num_envs, dtype=bool)
        
        # Get state shape from first environment for array pre-allocation
        sample_state = self.envs[0].reset()
        self.state_shape = sample_state.shape if hasattr(sample_state, 'shape') else ()
        self.state_dtype = sample_state.dtype if hasattr(sample_state, 'dtype') else np.float32
        
        # Store initial states
        self.current_states = self._reset_all()
        
        # Performance tracking
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)
        self.total_steps = 0
        
    def _reset_all(self) -> np.ndarray:
        """Reset all environments and return states as array"""
        states = []
        for env in self.envs:
            state = env.reset()
            states.append(state)
        return np.array(states, dtype=self.state_dtype)
    
    def reset(self, env_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset environments. If env_indices is None, reset all environments.
        
        Args:
            env_indices: Optional array of environment indices to reset
            
        Returns:
            Array of initial states
        """
        if env_indices is None:
            # Reset all environments
            self.current_states = self._reset_all()
            self.dones[:] = False
            self.episode_lengths[:] = 0
            self.episode_rewards[:] = 0.0
        else:
            # Reset only specified environments
            for idx in env_indices:
                if 0 <= idx < self.num_envs:
                    self.current_states[idx] = self.envs[idx].reset()
                    self.dones[idx] = False
                    self.episode_lengths[idx] = 0
                    self.episode_rewards[idx] = 0.0
        
        return self.current_states.copy()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Execute actions in all environments.
        
        Args:
            actions: Array of actions of shape (num_envs, action_dim) or (num_envs,)
                    Must be integer actions for discrete environments
            
        Returns:
            next_states: Array of shape (num_envs, *state_shape)
            rewards: Array of shape (num_envs,)
            dones: Boolean array of shape (num_envs,)
            infos: Dictionary with additional info arrays
        """
        # Input validation
        assert len(actions) == self.num_envs, \
            f"Number of actions ({len(actions)}) must match number of environments ({self.num_envs})"
        
        assert isinstance(actions, np.ndarray), "Actions must be a numpy array"
        
        # Pre-allocate arrays for efficiency
        next_states = np.zeros((self.num_envs,) + self.state_shape, dtype=self.state_dtype)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        
        # Track which environments need reset
        reset_indices = []
        
        # Process each environment
        for i in range(self.num_envs):
            if self.dones[i]:
                # Environment was already done, keep it in done state
                next_states[i] = self.current_states[i]
                rewards[i] = 0.0
                dones[i] = True
                continue
            
            # Execute step
            obs, reward, done, _ = self.envs[i].step(actions[i])

            # Update tracking
            self.episode_lengths[i] += 1
            self.episode_rewards[i] += reward
            self.total_steps += 1

            # Store results
            next_states[i] = obs
            rewards[i] = reward
            dones[i] = done
            
            # Check if environment is done
            if done:
                self.dones[i] = True
                reset_indices.append(i)
        
        # Auto-reset done environments immediately
        if reset_indices:
            reset_indices_arr = np.array(reset_indices, dtype=np.int32)
            reset_states = self.reset(reset_indices_arr)
            # Update next_states for reset environments
            for idx in reset_indices:
                next_states[idx] = reset_states[idx]
                dones[idx] = False  # Reset flag for next step
        
        # Update current states
        self.current_states = next_states.copy()
        
        # Compile additional info
        infos = {
            'episode_lengths': self.episode_lengths.copy(),
            'episode_rewards': self.episode_rewards.copy(),
            'total_steps': self.total_steps
        }
        
        return next_states, rewards, dones, infos
    
    def step_batch(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """Alias for step() for consistency with other frameworks"""
        return self.step(actions)
    
    def render(self, env_index: int = 0, mode: str = 'human'):
        """
        Render a specific environment.
        
        Args:
            env_index: Index of environment to render
            mode: Rendering mode
        """
        assert 0 <= env_index < self.num_envs, f"Invalid environment index: {env_index}"
        return self.envs[env_index].render(mode=mode)
    
    def get_observations(self) -> np.ndarray:
        """Get current observations from all environments"""
        return self.current_states.copy()
    
    def get_done_envs(self) -> np.ndarray:
        """Get indices of environments that are done"""
        return np.where(self.dones)[0]
    
    def get_active_envs(self) -> np.ndarray:
        """Get indices of environments that are still active"""
        return np.where(~self.dones)[0]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get environment statistics"""
        active_envs = self.get_active_envs()
        
        if len(active_envs) > 0:
            active_rewards = self.episode_rewards[active_envs]
            active_lengths = self.episode_lengths[active_envs]
        else:
            active_rewards = np.array([0.0])
            active_lengths = np.array([0])
        
        done_envs = self.get_done_envs()
        if len(done_envs) > 0:
            done_rewards = self.episode_rewards[done_envs]
            done_lengths = self.episode_lengths[done_envs]
        else:
            done_rewards = np.array([0.0])
            done_lengths = np.array([0])
        
        return {
            'total_steps': self.total_steps,
            'active_environments': len(active_envs),
            'done_environments': len(done_envs),
            'mean_active_reward': float(np.mean(active_rewards)),
            'mean_done_reward': float(np.mean(done_rewards)) if len(done_envs) > 0 else 0.0,
            'mean_active_length': float(np.mean(active_lengths)),
            'mean_done_length': float(np.mean(done_lengths)) if len(done_envs) > 0 else 0.0,
            'max_reward': float(np.max(self.episode_rewards)),
            'min_reward': float(np.min(self.episode_rewards)),
        }
    
    def close(self):
        """Close all environments"""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()

        # Clear references
        self.envs.clear()
    
    def __len__(self) -> int:
        return self.num_envs
    
    def __getitem__(self, index: int) -> ArenaEnvCore:
        """Get individual environment"""
        return self.envs[index]
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.close()


# ⚠️ DEPRECATED: DO NOT USE AsyncVectorEnv - Use DeterministicVectorizedArena instead
try:
    import multiprocessing as mp
    from concurrent.futures import ThreadPoolExecutor
    import threading

    class AsyncVectorEnv(VectorEnv):
        """
        ⚠️ DEPRECATED: AsyncVectorEnv is SLOWER than DeterministicVectorizedArena.
        
        DO NOT USE THIS CLASS. Use DeterministicVectorizedArena instead.
        Python GIL and thread overhead make this significantly slower than NumPy vectorization.
        
        Use: from deterministic_env import DeterministicVectorizedArena
        """
        def __init__(self, num_envs: int, stage: Any,
                    config: Optional[Dict[str, Any]] = None,
                    max_workers: Optional[int] = None):
            """
            ⚠️ DEPRECATED: Use DeterministicVectorizedArena instead.

            Args:
                num_envs: Number of environments
                stage: Curriculum stage
                config: Environment configuration
                max_workers: Maximum number of worker threads (default: num_envs)
            """
            warnings.warn(
                "AsyncVectorEnv is DEPRECATED and SLOW. Use DeterministicVectorizedArena instead. "
                "Import with: from deterministic_env import DeterministicVectorizedArena",
                DeprecationWarning,
                stacklevel=2
            )
            super().__init__(num_envs, stage, config)
            self.max_workers = max_workers or num_envs
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self.lock = threading.Lock()

        def step_async(self, actions: np.ndarray):
            """
            Execute actions asynchronously.

            Returns:
                Future object that will contain step results
            """
            assert len(actions) == self.num_envs

            # Submit all step operations to thread pool
            futures = []
            for i in range(self.num_envs):
                if not self.dones[i]:
                    future = self.executor.submit(self.envs[i].step, actions[i])
                    futures.append((i, future))
                else:
                    # Already done environments
                    futures.append((i, None))

            return futures

        def step_wait(self, futures):
            """
            Wait for async step results.

            Returns:
                Same as VectorEnv.step()
            """
            next_states = np.zeros((self.num_envs,) + self.state_shape, dtype=self.state_dtype)
            rewards = np.zeros(self.num_envs, dtype=np.float32)
            dones = np.zeros(self.num_envs, dtype=bool)

            reset_indices = []

            for i, future in futures:
                if future is None:
                    # Environment was already done
                    next_states[i] = self.current_states[i]
                    rewards[i] = 0.0
                    dones[i] = True
                    continue

            # Get result from future
                try:
                    obs, reward, done, _ = future.result(timeout=30)
                except TimeoutError:
                    print("[ERROR] Worker hung, skipping genome")
                    continue

                # Update tracking (thread-safe with lock)
                with self.lock:
                    self.episode_lengths[i] += 1
                    self.episode_rewards[i] += reward
                    self.total_steps += 1

                # Store results
                next_states[i] = obs
                rewards[i] = reward
                dones[i] = done

                if done:
                    self.dones[i] = True
                    reset_indices.append(i)

            # Auto-reset done environments
            if reset_indices:
                reset_states = self.reset(np.array(reset_indices))
                for idx in reset_indices:
                    next_states[idx] = reset_states[idx]
                    dones[idx] = False

            self.current_states = next_states.copy()

            infos = {
                'episode_lengths': self.episode_lengths.copy(),
                'episode_rewards': self.episode_rewards.copy(),
                'total_steps': self.total_steps
            }

            return next_states, rewards, dones, infos

        def close(self):
            """Close executor and environments"""
            self.executor.shutdown(wait=True)
            super().close()

except ImportError:
    # ThreadPoolExecutor not available
    pass
