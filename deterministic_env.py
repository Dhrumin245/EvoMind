"""
Deterministic environment implementations with seed control
"""
import numpy as np
import random
from typing import Optional, Dict, Any, Tuple, List
import torch
import matplotlib.pyplot as plt


class DeterministicVectorizedArena:
    """
    Fully deterministic vectorized environment with controlled random seeds
    """
    
    def __init__(self, 
                 num_envs: int = 100, 
                 max_steps: int = 1000,
                 seed: Optional[int] = None,
                 stage_config: Optional[Dict[str, Any]] = None,
                 enable_diagnostics: bool = False):
        """
        Args:
            num_envs: Number of parallel environments
            max_steps: Maximum steps per episode
            seed: Base random seed (deterministic if provided)
            stage_config: Curriculum stage configuration
            enable_diagnostics: Enable diagnostic logging (default: False for training)
        """
        self.n = num_envs
        self.max_steps = max_steps
        
        # Store seed for reproducibility
        self.base_seed = seed
        
        # Create independent RNGs for each environment
        if seed is not None:
            # Create deterministic seeds for each environment
            self.env_seeds = [seed + i * 1000 for i in range(num_envs)]
            self.rngs = [np.random.RandomState(s) for s in self.env_seeds]
            
            # Set global seeds for consistency
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        else:
            self.env_seeds = [None] * num_envs
            self.rngs = [np.random.RandomState() for _ in range(num_envs)]
        
        # Stage configuration
        self.config = stage_config or {
            'wall_penalty': -5.0,  # Increased wall penalty
            'food_reward': 100.0,  # Increased food reward
            'step_penalty': -0.1   # Increased step penalty
        }
        
        # Reset all environments
        self.reset()

        # Diagnostic 1: In-Lifetime Recovery Curve
        self.enable_diagnostics = enable_diagnostics
        if self.enable_diagnostics:
            self.diagnostic_logs = []
            self.shock_step = 300
            self.step_count = 0

        print(f"DeterministicVectorizedArena initialized with seed={seed}")
    
    def reset(self) -> np.ndarray:
        """Reset all environments deterministically"""
        self.agent_x = np.zeros(self.n, dtype=np.float32)
        self.agent_y = np.zeros(self.n, dtype=np.float32)
        self.agent_dir = np.zeros(self.n, dtype=np.float32)
        self.agent_speed = np.full(self.n, 2.0, dtype=np.float32)
        
        self.food_x = np.zeros(self.n, dtype=np.float32)
        self.food_y = np.zeros(self.n, dtype=np.float32)
        
        # Use each environment's RNG for deterministic initialization
        for i, rng in enumerate(self.rngs):
            self.agent_x[i] = 400.0  # SCREEN_WIDTH / 2
            self.agent_y[i] = 300.0  # SCREEN_HEIGHT / 2
             
            # Deterministic food placement
            self.food_x[i] = rng.uniform(50, 750)  # SCREEN_WIDTH - 50
            self.food_y[i] = rng.uniform(50, 550)  # SCREEN_HEIGHT - 50
        
        self.steps = np.zeros(self.n, dtype=np.int32)
        self.done = np.zeros(self.n, dtype=bool)
        
        return self.get_state()
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic step function
        """
        # Initialize rewards with step penalty
        rewards = np.full(self.n, self.config['step_penalty'], dtype=np.float32)
        
        # Update direction
        self.agent_dir += np.where(actions == 1, -0.2, 0.0)
        self.agent_dir += np.where(actions == 2, 0.2, 0.0)
        
        # Normalize direction
        self.agent_dir = ((self.agent_dir + np.pi) % (2 * np.pi)) - np.pi
        
        # Move forward
        moving = actions == 0
        cos_dir = np.cos(self.agent_dir)
        sin_dir = np.sin(self.agent_dir)
        
        self.agent_x[moving] += cos_dir[moving] * self.agent_speed[moving]
        self.agent_y[moving] += sin_dir[moving] * self.agent_speed[moving]
        
        # Wall collisions (deterministic)
        hit_left = self.agent_x < 0
        hit_right = self.agent_x > 800
        hit_top = self.agent_y < 0
        hit_bottom = self.agent_y > 600
        
        # Apply wall penalties deterministically
        wall_hit = hit_left | hit_right | hit_top | hit_bottom
        rewards[wall_hit] += self.config['wall_penalty']
        
        # Bounce physics (deterministic)
        self.agent_dir[hit_left | hit_right] = np.pi - self.agent_dir[hit_left | hit_right]
        self.agent_dir[hit_top | hit_bottom] = -self.agent_dir[hit_top | hit_bottom]
        
        # Clamp positions
        self.agent_x = np.clip(self.agent_x, 0, 800)
        self.agent_y = np.clip(self.agent_y, 0, 600)
        
        # Food collision (deterministic)
        dx = self.agent_x - self.food_x
        dy = self.agent_y - self.food_y
        dist = np.sqrt(dx * dx + dy * dy)
        
        food_eaten = dist < 18.0
        
        # Apply food rewards and respawn deterministically
        if np.any(food_eaten):
            rewards[food_eaten] += self.config['food_reward']
            
            # Deterministic respawn using each environment's RNG
            idx = np.where(food_eaten)[0]
            if len(idx) > 0:
                self.food_x[idx] = np.array([self.rngs[i].uniform(50, 750) for i in idx])
                self.food_y[idx] = np.array([self.rngs[i].uniform(50, 550) for i in idx])
        
        # Update steps
        self.steps += 1

        # Diagnostic 1: Downsampled logging for recovery curve
        if self.enable_diagnostics:
            if self.step_count % 5 == 0:  # downsample heavily
                self.diagnostic_logs.append({
                    "timestep": self.step_count,
                    "avg_reward": float(np.mean(rewards)),
                    "shock": self.step_count == self.shock_step
                })
            self.step_count += 1

        # Check done
        self.done = self.steps >= self.max_steps

        return self.get_state(), rewards, self.done.copy()
    
    def get_state(self) -> np.ndarray:
        """Get deterministic state"""
        # Food distance and angle
        dx_food = self.food_x - self.agent_x
        dy_food = self.food_y - self.agent_y
        
        dist_food = np.sqrt(dx_food * dx_food + dy_food * dy_food)
        angle_food = np.arctan2(dy_food, dx_food) - self.agent_dir
        
        # Normalize
        max_dist = float(np.sqrt(800**2 + 600**2))
        dist_food_norm = dist_food / max_dist
        angle_food_norm = ((angle_food + np.pi) % (2 * np.pi) - np.pi) / np.pi
        
        # Wall distance
        left = self.agent_x
        right = 800 - self.agent_x
        top = self.agent_y
        bottom = 600 - self.agent_y
        wall_dist = np.minimum.reduce([left, right, top, bottom])
        wall_dist_norm = wall_dist / 400.0  # SCREEN_WIDTH / 2
        
        # Speed
        speed_norm = self.agent_speed / 2.0
        
        # Stack states
        states = np.column_stack([
            dist_food_norm,
            angle_food_norm,
            wall_dist_norm,
            np.zeros(self.n),  # reserved
            speed_norm,
            np.ones(self.n),   # bias
            np.zeros(self.n),  # reserved2
            np.zeros(self.n)   # reserved3
        ])
        
        return states.astype(np.float32)
    
    def get_seed_info(self) -> Dict:
        """Get seed information for reproducibility"""
        return {
            'base_seed': self.base_seed,
            'env_seeds': self.env_seeds,
            'num_envs': self.n,
            'config': self.config
        }
    
    def save_checkpoint(self) -> Dict:
        """Save current state for deterministic replay"""
        return {
            'agent_x': self.agent_x.copy(),
            'agent_y': self.agent_y.copy(),
            'agent_dir': self.agent_dir.copy(),
            'agent_speed': self.agent_speed.copy(),
            'food_x': self.food_x.copy(),
            'food_y': self.food_y.copy(),
            'steps': self.steps.copy(),
            'done': self.done.copy(),
            'seed_info': self.get_seed_info()
        }
    
    def load_checkpoint(self, checkpoint: Dict):
        """Load state for deterministic replay"""
        self.agent_x = checkpoint['agent_x'].copy()
        self.agent_y = checkpoint['agent_y'].copy()
        self.agent_dir = checkpoint['agent_dir'].copy()
        self.agent_speed = checkpoint['agent_speed'].copy()
        self.food_x = checkpoint['food_x'].copy()
        self.food_y = checkpoint['food_y'].copy()
        self.steps = checkpoint['steps'].copy()
        self.done = checkpoint['done'].copy()

    def close(self):
        """Close the environment (no-op for deterministic env)"""
        pass

    def plot_recovery_curve(self, filename="recovery_curve.png"):
        """Plot the in-lifetime recovery curve: average reward vs timestep"""
        # Aggregate rewards per timestep across all agents
        timestep_rewards = {}
        for log in self.diagnostic_logs:
            t = log["timestep"]
            r = log["avg_reward"]
            if t not in timestep_rewards:
                timestep_rewards[t] = []
            timestep_rewards[t].append(r)

        # Average rewards per timestep
        timesteps = sorted(timestep_rewards.keys())
        avg_rewards = [np.mean(timestep_rewards[t]) for t in timesteps]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, avg_rewards, label="Average Reward")
        plt.axvline(x=self.shock_step, color='r', linestyle='--', label=f"Shock at t={self.shock_step}")
        plt.xlabel("Timestep")
        plt.ylabel("Average Reward")
        plt.title("DIAGNOSTIC 1: In-Lifetime Recovery Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()


class DeterministicSeedManager:
    """
    Manages deterministic seeds for the entire training process
    """
    
    def __init__(self, base_seed: int = 42):
        """
        Args:
            base_seed: Base random seed for entire experiment
        """
        self.base_seed = base_seed
        self.seed_registry = {}
        
        # Set global seeds
        self._set_global_seeds()
    
    def _set_global_seeds(self):
        """Set all random seeds for reproducibility"""
        np.random.seed(self.base_seed)
        random.seed(self.base_seed)
        
        # PyTorch seeds
        import torch
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.base_seed)
            torch.cuda.manual_seed_all(self.base_seed)
            
            # Additional CUDA settings for determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def register_seed(self, component: str, seed: int):
        """Register a seed for a component"""
        self.seed_registry[component] = seed
    
    def get_seed(self, component: str, offset: int = 0) -> int:
        """Get deterministic seed for a component"""
        if component in self.seed_registry:
            return self.seed_registry[component] + offset
        else:
            # Generate new seed based on base seed and component name
            component_hash = hash(component) % 1000000
            seed = self.base_seed + component_hash + offset
            self.register_seed(component, seed)
            return seed
    
    def get_env_seeds(self, num_envs: int, component: str = "env") -> List[int]:
        """Get deterministic seeds for multiple environments"""
        base_env_seed = self.get_seed(component)
        return [base_env_seed + i * 1000 for i in range(num_envs)]
    
    def get_genome_seeds(self, num_genomes: int, component: str = "genome") -> List[int]:
        """Get deterministic seeds for multiple genomes"""
        base_genome_seed = self.get_seed(component)
        return [base_genome_seed + i * 100 for i in range(num_genomes)]
    
    def get_seed_registry(self) -> Dict:
        """Get complete seed registry"""
        return self.seed_registry.copy()
    
    def save_seeds(self, filename: str):
        """Save seed registry to file"""
        import json
        registry = {
            'base_seed': self.base_seed,
            'seed_registry': self.seed_registry
        }
        with open(filename, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def load_seeds(self, filename: str):
        """Load seed registry from file"""
        import json
        with open(filename, 'r') as f:
            registry = json.load(f)
        
        self.base_seed = registry['base_seed']
        self.seed_registry = registry['seed_registry']
        
        # Reset global seeds
        self._set_global_seeds()