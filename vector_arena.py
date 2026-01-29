import numpy as np
from typing import Tuple, Dict, Any, Optional, List, Union, cast
from dataclasses import dataclass, field
import warnings

# Import curriculum for config
from curriculum import get_stage_config, CurriculumStage


@dataclass
class EntityState:
    """State of a single entity (agent, predator, food, etc.)"""
    pos_x: np.ndarray  # [n_entities] or scalar
    pos_y: np.ndarray  # [n_entities] or scalar
    vel_x: np.ndarray  # [n_entities] or scalar
    vel_y: np.ndarray  # [n_entities] or scalar
    heading: np.ndarray  # [n_entities] or scalar
    active: np.ndarray  # [n_entities] or bool
    radius: float
    entity_type: str
    
    @classmethod
    def create_batch(cls, n: int, radius: float, entity_type: str,
                    x_range: Tuple[float, float], y_range: Tuple[float, float],
                    rng: Union['DeterministicRNG', np.random.Generator]) -> 'EntityState':
        """Create a batch of entities"""
        pos_x = rng.uniform(x_range[0], x_range[1], n)
        pos_y = rng.uniform(y_range[0], y_range[1], n)
        return cls(
            pos_x=pos_x.astype(np.float32),
            pos_y=pos_y.astype(np.float32),
            vel_x=np.zeros(n, dtype=np.float32),
            vel_y=np.zeros(n, dtype=np.float32),
            heading=rng.uniform(0, 2*np.pi, n).astype(np.float32),
            active=np.ones(n, dtype=bool),
            radius=radius,
            entity_type=entity_type
        )


class DeterministicRNG:
    """Deterministic random number generator with per-environment seeds"""
    
    def __init__(self, num_envs: int, base_seed: int):
        self.num_envs = num_envs
        self.base_seed = base_seed
        self.generators = []
        
        # Create independent generators for each environment
        for i in range(num_envs):
            seed = base_seed + i * 1000
            self.generators.append(np.random.Generator(np.random.PCG64(seed)))
    
    def uniform(self, low: float, high: float, size: Optional[Union[int, Tuple[int, ...]]] = None,
                env_idx: Optional[int] = None) -> np.ndarray:
        """Generate uniform random numbers"""
        if env_idx is not None:
            return self.generators[env_idx].uniform(low, high, size)

        # Normalize size to tuple
        if size is None:
            size = (self.num_envs,)
        elif isinstance(size, int):
            size = (size,)

        # For batch operations, generate for all environments
        result = np.zeros(size, dtype=np.float32)
        if len(size) == 1:
            # 1D array: one value per environment
            for i in range(self.num_envs):
                result[i] = self.generators[i].uniform(low, high)
        else:
            # Multi-dimensional: generate independently per environment
            for i in range(self.num_envs):
                slice_idx = (i,) + (slice(None),) * (len(size) - 1)
                result[slice_idx] = self.generators[i].uniform(low, high, size[1:])

        return result
    
    def random(self, size: Optional[Tuple[int, ...]] = None, 
               env_idx: Optional[int] = None) -> np.ndarray:
        """Generate random numbers in [0, 1)"""
        return self.uniform(0.0, 1.0, size, env_idx)
    
    def choice(self, a: int, size: Optional[Union[int, Tuple[int, ...]]] = None,
               replace: bool = True, p: Optional[np.ndarray] = None,
               env_idx: Optional[int] = None) -> np.ndarray:
        """Random choice"""
        if env_idx is not None:
            return self.generators[env_idx].choice(a, size, replace, p)

        # Normalize size to tuple
        if size is None:
            size = (self.num_envs,)
        elif isinstance(size, int):
            size = (size,)

        # Batch operation
        result = np.zeros(size, dtype=int)
        if len(size) == 1:
            for i in range(self.num_envs):
                result[i] = self.generators[i].choice(a, size=None, replace=replace, p=p)
        else:
            for i in range(self.num_envs):
                slice_idx = (i,) + (slice(None),) * (len(size) - 1)
                result[slice_idx] = self.generators[i].choice(a, size[1:], replace, p)

        return result


class VectorizedArena:
    """
    Fully vectorized multi-agent environment with curriculum support,
    deterministic execution, and proper done state handling.
    """
    
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    BOUNDARY_MARGIN = 20
    
    def __init__(self, 
                 num_envs: int = 100,
                 stage: CurriculumStage = CurriculumStage.FORAGING,
                 config: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None,
                 deterministic: bool = True):
        """
        Args:
            num_envs: Number of parallel environments
            stage: Curriculum stage for difficulty
            config: Environment configuration (overrides stage defaults)
            seed: Random seed for deterministic mode
            deterministic: Whether to use deterministic RNG
        """
        self.num_envs = num_envs
        self.stage = stage
        self.deterministic = deterministic
        
        # Load configuration from curriculum
        self.config = config or get_stage_config(stage)
        
        # Set up RNG
        if deterministic:
            if seed is None:
                seed = 42  # Default deterministic seed
            self.rng = DeterministicRNG(num_envs, seed)
        else:
            # Non-deterministic mode - use global numpy RNG
            self.rng = np.random.default_rng(seed)
        
        # Extract config values
        self.max_steps = self.config.get('max_steps', 1000)
        self.food_count = self.config.get('food_count', 10)
        self.predator_count = self.config.get('predator_count', 0)
        self.has_predators = self.config.get('predator', False) and self.predator_count > 0
        
        # Rewards and penalties from config
        self.step_penalty = self.config.get('step_penalty', -0.01)
        self.wall_penalty = self.config.get('wall_penalty', -0.5)
        self.food_reward = self.config.get('food_reward', 10.0)
        self.predator_penalty = self.config.get('predator_penalty', -5.0)
        
        # Game parameters
        self.agent_radius = 10.0
        self.food_radius = 8.0
        self.predator_radius = 15.0
        self.agent_speed = 2.0
        self.predator_speed = self.config.get('predator_speed', 1.5)
        self.predator_vision = self.config.get('predator_vision', 150.0)
        
        # Food respawn strategy
        self.food_respawn_rate = self.config.get('food_respawn_rate', 0.05)
        self.food_competitive_placement = self.config.get('competitive_placement', True)
        
        # Initialize entity states
        self.reset()
        
        # Pre-allocate arrays for performance
        self._cos_cache = np.zeros(num_envs, dtype=np.float32)
        self._sin_cache = np.zeros(num_envs, dtype=np.float32)
        self._temp_distances = np.zeros((num_envs, self.food_count), dtype=np.float32)
        
        # Statistics tracking
        self.episode_stats = {
            'total_reward': np.zeros(num_envs, dtype=np.float32),
            'food_collected': np.zeros(num_envs, dtype=np.int32),
            'wall_collisions': np.zeros(num_envs, dtype=np.int32),
            'predator_escapes': np.zeros(num_envs, dtype=np.int32),
            'steps_survived': np.zeros(num_envs, dtype=np.int32)
        }
    
    def reset(self) -> np.ndarray:
        """Reset all environments and return initial states"""
        # Reset time steps
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.done = np.zeros(self.num_envs, dtype=bool)
        
        # Reset agents (1 agent per environment for now, but structure supports multi-agent)
        self.agents = EntityState.create_batch(
            n=self.num_envs,
            radius=self.agent_radius,
            entity_type='agent',
            x_range=(50, self.SCREEN_WIDTH - 50),
            y_range=(50, self.SCREEN_HEIGHT - 50),
            rng=self.rng if isinstance(self.rng, np.random.Generator) else self.rng
        )
        self.agents.vel_x[:] = 0.0
        self.agents.vel_y[:] = 0.0
        
        # Reset foods (multiple foods per environment)
        self.foods = []
        for env_idx in range(self.num_envs):
            if isinstance(self.rng, DeterministicRNG):
                rng = self.rng.generators[env_idx]
            else:
                rng = self.rng
            foods_env = EntityState.create_batch(
                n=self.food_count,
                radius=self.food_radius,
                entity_type='food',
                x_range=(30, self.SCREEN_WIDTH - 30),
                y_range=(30, self.SCREEN_HEIGHT - 30),
                rng=rng
            )
            self.foods.append(foods_env)

        # Reset predators if enabled
        self.predators = []
        if self.has_predators:
            for env_idx in range(self.num_envs):
                if isinstance(self.rng, DeterministicRNG):
                    rng = self.rng.generators[env_idx]
                else:
                    rng = self.rng
                predators_env = EntityState.create_batch(
                    n=self.predator_count,
                    radius=self.predator_radius,
                    entity_type='predator',
                    x_range=(50, self.SCREEN_WIDTH - 50),
                    y_range=(50, self.SCREEN_HEIGHT - 50),
                    rng=rng
                )
                predators_env.vel_x[:] = 0.0
                predators_env.vel_y[:] = 0.0
                self.predators.append(predators_env)
        
        # Reset statistics
        for key in self.episode_stats:
            self.episode_stats[key][:] = 0
        
        return self.get_state()
    
    def _get_active_mask(self) -> np.ndarray:
        """Get mask of active (not done) environments"""
        return ~self.done
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute actions in all active environments.
        
        Args:
            actions: Array of shape (num_envs, action_dim) for continuous actions
                    or (num_envs,) for discrete actions
        
        Returns:
            states: Array of observations
            rewards: Array of rewards per environment
            dones: Array of done flags
            infos: Dictionary with additional information
        """
        # Validate input
        assert len(actions) == self.num_envs, \
            f"Expected {self.num_envs} actions, got {len(actions)}"
        
        # Get active mask
        active = self._get_active_mask()
        rewards = np.full(self.num_envs, self.step_penalty, dtype=np.float32)

        # META-4 PRESSURE INJECTION
        # Initialize reward_sign if not exists
        if not hasattr(self, 'reward_sign'):
            self.reward_sign = np.ones(self.num_envs, dtype=np.float32)

        # META-4.1 Mid-Episode World Shift
        mid_episode_step = self.max_steps // 2
        world_shift_triggered = (self.steps == mid_episode_step) & active
        if np.any(world_shift_triggered):
            # Reduce food reward by 30%
            self.food_reward *= 0.7
            # Increase predator speed by 40%
            self.predator_speed *= 1.4

        # META-4.3 Rule Flip
        rule_flip_triggered = (self.steps == 300) & active
        if np.any(rule_flip_triggered):
            # Flip reward signs
            self.reward_sign[rule_flip_triggered] *= -1
        
        # Process only active environments
        if np.any(active):
            # Update agent states based on actions
            self._process_actions(actions, active)
            
            # Update predator AI
            if self.has_predators:
                self._update_predators(active)
            
            # Apply movement
            self._apply_movement(active)
            
            # Handle collisions
            self._handle_collisions(active, rewards)
            
            # Handle boundary conditions
            self._handle_boundaries(active, rewards)

        # Update total reward statistics
        self.episode_stats['total_reward'][active] += rewards[active]

        # Update steps and check termination
        self.steps[active] += 1
        step_limit_reached = self.steps >= self.max_steps
        
        # Mark done environments
        new_dones = step_limit_reached
        self.done = self.done | new_dones
        
        # Zero out rewards for already done environments
        rewards[self.done] = 0.0
        
        # Get observations
        states = self.get_state()
        
        # Compile info
        infos = self._compile_info(new_dones)
        
        return states, rewards, self.done.copy(), infos
    
    def _process_actions(self, actions: np.ndarray, active: np.ndarray):
        """Process agent actions"""
        # Handle both discrete and continuous actions
        if actions.ndim == 1:
            # Discrete actions: 0=forward, 1=left, 2=right
            turn_left = actions == 1
            turn_right = actions == 2
            move_forward = actions == 0
            
            # Update headings for active agents
            active_turn_left = active & turn_left
            active_turn_right = active & turn_right
            
            self.agents.heading[active_turn_left] -= 0.2
            self.agents.heading[active_turn_right] += 0.2
            
            # Normalize headings
            self.agents.heading[active] = np.mod(self.agents.heading[active], 2 * np.pi)
            
            # Set velocities for moving agents
            active_move = active & move_forward
            self._cos_cache[active_move] = np.cos(self.agents.heading[active_move])
            self._sin_cache[active_move] = np.sin(self.agents.heading[active_move])
            
            self.agents.vel_x[active_move] = self._cos_cache[active_move] * self.agent_speed
            self.agents.vel_y[active_move] = self._sin_cache[active_move] * self.agent_speed
            
            # Stop agents that aren't moving forward
            not_moving = active & ~move_forward
            self.agents.vel_x[not_moving] = 0.0
            self.agents.vel_y[not_moving] = 0.0
            
        else:
            # Continuous actions: [turn_rate, thrust]
            # For now, assume continuous actions are not yet implemented
            warnings.warn("Continuous actions not fully implemented, using default behavior")
    
    def _update_predators(self, active: np.ndarray):
        """Update predator positions with simple chasing AI"""
        if not self.has_predators:
            return
        
        for env_idx in range(self.num_envs):
            if not active[env_idx]:
                continue
            
            predators = self.predators[env_idx]
            agent_x = self.agents.pos_x[env_idx]
            agent_y = self.agents.pos_y[env_idx]
            
            # Simple chasing behavior
            for pred_idx in range(self.predator_count):
                if not predators.active[pred_idx]:
                    continue
                
                dx = agent_x - predators.pos_x[pred_idx]
                dy = agent_y - predators.pos_y[pred_idx]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < self.predator_vision:
                    # Chase the agent
                    direction = np.arctan2(dy, dx)
                    predators.vel_x[pred_idx] = np.cos(direction) * self.predator_speed
                    predators.vel_y[pred_idx] = np.sin(direction) * self.predator_speed
                else:
                    # Random wandering
                    if isinstance(self.rng, DeterministicRNG):
                        rng = self.rng.generators[env_idx]
                    else:
                        rng = self.rng

                    predators.vel_x[pred_idx] = rng.uniform(-1, 1) * self.predator_speed * 0.5
                    predators.vel_y[pred_idx] = rng.uniform(-1, 1) * self.predator_speed * 0.5
    
    def _apply_movement(self, active: np.ndarray):
        """Apply velocity to all entities"""
        # Update agents
        self.agents.pos_x[active] += self.agents.vel_x[active]
        self.agents.pos_y[active] += self.agents.vel_y[active]
        
        # Update predators
        if self.has_predators:
            for env_idx in range(self.num_envs):
                if not active[env_idx]:
                    continue
                
                predators = self.predators[env_idx]
                active_preds = predators.active
                predators.pos_x[active_preds] += predators.vel_x[active_preds]
                predators.pos_y[active_preds] += predators.vel_y[active_preds]
    
    def _handle_collisions(self, active: np.ndarray, rewards: np.ndarray):
        """Handle collisions between entities"""
        # Food collisions
        for env_idx in range(self.num_envs):
            if not active[env_idx]:
                continue
            
            agent_x = self.agents.pos_x[env_idx]
            agent_y = self.agents.pos_y[env_idx]
            foods = self.foods[env_idx]
            
            # Calculate distances to all foods
            dx = foods.pos_x - agent_x
            dy = foods.pos_y - agent_y
            distances = np.sqrt(dx*dx + dy*dy)
            
            # Check for collisions
            collision_radius = self.agent_radius + self.food_radius
            food_collisions = distances < collision_radius
            
            if np.any(food_collisions & foods.active):
                # Award reward
                rewards[env_idx] += self.food_reward
                self.episode_stats['food_collected'][env_idx] += np.sum(food_collisions)
                
                # Respawn collided foods
                for food_idx in np.where(food_collisions & foods.active)[0]:
                    self._respawn_food(env_idx, food_idx)
        
        # Predator collisions
        if self.has_predators:
            for env_idx in range(self.num_envs):
                if not active[env_idx]:
                    continue
                
                agent_x = self.agents.pos_x[env_idx]
                agent_y = self.agents.pos_y[env_idx]
                predators = self.predators[env_idx]
                
                # Calculate distances to all predators
                dx = predators.pos_x - agent_x
                dy = predators.pos_y - agent_y
                distances = np.sqrt(dx*dx + dy*dy)
                
                # Check for collisions
                collision_radius = self.agent_radius + self.predator_radius
                predator_collisions = distances < collision_radius
                
                if np.any(predator_collisions & predators.active):
                    # Apply penalty
                    rewards[env_idx] += self.predator_penalty
                    self.episode_stats['predator_escapes'][env_idx] += 1
                    
                    # Could deactivate agent or predator here
                    # For now, just apply penalty
    
    def _respawn_food(self, env_idx: int, food_idx: int):
        """Respawn food at a new location"""
        foods = self.foods[env_idx]
        
        if self.food_competitive_placement:
            # Competitive placement: avoid agents and other foods
            max_attempts = 10
            for attempt in range(max_attempts):
                # Get RNG for this environment
                if isinstance(self.rng, DeterministicRNG):
                    rng = self.rng.generators[env_idx]
                else:
                    rng = self.rng

                new_x = rng.uniform(30, self.SCREEN_WIDTH - 30)
                new_y = rng.uniform(30, self.SCREEN_HEIGHT - 30)
                
                # Check distance to agent
                dx = new_x - self.agents.pos_x[env_idx]
                dy = new_y - self.agents.pos_y[env_idx]
                dist_to_agent = np.sqrt(dx*dx + dy*dy)
                
                # Check distance to other foods
                other_foods = np.arange(self.food_count) != food_idx  # Get other foods
                if np.any(other_foods):
                    dx = new_x - foods.pos_x[other_foods]
                    dy = new_y - foods.pos_y[other_foods]
                    dist_to_foods = np.min(np.sqrt(dx*dx + dy*dy))
                else:
                    dist_to_foods = float('inf')
                
                # Place food if sufficiently distant
                if dist_to_agent > 100 and dist_to_foods > 50:
                    foods.pos_x[food_idx] = new_x
                    foods.pos_y[food_idx] = new_y
                    foods.active[food_idx] = True
                    break
                
                if attempt == max_attempts - 1:
                    # Last attempt: just place randomly
                    foods.pos_x[food_idx] = new_x
                    foods.pos_y[food_idx] = new_y
                    foods.active[food_idx] = True
        else:
            # Random placement
            if isinstance(self.rng, DeterministicRNG):
                rng = self.rng.generators[env_idx]
            else:
                rng = self.rng

            foods.pos_x[food_idx] = rng.uniform(30, self.SCREEN_WIDTH - 30)
            foods.pos_y[food_idx] = rng.uniform(30, self.SCREEN_HEIGHT - 30)
            foods.active[food_idx] = True
    
    def _handle_boundaries(self, active: np.ndarray, rewards: np.ndarray):
        """Handle wall collisions and boundary conditions"""
        # Agent boundaries
        hit_left = self.agents.pos_x < self.BOUNDARY_MARGIN
        hit_right = self.agents.pos_x > self.SCREEN_WIDTH - self.BOUNDARY_MARGIN
        hit_top = self.agents.pos_y < self.BOUNDARY_MARGIN
        hit_bottom = self.agents.pos_y > self.SCREEN_HEIGHT - self.BOUNDARY_MARGIN
        
        wall_hits = (hit_left | hit_right | hit_top | hit_bottom) & active
        
        if np.any(wall_hits):
            # Apply penalty
            rewards[wall_hits] += self.wall_penalty
            self.episode_stats['wall_collisions'][wall_hits] += 1
            
            # Bounce physics
            self.agents.heading[hit_left & active] = np.pi - self.agents.heading[hit_left & active]
            self.agents.heading[hit_right & active] = np.pi - self.agents.heading[hit_right & active]
            self.agents.heading[hit_top & active] = -self.agents.heading[hit_top & active]
            self.agents.heading[hit_bottom & active] = -self.agents.heading[hit_bottom & active]
            
            # Clamp positions
            self.agents.pos_x = np.clip(
                self.agents.pos_x,
                self.BOUNDARY_MARGIN,
                self.SCREEN_WIDTH - self.BOUNDARY_MARGIN
            )
            self.agents.pos_y = np.clip(
                self.agents.pos_y,
                self.BOUNDARY_MARGIN,
                self.SCREEN_HEIGHT - self.BOUNDARY_MARGIN
            )
        
        # Predator boundaries (simple wrap-around)
        if self.has_predators:
            for env_idx in range(self.num_envs):
                if not active[env_idx]:
                    continue
                
                predators = self.predators[env_idx]
                active_preds = predators.active
                
                predators.pos_x[active_preds] = np.clip(
                    predators.pos_x[active_preds],
                    self.BOUNDARY_MARGIN,
                    self.SCREEN_WIDTH - self.BOUNDARY_MARGIN
                )
                predators.pos_y[active_preds] = np.clip(
                    predators.pos_y[active_preds],
                    self.BOUNDARY_MARGIN,
                    self.SCREEN_HEIGHT - self.BOUNDARY_MARGIN
                )
    
    def get_state(self) -> np.ndarray:
        """Get vectorized state observations for all environments"""
        # Calculate maximum possible distance
        max_dist = np.sqrt(self.SCREEN_WIDTH**2 + self.SCREEN_HEIGHT**2)

        # Pre-allocate state array
        # [dist_food, angle_food, wall_dist, predator_dist, predator_angle, speed, bias]
        state_dim = 7
        states = np.zeros((self.num_envs, state_dim), dtype=np.float32)
        
        for env_idx in range(self.num_envs):
            if self.done[env_idx]:
                continue
            
            agent_x = self.agents.pos_x[env_idx]
            agent_y = self.agents.pos_y[env_idx]
            agent_heading = self.agents.heading[env_idx]
            
            # Find nearest food
            foods = self.foods[env_idx]
            if np.any(foods.active):
                dx = foods.pos_x[foods.active] - agent_x
                dy = foods.pos_y[foods.active] - agent_y
                distances = np.sqrt(dx*dx + dy*dy)
                
                if len(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    dist_food = distances[nearest_idx] / max_dist
                    
                    angle = np.arctan2(dy[nearest_idx], dx[nearest_idx]) - agent_heading
                    angle_food = ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi
                else:
                    dist_food = 1.0  # Max distance
                    angle_food = 0.0
            else:
                dist_food = 1.0
                angle_food = 0.0
            
            # Wall distances
            left = agent_x - self.BOUNDARY_MARGIN
            right = self.SCREEN_WIDTH - self.BOUNDARY_MARGIN - agent_x
            top = agent_y - self.BOUNDARY_MARGIN
            bottom = self.SCREEN_HEIGHT - self.BOUNDARY_MARGIN - agent_y
            wall_dist = min(left, right, top, bottom) / (self.SCREEN_WIDTH / 2)
            
            # Predator distance and angle
            predator_dist = 1.0  # Default: no predator
            predator_angle = 0.0
            
            if self.has_predators and np.any(self.predators[env_idx].active):
                predators = self.predators[env_idx]
                dx = predators.pos_x[predators.active] - agent_x
                dy = predators.pos_y[predators.active] - agent_y
                distances = np.sqrt(dx*dx + dy*dy)
                
                if len(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    predator_dist = distances[nearest_idx] / max_dist
                    
                    angle = np.arctan2(dy[nearest_idx], dx[nearest_idx]) - agent_heading
                    predator_angle = ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi
            
            # Speed (normalized)
            speed = np.sqrt(self.agents.vel_x[env_idx]**2 + self.agents.vel_y[env_idx]**2) / self.agent_speed
            
            # Compile state
            states[env_idx] = [
                dist_food,
                angle_food,
                wall_dist,
                predator_dist,
                predator_angle,
                speed,
                1.0  # Bias term
            ]

        # META-4.2 Sensor Drift: Add Gaussian noise after step > 200
        sensor_drift_active = self.steps > 200
        if np.any(sensor_drift_active):
            # Add Gaussian noise (mean=0, std=0.2) to observations
            noise = np.random.normal(0.0, 0.2, states.shape)
            states[sensor_drift_active] += noise[sensor_drift_active]
            # Clip to valid range [0, 1]
            states = np.clip(states, 0.0, 1.0)

        return states
    
    def _compile_info(self, new_dones: np.ndarray) -> Dict:
        """Compile additional information dictionary"""
        # Calculate statistics for newly done episodes
        done_indices = np.where(new_dones)[0]
        
        episode_rewards = []
        for idx in done_indices:
            episode_rewards.append(self.episode_stats['total_reward'][idx])
        
        return {
            'episode': {
                'r': np.array(episode_rewards) if episode_rewards else np.array([]),
                'l': self.steps[new_dones] if np.any(new_dones) else np.array([]),
                'food_collected': self.episode_stats['food_collected'][new_dones] if np.any(new_dones) else np.array([]),
                'wall_collisions': self.episode_stats['wall_collisions'][new_dones] if np.any(new_dones) else np.array([]),
            },
            'active_environments': np.sum(~self.done),
            'mean_food_collected': np.mean(self.episode_stats['food_collected'][~self.done]) if np.any(~self.done) else 0.0,
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Update environment configuration"""
        self.config.update(config)
        
        # Update parameters from config
        self.step_penalty = self.config.get('step_penalty', self.step_penalty)
        self.wall_penalty = self.config.get('wall_penalty', self.wall_penalty)
        self.food_reward = self.config.get('food_reward', self.food_reward)
        self.predator_penalty = self.config.get('predator_penalty', self.predator_penalty)
        
        if 'predator' in config:
            self.has_predators = config['predator'] and self.predator_count > 0
    
    def close(self):
        """Clean up resources"""
        pass
    
    def render(self, mode: str = 'human', env_idx: int = 0):
        """Render a specific environment"""
        if mode == 'human':
            # Simple text-based rendering
            print(f"Env {env_idx}: Step {self.steps[env_idx]}, " +
                  f"Pos ({self.agents.pos_x[env_idx]:.1f}, {self.agents.pos_y[env_idx]:.1f}), " +
                  f"Food: {self.episode_stats['food_collected'][env_idx]}")
        return None