import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
import warnings

# Import from updated curriculum
from curriculum.curriculum import get_stage_config, CurriculumStage


@dataclass
class Entity:
    """Base class for all entities in the arena"""
    pos: np.ndarray  # [x, y]
    vel: np.ndarray  # [vx, vy]
    radius: float
    entity_type: str  # 'agent', 'food', 'predator'
    active: bool = True
    id: Optional[int] = None
    
    def update_position(self, dt: float = 1.0):
        """Update position based on velocity"""
        self.pos += self.vel * dt
        return self.pos


@dataclass
class Agent(Entity):
    """Agent entity with additional properties"""
    heading: float = 0.0  # Direction in radians
    max_speed: float = 2.0
    max_turn_rate: float = 0.2
    health: float = 100.0
    energy: float = 100.0
    last_action: Optional[np.ndarray] = None
    
    def apply_action(self, action: np.ndarray):
        """
        Apply continuous action vector.
        action[0]: turn rate (-1 to 1)
        action[1]: thrust (0 to 1)
        action[2]: special action (if available)
        """
        self.last_action = action.copy()
        
        # Update heading (turn rate)
        turn = action[0] * self.max_turn_rate
        self.heading += turn
        
        # Normalize heading to [0, 2π]
        self.heading = self.heading % (2 * np.pi)
        
        # Apply thrust
        thrust = action[1] * self.max_speed
        self.vel = np.array([
            np.cos(self.heading) * thrust,
            np.sin(self.heading) * thrust
        ])
        
        # Apply special action (e.g., boost, brake)
        if len(action) > 2:
            # Example: boost consumes energy for temporary speed
            if action[2] > 0.5 and self.energy > 0:
                boost_factor = 1.5
                self.vel *= boost_factor
                self.energy -= 5.0


@dataclass
class ArenaState:
    """Complete state of the arena for vectorized operations"""
    agents: np.ndarray  # [num_agents, agent_state_dim]
    foods: np.ndarray   # [num_foods, food_state_dim]
    predators: np.ndarray  # [num_predators, predator_state_dim]
    walls: np.ndarray   # Wall positions and orientations
    time_step: int = 0
    
    @property
    def num_agents(self):
        return len(self.agents)
    
    @property
    def num_foods(self):
        return len(self.foods)
    
    @property
    def num_predators(self):
        return len(self.predators)


class VectorizedArenaCore:
    """
    Vectorized arena environment optimized for batch operations.
    Supports multiple agents, food, and predators simultaneously.
    """
    
    def __init__(self, 
                 stage: CurriculumStage = CurriculumStage.FORAGING,
                 config: Optional[Dict[str, Any]] = None,
                 num_agents: int = 1,
                 seed: Optional[int] = None):
        """
        Initialize vectorized arena.
        
        Args:
            stage: Curriculum stage
            config: Override configuration
            num_agents: Number of agents in this environment
            seed: Random seed for reproducibility
        """
        # Set random seed
        self.rng = np.random.RandomState(seed)
        
        # Load configuration
        self.stage = stage
        self.config = config or get_stage_config(stage)
        
        # Arena dimensions
        self.screen_width = 800
        self.screen_height = 600
        self.boundary_margin = 20
        
        # Game parameters from config
        self.wall_penalty = self.config.get("wall_penalty", 0.0)
        self.food_reward = self.config.get("food_reward", 10.0)
        self.step_penalty = self.config.get("step_penalty", 0.0)
        self.has_predators = self.config.get("predator", False)
        
        # Enhanced config with new parameters
        self.num_foods = self.config.get("food_count", 10)
        self.num_predators = self.config.get("predator_count", 0)
        self.max_steps = self.config.get("max_steps", 1000)
        self.food_respawn_rate = self.config.get("food_respawn_rate", 0.05)
        self.predator_speed = self.config.get("predator_speed", 1.0)
        self.predator_vision = self.config.get("predator_vision", 100.0)
        
        # Agent configuration
        self.num_agents = num_agents
        self.agent_radius = 10.0
        self.food_radius = 8.0
        self.predator_radius = 15.0
        
        # Initialize state containers
        self.reset()
        
        # Observation space parameters
        self.obs_num_foods = 5  # Number of nearest foods to observe
        self.obs_num_predators = 3  # Number of nearest predators to observe
        
        # Calculate observation dimension
        # [agent_state(6), nearest_foods(2*5), nearest_predators(2*3), walls(4)]
        self.observation_dim = 6 + (2 * self.obs_num_foods) + (2 * self.obs_num_predators) + 4
        
        # Action space: [turn_rate, thrust, special]
        self.action_dim = 3
        
        # Performance tracking
        self.episode_stats = {
            'total_reward': 0.0,
            'food_collected': 0,
            'wall_collisions': 0,
            'predator_escapes': 0,
            'steps_survived': 0
        }

        # Diagnostic 1: In-Lifetime Recovery Curve
        self.shock_step = 300  # Based on rule_flip_triggered at t=300
        self.diagnostic_logs = []
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observations"""
        self.time_step = 0
        self.episode_stats = {k: 0.0 for k in self.episode_stats.keys()}

        # Reset diagnostic logs
        self.diagnostic_logs = [[] for _ in range(self.num_agents)]

        # Initialize agents
        self.agents = []
        for i in range(self.num_agents):
            agent = Agent(
                pos=self._random_position(margin=50),
                vel=np.zeros(2),
                radius=self.agent_radius,
                entity_type='agent',
                id=i,
                heading=self.rng.uniform(0, 2 * np.pi)
            )
            self.agents.append(agent)

        # Initialize foods
        self.foods = []
        for i in range(self.num_foods):
            food = Entity(
                pos=self._random_position(margin=30),
                vel=np.zeros(2),
                radius=self.food_radius,
                entity_type='food',
                id=i
            )
            self.foods.append(food)

        # Initialize predators if enabled
        self.predators = []
        if self.has_predators and self.num_predators > 0:
            for i in range(self.num_predators):
                predator = Entity(
                    pos=self._random_position(margin=50),
                    vel=np.zeros(2),
                    radius=self.predator_radius,
                    entity_type='predator',
                    id=i
                )
                self.predators.append(predator)

        return self._get_observations()
    
    def _random_position(self, margin: float = 0.0) -> np.ndarray:
        """Generate random position within bounds"""
        return np.array([
            self.rng.uniform(margin, self.screen_width - margin),
            self.rng.uniform(margin, self.screen_height - margin)
        ])
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Execute actions for all agents.
        
        Args:
            actions: Array of shape (num_agents, action_dim)
            
        Returns:
            observations: Array of shape (num_agents, observation_dim)
            rewards: Array of shape (num_agents,)
            dones: Array of shape (num_agents,) or scalar
            info: Dictionary with additional information
        """
        # Validate input
        assert actions.shape == (self.num_agents, self.action_dim), \
            f"Expected actions shape ({self.num_agents}, {self.action_dim}), got {actions.shape}"
        
        # Initialize rewards
        rewards = np.zeros(self.num_agents, dtype=np.float32)

        # META-4 PRESSURE INJECTION
        # Initialize reward_sign if not exists
        if not hasattr(self, 'reward_sign'):
            self.reward_sign = np.ones(self.num_agents, dtype=np.float32)

        # META-4.1 Mid-Episode World Shift
        mid_episode_step = self.max_steps // 2
        world_shift_triggered = (self.time_step == mid_episode_step)
        if np.any(world_shift_triggered):
            # Reduce food reward by 30%
            self.food_reward *= 0.7
            # Increase predator speed by 40%
            self.predator_speed *= 1.4

        # META-4.3 Rule Flip
        rule_flip_triggered = (self.time_step == 300)
        if np.any(rule_flip_triggered):
            # Flip reward signs
            self.reward_sign *= -1

        # Apply step penalty
        rewards += self.step_penalty
        
        # Update all agents
        for i, agent in enumerate(self.agents):
            if agent.active:
                agent.apply_action(actions[i])
        
        # Update predators (simple AI)
        self._update_predators()
        
        # Update positions
        for entity_list in [self.agents, self.foods, self.predators]:
            for entity in entity_list:
                if entity.active:
                    entity.update_position()
        
        # Handle boundary collisions (vectorized approach)
        agent_positions = np.array([agent.pos for agent in self.agents if agent.active])
        if len(agent_positions) > 0:
            # Check wall collisions
            wall_collisions = self._check_boundary_collisions(agent_positions)
            
            # Apply wall penalties and bounce
            for i, agent in enumerate(self.agents):
                if agent.active and wall_collisions[i]:
                    rewards[i] += self.wall_penalty
                    self.episode_stats['wall_collisions'] += 1
                    
                    # Simple bounce: reverse velocity component
                    if agent.pos[0] <= self.boundary_margin or agent.pos[0] >= self.screen_width - self.boundary_margin:
                        agent.vel[0] *= -0.5
                    if agent.pos[1] <= self.boundary_margin or agent.pos[1] >= self.screen_height - self.boundary_margin:
                        agent.vel[1] *= -0.5
                    
                    # Clamp position
                    agent.pos = np.clip(
                        agent.pos,
                        [self.boundary_margin, self.boundary_margin],
                        [self.screen_width - self.boundary_margin, self.screen_height - self.boundary_margin]
                    )
        
        # Handle food collisions (vectorized for efficiency)
        if self.foods:
            agent_positions = np.array([agent.pos for agent in self.agents if agent.active])
            food_positions = np.array([food.pos for food in self.foods if food.active])
            
            if len(agent_positions) > 0 and len(food_positions) > 0:
                # Compute distances between all agents and foods
                # Using broadcasting for efficiency
                diff = agent_positions[:, np.newaxis, :] - food_positions[np.newaxis, :, :]
                distances = np.linalg.norm(diff, axis=2)
                
                # Find collisions (agent_radius + food_radius)
                collision_radius = self.agent_radius + self.food_radius
                collisions = distances < collision_radius
                
                # Process collisions
                for agent_idx in range(len(self.agents)):
                    if self.agents[agent_idx].active:
                        food_collisions = np.where(collisions[agent_idx])[0]
                        for food_idx in food_collisions:
                            if self.foods[food_idx].active:
                                # Collect food
                                rewards[agent_idx] += self.food_reward
                                self.episode_stats['food_collected'] += 1
                                
                                # Respawn food (competitive placement)
                                self._respawn_food(food_idx)
        
        # Handle predator interactions
        if self.predators:
            self._handle_predator_interactions(rewards)
        
        # Increment time
        self.time_step += 1
        self.episode_stats['steps_survived'] = self.time_step
        
        # Check termination conditions
        done = self.time_step >= self.max_steps
        dones = np.full(self.num_agents, done, dtype=np.bool_)
        
        # Get observations
        observations = self._get_observations()
        
        # Diagnostic 1: Log data for each active agent
        for i, agent in enumerate(self.agents):
            if agent.active:
                self.diagnostic_logs[i].append({
                    "timestep": self.time_step,
                    "reward": float(rewards[i]),
                    "is_shock": self.time_step == self.shock_step
                })

        # Compile info
        info = {
            'episode_stats': self.episode_stats.copy(),
            'time_step': self.time_step,
            'agents_active': sum(agent.active for agent in self.agents),
            'foods_active': sum(food.active for food in self.foods),
            'predators_active': sum(pred.active for pred in self.predators)
        }

        return observations, rewards, dones, info
    
    def _update_predators(self):
        """Update predator positions with simple chasing AI"""
        if not self.predators:
            return
        
        predator_positions = np.array([p.pos for p in self.predators if p.active])
        agent_positions = np.array([a.pos for a in self.agents if a.active])
        
        if len(agent_positions) == 0 or len(predator_positions) == 0:
            return
        
        # Vectorized distance calculation
        for i, predator in enumerate(self.predators):
            if predator.active:
                # Find nearest agent
                distances = np.linalg.norm(agent_positions - predator.pos, axis=1)
                nearest_idx = int(np.argmin(distances))
                
                if distances[nearest_idx] < self.predator_vision:
                    # Chase nearest agent
                    direction = agent_positions[nearest_idx] - predator.pos
                    direction_norm = np.linalg.norm(direction)
                    if direction_norm > 0:
                        predator.vel = (direction / direction_norm) * self.predator_speed
                else:
                    # Random wandering
                    predator.vel = self.rng.uniform(-1, 1, 2) * self.predator_speed * 0.5
    
    def _handle_predator_interactions(self, rewards: np.ndarray):
        """Handle predator-agent interactions"""
        if not self.predators:
            return
        
        predator_positions = np.array([p.pos for p in self.predators if p.active])
        agent_positions = np.array([a.pos for a in self.agents if a.active])
        
        if len(agent_positions) == 0 or len(predator_positions) == 0:
            return
        
        # Check collisions between predators and agents
        for i, agent in enumerate(self.agents):
            if agent.active:
                for predator in self.predators:
                    if predator.active:
                        distance = np.linalg.norm(agent.pos - predator.pos)
                        if distance < (self.agent_radius + self.predator_radius):
                            # Agent caught by predator
                            agent.active = False
                            rewards[i] += self.config.get("predator_penalty", -20.0)
                            self.episode_stats['predator_escapes'] += 1
    
    def _respawn_food(self, food_idx: int):
        """Respawn food with competitive positioning"""
        food = self.foods[food_idx]
        
        # Competitive placement: avoid clustering near agents
        agent_positions = np.array([agent.pos for agent in self.agents if agent.active])
        
        if len(agent_positions) > 0:
            # Try to place food away from agents
            max_attempts = 10
            for attempt in range(max_attempts):
                new_pos = self._random_position(margin=30)
                
                # Check distance to all agents
                distances = np.linalg.norm(agent_positions - new_pos, axis=1)
                min_distance = np.min(distances)
                
                # Place food at least 100 pixels from nearest agent
                if min_distance > 100 or attempt == max_attempts - 1:
                    food.pos = new_pos
                    food.active = True
                    break
        else:
            # No agents active, random placement
            food.pos = self._random_position(margin=30)
            food.active = True
    
    def _check_boundary_collisions(self, positions: np.ndarray) -> np.ndarray:
        """Check if positions are outside boundaries"""
        collisions = np.zeros(len(positions), dtype=np.bool_)
        
        # Left boundary
        collisions |= positions[:, 0] < self.boundary_margin
        # Right boundary
        collisions |= positions[:, 0] > self.screen_width - self.boundary_margin
        # Top boundary
        collisions |= positions[:, 1] < self.boundary_margin
        # Bottom boundary
        collisions |= positions[:, 1] > self.screen_height - self.boundary_margin
        
        return collisions
    
    def _get_observations(self) -> np.ndarray:
        """
        Get observations for all agents.
        Returns array of shape (num_agents, observation_dim)
        """
        observations = np.zeros((self.num_agents, self.observation_dim), dtype=np.float32)
        
        # Precompute entity positions
        agent_positions = np.array([agent.pos for agent in self.agents if agent.active])
        food_positions = np.array([food.pos for food in self.foods if food.active])
        predator_positions = np.array([pred.pos for pred in self.predators if pred.active])
        
        for i, agent in enumerate(self.agents):
            if not agent.active:
                continue
                
            obs_idx = 0
            
            # Agent state: [x, y, vx, vy, heading, energy]
            observations[i, obs_idx:obs_idx+6] = [
                agent.pos[0] / self.screen_width,  # Normalized x
                agent.pos[1] / self.screen_height,  # Normalized y
                agent.vel[0] / agent.max_speed,  # Normalized vx
                agent.vel[1] / agent.max_speed,  # Normalized vy
                agent.heading / (2 * np.pi),  # Normalized heading
                agent.energy / 100.0  # Normalized energy
            ]
            obs_idx += 6
            
            # Nearest foods
            if len(food_positions) > 0:
                # Compute distances to all foods
                distances = np.linalg.norm(food_positions - agent.pos, axis=1)
                
                # Get indices of nearest foods
                if len(distances) > self.obs_num_foods:
                    nearest_indices = np.argpartition(distances, self.obs_num_foods)[:self.obs_num_foods]
                else:
                    nearest_indices = np.arange(len(distances))
                    # Pad if not enough foods
                    if len(nearest_indices) < self.obs_num_foods:
                        nearest_indices = np.pad(
                            nearest_indices,
                            (0, self.obs_num_foods - len(nearest_indices)),
                            mode='constant',
                            constant_values=0
                        )
                
                # Add food observations
                for j, food_idx in enumerate(nearest_indices):
                    if food_idx < len(food_positions):
                        food_pos = food_positions[food_idx]
                        rel_pos = food_pos - agent.pos
                        distance = np.linalg.norm(rel_pos)
                        angle = np.arctan2(rel_pos[1], rel_pos[0]) - agent.heading
                        
                        # Normalize
                        normalized_distance = distance / np.sqrt(self.screen_width**2 + self.screen_height**2)
                        normalized_angle = (angle + np.pi) / (2 * np.pi)  # [0, 1]
                        
                        observations[i, obs_idx + 2*j] = normalized_distance
                        observations[i, obs_idx + 2*j + 1] = normalized_angle
                    else:
                        # Padding: max distance, zero angle
                        observations[i, obs_idx + 2*j] = 1.0
                        observations[i, obs_idx + 2*j + 1] = 0.0
            
            obs_idx += 2 * self.obs_num_foods
            
            # Nearest predators
            if len(predator_positions) > 0 and self.has_predators:
                # Similar logic for predators
                distances = np.linalg.norm(predator_positions - agent.pos, axis=1)
                
                if len(distances) > self.obs_num_predators:
                    nearest_indices = np.argpartition(distances, self.obs_num_predators)[:self.obs_num_predators]
                else:
                    nearest_indices = np.arange(len(distances))
                    if len(nearest_indices) < self.obs_num_predators:
                        nearest_indices = np.pad(
                            nearest_indices,
                            (0, self.obs_num_predators - len(nearest_indices)),
                            mode='constant',
                            constant_values=0
                        )
                
                for j, pred_idx in enumerate(nearest_indices):
                    if pred_idx < len(predator_positions):
                        pred_pos = predator_positions[pred_idx]
                        rel_pos = pred_pos - agent.pos
                        distance = np.linalg.norm(rel_pos)
                        angle = np.arctan2(rel_pos[1], rel_pos[0]) - agent.heading
                        
                        normalized_distance = distance / np.sqrt(self.screen_width**2 + self.screen_height**2)
                        normalized_angle = (angle + np.pi) / (2 * np.pi)
                        
                        observations[i, obs_idx + 2*j] = normalized_distance
                        observations[i, obs_idx + 2*j + 1] = normalized_angle
                    else:
                        observations[i, obs_idx + 2*j] = 1.0
                        observations[i, obs_idx + 2*j + 1] = 0.0
            
            obs_idx += 2 * self.obs_num_predators
            
            # Wall distances: [left, right, top, bottom]
            observations[i, obs_idx:obs_idx+4] = [
                agent.pos[0] / self.screen_width,  # Distance to left wall
                (self.screen_width - agent.pos[0]) / self.screen_width,  # Distance to right wall
                agent.pos[1] / self.screen_height,  # Distance to top wall
                (self.screen_height - agent.pos[1]) / self.screen_height  # Distance to bottom wall
            ]

        # META-4.2 Sensor Drift: Add Gaussian noise after step > 200
        if self.time_step > 200:
            # Add Gaussian noise (mean=0, std=0.2) to observations
            noise = np.random.normal(0.0, 0.2, observations.shape)
            observations += noise
            # Clip to valid range [0, 1]
            observations = np.clip(observations, 0.0, 1.0)

        return observations
    
    def render(self, mode: str = 'human'):
        """Render the environment (placeholder)"""
        if mode == 'human':
            # For now, print basic stats
            print(f"Step: {self.time_step}, " +
                  f"Food: {self.episode_stats['food_collected']}, " +
                  f"Reward: {self.episode_stats['total_reward']:.2f}")
        return None
    
    def close(self):
        """Clean up resources"""
        pass
    
    @property
    def action_space(self):
        """Get action space description"""
        return {
            'shape': (self.action_dim,),
            'low': np.array([-1.0, 0.0, 0.0]),
            'high': np.array([1.0, 1.0, 1.0]),
            'dtype': np.float32
        }
    
    @property
    def observation_space(self):
        """Get observation space description"""
        return {
            'shape': (self.observation_dim,),
            'low': 0.0,
            'high': 1.0,
            'dtype': np.float32
        }

    def get_diagnostic_logs(self) -> List[List[Dict[str, Any]]]:
        """Get diagnostic logs for all agents"""
        return self.diagnostic_logs.copy()


# Backward compatibility wrapper
class ArenaEnvCore(VectorizedArenaCore):
    """
    DEPRECATED: ArenaEnvCore is SLOW and should NOT be used.
    
    Use DeterministicVectorizedArena instead for true NumPy vectorization.
    Import with: from deterministic_env import DeterministicVectorizedArena
    """
    def __init__(self, stage=CurriculumStage.FORAGING, config=None):
        warnings.warn(
            "ArenaEnvCore is DEPRECATED and SLOW. Use DeterministicVectorizedArena instead. "
            "Import with: from deterministic_env import DeterministicVectorizedArena",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(stage=stage, config=config, num_agents=1)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Legacy step method with single action"""
        # Convert single action to batch of 1
        if isinstance(actions, (int, float)):
            # For discrete actions from old code, map to continuous
            if actions == 0:  # Move forward
                action_array = np.array([[0.0, 1.0, 0.0]])
            elif actions == 1:  # Turn left
                action_array = np.array([[-0.5, 0.5, 0.0]])
            elif actions == 2:  # Turn right
                action_array = np.array([[0.5, 0.5, 0.0]])
            else:
                action_array = np.array([[0.0, 0.0, 0.0]])
        else:
            action_array = np.array([actions])
        
        obs, rewards, dones, info = super().step(action_array)
        
        # Return single agent results (4 values: obs, rewards, dones, info)
        return obs[0], rewards[0], dones[0], info
    
    def get_state(self):
        """Legacy method - use reset/step instead"""
        obs = self._get_observations()
        return tuple(obs[0]) if len(obs) > 0 else ()