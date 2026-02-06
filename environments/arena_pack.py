import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Import curriculum for config
from curriculum.curriculum import CurriculumStage, get_stage_config


class PredatorRole(Enum):
    """Roles in predator packs"""
    CHASER = "chaser"  # Fast, pursues prey directly
    CUTTER = "cutter"  # Intercepts escape routes
    AMBUSHER = "ambusher"  # Waits in strategic positions
    SCOUT = "scout"  # Explores and locates prey


@dataclass
class PredatorTraits:
    """Individual traits for each predator"""
    speed: float
    turn_rate: float
    stamina: float  # How long they can sustain high speed
    aggression: float  # How aggressively they chase
    caution: float  # How carefully they approach
    role: PredatorRole
    vision_range: float
    energy_max: float
    energy_initial: float
    energy_move_cost: float
    energy_turn_cost: float
    capture_efficiency: float  # Chance to capture when in range
    pack_contribution: float  # How much they contribute to pack coordination


@dataclass
class ArenaConfig:
    """Configuration for pack arena"""
    screen_width: int = 800
    screen_height: int = 600
    boundary_margin: int = 20
    max_steps: int = 1000
    
    # Prey configuration
    prey_speed: float = 2.0
    prey_turn_rate: float = 0.2
    prey_vision_range: float = 250.0
    prey_energy_max: float = 80.0
    prey_energy_initial: float = 80.0
    prey_energy_move_cost: float = 0.08
    prey_energy_turn_cost: float = 0.04
    
    # Food configuration
    num_food: int = 15
    food_radius: float = 8.0
    food_energy_gain: float = 25.0
    food_respawn_rate: float = 0.05
    food_competitive_placement: bool = True
    
    # Capture mechanics
    base_capture_radius: float = 18.0
    capture_efficiency_range: Tuple[float, float] = (0.3, 0.9)  # Min/max efficiency
    
    # Reward structure
    prey_capture_penalty: float = -30.0
    predator_capture_reward: float = 10.0
    prey_escape_reward: float = 5.0
    predator_starvation_penalty: float = -20.0
    prey_starvation_penalty: float = -20.0
    step_penalty: float = -0.01
    predator_step_penalty: float = -0.02
    
    # Pack coordination
    pack_coordination_bonus: float = 0.5
    role_specialization_bonus: float = 0.3
    successful_interception_bonus: float = 2.0


class DeterministicTraitGenerator:
    """Generates deterministic predator traits based on seed and index"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def generate_traits(self, predator_index: int, total_predators: int, 
                       config: ArenaConfig) -> PredatorTraits:
        """Generate unique traits for each predator"""
        # Use predator index to create deterministic variations
        base_speed = 2.5 + (predator_index % 3) * 0.3  # 2.5, 2.8, 3.1
        
        # Assign roles based on position in pack
        if predator_index == 0:
            role = PredatorRole.CHASER
            aggression = 0.9
            caution = 0.1
        elif predator_index == 1:
            role = PredatorRole.CUTTER
            aggression = 0.6
            caution = 0.4
        else:
            role = PredatorRole.AMBUSHER if predator_index % 2 == 0 else PredatorRole.SCOUT
            aggression = 0.4
            caution = 0.6
        
        # Add some randomness (but deterministic based on index)
        rand_seed = predator_index * 1000
        local_rng = np.random.default_rng(rand_seed)
        
        stamina = 0.7 + local_rng.uniform(-0.2, 0.2)
        capture_efficiency = np.clip(
            0.5 + local_rng.uniform(-0.2, 0.2),
            config.capture_efficiency_range[0],
            config.capture_efficiency_range[1]
        )
        
        return PredatorTraits(
            speed=base_speed,
            turn_rate=0.15 + (predator_index % 2) * 0.05,
            stamina=stamina,
            aggression=aggression,
            caution=caution,
            role=role,
            vision_range=300.0 + (predator_index % 4) * 20,
            energy_max=120.0 + (predator_index % 3) * 10,
            energy_initial=120.0 + (predator_index % 3) * 10,
            energy_move_cost=0.12 + (predator_index % 3) * 0.02,
            energy_turn_cost=0.06 + (predator_index % 3) * 0.01,
            capture_efficiency=capture_efficiency,
            pack_contribution=0.5 + local_rng.uniform(-0.2, 0.2)
        )


class EnergySystem:
    """Manages energy and stamina for agents"""

    def __init__(self, max_energy: float, initial_energy: float):
        self.max_energy = max_energy
        self.energy: np.ndarray = np.array([], dtype=np.float32)
        self.alive: np.ndarray = np.array([], dtype=bool)
        self.starved: np.ndarray = np.array([], dtype=bool)
        self.stamina_factor = np.array([], dtype=np.float32)  # Current stamina multiplier
        
    def initialize(self, num_agents: int):
        self.energy = np.full(num_agents, self.max_energy, dtype=np.float32)
        self.alive = np.ones(num_agents, dtype=bool)
        self.starved = np.zeros(num_agents, dtype=bool)
        self.stamina_factor = np.ones(num_agents, dtype=np.float32)
    
    def apply_movement_cost(self, moving: np.ndarray, turning: np.ndarray, 
                           move_cost: float, turn_cost: float):
        """Apply energy costs considering current stamina"""
        if self.energy is None:
            return
        
        # Base costs
        energy_cost = np.zeros_like(self.energy)
        energy_cost[moving] += move_cost
        energy_cost[turning & moving] += turn_cost
        
        # Apply stamina factor (higher stamina = lower effective cost)
        effective_cost = energy_cost / self.stamina_factor
        
        self.energy -= effective_cost
        
        # Update stamina factor (fatigue)
        moving_mask = moving.astype(np.float32)
        self.stamina_factor = np.maximum(0.3, self.stamina_factor - moving_mask * 0.01)
        
        # Recover stamina when not moving
        recovery_mask = (~moving).astype(np.float32)
        self.stamina_factor = np.minimum(1.0, self.stamina_factor + recovery_mask * 0.02)
        
        # Check starvation
        self.starved = self.energy <= 0
        self.alive[self.starved] = False
        
        # Clamp energy
        self.energy = np.clip(self.energy, 0, self.max_energy)
    
    def apply_energy_gain(self, agents_idx: np.ndarray, amount: float):
        """Apply energy gain (from food or successful capture)"""
        if self.energy is None:
            return
        
        self.energy[agents_idx] += amount
        self.energy = np.clip(self.energy, 0, self.max_energy)
        
        # Restore stamina when gaining energy
        self.stamina_factor[agents_idx] = np.minimum(1.0, self.stamina_factor[agents_idx] + 0.1)
    
    def get_energy_normalized(self) -> np.ndarray:
        """Get normalized energy levels"""
        if self.energy is None:
            return np.array([])
        return self.energy / self.max_energy


class PackCoordinationSystem:
    """Manages predator pack coordination and role specialization"""
    
    def __init__(self, num_predators: int, traits: List[PredatorTraits]):
        self.num_predators = num_predators
        self.traits = traits
        self.coordination_level = 0.0
        self.current_strategy = None
        self.role_assignments = None
        
    def update(self, predator_positions: np.ndarray, prey_position: np.ndarray,
               prey_velocity: np.ndarray, time_step: int):
        """Update pack coordination and strategy"""
        if len(predator_positions) != self.num_predators:
            return
        
        # Calculate distances to prey
        distances = np.linalg.norm(predator_positions - prey_position, axis=1)
        
        # Determine strategy based on distances and prey velocity
        avg_distance = np.mean(distances)
        prey_speed = np.linalg.norm(prey_velocity)
        
        if avg_distance > 200:
            strategy = "search"  # Spread out and search
        elif avg_distance > 100:
            strategy = "encircle"  # Form a circle around prey
        elif avg_distance > 50:
            strategy = "pursuit"  # Direct chase
        else:
            strategy = "capture"  # Close-range capture attempt
        
        self.current_strategy = strategy
        
        # Calculate coordination level (0-1)
        # Based on how well predators are positioned relative to each other
        if self.num_predators > 1:
            # Calculate spacing between predators
            spacing = []
            for i in range(self.num_predators):
                for j in range(i+1, self.num_predators):
                    spacing.append(np.linalg.norm(predator_positions[i] - predator_positions[j]))
            
            if spacing:
                optimal_spacing = 100.0  # Optimal spacing for encirclement
                spacing_error = np.mean([abs(s - optimal_spacing) for s in spacing])
                self.coordination_level = np.clip(1.0 - spacing_error / 200.0, 0.0, 1.0)
        
        # Assign roles based on strategy and traits
        self._assign_roles(strategy, distances)
    
    def _assign_roles(self, strategy: str, distances_to_prey: np.ndarray):
        """Assign specific roles to each predator based on strategy"""
        self.role_assignments = []
        
        if strategy == "search":
            # Scouts take lead, others spread out
            for i, trait in enumerate(self.traits):
                if trait.role == PredatorRole.SCOUT:
                    self.role_assignments.append("lead_scout")
                else:
                    self.role_assignments.append("patrol")
        
        elif strategy == "encircle":
            # Position predators around prey based on their roles
            sorted_indices = np.argsort(distances_to_prey)
            for idx, trait_idx in enumerate(sorted_indices):
                trait = self.traits[trait_idx]
                if idx == 0:
                    self.role_assignments.append("front_chaser")
                elif idx == 1:
                    self.role_assignments.append("flank_cutter")
                else:
                    self.role_assignments.append("rear_ambusher")
        
        elif strategy == "pursuit":
            # Fastest predators lead the chase
            speed_order = np.argsort([-t.speed for t in self.traits])  # Descending speed
            for idx, trait_idx in enumerate(speed_order):
                if idx == 0:
                    self.role_assignments.append("primary_chaser")
                elif idx == 1:
                    self.role_assignments.append("secondary_chaser")
                else:
                    self.role_assignments.append("support")
        
        else:  # capture
            # All predators try to capture
            for trait in self.traits:
                self.role_assignments.append("capture_attempt")
    
    def get_role_bonus(self, predator_index: int) -> float:
        """Get bonus for performing assigned role well"""
        if not self.role_assignments or predator_index >= len(self.role_assignments):
            return 0.0
        
        role = self.role_assignments[predator_index]
        trait = self.traits[predator_index]
        
        # Base bonus for coordination
        bonus = self.coordination_level * 0.3
        
        # Additional bonus for role specialization
        if (role == "lead_scout" and trait.role == PredatorRole.SCOUT) or \
           (role == "front_chaser" and trait.role == PredatorRole.CHASER) or \
           (role == "flank_cutter" and trait.role == PredatorRole.CUTTER) or \
           (role == "rear_ambusher" and trait.role == PredatorRole.AMBUSHER):
            bonus += 0.2
        
        return float(bonus)
    
    def get_strategy_instructions(self, predator_index: int) -> np.ndarray:
        """Get strategy-specific instructions for each predator"""
        if not self.role_assignments or predator_index >= len(self.role_assignments):
            return np.zeros(3, dtype=np.float32)
        
        role = self.role_assignments[predator_index]
        
        if role in ["lead_scout", "patrol"]:
            # Explore and cover area
            return np.array([0.0, 0.7, 0.3], dtype=np.float32)  # [caution, aggression, coordination]
        elif role in ["front_chaser", "primary_chaser"]:
            # Aggressive pursuit
            return np.array([0.1, 0.9, 0.5], dtype=np.float32)
        elif role in ["flank_cutter", "secondary_chaser"]:
            # Strategic positioning
            return np.array([0.3, 0.7, 0.7], dtype=np.float32)
        elif role == "rear_ambusher":
            # Patient waiting
            return np.array([0.7, 0.3, 0.6], dtype=np.float32)
        elif role == "support":
            # Support primary chaser
            return np.array([0.4, 0.6, 0.8], dtype=np.float32)
        else:  # capture_attempt
            # All-out capture attempt
            return np.array([0.0, 1.0, 0.4], dtype=np.float32)


class PredatorPackArena:
    """
    Advanced predator-prey arena with pack coordination, individual traits,
    survival pressures, and realistic movement.
    """
    
    def __init__(self, 
                 batch_size: int,
                 predators_per_pack: int = 3,
                 stage: CurriculumStage = CurriculumStage.FORAGING,
                 config: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None,
                 deterministic: bool = True):
        """
        Initialize the predator pack arena.
        
        Args:
            batch_size: Number of parallel environments
            predators_per_pack: Number of predators in each pack
            stage: Curriculum stage for difficulty
            config: Override configuration
            seed: Random seed for reproducibility
            deterministic: Whether to use deterministic RNG
        """
        self.batch_size = batch_size
        self.predators_per_pack = predators_per_pack
        self.total_predators = batch_size * predators_per_pack
        self.deterministic = deterministic
        
        # Load configuration
        self.stage = stage
        self.base_config = config or get_stage_config(stage)
        self.config = self._create_config(self.base_config)
        
        # Setup RNG
        self.rng = np.random.default_rng(seed)
        
        # Generate predator traits (deterministic based on index)
        self.trait_generator = DeterministicTraitGenerator(seed)
        self.predator_traits = self._generate_predator_traits()
        
        # Initialize systems
        self.prey_energy_system = EnergySystem(
            self.config.prey_energy_max,
            self.config.prey_energy_initial
        )
        self.predator_energy_systems = [
            EnergySystem(trait.energy_max, trait.energy_initial)
            for trait in self.predator_traits
        ]
        
        self.pack_systems = []  # One per environment
        
        # State tracking
        self.step_count = 0
        self.prey_positions = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.predator_positions = np.zeros((self.total_predators, 2), dtype=np.float32)
        self.food_positions = np.zeros((self.batch_size * self.config.num_food, 2), dtype=np.float32)
        self.prey_headings = np.zeros(self.batch_size, dtype=np.float32)
        self.predator_headings = np.zeros(self.total_predators, dtype=np.float32)
        self.prey_velocities = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.predator_velocities = np.zeros((self.total_predators, 2), dtype=np.float32)

        # Termination tracking
        self.prey_done = np.zeros(self.batch_size, dtype=bool)
        self.predator_done = np.zeros(self.total_predators, dtype=bool)
        self.prey_captured = np.zeros(self.batch_size, dtype=bool)
        self.prey_escaped = np.zeros(self.batch_size, dtype=bool)
        self.prey_starved = np.zeros(self.batch_size, dtype=bool)
        self.predator_starved = np.zeros(self.total_predators, dtype=bool)
        
        # Statistics
        self.stats = {
            'prey_captures': np.zeros(self.total_predators, dtype=np.int32),
            'prey_escapes': np.zeros(self.batch_size, dtype=np.int32),
            'predator_starved': np.zeros(self.total_predators, dtype=np.int32),
            'prey_starved': np.zeros(self.batch_size, dtype=np.int32),
            'food_consumed': np.zeros(self.batch_size, dtype=np.int32),
            'pack_coordination': np.zeros(self.batch_size, dtype=np.float32),
            'capture_efficiency': np.zeros(self.batch_size, dtype=np.float32),
            'steps_survived': np.zeros(self.batch_size, dtype=np.int32)
        }
    
    def _create_config(self, base_config: Dict[str, Any]) -> ArenaConfig:
        """Create arena configuration from base config"""
        config = ArenaConfig()
        
        # Update from base config
        config.screen_width = base_config.get('screen_width', config.screen_width)
        config.screen_height = base_config.get('screen_height', config.screen_height)
        config.num_food = base_config.get('food_count', config.num_food)
        config.max_steps = base_config.get('max_steps', config.max_steps)
        
        # Update prey parameters
        if 'prey_speed' in base_config:
            config.prey_speed = base_config['prey_speed']
        if 'prey_energy' in base_config:
            config.prey_energy_max = base_config['prey_energy']
            config.prey_energy_initial = base_config['prey_energy']
        
        # Update capture mechanics
        if 'capture_radius' in base_config:
            config.base_capture_radius = base_config['capture_radius']
        if 'capture_efficiency_range' in base_config:
            config.capture_efficiency_range = base_config['capture_efficiency_range']
        
        # Update rewards
        if 'prey_capture_penalty' in base_config:
            config.prey_capture_penalty = base_config['prey_capture_penalty']
        if 'predator_capture_reward' in base_config:
            config.predator_capture_reward = base_config['predator_capture_reward']
        
        return config
    
    def _generate_predator_traits(self) -> List[PredatorTraits]:
        """Generate unique traits for each predator in a pack"""
        traits = []
        for predator_idx in range(self.predators_per_pack):
            trait = self.trait_generator.generate_traits(
                predator_idx, self.predators_per_pack, self.config
            )
            traits.append(trait)
        return traits
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments and return initial states"""
        self.step_count = 0
        
        # Initialize positions
        self._initialize_positions()
        
        # Initialize headings and velocities
        self.prey_headings = self.rng.uniform(0, 2*np.pi, (self.batch_size,))
        self.predator_headings = self.rng.uniform(0, 2*np.pi, (self.total_predators,))
        
        self.prey_velocities = np.zeros((self.batch_size, 2), dtype=np.float32)
        self.predator_velocities = np.zeros((self.total_predators, 2), dtype=np.float32)
        
        # Initialize energy systems
        self.prey_energy_system.initialize(self.batch_size)
        for system in self.predator_energy_systems:
            # Each system handles all predators of that type across all environments
            system.initialize(self.batch_size)
        
        # Initialize pack coordination systems (one per environment)
        self.pack_systems = []
        for env_idx in range(self.batch_size):
            # Each environment has its own pack system
            predator_indices = [env_idx * self.predators_per_pack + i 
                              for i in range(self.predators_per_pack)]
            env_traits = [self.predator_traits[i] for i in range(self.predators_per_pack)]
            self.pack_systems.append(
                PackCoordinationSystem(self.predators_per_pack, env_traits)
            )
        
        # Initialize termination tracking
        self.prey_done = np.zeros(self.batch_size, dtype=bool)
        self.predator_done = np.zeros(self.total_predators, dtype=bool)
        self.prey_captured = np.zeros(self.batch_size, dtype=bool)
        self.prey_escaped = np.zeros(self.batch_size, dtype=bool)
        self.prey_starved = np.zeros(self.batch_size, dtype=bool)
        self.predator_starved = np.zeros(self.total_predators, dtype=bool)
        
        # Reset statistics
        for key in self.stats:
            self.stats[key][:] = 0
        
        return self._get_observations()
    
    def _initialize_positions(self):
        """Initialize positions for prey, predators, and food"""
        # Prey positions (center of each environment)
        self.prey_positions = np.column_stack([
            self.rng.uniform(200, self.config.screen_width - 200, self.batch_size),
            self.rng.uniform(200, self.config.screen_height - 200, self.batch_size)
        ])
        
        # Predator positions (arranged around prey)
        self.predator_positions = np.zeros((self.total_predators, 2), dtype=np.float32)
        
        for env_idx in range(self.batch_size):
            prey_pos = self.prey_positions[env_idx]
            
            # Position predators in a circle around prey
            for pred_idx in range(self.predators_per_pack):
                global_idx = env_idx * self.predators_per_pack + pred_idx
                
                # Angle around prey
                angle = 2 * np.pi * pred_idx / self.predators_per_pack
                distance = 150.0  # Starting distance from prey
                
                offset = np.array([np.cos(angle), np.sin(angle)]) * distance
                self.predator_positions[global_idx] = prey_pos + offset
        
        # Food positions
        self.food_positions = np.column_stack([
            self.rng.uniform(50, self.config.screen_width - 50, 
                           self.config.num_food * self.batch_size),
            self.rng.uniform(50, self.config.screen_height - 50,
                           self.config.num_food * self.batch_size)
        ])
    
    def step(self, prey_actions: np.ndarray, predator_actions: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute actions for prey and predators.
        
        Args:
            prey_actions: Array of shape [batch_size, action_dim]
            predator_actions: Array of shape [total_predators, action_dim] or [batch_size, predators_per_pack, action_dim]
            
        Returns:
            (prey_states, predator_states): Observations for each agent type
            prey_rewards: Rewards for each prey
            predator_rewards: Rewards for each predator
            info: Additional information
        """
        self.step_count += 1
        
        # Ensure correct action dimensions
        if predator_actions.ndim == 3:
            # Reshape from [batch, predators_per_pack, action_dim] to [total_predators, action_dim]
            predator_actions = predator_actions.reshape(-1, predator_actions.shape[-1])
        
        # Update pack coordination
        self._update_pack_coordination()
        
        # Process actions
        prey_rewards, prey_info = self._process_prey_actions(prey_actions)
        predator_rewards, predator_info = self._process_predator_actions(predator_actions)
        
        # Update positions
        self._update_positions()
        
        # Handle food consumption
        self._handle_food_consumption(prey_rewards, predator_rewards)
        
        # Handle predator-prey interactions
        self._handle_interactions(prey_rewards, predator_rewards)
        
        # Handle boundaries
        self._handle_boundaries()
        
        # Update energy systems
        self._update_energy_systems(prey_info, predator_info)
        
        # Update done states
        self._update_done_states()
        
        # Apply step penalties
        prey_rewards += self.config.step_penalty
        for pred_idx in range(self.total_predators):
            predator_rewards[pred_idx] += self.config.predator_step_penalty
        
        # Zero rewards for done agents
        prey_rewards[self.prey_done] = 0.0
        predator_rewards[self.predator_done] = 0.0
        
        # Update statistics
        self.stats['steps_survived'] += 1
        
        # Check episode termination
        env_done = self._check_episode_termination()
        
        # Get observations
        prey_states, predator_states = self._get_observations()
        
        # Compile info
        info = self._compile_info(prey_info, predator_info, env_done)
        
        return (prey_states, predator_states), prey_rewards, predator_rewards, info
    
    def _update_pack_coordination(self):
        """Update pack coordination systems for each environment"""
        for env_idx in range(self.batch_size):
            if self.prey_done[env_idx]:
                continue
            
            # Get predator indices for this environment
            pred_start = env_idx * self.predators_per_pack
            pred_end = pred_start + self.predators_per_pack
            pred_indices = np.arange(pred_start, pred_end)
            
            # Get active predators in this environment
            active_predators = pred_indices[~self.predator_done[pred_indices]]
            if len(active_predators) == 0:
                continue
            
            # Get positions and velocities
            pred_positions = self.predator_positions[active_predators]
            prey_position = self.prey_positions[env_idx]
            prey_velocity = self.prey_velocities[env_idx]
            
            # Update pack system
            pack_system = self.pack_systems[env_idx]
            pack_system.update(pred_positions, prey_position, prey_velocity, self.step_count)
            
            # Update coordination statistics
            self.stats['pack_coordination'][env_idx] = pack_system.coordination_level
    
    def _process_prey_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process prey actions"""
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        info = {
            'moving': np.zeros(self.batch_size, dtype=bool),
            'turning': np.zeros(self.batch_size, dtype=bool)
        }
        
        # Continuous actions: [turn_rate, thrust]
        turn_rates = actions[:, 0] * self.config.prey_turn_rate
        thrust = np.clip(actions[:, 1], 0, 1) * self.config.prey_speed
        
        # Update headings
        self.prey_headings += turn_rates
        self.prey_headings = np.mod(self.prey_headings, 2 * np.pi)
        
        # Update velocities
        self.prey_velocities = np.column_stack([
            np.cos(self.prey_headings),
            np.sin(self.prey_headings)
        ]) * thrust[:, np.newaxis]
        
        info['moving'] = thrust > 0.1
        info['turning'] = np.abs(turn_rates) > 0.05
        
        return rewards, info
    
    def _process_predator_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process predator actions with individual traits"""
        rewards = np.zeros(self.total_predators, dtype=np.float32)
        info = {
            'moving': np.zeros(self.total_predators, dtype=bool),
            'turning': np.zeros(self.total_predators, dtype=bool),
            'role_bonus': np.zeros(self.total_predators, dtype=np.float32)
        }
        
        for env_idx in range(self.batch_size):
            if self.prey_done[env_idx]:
                continue
            
            pack_system = self.pack_systems[env_idx]
            
            for pack_idx in range(self.predators_per_pack):
                global_idx = env_idx * self.predators_per_pack + pack_idx
                
                if self.predator_done[global_idx]:
                    continue
                
                # Get predator traits
                traits = self.predator_traits[pack_idx]
                
                # Get actions for this predator
                action = actions[global_idx]
                
                # Apply pack strategy instructions
                strategy_instructions = pack_system.get_strategy_instructions(pack_idx)
                
                # Modify action based on strategy and traits
                effective_turn_rate = action[0] * traits.turn_rate
                effective_thrust = np.clip(action[1], 0, 1) * traits.speed
                
                # Blend with strategy instructions
                caution = strategy_instructions[0]
                aggression = strategy_instructions[1]
                coordination = strategy_instructions[2]
                
                # More cautious predators turn less aggressively
                effective_turn_rate *= (1.0 - caution * 0.5)
                
                # More aggressive predators thrust more
                effective_thrust *= (0.5 + aggression * 0.5)
                
                # Update heading
                self.predator_headings[global_idx] += effective_turn_rate
                self.predator_headings[global_idx] = np.mod(
                    self.predator_headings[global_idx], 2 * np.pi
                )
                
                # Update velocity
                self.predator_velocities[global_idx] = np.array([
                    np.cos(self.predator_headings[global_idx]),
                    np.sin(self.predator_headings[global_idx])
                ]) * effective_thrust
                
                # Record movement info
                info['moving'][global_idx] = effective_thrust > 0.1
                info['turning'][global_idx] = np.abs(effective_turn_rate) > 0.05
                
                # Apply role bonus
                role_bonus = pack_system.get_role_bonus(pack_idx)
                info['role_bonus'][global_idx] = role_bonus
                rewards[global_idx] += role_bonus * self.config.role_specialization_bonus
        
        return rewards, info
    
    def _update_positions(self):
        """Update agent positions"""
        # Update prey positions
        active_prey = ~self.prey_done
        self.prey_positions[active_prey] += self.prey_velocities[active_prey]
        
        # Update predator positions
        active_predator = ~self.predator_done
        self.predator_positions[active_predator] += self.predator_velocities[active_predator]
    
    def _handle_food_consumption(self, prey_rewards: np.ndarray, predator_rewards: np.ndarray):
        """Handle food consumption by prey"""
        for env_idx in range(self.batch_size):
            if self.prey_done[env_idx]:
                continue
            
            prey_pos = self.prey_positions[env_idx]
            
            # Check all food items
            for food_idx in range(self.config.num_food):
                global_food_idx = env_idx * self.config.num_food + food_idx
                food_pos = self.food_positions[global_food_idx]
                
                distance = np.linalg.norm(prey_pos - food_pos)
                if distance < (10.0 + self.config.food_radius):  # Prey radius + food radius
                    # Consume food
                    prey_rewards[env_idx] += 10.0
                    self.prey_energy_system.apply_energy_gain(
                        np.array([env_idx]), self.config.food_energy_gain
                    )
                    self.stats['food_consumed'][env_idx] += 1
                    
                    # Respawn food
                    self._respawn_food(global_food_idx, env_idx)
                    break  # Prey can only eat one food per step
    
    def _handle_interactions(self, prey_rewards: np.ndarray, predator_rewards: np.ndarray):
        """Handle predator-prey interactions"""
        for env_idx in range(self.batch_size):
            if self.prey_done[env_idx]:
                continue
            
            prey_pos = self.prey_positions[env_idx]
            
            # Get predator indices for this environment
            pred_start = env_idx * self.predators_per_pack
            pred_end = pred_start + self.predators_per_pack
            pred_indices = np.arange(pred_start, pred_end)
            
            # Check distances to all predators
            capture_attempts = []
            for pred_idx in pred_indices:
                if self.predator_done[pred_idx]:
                    continue
                
                pred_pos = self.predator_positions[pred_idx]
                distance = np.linalg.norm(prey_pos - pred_pos)
                
                # Base capture radius
                capture_radius = self.config.base_capture_radius
                
                # Adjust based on predator traits
                traits = self.predator_traits[pred_idx % self.predators_per_pack]
                effective_radius = capture_radius * traits.capture_efficiency
                
                if distance < effective_radius:
                    capture_attempts.append((pred_idx, distance, traits.capture_efficiency))
            
            # Handle capture attempts
            if capture_attempts:
                # Sort by distance (closest first)
                capture_attempts.sort(key=lambda x: x[1])
                
                # Calculate total capture probability
                total_efficiency = sum(attempt[2] for attempt in capture_attempts)
                capture_probability = min(1.0, total_efficiency)
                
                # Determine if capture succeeds
                if self.rng.random() < capture_probability:
                    # Successful capture
                    prey_rewards[env_idx] += self.config.prey_capture_penalty
                    self.prey_captured[env_idx] = True
                    self.prey_done[env_idx] = True
                    
                    # Distribute reward among participating predators
                    reward_per_predator = self.config.predator_capture_reward / len(capture_attempts)
                    for pred_idx, _, efficiency in capture_attempts:
                        predator_rewards[pred_idx] += reward_per_predator * efficiency
                        self.stats['prey_captures'][pred_idx] += 1
                    
                    # Update capture efficiency statistic
                    self.stats['capture_efficiency'][env_idx] = capture_probability
                else:
                    # Failed capture attempt - prey escapes
                    prey_rewards[env_idx] += self.config.prey_escape_reward
                    self.stats['prey_escapes'][env_idx] += 1
    
    def _handle_boundaries(self):
        """Handle boundary collisions"""
        # Prey boundaries (bounce)
        for env_idx in range(self.batch_size):
            if self.prey_done[env_idx]:
                continue
            
            pos = self.prey_positions[env_idx]
            vel = self.prey_velocities[env_idx]
            
            # Check boundaries
            if pos[0] < self.config.boundary_margin:
                pos[0] = self.config.boundary_margin
                vel[0] = -vel[0] * 0.5  # Bounce with energy loss
                self.prey_headings[env_idx] = np.pi - self.prey_headings[env_idx]
            elif pos[0] > self.config.screen_width - self.config.boundary_margin:
                pos[0] = self.config.screen_width - self.config.boundary_margin
                vel[0] = -vel[0] * 0.5
                self.prey_headings[env_idx] = np.pi - self.prey_headings[env_idx]
            
            if pos[1] < self.config.boundary_margin:
                pos[1] = self.config.boundary_margin
                vel[1] = -vel[1] * 0.5
                self.prey_headings[env_idx] = -self.prey_headings[env_idx]
            elif pos[1] > self.config.screen_height - self.config.boundary_margin:
                pos[1] = self.config.screen_height - self.config.boundary_margin
                vel[1] = -vel[1] * 0.5
                self.prey_headings[env_idx] = -self.prey_headings[env_idx]
            
            self.prey_positions[env_idx] = pos
            self.prey_velocities[env_idx] = vel
        
        # Predator boundaries (wrap around for simplicity)
        for pred_idx in range(self.total_predators):
            if self.predator_done[pred_idx]:
                continue
            
            pos = self.predator_positions[pred_idx]
            
            if pos[0] < 0:
                pos[0] = self.config.screen_width
            elif pos[0] > self.config.screen_width:
                pos[0] = 0
            
            if pos[1] < 0:
                pos[1] = self.config.screen_height
            elif pos[1] > self.config.screen_height:
                pos[1] = 0
            
            self.predator_positions[pred_idx] = pos
    
    def _update_energy_systems(self, prey_info: Dict, predator_info: Dict):
        """Update energy systems for prey and predators"""
        # Update prey energy
        self.prey_energy_system.apply_movement_cost(
            prey_info['moving'],
            prey_info['turning'],
            self.config.prey_energy_move_cost,
            self.config.prey_energy_turn_cost
        )
        self.prey_starved = self.prey_energy_system.starved

        # Update predator energy
        for pred_idx in range(self.total_predators):
            if self.predator_done[pred_idx]:
                continue

            system = self.predator_energy_systems[pred_idx % self.predators_per_pack]
            env_idx = pred_idx // self.predators_per_pack

            # Create full boolean arrays for the environment
            moving = np.zeros(self.batch_size, dtype=bool)
            moving[env_idx] = predator_info['moving'][pred_idx]
            turning = np.zeros(self.batch_size, dtype=bool)
            turning[env_idx] = predator_info['turning'][pred_idx]

            traits = self.predator_traits[pred_idx % self.predators_per_pack]
            system.apply_movement_cost(
                moving, turning,
                traits.energy_move_cost,
                traits.energy_turn_cost
            )

            # Check starvation for this predator
            if system.starved[env_idx]:
                self.predator_starved[pred_idx] = True

        # Update statistics
        self.stats['prey_starved'][self.prey_starved] += 1
        self.stats['predator_starved'][self.predator_starved] += 1
    
    def _update_done_states(self):
        """Update done states"""
        # Prey are done if captured or starved
        self.prey_done = self.prey_done | self.prey_captured | self.prey_starved
        
        # Predators are done if starved
        self.predator_done = self.predator_done | self.predator_starved
    
    def _check_episode_termination(self) -> np.ndarray:
        """Check if episodes should terminate"""
        env_done = np.zeros(self.batch_size, dtype=bool)
        
        for env_idx in range(self.batch_size):
            # Check if prey is done
            prey_done = self.prey_done[env_idx]
            
            # Check if all predators in this environment are done
            pred_start = env_idx * self.predators_per_pack
            pred_end = pred_start + self.predators_per_pack
            all_predators_done = np.all(self.predator_done[pred_start:pred_end])
            
            # Check step limit
            step_limit_reached = self.step_count >= self.config.max_steps
            
            # Episode ends if prey is done, all predators are done, or step limit reached
            env_done[env_idx] = prey_done or all_predators_done or step_limit_reached
        
        return env_done
    
    def _respawn_food(self, food_idx: int, env_idx: int):
        """Respawn a food item"""
        # Competitive placement: avoid agents
        max_attempts = 10
        for attempt in range(max_attempts):
            new_pos = np.array([
                self.rng.uniform(50, self.config.screen_width - 50),
                self.rng.uniform(50, self.config.screen_height - 50)
            ])
            
            # Check distance to prey
            prey_pos = self.prey_positions[env_idx]
            dist_to_prey = np.linalg.norm(new_pos - prey_pos)
            
            # Check distance to predators
            min_pred_dist = float('inf')
            pred_start = env_idx * self.predators_per_pack
            pred_end = pred_start + self.predators_per_pack
            for pred_idx in range(pred_start, pred_end):
                if not self.predator_done[pred_idx]:
                    pred_pos = self.predator_positions[pred_idx]
                    dist = np.linalg.norm(new_pos - pred_pos)
                    min_pred_dist = min(min_pred_dist, dist)
            
            # Place food if sufficiently distant from agents
            if dist_to_prey > 100 and min_pred_dist > 80:
                self.food_positions[food_idx] = new_pos
                return
            
            if attempt == max_attempts - 1:
                # Last attempt: place randomly
                self.food_positions[food_idx] = new_pos
    
    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get observations for prey and predators"""
        prey_observations = np.zeros((self.batch_size, 6), dtype=np.float32)
        predator_observations = np.zeros((self.total_predators, 6), dtype=np.float32)
        
        # Prey observations
        for env_idx in range(self.batch_size):
            if self.prey_done[env_idx]:
                continue
            
            prey_pos = self.prey_positions[env_idx]
            prey_heading = self.prey_headings[env_idx]
            
            # Get predator positions for this environment
            pred_start = env_idx * self.predators_per_pack
            pred_end = pred_start + self.predators_per_pack
            pred_positions = self.predator_positions[pred_start:pred_end]
            active_predators = ~self.predator_done[pred_start:pred_end]
            
            if np.any(active_predators):
                active_pred_positions = pred_positions[active_predators]
                
                # Calculate center of predator pack
                pack_center = np.mean(active_pred_positions, axis=0)
                
                # Distance and angle to pack center
                rel_pack = pack_center - prey_pos
                dist_pack = np.linalg.norm(rel_pack) / 1000.0
                angle_pack = np.arctan2(rel_pack[1], rel_pack[0]) - prey_heading
                angle_pack = ((angle_pack + np.pi) % (2 * np.pi) - np.pi) / np.pi
                
                # Distance to nearest predator
                distances = np.linalg.norm(active_pred_positions - prey_pos, axis=1)
                nearest_idx = np.argmin(distances)
                dist_nearest = distances[nearest_idx] / 1000.0
                rel_nearest = active_pred_positions[nearest_idx] - prey_pos
                angle_nearest = np.arctan2(rel_nearest[1], rel_nearest[0]) - prey_heading
                angle_nearest = ((angle_nearest + np.pi) % (2 * np.pi) - np.pi) / np.pi
            else:
                dist_pack = 1.0
                angle_pack = 0.0
                dist_nearest = 1.0
                angle_nearest = 0.0
            
            # Energy level
            energy = float(self.prey_energy_system.get_energy_normalized()[env_idx])
            
            # Wall distance
            left = prey_pos[0] / self.config.screen_width
            right = (self.config.screen_width - prey_pos[0]) / self.config.screen_width
            top = prey_pos[1] / self.config.screen_height
            bottom = (self.config.screen_height - prey_pos[1]) / self.config.screen_height
            wall_dist = min(left, right, top, bottom)
            
            prey_observations[env_idx] = [
                dist_pack, angle_pack,
                dist_nearest, angle_nearest,
                energy,
                wall_dist
            ]
        
        # Predator observations
        for pred_idx in range(self.total_predators):
            if self.predator_done[pred_idx]:
                continue
            
            pred_pos = self.predator_positions[pred_idx]
            pred_heading = self.predator_headings[pred_idx]
            env_idx = pred_idx // self.predators_per_pack
            pack_idx = pred_idx % self.predators_per_pack
            
            if not self.prey_done[env_idx]:
                prey_pos = self.prey_positions[env_idx]
                
                # Distance and angle to prey
                rel_prey = prey_pos - pred_pos
                dist_prey = np.linalg.norm(rel_prey) / 1000.0
                angle_prey = np.arctan2(rel_prey[1], rel_prey[0]) - pred_heading
                angle_prey = ((angle_prey + np.pi) % (2 * np.pi) - np.pi) / np.pi
            else:
                dist_prey = 1.0
                angle_prey = 0.0
            
            # Distance to nearest teammate
            pred_start = env_idx * self.predators_per_pack
            pred_end = pred_start + self.predators_per_pack
            teammate_indices = [i for i in range(pred_start, pred_end) 
                              if i != pred_idx and not self.predator_done[i]]
            
            if teammate_indices:
                teammate_positions = self.predator_positions[teammate_indices]
                distances = np.linalg.norm(teammate_positions - pred_pos, axis=1)
                nearest_idx = np.argmin(distances)
                dist_team = distances[nearest_idx] / 1000.0
                rel_team = teammate_positions[nearest_idx] - pred_pos
                angle_team = np.arctan2(rel_team[1], rel_team[0]) - pred_heading
                angle_team = ((angle_team + np.pi) % (2 * np.pi) - np.pi) / np.pi
            else:
                dist_team = 1.0
                angle_team = 0.0
            
            # Energy level
            system = self.predator_energy_systems[pack_idx]
            energy = system.get_energy_normalized()[env_idx]
            
            # Pack coordination level
            pack_system = self.pack_systems[env_idx]
            coordination = pack_system.coordination_level
            
            predator_observations[pred_idx] = [
                dist_prey, angle_prey,
                dist_team, angle_team,
                energy,
                coordination
            ]
        
        return prey_observations, predator_observations
    
    def _compile_info(self, prey_info: Dict, predator_info: Dict, 
                     env_done: np.ndarray) -> Dict[str, Any]:
        """Compile information dictionary"""
        return {
            'step': self.step_count,
            'env_done': env_done.copy(),
            'prey_done': self.prey_done.copy(),
            'predator_done': self.predator_done.copy(),
            'prey_captured': self.prey_captured.copy(),
            'prey_escaped': self.prey_escaped.copy(),
            'prey_starved': self.prey_starved.copy(),
            'predator_starved': self.predator_starved.copy(),
            'stats': {k: v.copy() for k, v in self.stats.items()},
            'prey_energy': self.prey_energy_system.get_energy_normalized().copy(),
            'pack_coordination': [ps.coordination_level for ps in self.pack_systems],
            'current_strategies': [ps.current_strategy for ps in self.pack_systems]
        }
    
    def update_config(self, stage: Optional[CurriculumStage] = None,
                     config: Optional[Dict[str, Any]] = None):
        """Update arena configuration"""
        if stage is not None:
            self.stage = stage
        
        if config is not None:
            self.base_config.update(config)
            self.config = self._create_config(self.base_config)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get arena statistics"""
        return {
            'step': self.step_count,
            'active_prey': np.sum(~self.prey_done),
            'active_predators': np.sum(~self.predator_done),
            'total_captures': np.sum(self.prey_captured),
            'total_escapes': np.sum(self.prey_escaped),
            'mean_pack_coordination': np.mean(self.stats['pack_coordination']),
            'mean_capture_efficiency': np.mean(self.stats['capture_efficiency']),
            'mean_food_consumed': np.mean(self.stats['food_consumed']),
            'prey_starved': np.sum(self.prey_starved),
            'predator_starved': np.sum(self.predator_starved)
        }