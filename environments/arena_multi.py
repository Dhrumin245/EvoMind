import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field

# Import curriculum for config
from curriculum.curriculum import CurriculumStage, get_stage_config


@dataclass
class AgentConfig:
    """Configuration for agent properties"""
    speed: float = 2.0
    turn_rate: float = 0.2
    vision_range: float = 300.0
    energy_max: float = 100.0
    energy_initial: float = 100.0
    energy_move_cost: float = 0.1
    energy_turn_cost: float = 0.05
    energy_gain_per_food: float = 30.0
    capture_radius: float = 18.0
    radius: float = 10.0


@dataclass
class ArenaConfig:
    """Configuration for the arena"""
    screen_width: int = 800
    screen_height: int = 600
    num_food: int = 10
    food_respawn_rate: float = 0.05
    food_radius: float = 8.0
    food_energy: float = 20.0
    boundary_margin: int = 20
    max_steps: int = 80

    # Noise configuration
    observation_noise_level: float = 0.0
    action_noise_level: float = 0.0
    dynamics_noise_level: float = 0.0
    noise_schedule_start: int = 0
    noise_schedule_end: int = 1000
    noise_schedule_type: str = 'linear'  # 'linear', 'step', 'exponential'
    
    # Prey configuration
    prey_config: AgentConfig = field(default_factory=lambda: AgentConfig(
        speed=2.0,
        turn_rate=0.2,
        vision_range=250.0,
        energy_max=80.0,
        energy_move_cost=0.08,
        energy_turn_cost=0.04,
        energy_gain_per_food=25.0,
        capture_radius=10.0,
        radius=8.0
    ))
    
    # Predator configuration
    predator_config: AgentConfig = field(default_factory=lambda: AgentConfig(
        speed=2.5,
        turn_rate=0.15,
        vision_range=350.0,
        energy_max=120.0,
        energy_move_cost=0.12,
        energy_turn_cost=0.06,
        energy_gain_per_food=40.0,
        capture_radius=15.0,
        radius=12.0
    ))


class EnergySystem:
    """Manages energy and survival mechanics for agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.energy = None
        self.alive = None
        self.starved = None

    
    def initialize(self, num_agents: int):
        """Initialize energy system for a batch of agents"""
        self.energy = np.full(num_agents, self.config.energy_initial, dtype=np.float32)
        self.alive = np.ones(num_agents, dtype=bool)
        self.starved = np.zeros(num_agents, dtype=bool)
    
    def apply_movement_cost(self, moving: np.ndarray, turning: np.ndarray):
        """Apply energy costs for movement and turning"""
        if self.energy is None:
            return
        
        # Cost for moving
        self.energy[moving] -= self.config.energy_move_cost
        
        # Additional cost for turning
        turning_moving = moving & turning
        self.energy[turning_moving] -= self.config.energy_turn_cost
        
        # Check for starvation
        self.starved = self.energy <= 0
        if self.alive is not None:
            self.alive[self.starved] = False
        
        # Clamp energy to [0, max]
        self.energy = np.clip(self.energy, 0, self.config.energy_max)
    
    def apply_food_consumption(self, agents_idx: np.ndarray):
        """Apply energy gain from food consumption"""
        if self.energy is None:
            return
        
        self.energy[agents_idx] += self.config.energy_gain_per_food
        self.energy = np.clip(self.energy, 0, self.config.energy_max)
    
    def get_energy_normalized(self) -> np.ndarray:
        """Get normalized energy levels (0 to 1)"""
        if self.energy is None:
            return np.array([])
        return self.energy / self.config.energy_max


class PredatorPackSystem:
    """
    Manages predator pack coordination.

    Coordination bonuses are earned through *actual physical cohesion* —
    how tightly clustered pack members are around their shared center of mass.
    A dispersed pack earns little or no bonus, forcing evolved brains to
    actively learn to stay near pack-mates rather than receiving the reward
    for free from spatial proximity to prey alone.
    """

    def __init__(self, pack_size: int = 3):
        self.pack_size = pack_size
        self.pack_assignments = None
        self.pack_coordination = None
        self.num_predators_per_env = None
        self.batch_size = None

    def initialize(self, num_predators: int, num_predators_per_env: Optional[int] = None, batch_size: Optional[int] = None):
        """Initialize pack assignments"""
        self.num_predators_per_env = num_predators_per_env or num_predators
        self.batch_size = batch_size or 1

        # Calculate packs per environment
        packs_per_env = (self.num_predators_per_env + self.pack_size - 1) // self.pack_size
        total_packs = packs_per_env * self.batch_size

        self.pack_assignments = np.zeros(num_predators, dtype=int)

        # Assign predators to packs within each environment
        for env_idx in range(self.batch_size):
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env
            pack_start = env_idx * packs_per_env

            for i in range(self.num_predators_per_env):
                predator_idx = predator_start + i
                self.pack_assignments[predator_idx] = pack_start + (i // self.pack_size)

        # Initialize pack coordination state
        self.pack_coordination = {
            'target_prey': np.full(total_packs, -1, dtype=int),
            'strategy': np.zeros(total_packs, dtype=int),  # 0=random, 1=encircle, 2=chase
            # cohesion: actual spatial tightness of the pack [0, 1].
            # 1.0 = all members at same point, 0.0 = spread >= MAX_COHESION_DIST pixels apart.
            'cohesion': np.zeros(total_packs, dtype=np.float32)
        }
        # Distance (pixels from pack center) at which cohesion reaches 0
        self.MAX_COHESION_DIST: float = 150.0

    def update_pack_strategy(self, predator_positions: np.ndarray,
                           prey_positions: np.ndarray, prey_alive: np.ndarray,
                           num_prey_per_env: int, num_predators_per_env: int, batch_size: int):
        """Update pack hunting strategies per environment"""
        if self.pack_assignments is None or self.pack_coordination is None:
            return

        packs_per_env = (num_predators_per_env + self.pack_size - 1) // self.pack_size

        for env_idx in range(batch_size):
            # Get indices for this environment
            prey_start = env_idx * num_prey_per_env
            prey_end = prey_start + num_prey_per_env
            predator_start = env_idx * num_predators_per_env
            predator_end = predator_start + num_predators_per_env
            pack_start = env_idx * packs_per_env

            # Get prey alive in this environment
            env_prey_alive = prey_alive[prey_start:prey_end]
            env_prey_positions = prey_positions[prey_start:prey_end]
            env_predator_positions = predator_positions[predator_start:predator_end]

            for pack_local_idx in range(packs_per_env):
                pack_global_idx = pack_start + pack_local_idx

                # Get predators in this pack
                pack_predator_start = pack_local_idx * self.pack_size
                pack_predator_end = min(pack_predator_start + self.pack_size, num_predators_per_env)
                pack_predators = env_predator_positions[pack_predator_start:pack_predator_end]

                if len(pack_predators) == 0:
                    continue

                # Find closest alive prey in this environment
                pack_center = np.mean(pack_predators, axis=0)

                # ── Spatial cohesion (brain must actually cluster to earn bonus) ──
                if len(pack_predators) > 1:
                    member_distances = np.linalg.norm(pack_predators - pack_center, axis=1)
                    mean_spread = float(np.mean(member_distances))
                else:
                    mean_spread = 0.0  # Single predator: perfect cohesion
                cohesion = float(np.clip(1.0 - mean_spread / self.MAX_COHESION_DIST, 0.0, 1.0))
                self.pack_coordination['cohesion'][pack_global_idx] = cohesion

                if np.any(env_prey_alive):
                    alive_prey_positions = env_prey_positions[env_prey_alive]
                    prey_distances = np.linalg.norm(alive_prey_positions - pack_center, axis=1)
                    closest_prey_local = np.argmin(prey_distances)

                    # Update target (global prey index)
                    alive_indices = np.where(env_prey_alive)[0]
                    self.pack_coordination['target_prey'][pack_global_idx] = prey_start + alive_indices[closest_prey_local]

                    # Update strategy based on distance
                    if prey_distances[closest_prey_local] < 100:
                        self.pack_coordination['strategy'][pack_global_idx] = 1  # Encircle
                    else:
                        self.pack_coordination['strategy'][pack_global_idx] = 2  # Chase
                else:
                    self.pack_coordination['target_prey'][pack_global_idx] = -1
                    self.pack_coordination['strategy'][pack_global_idx] = 0  # Random

    def get_coordination_bonus(self, predator_idx: int) -> float:
        """
        Get coordination bonus for a predator.

        The base bonus is determined by the current hunting strategy, but is
        scaled by the pack's *actual spatial cohesion* — how tightly clustered
        the members are around their center of mass.  A dispersed pack receives
        a near-zero bonus, incentivising evolved brains to stay close to
        pack-mates rather than hunting solo.
        """
        if self.pack_assignments is None or self.pack_coordination is None:
            return 0.0

        pack_idx = self.pack_assignments[predator_idx]
        strategy = self.pack_coordination['strategy'][pack_idx]
        cohesion = float(self.pack_coordination['cohesion'][pack_idx])

        # Base bonus by strategy (max values); scaled to zero when pack is dispersed
        if strategy == 1:   # Encircle
            base = 0.3
        elif strategy == 2:  # Chase
            base = 0.2
        else:               # Random / no prey
            return 0.0

        return base * cohesion


class DeterministicRNG:
    """Deterministic random number generator for reproducibility"""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.Generator(np.random.PCG64(seed)) if seed is not None else None
        self.global_rng = np.random.default_rng(seed)

    def uniform(self, low: float | np.ndarray, high: float | np.ndarray, size: Tuple[int, ...],
                deterministic: bool = True) -> np.ndarray:
        """Generate uniform random numbers"""
        if deterministic and self.rng is not None:
            return self.rng.uniform(low, high, size)
        return self.global_rng.uniform(low, high, size)

    def random(self, size: Tuple[int, ...], deterministic: bool = True) -> np.ndarray:
        """Generate random numbers in [0, 1)"""
        return self.uniform(0.0, 1.0, size, deterministic)

    def choice(self, a: int, size: Tuple[int, ...], deterministic: bool = True) -> np.ndarray:
        """Random choice"""
        if deterministic and self.rng is not None:
            return self.rng.choice(a, size)
        return self.global_rng.choice(a, size)


class NoiseInjector:
    """Systematic robustness testing"""

    def __init__(self, rng: Optional[DeterministicRNG] = None):
        self.rng = rng or DeterministicRNG()

    def inject_observation_noise(self, obs, noise_level=0.1):
        return obs + self.rng.random(obs.shape, deterministic=True) * noise_level

    def inject_action_noise(self, actions, noise_level=0.05):
        return actions + self.rng.random(actions.shape, deterministic=True) * noise_level

    def inject_dynamics_noise(self, positions, noise_level=0.1):
        noise = self.rng.random(positions.shape, deterministic=True) - 0.5  # -0.5 to 0.5
        return noise * noise_level


class MultiAgentArena:
    """
    Advanced multi-agent arena with curriculum support, energy systems,
    predator packs, and proper termination semantics.
    """
    
    def __init__(self,
                 batch_size: int,
                 num_prey_per_env: int = 1,
                 num_predators_per_env: int = 1,
                 stage: CurriculumStage = CurriculumStage.FORAGING,
                 config: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None,
                 deterministic: bool = True,
                 predator_pack_size: int = 3):
        """
        Initialize the multi-agent arena.
        
        Args:
            batch_size: Number of parallel environments
            stage: Curriculum stage for difficulty scaling
            config: Override configuration
            seed: Random seed for reproducibility
            deterministic: Whether to use deterministic RNG
            num_prey_per_env: Number of prey agents per environment
            num_predators_per_env: Number of predator agents per environment
            predator_pack_size: Size of predator packs (0 = no packs)
        """
        self.batch_size = batch_size
        self.num_prey_per_env = num_prey_per_env
        self.num_predators_per_env = num_predators_per_env
        self.total_prey = batch_size * num_prey_per_env
        self.total_predators = batch_size * num_predators_per_env
        
        # Load configuration from curriculum
        self.stage = stage
        self.base_config = config or get_stage_config(stage)
        self.arena_config = self._create_arena_config(self.base_config)
        
        # Setup RNG
        self.deterministic = deterministic
        self.rng = DeterministicRNG(seed)

        # Initialize noise injector
        self.noise_injector = NoiseInjector(self.rng)

        # Initialize systems
        self.energy_system_prey = EnergySystem(self.arena_config.prey_config)
        self.energy_system_predator = EnergySystem(self.arena_config.predator_config)
        
        # Initialize predator packs if enabled
        self.predator_pack_size = predator_pack_size
        if predator_pack_size > 0 and num_predators_per_env >= predator_pack_size:
            self.pack_system = PredatorPackSystem(pack_size=predator_pack_size)
        else:
            self.pack_system = None
        
        # State tracking
        self.step_count = 0
        self.obs_step_counter = 0
        self.last_prey_obs: Optional[np.ndarray] = None
        self.last_pred_obs: Optional[np.ndarray] = None
        self.prey_positions: np.ndarray = np.zeros((self.total_prey, 2), dtype=np.float32)
        self.predator_positions: np.ndarray = np.zeros((self.total_predators, 2), dtype=np.float32)
        self.food_positions: np.ndarray = np.zeros((self.arena_config.num_food * self.batch_size, 2), dtype=np.float32)
        self.prey_headings: np.ndarray = np.zeros(self.total_prey, dtype=np.float32)
        self.predator_headings: np.ndarray = np.zeros(self.total_predators, dtype=np.float32)
        self.prey_velocities: np.ndarray = np.zeros((self.total_prey, 2), dtype=np.float32)
        self.predator_velocities: np.ndarray = np.zeros((self.total_predators, 2), dtype=np.float32)

        # Termination tracking
        self.prey_done: np.ndarray = np.zeros(self.total_prey, dtype=bool)
        self.predator_done: np.ndarray = np.zeros(self.total_predators, dtype=bool)
        self.prey_captured: np.ndarray = np.zeros(self.total_prey, dtype=bool)
        self.predator_starved: np.ndarray = np.zeros(self.total_predators, dtype=bool)
        self.prey_starved: np.ndarray = np.zeros(self.total_prey, dtype=bool)

        # Non-adaptive tracking
        self.non_adaptive_timer = np.zeros(self.total_prey, dtype=np.int32)
        self.NON_ADAPTIVE_GRACE = 50  # steps allowed to adapt

        # Reward shaping / novelty (variance > magnitude)
        # Keep event rewards substantial but not saturating at ±5 clip
        self._food_reward_scale = float(self.base_config.get('food_reward_scale', 0.6))
        self._capture_reward_scale = float(self.base_config.get('capture_reward_scale', 0.5))
        self._predator_capture_reward_scale = float(
            self.base_config.get('predator_capture_reward_scale', max(self._capture_reward_scale, 0.7))
        )
        # Use small step penalties so they don't dominate the signal
        self._prey_step_penalty = float(self.base_config.get('step_penalty', -0.01))
        self._pred_step_penalty = float(self.base_config.get('predator_step_penalty', -0.005))

        # Directional shaping coefficients (per pixel improvement) - increased 5x
        self._shaping_food_scale = float(self.base_config.get('shaping_food_scale', 0.015))
        self._shaping_avoid_scale = float(self.base_config.get('shaping_avoid_scale', 0.010))
        self._shaping_hunt_scale = float(self.base_config.get('shaping_hunt_scale', 0.022))

        # Sparse novelty reward - increased to be meaningful
        self._novelty_cell_size = int(self.base_config.get('novelty_cell_size', 60))
        self._novelty_bonus_prey = float(self.base_config.get('novelty_bonus_prey', 0.2))
        self._novelty_bonus_predator = float(self.base_config.get('novelty_bonus_predator', 0.15))
        self._predator_coordination_scale = float(self.base_config.get('predator_coordination_scale', 0.25))
        # Use plain runtime sets for broad Python compatibility.
        self._visited_prey: List[set] = []
        self._visited_predator: List[set] = []

        # Cached distances for shaping (initialized on reset)
        self._prev_prey_food_dist: Optional[np.ndarray] = None
        self._prev_prey_pred_dist: Optional[np.ndarray] = None
        self._prev_pred_prey_dist: Optional[np.ndarray] = None

        # Diagnostic 1: In-Lifetime Recovery Curve
        self.shock_step = 300
        self.diagnostic_logs = [[] for _ in range(self.total_prey)]

        # Pressure Injection #1: Kill non-adaptive agents mid-episode
        self.ADAPTATION_POINT = 200  # Step when goal location changes
        self.adaptation_triggered = False
        self.pre_adaptation_positions = None  # Store positions before adaptation
        
        # Statistics
        self.stats = {
            'prey_food_collected': np.zeros(self.total_prey, dtype=np.int32),
            'predator_food_collected': np.zeros(self.total_predators, dtype=np.int32),
            'prey_captures': np.zeros(self.total_predators, dtype=np.int32),
            'predator_starved': np.zeros(self.total_predators, dtype=np.int32),
            'prey_starved': np.zeros(self.total_prey, dtype=np.int32),
            'steps_survived': np.zeros(batch_size, dtype=np.int32)
        }

        # Metrics contract attributes
        self.prey_alive = np.ones(self.total_prey, dtype=bool)
        self.predator_captures = np.zeros(self.total_predators, dtype=np.int32)
        self.prey_food_collected = np.zeros(self.total_prey, dtype=np.int32)
        self.prey_energy_initial = self.arena_config.prey_config.energy_initial
        self.predator_energy_initial = self.arena_config.predator_config.energy_initial
        self.prey_energy = np.full(self.total_prey, self.prey_energy_initial, dtype=np.float32)
        self.predator_energy = np.full(self.total_predators, self.predator_energy_initial, dtype=np.float32)
        self.prey_config = self.arena_config.prey_config
        self.predator_config = self.arena_config.predator_config
    
    def _create_arena_config(self, base_config: Dict[str, Any]) -> ArenaConfig:
        """Create arena configuration from base config"""
        config = ArenaConfig()

        # Update from base config
        config.screen_width = base_config.get('screen_width', config.screen_width)
        config.screen_height = base_config.get('screen_height', config.screen_height)
        config.num_food = base_config.get('food_count', config.num_food)
        config.max_steps = base_config.get('max_steps', config.max_steps)

        # Update noise configuration
        config.observation_noise_level = base_config.get('observation_noise_level', config.observation_noise_level)
        config.action_noise_level = base_config.get('action_noise_level', config.action_noise_level)
        config.dynamics_noise_level = base_config.get('dynamics_noise_level', config.dynamics_noise_level)
        config.noise_schedule_start = base_config.get('noise_schedule_start', config.noise_schedule_start)
        config.noise_schedule_end = base_config.get('noise_schedule_end', config.noise_schedule_end)
        config.noise_schedule_type = base_config.get('noise_schedule_type', config.noise_schedule_type)

        # Update prey config
        if 'prey_speed' in base_config:
            config.prey_config.speed = base_config['prey_speed']
        if 'prey_energy' in base_config:
            config.prey_config.energy_max = base_config['prey_energy']
            config.prey_config.energy_initial = base_config['prey_energy']

        # Update predator config
        if 'predator_speed' in base_config:
            config.predator_config.speed = base_config['predator_speed']
        if 'predator_energy' in base_config:
            config.predator_config.energy_max = base_config['predator_energy']
            config.predator_config.energy_initial = base_config['predator_energy']
        if 'predator_vision' in base_config:
            config.predator_config.vision_range = base_config['predator_vision']

        # Capture radius from config
        if 'capture_radius' in base_config:
            config.prey_config.capture_radius = base_config['capture_radius']
            config.predator_config.capture_radius = base_config['capture_radius']

        return config
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Reset all environments and return initial states."""
        if seed is not None:
            self.rng = DeterministicRNG(seed)

        self.step_count = 0
        
        # Initialize positions
        self._initialize_positions()
        
        # Initialize headings and velocities
        self.prey_headings = self.rng.uniform(0, 2*np.pi, (self.total_prey,), self.deterministic)
        self.predator_headings = self.rng.uniform(0, 2*np.pi, (self.total_predators,), self.deterministic)
        self.prey_velocities = np.zeros((self.total_prey, 2), dtype=np.float32)
        self.predator_velocities = np.zeros((self.total_predators, 2), dtype=np.float32)
        
        # Initialize energy systems
        self.energy_system_prey.initialize(self.total_prey)
        self.energy_system_predator.initialize(self.total_predators)
        
        # Initialize predator packs
        if self.pack_system:
            self.pack_system.initialize(self.total_predators, num_predators_per_env=self.num_predators_per_env, batch_size=self.batch_size)
        
        # Initialize termination tracking
        self.prey_done = np.zeros(self.total_prey, dtype=bool)
        self.predator_done = np.zeros(self.total_predators, dtype=bool)
        self.prey_captured = np.zeros(self.total_prey, dtype=bool)
        self.predator_starved = np.zeros(self.total_predators, dtype=bool)
        self.prey_starved = np.zeros(self.total_prey, dtype=bool)
        
        # Reset statistics
        for key in self.stats:
            self.stats[key][:] = 0

        # Reset novelty tracking
        self._visited_prey = [set() for _ in range(self.total_prey)]
        self._visited_predator = [set() for _ in range(self.total_predators)]

        # Initialize shaping baselines
        self._prev_prey_food_dist = self._nearest_food_distance_for_prey()
        self._prev_prey_pred_dist = self._nearest_predator_distance_for_prey()
        self._prev_pred_prey_dist = self._nearest_prey_distance_for_predators()
        
        return self._get_observations()

    def _nearest_food_distance_for_prey(self) -> np.ndarray:
        assert self.prey_positions is not None
        assert self.food_positions is not None
        dists = np.full(self.total_prey, np.inf, dtype=np.float32)

        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            food_start = env_idx * self.arena_config.num_food
            food_end = food_start + self.arena_config.num_food

            env_prey = self.prey_positions[prey_start:prey_end]
            env_food = self.food_positions[food_start:food_end]

            diff = env_prey[:, None, :] - env_food[None, :, :]
            env_d = np.linalg.norm(diff, axis=2)
            dists[prey_start:prey_end] = np.min(env_d, axis=1)

        return dists

    def _nearest_predator_distance_for_prey(self) -> np.ndarray:
        assert self.prey_positions is not None
        assert self.predator_positions is not None
        dists = np.full(self.total_prey, np.inf, dtype=np.float32)

        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            pred_start = env_idx * self.num_predators_per_env
            pred_end = pred_start + self.num_predators_per_env

            env_prey = self.prey_positions[prey_start:prey_end]
            env_pred = self.predator_positions[pred_start:pred_end]
            if env_pred.shape[0] == 0:
                continue

            diff = env_prey[:, None, :] - env_pred[None, :, :]
            env_d = np.linalg.norm(diff, axis=2)
            dists[prey_start:prey_end] = np.min(env_d, axis=1)

        return dists

    def _nearest_prey_distance_for_predators(self) -> np.ndarray:
        assert self.prey_positions is not None
        assert self.predator_positions is not None
        dists = np.full(self.total_predators, np.inf, dtype=np.float32)

        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            pred_start = env_idx * self.num_predators_per_env
            pred_end = pred_start + self.num_predators_per_env

            env_prey = self.prey_positions[prey_start:prey_end]
            env_pred = self.predator_positions[pred_start:pred_end]
            if env_prey.shape[0] == 0:
                continue

            diff = env_pred[:, None, :] - env_prey[None, :, :]
            env_d = np.linalg.norm(diff, axis=2)
            dists[pred_start:pred_end] = np.min(env_d, axis=1)

        return dists

    def _apply_shaping_and_novelty(self, prey_rewards: np.ndarray, predator_rewards: np.ndarray):
        """Directional shaping + sparse novelty bonuses to increase reward variance."""
        if self._prev_prey_food_dist is None or self._prev_prey_pred_dist is None or self._prev_pred_prey_dist is None:
            return

        # Directional shaping (distance improvements)
        curr_food = self._nearest_food_distance_for_prey()
        curr_pred = self._nearest_predator_distance_for_prey()
        curr_hunt = self._nearest_prey_distance_for_predators()

        # Prey: closer to food, farther from predators
        food_improve = (self._prev_prey_food_dist - curr_food)
        avoid_improve = (curr_pred - self._prev_prey_pred_dist)

        prey_rewards += self._shaping_food_scale * food_improve
        prey_rewards += self._shaping_avoid_scale * avoid_improve

        # Predator: closer to prey
        hunt_improve = (self._prev_pred_prey_dist - curr_hunt)
        predator_rewards += self._shaping_hunt_scale * hunt_improve

        # Sparse novelty reward (new grid cell)
        cell = self._novelty_cell_size
        if cell > 0:
            for idx in np.where(~self.prey_done)[0]:
                pos = self.prey_positions[idx]
                key = (int(pos[0] // cell), int(pos[1] // cell))
                if key not in self._visited_prey[idx]:
                    self._visited_prey[idx].add(key)
                    prey_rewards[idx] += self._novelty_bonus_prey

            for idx in np.where(~self.predator_done)[0]:
                pos = self.predator_positions[idx]
                key = (int(pos[0] // cell), int(pos[1] // cell))
                if key not in self._visited_predator[idx]:
                    self._visited_predator[idx].add(key)
                    predator_rewards[idx] += self._novelty_bonus_predator

        # Update baselines
        self._prev_prey_food_dist = curr_food
        self._prev_prey_pred_dist = curr_pred
        self._prev_pred_prey_dist = curr_hunt
    
    def _initialize_positions(self):
        """Initialize agent and food positions"""
        # Initialize prey positions (one per environment, then repeat for multi-prey)
        prey_env_pos = self.rng.uniform(
            np.array([50, 50]),
            np.array([self.arena_config.screen_width - 50, self.arena_config.screen_height - 50]),
            (self.batch_size, 2),
            self.deterministic
        )
        self.prey_positions = np.repeat(prey_env_pos, self.num_prey_per_env, axis=0)
        
        # Initialize predator positions (away from prey)
        predator_env_pos = np.zeros((self.batch_size, 2), dtype=np.float32)
        for i in range(self.batch_size):
            valid = False
            attempts = 0
            while not valid and attempts < 10:
                pos = self.rng.uniform(
                    np.array([100, 100]),
                    np.array([self.arena_config.screen_width - 100, self.arena_config.screen_height - 100]),
                    (2,),
                    self.deterministic
                )
                # Ensure predators start at least 200 pixels from prey
                prey_pos = prey_env_pos[i]
                if np.linalg.norm(pos - prey_pos) > 200:
                    predator_env_pos[i] = pos
                    valid = True
                attempts += 1
            if not valid:
                predator_env_pos[i] = self.rng.uniform(
                    np.array([100, 100]),
                    np.array([self.arena_config.screen_width - 100, self.arena_config.screen_height - 100]),
                    (2,),
                    self.deterministic
                )
        
        self.predator_positions = np.repeat(predator_env_pos, self.num_predators_per_env, axis=0)
        
        # Initialize food positions
        self.food_positions = self.rng.uniform(
            np.array([30, 30]),
            np.array([self.arena_config.screen_width - 30, self.arena_config.screen_height - 30]),
            (self.arena_config.num_food * self.batch_size, 2),
            self.deterministic
        )
    
    def step(self, prey_actions: np.ndarray, predator_actions: np.ndarray) -> Tuple[
        Tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute actions for both prey and predators.

        Args:
            prey_actions: Array of shape [total_prey, action_dim] or [total_prey]
            predator_actions: Array of shape [total_predators, action_dim] or [total_predators]

        Returns:
            (prey_states, predator_states): Next states for each agent type
            prey_rewards: Rewards for each prey agent
            predator_rewards: Rewards for each predator agent
            info: Dictionary with additional information
        """
        # Assertions to help type checker
        assert self.prey_positions is not None
        assert self.predator_positions is not None
        assert self.food_positions is not None
        assert self.prey_headings is not None
        assert self.predator_headings is not None
        assert self.prey_velocities is not None
        assert self.predator_velocities is not None
        assert self.prey_done is not None
        assert self.predator_done is not None
        assert self.prey_captured is not None
        assert self.predator_starved is not None
        assert self.prey_starved is not None

        # Early exit if all agents are already done
        if np.all(self.prey_done) and np.all(self.predator_done):
            return self._get_observations(), np.zeros(self.total_prey, dtype=np.float32), np.zeros(self.total_predators, dtype=np.float32), {}

        self.step_count += 1

        # ── Noise magnitudes for this step (computed once, reused below) ──────
        # All three channels default to 0.0, so this is a no-op unless the
        # curriculum stage config sets non-zero noise levels.
        _sched = self.get_current_noise_level(self.step_count)
        _obs_noise_mag      = self.arena_config.observation_noise_level * _sched
        _action_noise_mag   = self.arena_config.action_noise_level      * _sched
        _dynamics_noise_mag = self.arena_config.dynamics_noise_level    * _sched

        # PRESSURE INJECTION #1: Kill non-adaptive agents mid-episode
        if self.step_count == self.ADAPTATION_POINT and not self.adaptation_triggered:
            self._trigger_adaptation()
            self.adaptation_triggered = True

        # Update predator packs
        if self.pack_system:
            self.pack_system.update_pack_strategy(
                self.predator_positions,
                self.prey_positions,
                ~self.prey_done,
                self.num_prey_per_env,
                self.num_predators_per_env,
                self.batch_size
            )

        # ── NOISE INJECTION 1/3: action noise ────────────────────────────────
        # Corrupt action commands before physics so brains must be robust to
        # actuator jitter.  Work on copies to avoid mutating the caller's arrays.
        if _action_noise_mag > 0.0:
            prey_actions     = prey_actions     + self.noise_injector.inject_action_noise(
                prey_actions.astype(np.float32), _action_noise_mag)
            predator_actions = predator_actions + self.noise_injector.inject_action_noise(
                predator_actions.astype(np.float32), _action_noise_mag)

        # Process prey actions
        prey_rewards, prey_info = self._process_prey_actions(prey_actions)
        
        # Process predator actions
        predator_rewards, predator_info = self._process_predator_actions(predator_actions)
        
        # Update positions
        self._update_positions()

        # ── NOISE INJECTION 2/3: dynamics noise ──────────────────────────────
        # Perturb positions symmetrically (±) after the physics update but
        # *before* boundary clamping so agents never leave the arena.
        if _dynamics_noise_mag > 0.0:
            self.prey_positions[~self.prey_done] += (
                self.noise_injector.inject_dynamics_noise(
                    self.prey_positions[~self.prey_done], _dynamics_noise_mag))
            self.predator_positions[~self.predator_done] += (
                self.noise_injector.inject_dynamics_noise(
                    self.predator_positions[~self.predator_done], _dynamics_noise_mag))

        # Reward shaping (uses updated positions)
        self._apply_shaping_and_novelty(prey_rewards, predator_rewards)

        # Delayed pruning for non-adaptive agents
        for i in range(self.total_prey):
            if self.non_adaptive_timer[i] > 0:
                self.non_adaptive_timer[i] -= 1
                if self.non_adaptive_timer[i] == 0:
                    self.prey_done[i] = True
        
        # Handle food consumption
        self._handle_food_consumption(prey_rewards, predator_rewards)
        
        # Handle predator-prey interactions
        self._handle_captures(prey_rewards, predator_rewards)
        
        # Handle boundary collisions
        self._handle_boundaries()
        
        # Update energy systems
        self._update_energy_systems(prey_info, predator_info)
        
        # Update done states
        self._update_done_states()

        # Apply adaptation recovery bonuses (plastic agents that recover faster get rewarded)
        if self.adaptation_triggered and self.pre_adaptation_positions is not None:
            self._apply_adaptation_bonuses(prey_rewards, predator_rewards)

        # Clip extreme per-step rewards (generous range to preserve variance)
        np.clip(prey_rewards, -10.0, 10.0, out=prey_rewards)
        np.clip(predator_rewards, -10.0, 10.0, out=predator_rewards)

        # Zero rewards for done agents
        prey_rewards[self.prey_done] = 0.0
        predator_rewards[self.predator_done] = 0.0
        
        # Update statistics
        self.stats['steps_survived'] += 1
        
        # Check episode termination
        env_done = self._check_episode_termination()
        
        # Get observations
        prey_states, predator_states = self._get_observations()

        # ── NOISE INJECTION 3/3: observation noise ────────────────────────────
        # Add noise to the sensor readings returned to the brains so evolved
        # networks must generalise beyond exact values.
        if _obs_noise_mag > 0.0:
            prey_states     = self.noise_injector.inject_observation_noise(
                prey_states,     _obs_noise_mag)
            predator_states = self.noise_injector.inject_observation_noise(
                predator_states, _obs_noise_mag)

        # Compile info
        info = self._compile_info(prey_info, predator_info, env_done)
        
        return (prey_states, predator_states), prey_rewards, predator_rewards, info
    
    def _process_prey_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process prey actions and return rewards and info"""
        rewards = np.zeros(self.total_prey, dtype=np.float32)
        info = {
            'moving': np.zeros(self.total_prey, dtype=bool),
            'turning': np.zeros(self.total_prey, dtype=bool)
        }
        
        # Handle both discrete and continuous actions
        if actions.ndim == 1:
            # Discrete actions: 0=forward, 1=left, 2=right
            moving = actions == 0
            turning_left = actions == 1
            turning_right = actions == 2
            
            # Update headings
            self.prey_headings[turning_left] -= self.arena_config.prey_config.turn_rate
            self.prey_headings[turning_right] += self.arena_config.prey_config.turn_rate
            
            # Normalize headings
            self.prey_headings = np.mod(self.prey_headings, 2 * np.pi)
            
            # Set velocities for moving prey
            self.prey_velocities[moving] = np.column_stack([
                np.cos(self.prey_headings[moving]),
                np.sin(self.prey_headings[moving])
            ]) * self.arena_config.prey_config.speed
            
            # Stop non-moving prey
            self.prey_velocities[~moving] = 0.0
            
            info['moving'] = moving
            info['turning'] = turning_left | turning_right
        else:
            # Continuous actions: [turn_rate, thrust]
            # For simplicity, assume continuous for now
            turn_rates = actions[:, 0] * self.arena_config.prey_config.turn_rate
            thrust = np.clip(actions[:, 1], 0, 1) * self.arena_config.prey_config.speed
            
            # Update headings
            self.prey_headings += turn_rates
            self.prey_headings = np.mod(self.prey_headings, 2 * np.pi)
            
            # Set velocities
            self.prey_velocities = np.column_stack([
                np.cos(self.prey_headings),
                np.sin(self.prey_headings)
            ]) * thrust[:, np.newaxis]
            
            info['moving'] = thrust > 0
            info['turning'] = np.abs(turn_rates) > 0.01
        
        # Apply step penalty (very small to avoid dominating signal)
        rewards[:] = self._prey_step_penalty
        
        return rewards, info
    
    def _process_predator_actions(self, actions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process predator actions and return rewards and info"""
        rewards = np.zeros(self.total_predators, dtype=np.float32)
        info = {
            'moving': np.zeros(self.total_predators, dtype=bool),
            'turning': np.zeros(self.total_predators, dtype=bool),
            'pack_coordination': np.zeros(self.total_predators, dtype=np.float32)
        }
        
        # Initialize with step penalty
        rewards[:] = self._pred_step_penalty

        # Apply pack coordination bonuses
        if self.pack_system:
            for i in range(self.total_predators):
                info['pack_coordination'][i] = self.pack_system.get_coordination_bonus(i)
                rewards[i] += info['pack_coordination'][i] * self._predator_coordination_scale
        
        # Handle actions (similar to prey)
        if actions.ndim == 1:
            moving = actions == 0
            turning_left = actions == 1
            turning_right = actions == 2
            
            self.predator_headings[turning_left] -= self.arena_config.predator_config.turn_rate
            self.predator_headings[turning_right] += self.arena_config.predator_config.turn_rate
            self.predator_headings = np.mod(self.predator_headings, 2 * np.pi)
            
            self.predator_velocities[moving] = np.column_stack([
                np.cos(self.predator_headings[moving]),
                np.sin(self.predator_headings[moving])
            ]) * self.arena_config.predator_config.speed
            
            self.predator_velocities[~moving] = 0.0
            
            info['moving'] = moving
            info['turning'] = turning_left | turning_right
        else:
            turn_rates = actions[:, 0] * self.arena_config.predator_config.turn_rate
            thrust = np.clip(actions[:, 1], 0, 1) * self.arena_config.predator_config.speed
            
            self.predator_headings += turn_rates
            self.predator_headings = np.mod(self.predator_headings, 2 * np.pi)
            
            self.predator_velocities = np.column_stack([
                np.cos(self.predator_headings),
                np.sin(self.predator_headings)
            ]) * thrust[:, np.newaxis]
            
            info['moving'] = thrust > 0
            info['turning'] = np.abs(turn_rates) > 0.01
        
        return rewards, info
    
    def _update_positions(self):
        """Update agent positions based on velocities"""
        # Assertions to help type checker
        assert self.prey_positions is not None
        assert self.predator_positions is not None
        assert self.prey_velocities is not None
        assert self.predator_velocities is not None
        assert self.prey_done is not None
        assert self.predator_done is not None

        # Update prey positions
        active_prey = ~self.prey_done
        self.prey_positions[active_prey] += self.prey_velocities[active_prey]

        # Update predator positions
        active_predator = ~self.predator_done
        self.predator_positions[active_predator] += self.predator_velocities[active_predator]
    
    def _handle_food_consumption(self, prey_rewards: np.ndarray, predator_rewards: np.ndarray):
        """Handle food consumption by both prey and predators"""
        # Assertions to help type checker
        assert self.prey_positions is not None
        assert self.predator_positions is not None
        assert self.food_positions is not None
        assert self.prey_done is not None
        assert self.predator_done is not None

        # Prey food consumption
        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            food_start = env_idx * self.arena_config.num_food
            food_end = food_start + self.arena_config.num_food

            env_prey_positions = self.prey_positions[prey_start:prey_end]
            env_food_positions = self.food_positions[food_start:food_end]
            env_prey_done = self.prey_done[prey_start:prey_end]

            active_prey_mask = ~env_prey_done
            if not np.any(active_prey_mask):
                continue

            active_prey_pos = env_prey_positions[active_prey_mask]

            # Compute distances: [num_active_prey, num_food]
            diff = active_prey_pos[:, None, :] - env_food_positions[None, :, :]
            food_distances = np.linalg.norm(diff, axis=2)

            # For each active prey, find closest food
            closest_food_indices = np.argmin(food_distances, axis=1)
            min_distances = np.min(food_distances, axis=1)

            for local_prey_idx, (dist, food_idx) in enumerate(zip(min_distances, closest_food_indices)):
                global_prey_idx = prey_start + np.where(active_prey_mask)[0][local_prey_idx]
                if dist < (self.arena_config.prey_config.radius + self.arena_config.food_radius):
                    # Consume food
                    base_food = float(self.base_config.get('food_reward', 10.0))
                    prey_rewards[global_prey_idx] += base_food * self._food_reward_scale
                    self.energy_system_prey.apply_food_consumption(np.array([global_prey_idx]))
                    self.stats['prey_food_collected'][global_prey_idx] += 1

                    # Respawn food
                    closest_food_global = food_start + food_idx
                    self._respawn_food(closest_food_global)

        # Predator food consumption (predators can eat too if configured)
        # This creates survival pressure on predators
        if self.base_config.get('predators_need_food', False):
            for env_idx in range(self.batch_size):
                predator_start = env_idx * self.num_predators_per_env
                predator_end = predator_start + self.num_predators_per_env
                food_start = env_idx * self.arena_config.num_food
                food_end = food_start + self.arena_config.num_food

                env_predator_positions = self.predator_positions[predator_start:predator_end]
                env_food_positions = self.food_positions[food_start:food_end]
                env_predator_done = self.predator_done[predator_start:predator_end]

                active_predator_mask = ~env_predator_done
                if not np.any(active_predator_mask):
                    continue

                active_predator_pos = env_predator_positions[active_predator_mask]

                # Compute distances: [num_active_predators, num_food]
                diff = active_predator_pos[:, None, :] - env_food_positions[None, :, :]
                food_distances = np.linalg.norm(diff, axis=2)

                # For each active predator, find closest food
                closest_food_indices = np.argmin(food_distances, axis=1)
                min_distances = np.min(food_distances, axis=1)

                for local_predator_idx, (dist, food_idx) in enumerate(zip(min_distances, closest_food_indices)):
                    global_predator_idx = predator_start + np.where(active_predator_mask)[0][local_predator_idx]
                    if dist < (self.arena_config.predator_config.radius + self.arena_config.food_radius):
                        base_food = float(self.base_config.get('food_reward', 10.0))
                        predator_rewards[global_predator_idx] += (base_food * self._food_reward_scale) * 0.8
                        self.energy_system_predator.apply_food_consumption(np.array([global_predator_idx]))
                        self.stats['predator_food_collected'][global_predator_idx] += 1

                        # Respawn food
                        closest_food_global = food_start + food_idx
                        self._respawn_food(closest_food_global)
    
    def _handle_captures(self, prey_rewards: np.ndarray, predator_rewards: np.ndarray):
        """Handle predator-prey capture interactions"""
        capture_radius = (
            self.arena_config.prey_config.capture_radius
            + self.arena_config.predator_config.capture_radius
        )

        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env

            prey_indices = np.arange(prey_start, prey_end)
            predator_indices = np.arange(predator_start, predator_end)

            active_prey = prey_indices[~self.prey_done[prey_indices]]
            active_predators = predator_indices[~self.predator_done[predator_indices]]

            if active_prey.size == 0 or active_predators.size == 0:
                continue

            prey_pos = self.prey_positions[active_prey]  # [P,2]
            pred_pos = self.predator_positions[active_predators]  # [Q,2]

            # Pairwise distances [P,Q]
            diff = prey_pos[:, None, :] - pred_pos[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            capture_mask = dists < capture_radius

            if not np.any(capture_mask):
                continue

            # Prey captured if any predator is within radius.
            captured_prey_local = np.any(capture_mask, axis=1)
            captured_prey = active_prey[captured_prey_local]

            if captured_prey.size:
                base_prey_penalty = float(self.base_config.get('prey_capture_penalty', -30.0))
                prey_rewards[captured_prey] += base_prey_penalty * self._capture_reward_scale
                self.prey_captured[captured_prey] = True
                self.prey_done[captured_prey] = True

            # Predator reward: +15 per captured prey within range (count captures).
            pred_capture_counts = np.sum(capture_mask, axis=0).astype(np.int32)  # [Q]
            if np.any(pred_capture_counts > 0):
                base_pred_reward = float(self.base_config.get('predator_capture_reward', 10.0))
                predator_rewards[active_predators] += (
                    base_pred_reward * self._predator_capture_reward_scale
                ) * pred_capture_counts.astype(np.float32)

                capturing_predators = active_predators[pred_capture_counts > 0]
                # One energy boost per predator that captured at least once this step.
                self.energy_system_predator.apply_food_consumption(capturing_predators)
                self.stats['prey_captures'][active_predators] += pred_capture_counts
    
    def _handle_boundaries(self):
        """Handle boundary collisions"""
        # Prey boundaries
        self.prey_positions[:, 0] = np.clip(
            self.prey_positions[:, 0],
            self.arena_config.boundary_margin,
            self.arena_config.screen_width - self.arena_config.boundary_margin
        )
        self.prey_positions[:, 1] = np.clip(
            self.prey_positions[:, 1],
            self.arena_config.boundary_margin,
            self.arena_config.screen_height - self.arena_config.boundary_margin
        )
        
        # Predator boundaries
        self.predator_positions[:, 0] = np.clip(
            self.predator_positions[:, 0],
            self.arena_config.boundary_margin,
            self.arena_config.screen_width - self.arena_config.boundary_margin
        )
        self.predator_positions[:, 1] = np.clip(
            self.predator_positions[:, 1],
            self.arena_config.boundary_margin,
            self.arena_config.screen_height - self.arena_config.boundary_margin
        )
        
        # Food boundaries (wrap around)
        self.food_positions[:, 0] = np.mod(
            self.food_positions[:, 0],
            self.arena_config.screen_width
        )
        self.food_positions[:, 1] = np.mod(
            self.food_positions[:, 1],
            self.arena_config.screen_height
        )
    
    def _update_energy_systems(self, prey_info: Dict[str, Any], predator_info: Dict[str, Any]):
        """Update energy systems for both prey and predators"""
        # Use actual moving/turning masks from action processing so turn costs are charged correctly
        active_prey = ~self.prey_done
        moving_prey = prey_info['moving'] & active_prey
        turning_prey = prey_info['turning'] & active_prey

        self.energy_system_prey.apply_movement_cost(moving_prey, turning_prey)

        # Update predator energy
        active_predator = ~self.predator_done
        moving_predator = predator_info['moving'] & active_predator
        turning_predator = predator_info['turning'] & active_predator

        self.energy_system_predator.apply_movement_cost(moving_predator, turning_predator)
        
        # Update starvation tracking
        if self.energy_system_prey.starved is not None:
            self.prey_starved = self.energy_system_prey.starved
        if self.energy_system_predator.starved is not None:
            self.predator_starved = self.energy_system_predator.starved
        
        # Update statistics
        self.stats['prey_starved'][self.prey_starved] += 1
        self.stats['predator_starved'][self.predator_starved] += 1
    
    def _update_done_states(self):
        """Update done states for agents"""
        # Prey are done if captured or starved
        self.prey_done = self.prey_done | self.prey_captured | self.prey_starved
        
        # Predators are done if starved
        self.predator_done = self.predator_done | self.predator_starved
    
    def _check_episode_termination(self) -> np.ndarray:
        """Check if episodes should terminate"""
        env_done = np.zeros(self.batch_size, dtype=bool)
        
        for env_idx in range(self.batch_size):
            # Get agent indices for this environment
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env
            
            # Check if all prey are done
            all_prey_done = np.all(self.prey_done[prey_start:prey_end])
            
            # Check if all predators are done
            all_predators_done = np.all(self.predator_done[predator_start:predator_end])
            
            # Check step limit
            step_limit_reached = self.step_count >= self.arena_config.max_steps
            
            # Episode ends if all agents are done or step limit reached
            env_done[env_idx] = (all_prey_done and all_predators_done) or step_limit_reached
        
        return env_done
    
    def _respawn_food(self, food_idx: int):
        """Respawn a food item at a new location"""
        self.food_positions[food_idx] = self.rng.uniform(
            np.array([30, 30]),
            np.array([self.arena_config.screen_width - 30, self.arena_config.screen_height - 30]),
            (2,),
            self.deterministic
        )
    
    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get observations for all prey and predators using vectorized operations"""
        # Early exit if all agents are done to prevent unnecessary computation
        if np.all(self.prey_done) and np.all(self.predator_done):
            prey_observations = np.zeros((self.total_prey, 8), dtype=np.float32)
            predator_observations = np.zeros((self.total_predators, 8), dtype=np.float32)
            return prey_observations, predator_observations

        prey_observations = np.zeros((self.total_prey, 8), dtype=np.float32)
        predator_observations = np.zeros((self.total_predators, 8), dtype=np.float32)

        # Prey observations - vectorized per environment
        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            food_start = env_idx * self.arena_config.num_food
            food_end = food_start + self.arena_config.num_food
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env

            env_prey_positions = self.prey_positions[prey_start:prey_end]
            env_food_positions = self.food_positions[food_start:food_end]
            env_predator_positions = self.predator_positions[predator_start:predator_end]
            env_prey_headings = self.prey_headings[prey_start:prey_end]
            env_prey_done = self.prey_done[prey_start:prey_end]

            active_prey_mask = ~env_prey_done
            if not np.any(active_prey_mask):
                continue

            active_prey_pos = env_prey_positions[active_prey_mask]
            active_prey_headings = env_prey_headings[active_prey_mask]

            # Distance and angle to nearest food
            # food_diff = active_prey_pos[:, None, :] - env_food_positions[None, :, :]
            # Randomly sample fixed K food items
            # Distance and angle to nearest K food

            TOP_K_FOOD = 3
            TOP_K_PREDATORS = 3

            # Initialize defaults
            min_food_distances = np.full(len(active_prey_pos), 1000.0, dtype=np.float32)
            food_angles = np.zeros(len(active_prey_pos), dtype=np.float32)

            if len(env_food_positions) > 0:
                food_diff = active_prey_pos[:, None, :] - env_food_positions[None, :, :]
                food_distances = np.linalg.norm(food_diff, axis=2)

                k_food = min(TOP_K_FOOD, food_distances.shape[1])

                if k_food > 0:
                    nearest_food_indices = np.argpartition(
                        food_distances, kth=k_food - 1, axis=1
                    )[:, :k_food]

                    nearest_food_distances = np.take_along_axis(
                        food_distances, nearest_food_indices, axis=1
                    )

                    nearest_food_positions = env_food_positions[nearest_food_indices]
                    # Use closest food among TOP-K
                    best_food_idx = np.argmin(nearest_food_distances, axis=1)
                    min_food_distances = nearest_food_distances[
                        np.arange(len(best_food_idx)), best_food_idx
                    ]

                    best_food_positions = nearest_food_positions[
                        np.arange(len(best_food_idx)), best_food_idx
                    ]
                    food_angle_diff = np.arctan2(
                        best_food_positions[:, 1] - active_prey_pos[:, 1],
                        best_food_positions[:, 0] - active_prey_pos[:, 0]
                    ) - active_prey_headings
                    food_angles = ((food_angle_diff + np.pi) % (2 * np.pi) - np.pi) / np.pi


            # Distance and angle to nearest predator
            # Distance and angle to nearest K food
            # Distance and angle to nearest K predators
            predator_diff = active_prey_pos[:, None, :] - env_predator_positions[None, :, :]
            predator_distances = np.linalg.norm(predator_diff, axis=2)

            k_pred = min(TOP_K_PREDATORS, predator_distances.shape[1])

            nearest_pred_indices = np.argpartition(
                predator_distances, kth=k_pred - 1, axis=1
            )[:, :k_pred]

            nearest_pred_distances = np.take_along_axis(
                predator_distances, nearest_pred_indices, axis=1
            )
            nearest_pred_positions = env_predator_positions[nearest_pred_indices]

            best_pred_idx = np.argmin(nearest_pred_distances, axis=1)
            min_predator_distances = nearest_pred_distances[
                np.arange(len(best_pred_idx)), best_pred_idx
            ]

            best_pred_positions = nearest_pred_positions[
                np.arange(len(best_pred_idx)), best_pred_idx
            ]
            predator_angle_diff = np.arctan2(
                best_pred_positions[:, 1] - active_prey_pos[:, 1],
                best_pred_positions[:, 0] - active_prey_pos[:, 0]
            ) - active_prey_headings
            predator_angles = ((predator_angle_diff + np.pi) % (2 * np.pi) - np.pi) / np.pi

            # Wall distances
            left = active_prey_pos[:, 0] / self.arena_config.screen_width
            right = (self.arena_config.screen_width - active_prey_pos[:, 0]) / self.arena_config.screen_width
            top = active_prey_pos[:, 1] / self.arena_config.screen_height
            bottom = (self.arena_config.screen_height - active_prey_pos[:, 1]) / self.arena_config.screen_height
            wall_distances = np.minimum(np.minimum(left, right), np.minimum(top, bottom))

            # Energy, speed, and bias
            active_prey_indices = np.where(active_prey_mask)[0]
            global_prey_indices = prey_start + active_prey_indices
            energies = self.energy_system_prey.get_energy_normalized()[global_prey_indices]
            speeds = np.linalg.norm(self.prey_velocities[global_prey_indices], axis=1) / self.arena_config.prey_config.speed
            bias = np.ones(len(active_prey_indices), dtype=np.float32)

            # Fill observations
            prey_observations[global_prey_indices] = np.column_stack([
                min_food_distances / 1000.0, food_angles,
                min_predator_distances / 1000.0, predator_angles,
                wall_distances, energies, speeds, bias
            ])

        # Predator observations - vectorized per environment
        for env_idx in range(self.batch_size):
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            food_start = env_idx * self.arena_config.num_food
            food_end = food_start + self.arena_config.num_food

            env_predator_positions = self.predator_positions[predator_start:predator_end]
            env_prey_positions = self.prey_positions[prey_start:prey_end]
            env_food_positions = self.food_positions[food_start:food_end]
            env_predator_headings = self.predator_headings[predator_start:predator_end]
            env_predator_done = self.predator_done[predator_start:predator_end]

            active_predator_mask = ~env_predator_done
            if not np.any(active_predator_mask):
                continue

            active_predator_pos = env_predator_positions[active_predator_mask]
            active_predator_headings = env_predator_headings[active_predator_mask]

            # Distance and angle to nearest prey
            if len(env_prey_positions) > 0:
                prey_diff = active_predator_pos[:, None, :] - env_prey_positions[None, :, :]
                prey_distances = np.linalg.norm(prey_diff, axis=2)
                nearest_prey_indices = np.argmin(prey_distances, axis=1)
                min_prey_distances = np.min(prey_distances, axis=1)

                nearest_prey_positions = env_prey_positions[nearest_prey_indices]
                prey_angle_diff = np.arctan2(
                    nearest_prey_positions[:, 1] - active_predator_pos[:, 1],
                    nearest_prey_positions[:, 0] - active_predator_pos[:, 0]
                ) - active_predator_headings
                prey_angles = ((prey_angle_diff + np.pi) % (2 * np.pi) - np.pi) / np.pi
            else:
                min_prey_distances = np.full(len(active_predator_pos), 1000.0, dtype=np.float32)  # Large distance
                prey_angles = np.zeros(len(active_predator_pos), dtype=np.float32)

            # Distance and angle to nearest food (for predators that need food)
            if len(env_food_positions) > 0:
                food_diff = active_predator_pos[:, None, :] - env_food_positions[None, :, :]
                food_distances = np.linalg.norm(food_diff, axis=2)
                nearest_food_indices = np.argmin(food_distances, axis=1)
                min_food_distances = np.min(food_distances, axis=1)

                nearest_food_positions = env_food_positions[nearest_food_indices]
                food_angle_diff = np.arctan2(
                    nearest_food_positions[:, 1] - active_predator_pos[:, 1],
                    nearest_food_positions[:, 0] - active_predator_pos[:, 0]
                ) - active_predator_headings
                food_angles = ((food_angle_diff + np.pi) % (2 * np.pi) - np.pi) / np.pi
            else:
                min_food_distances = np.full(len(active_predator_pos), 1000.0, dtype=np.float32)
                food_angles = np.zeros(len(active_predator_pos), dtype=np.float32)

            # Wall distances
            left = active_predator_pos[:, 0] / self.arena_config.screen_width
            right = (self.arena_config.screen_width - active_predator_pos[:, 0]) / self.arena_config.screen_width
            top = active_predator_pos[:, 1] / self.arena_config.screen_height
            bottom = (self.arena_config.screen_height - active_predator_pos[:, 1]) / self.arena_config.screen_height
            wall_distances = np.minimum(np.minimum(left, right), np.minimum(top, bottom))

            # Energy, speed, and pack coordination
            active_predator_indices = np.where(active_predator_mask)[0]
            global_predator_indices = predator_start + active_predator_indices
            energies = self.energy_system_predator.get_energy_normalized()[global_predator_indices]
            speeds = np.linalg.norm(self.predator_velocities[global_predator_indices], axis=1) / self.arena_config.predator_config.speed

            pack_coords = np.zeros(len(active_predator_indices), dtype=np.float32)
            if self.pack_system:
                for local_idx, global_idx in enumerate(global_predator_indices):
                    pack_coords[local_idx] = self.pack_system.get_coordination_bonus(global_idx)

            # Fill observations
            predator_observations[global_predator_indices] = np.column_stack([
                min_prey_distances / 1000.0, prey_angles,
                min_food_distances / 1000.0, food_angles,
                wall_distances, energies, speeds, pack_coords
            ])

        return prey_observations, predator_observations
    
    def _compile_info(self, prey_info: Dict, predator_info: Dict,
                     env_done: np.ndarray) -> Dict[str, Any]:
        """Compile information dictionary"""
        # Get success signals and metrics for the current episode
        success_signals = self.get_episode_success_signals()
        per_env = self._compute_per_env_success_signals()

        # Get success definition for current stage
        success_definition = self.get_success_definition(self.stage.name if hasattr(self.stage, 'name') else str(self.stage))

        return {
            'step': self.step_count,
            'env_done': env_done.copy(),
            'prey_done': self.prey_done.copy(),
            'predator_done': self.predator_done.copy(),
            'prey_captured': self.prey_captured.copy(),
            'predator_starved': self.predator_starved.copy(),
            'prey_starved': self.prey_starved.copy(),
            'stats': {k: v.copy() for k, v in self.stats.items()},
            'prey_energy': self.energy_system_prey.get_energy_normalized().copy(),
            'predator_energy': self.energy_system_predator.get_energy_normalized().copy(),
            'prey_info': prey_info,
            'predator_info': predator_info,
            # Metrics contract additions
            'success_signals': success_signals,
            'success_definition': success_definition,
            'energy_usage': {
                'prey_energy_used': success_signals['prey_energy_used'],
                'predator_energy_used': success_signals['predator_energy_used']
            },
            'energy_usage_per_env': {
                'prey_energy_used': per_env['prey_energy_used'],
                'predator_energy_used': per_env['predator_energy_used']
            },
            'novelty_hits': success_signals['novelty_hits'],
            'novelty_hits_per_env': per_env['novelty_hits'],
            'adaptation_recovery': {
                'prey_adaptation_recovery': success_signals['prey_adaptation_recovery'],
                'predator_adaptation_recovery': success_signals['predator_adaptation_recovery']
            },
            'adaptation_recovery_per_env': {
                'prey_adaptation_recovery': per_env['prey_adaptation_recovery'],
                'predator_adaptation_recovery': per_env['predator_adaptation_recovery']
            }
        }
    
    def update_config(self, stage: Optional[CurriculumStage] = None, 
                     config: Optional[Dict[str, Any]] = None):
        """Update arena configuration"""
        if stage is not None:
            self.stage = stage
        
        if config is not None:
            self.base_config.update(config)
            self.arena_config = self._create_arena_config(self.base_config)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get arena statistics"""
        return {
            'step': self.step_count,
            'active_prey': np.sum(~self.prey_done),
            'active_predators': np.sum(~self.predator_done),
            'total_prey_captured': np.sum(self.prey_captured),
            'total_predator_starved': np.sum(self.predator_starved),
            'total_prey_starved': np.sum(self.prey_starved),
            'mean_prey_food': np.mean(self.stats['prey_food_collected']),
            'mean_predator_food': np.mean(self.stats['predator_food_collected']),
            'mean_prey_captures': np.mean(self.stats['prey_captures'])
        }

    def _trigger_adaptation(self):
        """Trigger adaptation by changing goal locations and killing non-adaptive agents"""
        print(f"ADAPTATION TRIGGERED at step {self.step_count}!")

        # Store pre-adaptation positions for recovery measurement
        self.pre_adaptation_positions = {
            'prey': self.prey_positions.copy(),
            'predator': self.predator_positions.copy()
        }

        # Change goal locations - move all food to new random positions
        self.food_positions = self.rng.uniform(
            np.array([50, 50]),
            np.array([self.arena_config.screen_width - 50, self.arena_config.screen_height - 50]),
            (self.arena_config.num_food * self.batch_size, 2),
            self.deterministic
        )

        # Kill non-adaptive agents (those far from food)
        for env_idx in range(self.batch_size):
            food_start = env_idx * self.arena_config.num_food
            food_end = food_start + self.arena_config.num_food
            env_food_positions = self.food_positions[food_start:food_end]

            # Kill prey that are far from any food
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            for prey_idx in range(prey_start, prey_end):
                if self.prey_done[prey_idx]:
                    continue

                prey_pos = self.prey_positions[prey_idx]
                distances_to_food = np.linalg.norm(env_food_positions - prey_pos, axis=1)
                min_distance = np.min(distances_to_food)

                # Kill if too far from food (non-adaptive)
                if min_distance > 200:  # Threshold for being "non-adaptive"
                    self.non_adaptive_timer[prey_idx] = self.NON_ADAPTIVE_GRACE
                    if self.step_count < 5:
                        print(f"Prey {prey_idx} killed for being non-adaptive (dist={min_distance:.1f})")

            # Kill predators that are far from prey or food
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env

            for pred_idx in range(predator_start, predator_end):
                if self.predator_done[pred_idx]:
                    continue

                pred_pos = self.predator_positions[pred_idx]

                # Distance to nearest prey
                env_prey_positions = self.prey_positions[prey_start:prey_end]
                if np.any(~self.prey_done[prey_start:prey_end]):
                    active_prey_pos = env_prey_positions[~self.prey_done[prey_start:prey_end]]
                    prey_distances = np.linalg.norm(active_prey_pos - pred_pos, axis=1)
                    min_prey_distance = np.min(prey_distances) if len(prey_distances) > 0 else 1000.0
                else:
                    min_prey_distance = 1000.0

                # Distance to nearest food
                food_distances = np.linalg.norm(env_food_positions - pred_pos, axis=1)
                min_food_distance = np.min(food_distances)

                # Kill if too far from both prey and food
                if min_prey_distance > 250 and min_food_distance > 250:
                    self.predator_done[pred_idx] = True
                    if self.step_count < 5:
                        print(f"Predator {pred_idx} killed for being non-adaptive (prey_dist={min_prey_distance:.1f}, food_dist={min_food_distance:.1f})")

    def _apply_adaptation_bonuses(self, prey_rewards: np.ndarray, predator_rewards: np.ndarray):
        """Apply bonuses to agents that recovered well after adaptation"""
        if self.pre_adaptation_positions is None:
            return

        steps_since_adaptation = self.step_count - self.ADAPTATION_POINT

        for env_idx in range(self.batch_size):
            food_start = env_idx * self.arena_config.num_food
            food_end = food_start + self.arena_config.num_food
            env_food_positions = self.food_positions[food_start:food_end]

            # Reward prey that moved closer to food
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            for prey_idx in range(prey_start, prey_end):
                if self.prey_done[prey_idx]:
                    continue

                current_pos = self.prey_positions[prey_idx]
                pre_pos = self.pre_adaptation_positions['prey'][prey_idx]

                # Distance to nearest food before and after
                pre_distances = np.linalg.norm(env_food_positions - pre_pos, axis=1)
                current_distances = np.linalg.norm(env_food_positions - current_pos, axis=1)

                pre_min_dist = np.min(pre_distances)
                current_min_dist = np.min(current_distances)

                # Reward for getting closer to food
                distance_improvement = pre_min_dist - current_min_dist
                if distance_improvement > 0:
                    bonus = min(distance_improvement * 0.05, 2.0)  # Scale bonus
                    prey_rewards[prey_idx] += bonus

            # Reward predators that moved closer to prey or food
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env

            for pred_idx in range(predator_start, predator_end):
                if self.predator_done[pred_idx]:
                    continue

                current_pos = self.predator_positions[pred_idx]
                pre_pos = self.pre_adaptation_positions['predator'][pred_idx]

                # Distance to nearest prey before and after
                env_prey_positions = self.prey_positions[prey_start:prey_end]
                if np.any(~self.prey_done[prey_start:prey_end]):
                    active_prey_pos = env_prey_positions[~self.prey_done[prey_start:prey_end]]
                    pre_prey_distances = np.linalg.norm(active_prey_pos - pre_pos, axis=1)
                    current_prey_distances = np.linalg.norm(active_prey_pos - current_pos, axis=1)

                    pre_min_prey_dist = np.min(pre_prey_distances) if len(pre_prey_distances) > 0 else 1000.0
                    current_min_prey_dist = np.min(current_prey_distances) if len(current_prey_distances) > 0 else 1000.0
                else:
                    pre_min_prey_dist = 1000.0
                    current_min_prey_dist = 1000.0

                # Distance to nearest food before and after
                pre_food_distances = np.linalg.norm(env_food_positions - pre_pos, axis=1)
                current_food_distances = np.linalg.norm(env_food_positions - current_pos, axis=1)

                pre_min_food_dist = np.min(pre_food_distances)
                current_min_food_dist = np.min(current_food_distances)

                # Reward for getting closer to prey or food
                prey_improvement = pre_min_prey_dist - current_min_prey_dist
                food_improvement = pre_min_food_dist - current_min_food_dist

                max_improvement = max(prey_improvement, food_improvement)
                if max_improvement > 0:
                    bonus = min(max_improvement * 0.01, 0.8)
                    predator_rewards[pred_idx] += bonus

    def close(self):
        """Cleanup resources (placeholder for compatibility)"""
        pass

    def get_episode_success_signals(self) -> Dict[str, Any]:
        """Get success signals and metrics for the current episode"""
        # Calculate success signals
        prey_alive = np.sum(self.prey_alive)
        predator_captures = np.sum(self.predator_captures)
        food_collected = np.sum(self.prey_food_collected)

        # Energy usage summaries
        prey_energy_used = np.sum(self.prey_energy_initial - self.prey_energy)
        predator_energy_used = np.sum(self.predator_energy_initial - self.predator_energy)

        # Novelty hit counts based on unique grid cells visited
        prey_novelty_hits = sum(len(s) for s in self._visited_prey) if self._visited_prey else 0
        predator_novelty_hits = sum(len(s) for s in self._visited_predator) if self._visited_predator else 0
        novelty_hits = prey_novelty_hits + predator_novelty_hits

        # Adaptation recovery score (how well agents recovered from energy depletion)
        prey_recovery = np.mean(np.maximum(0, self.prey_energy - self.prey_config.energy_max * 0.1))
        predator_recovery = np.mean(np.maximum(0, self.predator_energy - self.predator_config.energy_max * 0.1))

        return {
            'prey_alive': int(prey_alive),
            'predator_captures': int(predator_captures),
            'food_collected': int(food_collected),
            'prey_energy_used': float(prey_energy_used),
            'predator_energy_used': float(predator_energy_used),
            'novelty_hits': int(novelty_hits),
            'prey_novelty_hits': int(prey_novelty_hits),
            'predator_novelty_hits': int(predator_novelty_hits),
            'prey_adaptation_recovery': float(prey_recovery),
            'predator_adaptation_recovery': float(predator_recovery)
        }

    def _compute_per_env_success_signals(self) -> Dict[str, np.ndarray]:
        """Compute per-environment success signals and energy usage."""
        prey_alive_per_env = np.zeros(self.batch_size, dtype=np.int32)
        predator_captures_per_env = np.zeros(self.batch_size, dtype=np.int32)
        food_collected_per_env = np.zeros(self.batch_size, dtype=np.int32)
        prey_energy_used_per_env = np.zeros(self.batch_size, dtype=np.float32)
        predator_energy_used_per_env = np.zeros(self.batch_size, dtype=np.float32)
        prey_recovery_per_env = np.zeros(self.batch_size, dtype=np.float32)
        predator_recovery_per_env = np.zeros(self.batch_size, dtype=np.float32)
        prey_novelty_per_env = np.zeros(self.batch_size, dtype=np.int32)
        predator_novelty_per_env = np.zeros(self.batch_size, dtype=np.int32)

        for env_idx in range(self.batch_size):
            prey_start = env_idx * self.num_prey_per_env
            prey_end = prey_start + self.num_prey_per_env
            predator_start = env_idx * self.num_predators_per_env
            predator_end = predator_start + self.num_predators_per_env

            prey_alive_per_env[env_idx] = int(np.sum(self.prey_alive[prey_start:prey_end]))
            predator_captures_per_env[env_idx] = int(np.sum(self.predator_captures[predator_start:predator_end]))
            food_collected_per_env[env_idx] = int(np.sum(self.prey_food_collected[prey_start:prey_end]))

            prey_energy_used_per_env[env_idx] = float(
                np.sum(self.prey_energy_initial - self.prey_energy[prey_start:prey_end])
            )
            predator_energy_used_per_env[env_idx] = float(
                np.sum(self.predator_energy_initial - self.predator_energy[predator_start:predator_end])
            )

            prey_recovery_per_env[env_idx] = float(
                np.mean(np.maximum(0, self.prey_energy[prey_start:prey_end] - self.prey_config.energy_max * 0.1))
            )
            predator_recovery_per_env[env_idx] = float(
                np.mean(np.maximum(0, self.predator_energy[predator_start:predator_end] - self.predator_config.energy_max * 0.1))
            )

            prey_novelty_per_env[env_idx] = int(
                sum(len(self._visited_prey[idx]) for idx in range(prey_start, prey_end))
            )
            predator_novelty_per_env[env_idx] = int(
                sum(len(self._visited_predator[idx]) for idx in range(predator_start, predator_end))
            )

        return {
            'prey_alive': prey_alive_per_env,
            'predator_captures': predator_captures_per_env,
            'food_collected': food_collected_per_env,
            'prey_energy_used': prey_energy_used_per_env,
            'predator_energy_used': predator_energy_used_per_env,
            'prey_adaptation_recovery': prey_recovery_per_env,
            'predator_adaptation_recovery': predator_recovery_per_env,
            'prey_novelty_hits': prey_novelty_per_env,
            'predator_novelty_hits': predator_novelty_per_env,
            'novelty_hits': prey_novelty_per_env + predator_novelty_per_env
        }

    def get_success_definition(self, stage: str) -> Dict[str, Any]:
        """Get configurable per-episode success definition based on stage"""
        stage_configs = {
            'FORAGING': {
                'prey_success': lambda signals: signals['food_collected'] > 0,
                'predator_success': lambda signals: signals['predator_captures'] > 0,
                'description': 'Basic survival and resource collection'
            },
            'PRECISION': {
                'prey_success': lambda signals: signals['food_collected'] >= 2,
                'predator_success': lambda signals: signals['predator_captures'] >= 1,
                'description': 'Refined control and efficiency'
            },
            'SCARCITY': {
                'prey_success': lambda signals: signals['food_collected'] >= 3 and signals['prey_alive'] > 0,
                'predator_success': lambda signals: signals['predator_captures'] >= 1 and signals['predator_energy_used'] < 50,
                'description': 'Coordination under resource constraints'
            },
            'THREAT': {
                'prey_success': lambda signals: signals['prey_alive'] > 0 and signals['food_collected'] > 0,
                'predator_success': lambda signals: signals['predator_captures'] >= 2,
                'description': 'Direct competition and evasion'
            },
            'ADVERSARIAL': {
                'prey_success': lambda signals: signals['prey_alive'] > 0,
                'predator_success': lambda signals: signals['predator_captures'] >= 3,
                'description': 'Advanced pack dynamics and strategy'
            }
        }

        return stage_configs.get(stage, stage_configs['FORAGING'])

    def get_robustness_metrics(self) -> Dict[str, Any]:
        """Get robustness metrics for fitness evaluation.

        robustness_score reflects how much noise is actually active across all
        three channels.  When all base levels are 0.0 (the default) the score
        stays at 1.0 regardless of the schedule position.
        """
        noise_level = self.get_current_noise_level(self.step_count)

        # Effective magnitudes for each active channel at this step
        obs_mag      = self.arena_config.observation_noise_level * noise_level
        action_mag   = self.arena_config.action_noise_level      * noise_level
        dynamics_mag = self.arena_config.dynamics_noise_level    * noise_level

        # Mean active noise across channels that have non-zero base levels.
        # Uses only configured channels so a single-channel setup isn't
        # deflated by the two silent channels.
        base_levels = [
            self.arena_config.observation_noise_level,
            self.arena_config.action_noise_level,
            self.arena_config.dynamics_noise_level,
        ]
        active_mags = [obs_mag, action_mag, dynamics_mag]
        configured  = [m for b, m in zip(base_levels, active_mags) if b > 0.0]
        mean_active_noise = float(np.mean(configured)) if configured else 0.0

        # robustness_score: 1.0 at silence, approaches 0 as noise saturates
        robustness_score = max(0.0, 1.0 - mean_active_noise)

        base_performance = self.get_episode_success_signals()
        return {
            'noise_level': float(noise_level),
            'obs_noise_mag': float(obs_mag),
            'action_noise_mag': float(action_mag),
            'dynamics_noise_mag': float(dynamics_mag),
            'robustness_score': float(robustness_score),
            'performance_under_noise': base_performance['food_collected'] * robustness_score,
            'noise_penalty': mean_active_noise * 0.1
        }

    def get_current_noise_level(self, step: int) -> float:
        """Get current noise level based on schedule"""
        if step < self.arena_config.noise_schedule_start:
            return 0.0
        elif step >= self.arena_config.noise_schedule_end:
            return 1.0
        else:
            progress = (step - self.arena_config.noise_schedule_start) / (self.arena_config.noise_schedule_end - self.arena_config.noise_schedule_start)
            if self.arena_config.noise_schedule_type == 'linear':
                return progress
            elif self.arena_config.noise_schedule_type == 'step':
                return 1.0 if progress > 0.5 else 0.0
            elif self.arena_config.noise_schedule_type == 'exponential':
                return progress ** 2
            else:
                return progress

    def apply_noise_modes(self, observations: np.ndarray, actions: np.ndarray, positions: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply noise to observations, actions, and dynamics"""
        noise_level = self.get_current_noise_level(step)

        obs_noise = self.arena_config.observation_noise_level * noise_level
        action_noise = self.arena_config.action_noise_level * noise_level
        dynamics_noise = self.arena_config.dynamics_noise_level * noise_level

        noisy_obs = self.noise_injector.inject_observation_noise(observations, obs_noise)
        noisy_actions = self.noise_injector.inject_action_noise(actions, action_noise)
        noisy_positions = positions + self.noise_injector.inject_dynamics_noise(positions, dynamics_noise)

        return noisy_obs, noisy_actions, noisy_positions

    def get_episode_info(self) -> Dict[str, Any]:
        """Get episode-level metrics for evaluation"""
        # Provide comprehensive info collection for metrics contract
        robustness_metrics = self.get_robustness_metrics()
        success_signals = self.get_episode_success_signals()
        success_definition = self.get_success_definition(self.stage.name if hasattr(self.stage, 'name') else str(self.stage))
        per_env = self._compute_per_env_success_signals()
        env_done = self._check_episode_termination()
        return {
            'prey_energy': self.energy_system_prey.get_energy_normalized().copy(),
            'predator_energy': self.energy_system_predator.get_energy_normalized().copy(),
            'env_done': env_done.copy(),
            'robustness_metrics': robustness_metrics,
            'success_signals': success_signals,
            'success_definition': success_definition,
            'success_signals_per_env': per_env,
            'energy_usage': {
                'prey_energy_used': success_signals['prey_energy_used'],
                'predator_energy_used': success_signals['predator_energy_used']
            },
            'energy_usage_per_env': {
                'prey_energy_used': per_env['prey_energy_used'],
                'predator_energy_used': per_env['predator_energy_used']
            },
            'novelty_hits': success_signals['novelty_hits'],
            'novelty_hits_per_env': per_env['novelty_hits'],
            'adaptation_recovery': {
                'prey_adaptation_recovery': success_signals['prey_adaptation_recovery'],
                'predator_adaptation_recovery': success_signals['predator_adaptation_recovery']
            },
            'adaptation_recovery_per_env': {
                'prey_adaptation_recovery': per_env['prey_adaptation_recovery'],
                'predator_adaptation_recovery': per_env['predator_adaptation_recovery']
            }
        }
