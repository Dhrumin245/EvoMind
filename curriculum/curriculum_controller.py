"""
Curriculum Controller Module
Adaptive curriculum progression based on learning progress.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from dataclasses import dataclass, field
import json
import time

# Import from curriculum.py
from curriculum.curriculum import CurriculumStage, get_stage_config, get_stage_transition_thresholds, get_stage_transition_graph


@dataclass
class StagePerformance:
    """Track performance metrics for a curriculum stage"""
    stage: CurriculumStage
    start_time: float
    end_time: Optional[float] = None
    generations_in_stage: int = 0
    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)
    success_rate_history: List[float] = field(default_factory=list)
    best_fitness_achieved: float = 0.0
    stagnation_count: int = 0
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def mean_fitness(self) -> float:
        """Get mean fitness across history"""
        if not self.fitness_history:
            return 0.0
        return float(np.mean(self.fitness_history[-50:]))  # Last 50 generations
    
    @property
    def mean_diversity(self) -> float:
        """Get mean diversity across history"""
        if not self.diversity_history:
            return 0.0
        return float(np.mean(self.diversity_history[-50:]))
    
    @property
    def mean_success_rate(self) -> float:
        """Get mean success rate across history"""
        if not self.success_rate_history:
            return 0.0
        return float(np.mean(self.success_rate_history[-50:]))
    
    @property
    def learning_speed(self) -> float:
        """Calculate learning speed (fitness improvement per generation)"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        # Calculate slope of fitness improvement
        recent_history = self.fitness_history[-20:]  # Last 20 generations
        if len(recent_history) < 2:
            return 0.0
        
        x = np.arange(len(recent_history))
        slope, _ = np.polyfit(x, recent_history, 1)
        return float(slope)
    
    def is_stagnant(self, threshold: float = 0.01, window: int = 10) -> bool:
        """Check if learning has stagnated"""
        if len(self.fitness_history) < window:
            return False
        
        recent_fitness = self.fitness_history[-window:]
        improvement = max(recent_fitness) - min(recent_fitness)
        return improvement < threshold


class TransitionReason(Enum):
    """Reasons for curriculum stage transition"""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    STAGNATION = "stagnation"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    MANUAL = "manual"
    RESET = "reset"
    FAILURE = "failure"


@dataclass
class TransitionRecord:
    """Record of a curriculum stage transition"""
    from_stage: CurriculumStage
    to_stage: CurriculumStage
    reason: TransitionReason
    timestamp: float
    performance_stats: Dict[str, float]
    generation: int


class CurriculumController:
    """
    Adaptive controller for curriculum progression.
    Monitors learning progress and decides when to advance or regress stages.
    """
    
    def __init__(self,
                 initial_stage: CurriculumStage = CurriculumStage.FORAGING,
                 adaptation_rate: float = 0.1,
                 enable_self_analysis: bool = True):

        self.current_stage = initial_stage
        self.initial_stage = initial_stage
        self.adaptation_rate = adaptation_rate
        self.enable_self_analysis = enable_self_analysis

        # Performance tracking
        self.current_performance = StagePerformance(stage=initial_stage, start_time=time.time())
        self.performance_history: Dict[CurriculumStage, StagePerformance] = {}
        self.transition_history: List[TransitionRecord] = []

        # Stage transition graph
        self.allowed_transitions = get_stage_transition_graph()

        # Adaptive thresholds
        self.base_thresholds = get_stage_transition_thresholds()
        self.adapted_thresholds = self.base_thresholds.copy()

        # Memory of what worked
        self.successful_transitions: List[Tuple[CurriculumStage, CurriculumStage]] = []
        self.failed_transitions: List[Tuple[CurriculumStage, CurriculumStage]] = []
        self.collapse_conditions: Dict[CurriculumStage, Dict[str, Any]] = {}

        # Current generation counter
        self.generation = 0
        
    def update(self, 
               population_stats: Dict[str, float],
               diversity_score: float,
               success_rate: float) -> Optional[CurriculumStage]:
        """
        Update curriculum based on current performance.
        Returns new stage if transition occurred, None otherwise.
        
        Args:
            population_stats: Dictionary with 'mean', 'max', 'min', 'std' of fitness
            diversity_score: Population diversity metric (0-1)
            success_rate: Task success rate (0-1)
        """
        self.generation += 1
        
        # Update current performance tracking
        self.current_performance.generations_in_stage += 1
        self.current_performance.fitness_history.append(population_stats['mean'])
        self.current_performance.diversity_history.append(diversity_score)
        self.current_performance.success_rate_history.append(success_rate)
        
        if population_stats['max'] > self.current_performance.best_fitness_achieved:
            self.current_performance.best_fitness_achieved = population_stats['max']
        
        # Check for stagnation
        if self.current_performance.is_stagnant():
            self.current_performance.stagnation_count += 1
        else:
            self.current_performance.stagnation_count = 0
        
        # Decide on transition
        new_stage = self._evaluate_transition(population_stats, diversity_score, success_rate)
        
        if new_stage and new_stage != self.current_stage:
            self._perform_transition(new_stage, population_stats, diversity_score, success_rate)
            return new_stage
        
        return None
    
    def _evaluate_transition(self,
                            population_stats: Dict[str, float],
                            diversity_score: float,
                            success_rate: float) -> Optional[CurriculumStage]:
        """
        Evaluate whether to transition to a different stage.
        Returns new stage if transition should occur, None otherwise.
        """
        current_thresholds = self.adapted_thresholds.get(self.current_stage, {})
        upper_thresholds, lower_thresholds = self._get_hysteresis_thresholds(current_thresholds)
        
        # Check if we should advance
        if self._should_advance(population_stats, diversity_score, success_rate, upper_thresholds):
            next_stage = self._get_next_stage(self.current_stage)
            if next_stage:
                return next_stage
        
        # Check if we should regress (learning is too hard)
        if self._should_regress(population_stats, diversity_score, success_rate, lower_thresholds):
            prev_stage = self._get_previous_stage(self.current_stage)
            if prev_stage:
                return prev_stage
        
        return None
    
    def _should_advance(self,
                       population_stats: Dict[str, float],
                       diversity_score: float,
                       success_rate: float,
                       thresholds: Dict[str, float]) -> bool:
        """Determine if population should advance to next stage"""
        
        # Check performance thresholds
        if population_stats['mean'] < thresholds.get('min_mean_fitness', 0):
            return False
        
        if success_rate < thresholds.get('min_success_rate', 0):
            return False
        
        if diversity_score < thresholds.get('min_diversity', 0):
            return False
        
        # Check stagnation
        stagnation_limit = thresholds.get('max_stagnation', 30)
        if self.current_performance.stagnation_count > stagnation_limit:
            # Too stagnant, might need to advance or change approach
            return population_stats['mean'] > thresholds.get('min_mean_fitness', 0) * 1.5
        
        # Check learning is still active
        learning_speed = self.current_performance.learning_speed
        if learning_speed > 0:  # Still learning
            return True
        elif learning_speed == 0 and self.current_performance.generations_in_stage > 20:
            # Plateaued but met thresholds
            return True
        
        return False
    
    def _should_regress(self,
                       population_stats: Dict[str, float],
                       diversity_score: float,
                       success_rate: float,
                       thresholds: Dict[str, float]) -> bool:
        """Determine if population should regress to easier stage"""
        
        # Check for collapse
        if diversity_score < 0.05:  # Extreme diversity collapse
            return True
        
        if success_rate < 0.1:  # Complete failure
            return True
        
        # Regress if performance is below hysteresis lower bounds
        if population_stats['mean'] < thresholds.get('min_mean_fitness', 0):
            return True

        if success_rate < thresholds.get('min_success_rate', 0):
            return True

        if diversity_score < thresholds.get('min_diversity', 0):
            return True

        # Check if stuck for too long without improvement
        if (self.current_performance.stagnation_count > 
            thresholds.get('max_stagnation', 30) * 2):
            return True
        
        # Too many generations without reaching threshold
        if (self.current_performance.generations_in_stage > 100 and
            population_stats['mean'] < thresholds.get('min_mean_fitness', 0) * 0.5):
            return True
        
        return False

    def _get_hysteresis_thresholds(self,
                                   thresholds: Dict[str, float],
                                   lower_factor: float = 0.7) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Build upper and lower thresholds to prevent oscillation."""
        upper = thresholds.copy()
        lower = thresholds.copy()

        for key in ['min_mean_fitness', 'min_success_rate', 'min_diversity']:
            if key in lower:
                lower[key] = lower[key] * lower_factor

        return upper, lower
    
    def _get_next_stage(self, current: CurriculumStage) -> Optional[CurriculumStage]:
        """Get next stage in curriculum, if exists"""
        try:
            stages = list(CurriculumStage)
            current_idx = stages.index(current)
            if current_idx + 1 < len(stages):
                return stages[current_idx + 1]
        except ValueError:
            pass
        return None
    
    def _get_previous_stage(self, current: CurriculumStage) -> Optional[CurriculumStage]:
        """Get previous stage in curriculum, if exists"""
        try:
            stages = list(CurriculumStage)
            current_idx = stages.index(current)
            if current_idx - 1 >= 0:
                return stages[current_idx - 1]
        except ValueError:
            pass
        return None
    
    def _perform_transition(self,
                           new_stage: CurriculumStage,
                           population_stats: Dict[str, float],
                           diversity_score: float,
                           success_rate: float):
        """Execute stage transition with proper bookkeeping"""
        
        # Determine transition reason
        if new_stage.value > self.current_stage.value:
            reason = TransitionReason.PERFORMANCE_THRESHOLD
        else:
            reason = TransitionReason.STAGNATION
        
        # Record the transition
        transition_record = TransitionRecord(
            from_stage=self.current_stage,
            to_stage=new_stage,
            reason=reason,
            timestamp=time.time(),
            performance_stats={
                'mean_fitness': population_stats['mean'],
                'max_fitness': population_stats['max'],
                'diversity': diversity_score,
                'success_rate': success_rate,
                'generations_in_stage': self.current_performance.generations_in_stage
            },
            generation=self.generation
        )
        self.transition_history.append(transition_record)
        
        # Finalize current performance record
        self.current_performance.end_time = time.time()
        self.performance_history[self.current_stage] = self.current_performance
        
        # Store memory of transition
        if reason == TransitionReason.PERFORMANCE_THRESHOLD:
            self.successful_transitions.append((self.current_stage, new_stage))
        else:
            self.failed_transitions.append((self.current_stage, new_stage))
        
        # Update to new stage
        old_stage = self.current_stage
        self.current_stage = new_stage
        
        # Create new performance tracker
        self.current_performance = StagePerformance(
            stage=new_stage,
            start_time=time.time()
        )
        
        # Adapt thresholds based on experience
        self._adapt_thresholds(old_stage, new_stage, transition_record)
        
        print(f"[Curriculum] Transition: {old_stage.name} â†’ {new_stage.name} "
              f"(Reason: {reason.value}, Gen: {self.generation})")
    
    def _adapt_thresholds(self,
                         from_stage: CurriculumStage,
                         to_stage: CurriculumStage,
                         transition: TransitionRecord):
        """Adapt thresholds based on transition experience"""
        if not self.enable_self_analysis:
            return
        
        # Get the stage we transitioned to (or from)
        target_stage = to_stage if to_stage.value > from_stage.value else from_stage
        
        if target_stage in self.adapted_thresholds:
            # Adjust based on difficulty of transition
            stats = transition.performance_stats
            
            # If transition was easy, make thresholds stricter
            if stats['generations_in_stage'] < 20:
                for key in ['min_mean_fitness', 'min_success_rate', 'min_diversity']:
                    if key in self.adapted_thresholds[target_stage]:
                        self.adapted_thresholds[target_stage][key] *= 1.1
            
            # If transition was hard, make thresholds easier
            elif stats['generations_in_stage'] > 50:
                for key in ['min_mean_fitness', 'min_success_rate', 'min_diversity']:
                    if key in self.adapted_thresholds[target_stage]:
                        self.adapted_thresholds[target_stage][key] *= 0.9
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get configuration for current stage"""
        return get_stage_config(self.current_stage)
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate progress report"""
        return {
            'current_stage': self.current_stage.name,
            'current_difficulty': self.current_stage.value,
            'generation': self.generation,
            'generations_in_current_stage': self.current_performance.generations_in_stage,
            'current_performance': {
                'mean_fitness': self.current_performance.mean_fitness,
                'best_fitness': self.current_performance.best_fitness_achieved,
                'mean_diversity': self.current_performance.mean_diversity,
                'mean_success_rate': self.current_performance.mean_success_rate,
                'learning_speed': self.current_performance.learning_speed,
                'stagnation_count': self.current_performance.stagnation_count
            },
            'transition_count': len(self.transition_history),
            'successful_transitions': len(self.successful_transitions),
            'failed_transitions': len(self.failed_transitions),
            'recent_transition': (self.transition_history[-1].to_stage.name 
                                 if self.transition_history else None)
        }
    
    def save_state(self, filepath: str):
        """Save controller state to file"""
        state = {
            'current_stage': self.current_stage.name,
            'generation': self.generation,
            'performance_history': {
                stage.name: {
                    'generations': perf.generations_in_stage,
                    'best_fitness': perf.best_fitness_achieved,
                    'duration': perf.duration
                }
                for stage, perf in self.performance_history.items()
            },
            'transition_history': [
                {
                    'from': trans.from_stage.name,
                    'to': trans.to_stage.name,
                    'reason': trans.reason.value,
                    'generation': trans.generation
                }
                for trans in self.transition_history[-50:]  # Last 50 transitions
            ],
            'adapted_thresholds': self.adapted_thresholds
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load controller state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.current_stage = CurriculumStage[state['current_stage']]
            self.generation = state['generation']
            # Note: More complex state restoration would be needed for full recovery
        except Exception as e:
            print(f"[Curriculum] Failed to load state: {e}")
    
    def reset_to_stage(self, stage: CurriculumStage):
        """Manually reset to a specific stage"""
        transition_record = TransitionRecord(
            from_stage=self.current_stage,
            to_stage=stage,
            reason=TransitionReason.MANUAL,
            timestamp=time.time(),
            performance_stats={},
            generation=self.generation
        )
        self.transition_history.append(transition_record)
        
        self.current_performance.end_time = time.time()
        self.performance_history[self.current_stage] = self.current_performance
        
        self.current_stage = stage
        self.current_performance = StagePerformance(
            stage=stage,
            start_time=time.time()
        )
        
        print(f"[Curriculum] Manual reset to stage: {stage.name}")
