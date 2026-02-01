from enum import Enum
from typing import Dict, Any, List
import json

class CurriculumStage(Enum):
    FORAGING = 1
    PRECISION = 2
    SCARCITY = 3
    THREAT = 4
    ADVERSARIAL = 5
    
    @classmethod
    def from_name(cls, name: str) -> 'CurriculumStage':
        """Get stage from string name"""
        return cls[name.upper()]


def get_stage_config(stage: CurriculumStage) -> Dict[str, Any]:
    """
    Returns environment configuration per stage with metadata.
    """
    configs = {
        CurriculumStage.FORAGING: dict(
            name=stage.name,
            difficulty=stage.value,
            description="Basic foraging with no penalties",
            wall_penalty=0.0,
            food_reward=10.0,
            step_penalty=0.0,
            predator=False,
            max_steps=80,
            food_count=20,
            initial_learning_rate=0.01
        ),
        CurriculumStage.PRECISION: dict(
            name=stage.name,
            difficulty=stage.value,
            description="Adds wall penalties and step costs",
            wall_penalty=-2.0,
            food_reward=8.0,
            step_penalty=-0.01,
            predator=False,
            max_steps=80,
            food_count=15,
            initial_learning_rate=0.008
        ),
        CurriculumStage.SCARCITY: dict(
            name=stage.name,
            difficulty=stage.value,
            description="Less food, higher rewards",
            wall_penalty=-2.0,
            food_reward=12.0,
            step_penalty=-0.02,
            predator=False,
            max_steps=80,
            food_count=10,
            initial_learning_rate=0.006
        ),
        CurriculumStage.THREAT: dict(
            name=stage.name,
            difficulty=stage.value,
            description="Introduces predators with low threat",
            wall_penalty=-3.0,
            food_reward=15.0,
            step_penalty=-0.03,
            predator=True,
            predator_speed=0.5,
            predator_count=1,
            max_steps=80,
            food_count=8,
            initial_learning_rate=0.005
        ),
        CurriculumStage.ADVERSARIAL: dict(
            name=stage.name,
            difficulty=stage.value,
            description="High-threat predators, severe penalties",
            wall_penalty=-4.0,
            food_reward=18.0,
            step_penalty=-0.04,
            predator=True,
            predator_speed=0.8,
            predator_count=2,
            max_steps=80,
            food_count=6,
            initial_learning_rate=0.003
        )
    }
    
    return configs.get(stage, configs[CurriculumStage.FORAGING]).copy()


def get_all_stages() -> Dict[CurriculumStage, Dict[str, Any]]:
    """Get configuration for all stages"""
    return {stage: get_stage_config(stage) for stage in CurriculumStage}


def get_stage_transition_thresholds() -> Dict[CurriculumStage, Dict[str, float]]:
    """
    Define thresholds for stage transitions.
    These are default values that can be adapted by the controller.
    """
    return {
        CurriculumStage.FORAGING: {
            'min_mean_fitness': 50.0,
            'min_success_rate': 0.7,
            'min_diversity': 0.3,
            'max_stagnation': 20
        },
        CurriculumStage.PRECISION: {
            'min_mean_fitness': 40.0,
            'min_success_rate': 0.6,
            'min_diversity': 0.25,
            'max_stagnation': 25
        },
        CurriculumStage.SCARCITY: {
            'min_mean_fitness': 35.0,
            'min_success_rate': 0.5,
            'min_diversity': 0.2,
            'max_stagnation': 30
        },
        CurriculumStage.THREAT: {
            'min_mean_fitness': 30.0,
            'min_success_rate': 0.4,
            'min_diversity': 0.15,
            'max_stagnation': 35
        },
        CurriculumStage.ADVERSARIAL: {
            'min_mean_fitness': 25.0,
            'min_success_rate': 0.3,
            'min_diversity': 0.1,
            'max_stagnation': 40
        }
    }


def get_stage_transition_graph() -> Dict[CurriculumStage, List[CurriculumStage]]:
    """
    Define the allowed transitions between curriculum stages.
    This creates a directed graph where stages can transition to multiple possible next stages.
    """
    return {
        CurriculumStage.FORAGING: [CurriculumStage.PRECISION, CurriculumStage.FORAGING],
        CurriculumStage.PRECISION: [CurriculumStage.SCARCITY, CurriculumStage.PRECISION, CurriculumStage.FORAGING],
        CurriculumStage.SCARCITY: [CurriculumStage.THREAT, CurriculumStage.SCARCITY, CurriculumStage.PRECISION],
        CurriculumStage.THREAT: [CurriculumStage.ADVERSARIAL, CurriculumStage.THREAT, CurriculumStage.SCARCITY],
        CurriculumStage.ADVERSARIAL: [CurriculumStage.ADVERSARIAL, CurriculumStage.THREAT]
    }


def get_stage_skill_tags() -> Dict[CurriculumStage, List[str]]:
    """
    Define skill tags for each curriculum stage.
    These represent the competencies that should be developed at each stage.
    """
    return {
        CurriculumStage.FORAGING: ["basic_movement", "food_detection", "simple_navigation"],
        CurriculumStage.PRECISION: ["precise_control", "obstacle_avoidance", "efficiency"],
        CurriculumStage.SCARCITY: ["resource_optimization", "strategic_planning", "competition"],
        CurriculumStage.THREAT: ["threat_awareness", "evasion", "coordination"],
        CurriculumStage.ADVERSARIAL: ["advanced_strategy", "pack_hunting", "counter_strategy"]
    }


def get_stage_metadata() -> Dict[CurriculumStage, Dict[str, Any]]:
    """
    Get comprehensive metadata for each stage including skill tags and transition info.
    """
    skill_tags = get_stage_skill_tags()
    transition_graph = get_stage_transition_graph()
    thresholds = get_stage_transition_thresholds()

    metadata = {}
    for stage in CurriculumStage:
        metadata[stage] = {
            'skill_tags': skill_tags.get(stage, []),
            'allowed_transitions': transition_graph.get(stage, []),
            'thresholds': thresholds.get(stage, {}),
            'difficulty_score': stage.value,
            'is_terminal': stage == CurriculumStage.ADVERSARIAL
        }

    return metadata


def stage_to_json(stage: CurriculumStage) -> str:
    """Serialize stage configuration to JSON"""
    config = get_stage_config(stage)
    return json.dumps(config, indent=2, default=str)