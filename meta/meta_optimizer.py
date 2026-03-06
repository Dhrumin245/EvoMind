import random
from typing import Dict, List, Any, Tuple


class EvolutionModifier:
    """Modify evolution parameters based on experiments"""

    def __init__(self, evolution_engine=None):
        self.evolution_engine = evolution_engine
        self.parameter_space = {
            'mutation_rate': (0.01, 0.5),
            'mutation_strength': (0.01, 0.3),
            'selection_pressure': (1.0, 5.0),
            'novelty_weight': (0.0, 1.0),
        }
        self.modification_history = []

    def optimize_evolution(self, experiment_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Modify evolution based on what worked"""

        # Extract successful strategies
        successful_params = self._extract_success_patterns(experiment_results)

        # Propose modifications
        modifications = []
        for param, value_range in self.parameter_space.items():
            current_value = self._get_current_value(param)
            new_value = self._propose_new_value(
                param,
                current_value,
                successful_params
            )
            modifications.append({
                'parameter': param,
                'old_value': current_value,
                'new_value': new_value,
                'reason': self._explain_modification(param, successful_params)
            })

        return modifications

    def apply_modifications(self, modifications: List[Dict[str, Any]], evolution_engine) -> None:
        """Apply modifications to evolution engine"""
        self.evolution_engine = evolution_engine
        for mod in modifications:
            self._set_param(evolution_engine, mod['parameter'], mod['new_value'])
            self.modification_history.append(mod)

    def rollback_modifications(self, evolution_engine) -> None:
        """Rollback to previous parameter values"""
        if not self.modification_history:
            return

        # Get the last set of modifications
        last_mods = {}
        for mod in reversed(self.modification_history):
            if mod['parameter'] not in last_mods:
                last_mods[mod['parameter']] = mod['old_value']

        # Apply rollback
        for param, old_value in last_mods.items():
            if old_value is not None:
                self._set_param(evolution_engine, param, old_value)

        # Remove the last modifications from history
        self.modification_history = self.modification_history[:-len(last_mods)]

    def _extract_success_patterns(self, experiment_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract successful parameter patterns from experiment results"""
        successful_params = {}

        # Assume experiment_results contains lists of parameter sets and their fitness
        if 'experiments' in experiment_results:
            experiments = experiment_results['experiments']
            if experiments:
                # Find the experiment with highest fitness
                best_experiment = max(experiments, key=lambda x: x.get('fitness', 0))
                successful_params = best_experiment.get('parameters', {})

        # If no experiments, use defaults
        if not successful_params:
            successful_params = {
                'mutation_rate': 0.1,
                'mutation_strength': 0.1,
                'selection_pressure': 2.0,
                'novelty_weight': 0.5
            }

        return successful_params

    def _get_current_value(self, param: str) -> float:
        """Get current value of a parameter from the live evolution engine, with defaults as fallback."""
        defaults = {
            'mutation_rate': 0.1,
            'mutation_strength': 0.1,
            'selection_pressure': 2.0,
            'novelty_weight': 0.5
        }
        if self.evolution_engine is not None:
            if param == 'selection_pressure':
                selector = getattr(self.evolution_engine, 'selector', None)
                if selector is not None:
                    return float(getattr(selector, 'selection_pressure', defaults[param]))
            else:
                value = getattr(self.evolution_engine, param, None)
                if value is not None:
                    return float(value)
        return defaults.get(param, 0.0)

    def _set_param(self, evolution_engine, param: str, value: float) -> None:
        """Set a parameter on the evolution engine, routing selection_pressure to engine.selector."""
        if param == 'selection_pressure':
            selector = getattr(evolution_engine, 'selector', None)
            if selector is not None:
                setattr(selector, 'selection_pressure', value)
        else:
            setattr(evolution_engine, param, value)

    def _propose_new_value(self, param: str, current_value: float, successful_params: Dict[str, float]) -> float:
        """Propose new value for parameter based on successful patterns"""
        min_val, max_val = self.parameter_space[param]

        # If we have a successful value for this parameter, move towards it
        if param in successful_params:
            successful_value = successful_params[param]
            # Move 20% towards the successful value
            adjustment_factor = 0.2
            new_value = current_value + adjustment_factor * (successful_value - current_value)
        else:
            # Small random adjustment
            adjustment = random.uniform(-0.05, 0.05) * current_value
            new_value = current_value + adjustment

        # Clamp to parameter space
        new_value = max(min_val, min(max_val, new_value))

        return round(new_value, 4)

    def _explain_modification(self, param: str, successful_params: Dict[str, float]) -> str:
        """Explain why this parameter is being modified"""
        if param in successful_params:
            successful_value = successful_params[param]
            return f"Adjusting towards successful value {successful_value} from experiments"
        else:
            return "Small exploratory adjustment based on parameter space"
