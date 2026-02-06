from environments.arena_multi import MultiAgentArena
import numpy as np

def evaluate_self_play(prey_population, predator_population, steps=500, batch=64):
    """
    Evaluate co-evolution between populations
    prey_population: list of PreyGenome objects
    predator_population: list of PredatorGenome objects
    """
    
    # Track fitness for all individuals
    prey_fitnesses = np.zeros(len(prey_population))
    predator_fitnesses = np.zeros(len(predator_population))
    
    # Sample opponents (population-level co-evolution)
    num_opponents = min(5, len(predator_population))
    
    # Create ONE arena and reuse it for all evaluations
    arena = MultiAgentArena(batch)
    
    for prey_idx, prey_genome in enumerate(prey_population):
        # Sample multiple predators to compete against
        predator_opponents = np.random.choice(
            predator_population, 
            size=num_opponents,
            replace=False
        )
        
        total_prey_fitness = 0
        
        for predator_genome in predator_opponents:
            prey_state, pred_state = arena.reset()
            
            step_fitness = 0.0
            survival_bonus = 0.0
            last_positions = None
            step = 0
            
            for step in range(steps):
                prey_actions = prey_genome.act_batch_gpu(prey_state)
                pred_actions = predator_genome.act_batch_gpu(pred_state)
                
                (prey_state, pred_state), r_prey, r_pred, done = arena.step(
                    prey_actions, pred_actions
                )
                
                # Enhanced fitness calculation
                survival_bonus += 0.01  # Reward for staying alive
                
                # Calculate anti-camping penalty for predators
                # (Add logic to detect if predators aren't moving)
                
                step_fitness += r_prey.mean() + survival_bonus
                
                if all(done.values()):
                    break
            
            # Add time-based bonus (surviving longer is better for prey)
            time_bonus = step / steps * 0.5
            total_prey_fitness += step_fitness + time_bonus
        
        prey_fitnesses[prey_idx] = total_prey_fitness / num_opponents
    
    # Similar evaluation for predators (reuse the same arena)
    for pred_idx, predator_genome in enumerate(predator_population):
        # Sample multiple prey to compete against
        prey_opponents = np.random.choice(
            prey_population,
            size=num_opponents,
            replace=False
        )
        
        total_pred_fitness = 0
        
        for prey_genome in prey_opponents:
            prey_state, pred_state = arena.reset()
            
            step_fitness = 0.0
            coordination_bonus = 0.0
            
            for step in range(steps):
                prey_actions = prey_genome.act_batch_gpu(prey_state)
                pred_actions = predator_genome.act_batch_gpu(pred_state)
                
                (prey_state, pred_state), r_prey, r_pred, done = arena.step(
                    prey_actions, pred_actions
                )
                
                # Enhanced predator fitness
                # Add coordination bonus (reward when multiple predators are close to same prey)
                # Add anti-camping penalty
                # Add time penalty (faster captures are better)
                
                time_penalty = -0.001 * step  # Penalize taking too long
                step_fitness += r_pred.mean() + time_penalty + coordination_bonus
                
                if all(done.values()):
                    capture_bonus = 1.0  # Bonus for successful capture
                    step_fitness += capture_bonus
                    break
            
            total_pred_fitness += step_fitness
        
        predator_fitnesses[pred_idx] = total_pred_fitness / num_opponents
    
    return prey_fitnesses, predator_fitnesses


def evolve_population(population, fitnesses, mutation_rate=0.1):
    """Evolve population based on fitness"""
    # Tournament selection
    new_population = []
    
    for _ in range(len(population)):
        # Select parents
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        parent1 = population[idx1] if fitnesses[idx1] > fitnesses[idx2] else population[idx2]
        
        idx3, idx4 = np.random.choice(len(population), 2, replace=False)
        parent2 = population[idx3] if fitnesses[idx3] > fitnesses[idx4] else population[idx4]
        
        # Crossover
        child = parent1.crossover(parent2)
        
        # Mutate (weights AND architecture)
        if np.random.random() < mutation_rate:
            child.mutate_weights()
        
        if np.random.random() < mutation_rate * 0.5:  # Lower probability for architecture mutation
            child.mutate_architecture()
        
        new_population.append(child)
    
    return new_population