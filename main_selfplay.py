from population import Population
from genome_prey import PreyGenome
from genome_predator import PredatorGenome
from self_play import evaluate_self_play
import random

prey_pop = Population(100).genomes
pred_pop = Population(100).genomes

for gen in range(1000):
    for prey in prey_pop:
        predator = random.choice(pred_pop)
        prey.fitness, _ = evaluate_self_play(prey, predator)

    for pred in pred_pop:
        prey = random.choice(prey_pop)
        _, pred.fitness = evaluate_self_play(prey, pred)

    # prey_pop = evolve(prey_pop)
    # pred_pop = evolve(pred_pop)

    print(f"Gen {gen} | Prey {max(g.fitness for g in prey_pop):.2f} "
          f"| Pred {max(g.fitness for g in pred_pop):.2f}")
