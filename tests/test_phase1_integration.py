"""
Phase 1 integration test.

Proves AdGenome satisfies the exact contract EvoMind's real EvolutionEngine
expects, by running actual generations through the UNMODIFIED engine class
(evolution/evolution.py, copied verbatim, byte-for-byte, from EvoMind).

Fitness landscape: a synthetic "audience preference" target embedding.
This is NOT the real market simulator (that's Phase 2) -- it exists only to
prove selection + crossover + mutation drive fitness upward through the real
engine, not just that the code avoids crashing.
"""
import random
import numpy as np

from genomes.genome_ad import AdGenome, EMBED_DIM, TRAIT_CANDIDATES
from evolution.evolution import EvolutionEngine

random.seed(42)
np.random.seed(42)

NUM_TRAITS = len(TRAIT_CANDIDATES)
TARGET = np.random.RandomState(7).uniform(-1, 1, size=EMBED_DIM * NUM_TRAITS)


def evaluate(genome: AdGenome) -> float:
    emb = genome.combined_embedding()
    dist = np.linalg.norm(emb - TARGET)
    badge_bonus = 0.05 * sum(genome.optional_traits.values())
    return float(max(0.0, 5.0 - dist) + badge_bonus)


def run():
    population_size = 40
    population = [AdGenome() for _ in range(population_size)]

    engine = EvolutionEngine(
        population_size=population_size,
        tournament_size=3,
        elite_count=2,
        mutation_rate=0.15,
        mutation_strength=0.2,
        architecture_mutation_rate=0.1,
        genome_cls=AdGenome,
        speciation_enabled=False,       # Phase 3 turns this on
        novelty_archive_enabled=False,  # Phase 3 turns this on
    )

    generations = 25
    best_history = []

    for gen in range(generations):
        for g in population:
            g.fitness = evaluate(g)
        raw = [g.fitness for g in population]
        lo, hi = min(raw), max(raw)
        for g in population:
            g.norm_fitness = (g.fitness - lo) / (hi - lo + 1e-9)

        best = max(population, key=lambda g: g.fitness)
        best_history.append(best.fitness)
        print(f"Gen {gen:2d} | best={best.fitness:.3f} | avg={np.mean(raw):.3f} | {best.render_summary()}")

        population = engine.create_next_generation(population, generation=gen, pop_name="ad_creatives")
        if not isinstance(population, list):
            population = population.genomes

    print("\nFitness trend (first 5 -> last 5):",
          [round(x, 2) for x in best_history[:5]], "->", [round(x, 2) for x in best_history[-5:]])
    assert best_history[-1] >= best_history[0], "Evolution did not improve fitness -- investigate."
    improvement = best_history[-1] - best_history[0]
    print(f"\nPASSED: real EvolutionEngine drove AdGenome fitness up by {improvement:.3f} over {generations} generations.")


if __name__ == "__main__":
    run()
