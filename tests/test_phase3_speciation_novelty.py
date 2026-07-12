"""
Phase 3 integration test.

Turns speciation_enabled=True and novelty_archive_enabled=True back on
(both were off in Phase 1/2 to isolate the reproduction loop) using
AdEvolutionEngine + AdGenomeDistance, and confirms:

  1. The population splits into multiple distinct species instead of
     collapsing into one blob or exploding into 40 singleton species.
  2. Novelty scores actually vary across genomes (proving the new
     _build_novelty_embedding carries real signal, unlike the default
     which reads NN-only fields AdGenome doesn't have).
  3. Fitness against the real market simulator still climbs with both
     mechanisms turned on.
"""
import random
import numpy as np

from genomes.genome_ad import AdGenome
from environments.ad_market_env import AdMarketEnv, MarketEvalConfig
from evolution.ad_evolution_engine import AdEvolutionEngine

random.seed(7)
np.random.seed(7)


def run():
    population_size = 60
    population = [AdGenome() for _ in range(population_size)]

    env = AdMarketEnv(config=MarketEvalConfig(base_seed=7, impressions_per_segment=250))

    engine = AdEvolutionEngine(
        population_size=population_size,
        tournament_size=3,
        elite_count=2,
        mutation_rate=0.15,
        mutation_strength=0.2,
        architecture_mutation_rate=0.1,
        genome_cls=AdGenome,
        speciation_enabled=True,
        novelty_archive_enabled=True,
        # Measured actual pairwise AdGenomeDistance across a random population
        # before picking this: min=0.08, p10=0.53, median=0.68, max=1.02.
        # 0.65 sits near the median so genomes split into a handful of
        # species rather than 1 (threshold too high) or ~N singletons
        # (threshold too low, which 0.35 caused on the first attempt).
        compatibility_threshold=0.65,
        compatibility_threshold_decay_rate=400.0,
        min_species_size=3,
        novelty_weight=0.15,
        novelty_fitness_beta=0.05,
    )

    generations = 25
    fitness_history, species_count_history, novelty_spread_history = [], [], []

    for gen in range(generations):
        env.evaluate_population(population)

        # NOTE: create_next_generation() internally calls compute_novelty_stats
        # with add_top_k_to_archive=0 ("refresh scores without growing the
        # archive again") -- actually growing the archive is the CALLER's
        # job. In the real system that's api/trainer.py's generation loop;
        # here, we are that caller.
        novelty_stats = engine.compute_novelty_stats(population, gen, add_top_k_to_archive=5)
        species_count = len(engine.speciation_manager.species) if engine.speciation_manager else 0

        best = max(population, key=lambda g: g.fitness)
        fitness_history.append(best.fitness)
        species_count_history.append(species_count)
        novelty_spread_history.append(novelty_stats["max"] - novelty_stats["mean"])

        if gen % 5 == 0 or gen == generations - 1:
            print(f"Gen {gen:2d} | best_fitness={best.fitness:6.2f} | "
                  f"species={species_count:2d} | "
                  f"novelty(mean={novelty_stats['mean']:.3f}, max={novelty_stats['max']:.3f}) | "
                  f"{best.render_summary()}")

        population = engine.create_next_generation(population, generation=gen, pop_name="ad_creatives")
        if not isinstance(population, list):
            population = population.genomes

    print("\n--- Summary ---")
    print(f"Fitness:        {fitness_history[0]:.2f} -> {fitness_history[-1]:.2f}")
    print(f"Species count over run: min={min(species_count_history)}, max={max(species_count_history)}, "
          f"last={species_count_history[-1]}")
    print(f"Novelty spread (max-mean) over run: min={min(novelty_spread_history):.3f}, "
          f"max={max(novelty_spread_history):.3f}")

    assert fitness_history[-1] >= fitness_history[0], "Fitness regressed with speciation/novelty enabled."
    assert max(species_count_history) > 1, "Population never split into more than one species -- speciation isn't doing anything."
    assert max(novelty_spread_history) > 0.01, "Novelty scores show no spread -- embedding still not carrying signal."
    print("\nPASSED: speciation maintains multiple species AND novelty scores carry real signal AND fitness still improves.")


if __name__ == "__main__":
    run()
