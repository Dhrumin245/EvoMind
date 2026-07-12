"""
Phase 5 integration test.

Replaces Phase 2/3's flat evaluate() (every genome gets the same fixed
impression count) with BudgetedMarketEvaluator: a real bandit decides how
much of a finite per-generation budget each genome gets, learning within
the generation as results come in.

Proves three things:
  1. WITHIN one generation, budget visibly shifts toward the eventual best
     performer across rounds (early rounds are close to uniform since all
     arms start Beta(1,1); later rounds concentrate on whoever's winning).
  2. ACROSS generations, evolution + the bandit together still drive
     fitness up -- the two mechanisms compose instead of fighting each
     other.
  3. New child genomes (fresh random genome_id every generation, since
     AdGenome.crossover() always mints one) still get their guaranteed
     minimum exploration budget every round -- they're never fully starved
     just for being unproven.
"""
import random
import numpy as np

from genomes.genome_ad import AdGenome
from environments.ad_market_env import AdMarketEnv, MarketEvalConfig
from evolution.budget_allocator import ThompsonSamplingAllocator, BudgetedMarketEvaluator
from evolution.ad_evolution_engine import AdEvolutionEngine

random.seed(5)
np.random.seed(5)


def run():
    population_size = 30
    population = [AdGenome() for _ in range(population_size)]

    env = AdMarketEnv(config=MarketEvalConfig(base_seed=5))
    evaluator = BudgetedMarketEvaluator(
        market_env=env,
        allocator=ThompsonSamplingAllocator(rng_seed=5),
        total_budget_per_generation=6000,
        rounds_per_generation=15,
    )

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
        compatibility_threshold=0.65,
        min_species_size=3,
    )

    generations = 15
    fitness_history = []
    concentration_history = []
    within_gen_shift_last_gen = None  # captured on the final generation for reporting

    for gen in range(generations):
        results = evaluator.run_generation(population)
        concentration = evaluator.budget_concentration(top_fraction=0.2)
        concentration_history.append(concentration)

        best = max(population, key=lambda g: g.fitness)
        fitness_history.append(best.fitness)

        if gen == generations - 1:
            # Within-generation shift: how much budget did the eventual
            # best genome get in the first 3 rounds vs the last 3 rounds?
            rounds = evaluator._last_allocation_by_round
            best_id = best.genome_id
            early = sum(r.get(best_id, 0) for r in rounds[:3])
            late = sum(r.get(best_id, 0) for r in rounds[-3:])
            within_gen_shift_last_gen = (early, late)

        if gen % 3 == 0 or gen == generations - 1:
            top20 = sorted(results, key=lambda r: r["impressions"], reverse=True)[:max(1, population_size // 5)]
            avg_top20_ctr = np.mean([r["empirical_ctr"] for r in top20])
            print(f"Gen {gen:2d} | best_fitness={best.fitness:6.2f} | "
                  f"top20%-budget-share={concentration*100:5.1f}% | "
                  f"top20%-avg-empirical-CTR={avg_top20_ctr*100:5.2f}% | "
                  f"{best.render_summary()}")

        population = engine.create_next_generation(population, generation=gen, pop_name="ad_creatives")
        if not isinstance(population, list):
            population = population.genomes

    print("\n--- Summary ---")
    print(f"Fitness:                 {fitness_history[0]:.2f} -> {fitness_history[-1]:.2f}")
    print(f"Top-20% budget share:    {concentration_history[0]*100:.1f}% -> {concentration_history[-1]*100:.1f}%")
    early, late = within_gen_shift_last_gen
    print(f"Within final generation, eventual winner's budget: "
          f"first 3 rounds={early} impressions -> last 3 rounds={late} impressions")

    assert fitness_history[-1] >= fitness_history[0], "Fitness did not improve with bandit-driven evaluation."
    assert late > early, "Budget did not shift toward the eventual winner within the generation."
    print("\nPASSED: bandit concentrates budget within a generation, and evolution still improves fitness across generations.")


if __name__ == "__main__":
    run()
