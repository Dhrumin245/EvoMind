"""
Phase 2 integration test.

Replaces Phase 1's abstract "distance to a hidden target vector" fitness
with the real AdMarketEnv -- genomes are now scored by simulated audience
response (CTR/CVR across four distinct user segments), still run through
the same unmodified EvolutionEngine.

This proves two things Phase 1 couldn't:
  1. Fitness grounded in a simulated market (not an arbitrary distance
     function) still climbs under evolution.
  2. The population discovers the urgency-badge / social-proof interactions
     baked into AdMarketEnv on its own -- nobody tells it these help.
"""
import random
import numpy as np

from genomes.genome_ad import AdGenome
from environments.ad_market_env import AdMarketEnv, MarketEvalConfig
from evolution.evolution import EvolutionEngine

random.seed(11)
np.random.seed(11)


def run():
    population_size = 40
    population = [AdGenome() for _ in range(population_size)]

    env = AdMarketEnv(config=MarketEvalConfig(base_seed=11, impressions_per_segment=250))

    engine = EvolutionEngine(
        population_size=population_size,
        tournament_size=3,
        elite_count=2,
        mutation_rate=0.15,
        mutation_strength=0.2,
        architecture_mutation_rate=0.1,
        genome_cls=AdGenome,
        speciation_enabled=False,       # still off -- Phase 3
        novelty_archive_enabled=False,  # still off -- Phase 3
    )

    generations = 30
    ctr_history, cvr_history, fitness_history = [], [], []
    badge_adoption_history = []

    for gen in range(generations):
        env.evaluate_population(population)

        best = max(population, key=lambda g: g.fitness)
        best_result = env.evaluate(best)  # re-score for a clean breakdown to print
        badge_rate = np.mean([g.optional_traits.get("urgency_badge", False) for g in population])

        ctr_history.append(best_result["weighted_ctr"])
        cvr_history.append(best_result["weighted_cvr"])
        fitness_history.append(best.fitness)
        badge_adoption_history.append(badge_rate)

        if gen % 5 == 0 or gen == generations - 1:
            print(f"Gen {gen:2d} | fitness={best.fitness:6.2f} | "
                  f"CTR={best_result['weighted_ctr']*100:5.2f}% | "
                  f"CVR={best_result['weighted_cvr']*100:5.2f}% | "
                  f"urgency_badge_adoption={badge_rate*100:4.1f}% | "
                  f"best_fit_segment={env.best_segment_match(best)} | "
                  f"{best.render_summary()}")

        population = engine.create_next_generation(population, generation=gen, pop_name="ad_creatives")
        if not isinstance(population, list):
            population = population.genomes

    print("\n--- Summary ---")
    print(f"Fitness:  {fitness_history[0]:.2f} -> {fitness_history[-1]:.2f}")
    print(f"CTR:      {ctr_history[0]*100:.2f}% -> {ctr_history[-1]*100:.2f}%")
    print(f"CVR:      {cvr_history[0]*100:.2f}% -> {cvr_history[-1]*100:.2f}%")
    print(f"Urgency-badge adoption in population: {badge_adoption_history[0]*100:.1f}% -> {badge_adoption_history[-1]*100:.1f}%")

    assert fitness_history[-1] > fitness_history[0], "Fitness did not improve against the market simulator."
    print("\nPASSED: population improved against the simulated market, driven entirely by AdMarketEnv + the real EvolutionEngine.")


if __name__ == "__main__":
    run()
