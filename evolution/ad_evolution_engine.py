"""
Phase 3 -- makes speciation and novelty scoring actually work on AdGenome.

Two real problems were found by reading the source, not by guessing:

1. GenomeDistance._architecture_distance / _parameter_distance / _behavior_distance
   read gene.activation, gene.input_dim, gene.output_dim, gene.weights, and
   genome.meta DIRECTLY (no getattr/hasattr guard). All of those are
   NN-specific and absent on AdTraitGene/AdGenome -- calling the stock
   GenomeDistance on two AdGenomes throws AttributeError. Fix: a fresh
   AdGenomeDistance class, swapped in after EvolutionEngine's own __init__
   builds the default one (SpeciationManager only ever calls
   `self.distance_calculator.calculate_distance(g1, g2)` -- pure duck typing,
   confirmed by inspection, so nothing else needs to change).

2. EvolutionEngine._build_novelty_embedding IS defensive (getattr everywhere,
   try/except around the NN-specific part) -- it will NOT crash on AdGenome.
   But it also won't produce anything meaningful: every AdGenome has no
   `.meta`/`.behavior_stats`, so every genome gets an almost-identical
   near-zero embedding, and the novelty archive can't tell creatives apart.
   Fix: override just this one method in a thin EvolutionEngine subclass so
   novelty is computed from the genome's actual trait embedding instead.

3. NoveltyInjector._calculate_population_diversity has the SAME bug as (1) --
   direct `gene.input_dim * gene.output_dim` access, no guard. Unlike
   GenomeDistance this one is NOT dormant: EvolutionEngine.create_next_generation
   calls `self.novelty_injector.should_inject(...)` unconditionally whenever
   novelty_archive_enabled=True, so this crashes on the very first generation.
   Found by actually running the test against the real engine, not by static
   reading -- confirms why "run it for real" mattered here. Fixed the same
   way: subclass, override the one broken method, swap the instance in.

Note on AdaptiveMutation._calculate_diversity(): traced the call graph and
confirmed `self.mutator` (a standalone AdaptiveMutation instance) is dead
code in the default engine -- adaptive-rate smoothing happens directly on
EvolutionEngine's own fields (_smoothed_weight_rate etc.) and only consults
a MutatorPopulation if one is explicitly attached. We're not using that
meta-evolution feature, so AdaptiveMutation is genuinely not exercised here.
Documented rather than "fixed" -- there's nothing running that would break.
"""

import numpy as np
from typing import List

from evolution.evolution import EvolutionEngine, GenomeDistance, BehaviorEmbedding, NoveltyInjector
from genomes.genome_ad import AdGenome, OPTIONAL_TRAITS



class AdGenomeDistance(GenomeDistance):
    """Drop-in replacement for GenomeDistance, same public interface
    (calculate_distance(genome1, genome2) -> float), but computed from
    ad-trait embeddings and optional-module toggles instead of NN weights.

    Inherits from GenomeDistance only to keep isinstance-friendly typing
    elsewhere in the codebase happy -- none of the parent's NN-specific
    methods are called; calculate_distance is fully overridden.
    """

    def __init__(self, embedding_weight: float = 0.75, optional_trait_weight: float = 0.25):
        # Skip GenomeDistance.__init__ (it only normalizes NN-specific
        # weight fields we don't use) and set our own weights directly.
        total = embedding_weight + optional_trait_weight
        self.embedding_weight = embedding_weight / total
        self.optional_trait_weight = optional_trait_weight / total

    def calculate_distance(self, genome1: AdGenome, genome2: AdGenome) -> float:
        emb1, emb2 = genome1.combined_embedding(), genome2.combined_embedding()
        # Normalize by vector length so distance is comparable regardless of
        # how many traits exist -- keeps compatibility_threshold meaningful
        # if the trait catalog grows later.
        embedding_dist = float(np.linalg.norm(emb1 - emb2) / np.sqrt(len(emb1)))

        diffs = sum(
            1 for t in OPTIONAL_TRAITS
            if genome1.optional_traits.get(t, False) != genome2.optional_traits.get(t, False)
        )
        optional_dist = diffs / max(len(OPTIONAL_TRAITS), 1)

        return float(self.embedding_weight * embedding_dist +
                     self.optional_trait_weight * optional_dist)

    def _architecture_distance(self, genome1: AdGenome, genome2: AdGenome) -> float:
        """EvolutionEngine._enforce_minimum_species_diversity() calls this
        DIRECTLY (bypassing calculate_distance()) when it needs to split an
        overcrowded species -- found by running the API test with a smaller
        population than Phase 3 used, which was the first scenario to
        actually trigger that code path. There's no clean "architecture
        only" vs "behavior only" split for ad traits the way there is for
        NN topology vs learned weights, so this just delegates to the same
        embedding-based distance."""
        return self.calculate_distance(genome1, genome2)


class AdNoveltyInjector(NoveltyInjector):
    """Fixes the same class of bug as AdGenomeDistance, in a different class:
    _calculate_population_diversity() reads gene.input_dim/output_dim
    directly with no guard, which crashes the instant novelty_archive_enabled
    is turned on (confirmed by an actual traceback, not by inspection alone).
    """

    def _calculate_population_diversity(self, population: List[AdGenome]) -> float:
        if len(population) < 2:
            return 0.0

        fitnesses = [g.fitness for g in population]
        fitness_div = float(np.std(fitnesses) / (np.mean(fitnesses) + 1e-10)) if fitnesses else 0.0

        embeddings = np.array([g.combined_embedding() for g in population])
        # Average per-dimension std, normalized into a roughly 0-1 range --
        # embeddings are drawn from uniform(-1, 1) so per-dim std tops out
        # well under 1.0 in practice.
        embedding_div = float(np.mean(np.std(embeddings, axis=0)))

        optional_matrix = np.array([
            [float(g.optional_traits.get(t, False)) for t in OPTIONAL_TRAITS]
            for g in population
        ])
        optional_div = float(np.mean(np.std(optional_matrix, axis=0)))

        total_diversity = (fitness_div + embedding_div + optional_div) / 3.0
        return float(min(total_diversity, 1.0))


class AdEvolutionEngine(EvolutionEngine):
    """EvolutionEngine, unmodified except for one method override.

    Everything else -- selection, crossover/mutate dispatch, speciation
    bookkeeping, novelty archive mechanics, stagnation-based rate adjustment
    -- is the real EvoMind engine running as-is.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap in the ad-aware distance metric wherever the parent stashed one.
        ad_distance = AdGenomeDistance()
        self.distance_calculator = ad_distance
        if self.speciation_manager is not None:
            self.speciation_manager.distance_calculator = ad_distance

        # Swap in the ad-aware novelty injector (same diversity_threshold/
        # stagnation_window defaults the parent hardcoded) so
        # create_next_generation's unconditional should_inject() call
        # doesn't crash on gene.input_dim.
        if self.novelty_archive is not None:
            self.novelty_injector = AdNoveltyInjector(
                novelty_archive=self.novelty_archive,
                diversity_threshold=0.1,
                stagnation_window=10,
            )

        # See _check_stagnation override below -- tracks population MEAN
        # fitness, which the parent's stagnation logic doesn't.
        self._ad_mean_fitness_history: List[float] = []

    def _build_novelty_embedding(self, genome: AdGenome, generation: int) -> BehaviorEmbedding:
        """Real trait-based novelty embedding, replacing the near-constant
        default (which reads NN-only fields that don't exist on AdGenome)."""
        emb = genome.combined_embedding()
        # tanh keeps it bounded like the parent implementation does, so
        # NoveltyArchive's distance math stays on a sane scale.
        bounded = np.tanh(emb)
        optional_flags = np.array(
            [float(genome.optional_traits.get(t, False)) for t in OPTIONAL_TRAITS],
            dtype=np.float32,
        )
        embedding = np.concatenate([bounded, optional_flags]).astype(np.float32)

        return BehaviorEmbedding(
            genome_id=str(genome.genome_id),
            fitness=float(genome.fitness),
            embedding=embedding,
            generation=int(generation),
        )

    def _check_stagnation(self, population: List[AdGenome]) -> None:
        """Overrides EvolutionEngine._check_stagnation. The parent version
        treats 'the single best genome's raw fitness didn't strictly
        increase this generation' as stagnation, then ratchets mutation
        rates toward a 0.5 ceiling the longer that continues -- with no
        path back down once the counter is high.

        That's a reasonable heuristic for effectively-unbounded fitness
        (AdMarketEnv's CTR/CVR formula has no real ceiling in practice, so
        this never misfired in Phases 2/3/5). It actively breaks for
        LLMJudgeFitnessBackend: judge scores are clamped to [1, 10] BY
        DESIGN -- a real rubric-based judge should have a ceiling. With
        elitism preserving the best genome found, hitting that ceiling even
        once means current_best can mathematically never 'improve' again,
        so every later generation reads as stagnation forever. Confirmed by
        actually running Phase 9: mutation rates hit their 0.5/0.5 ceiling
        around generation 16 and species count climbed to 16 out of a
        population of 30 shortly after -- a real vicious cycle (stuck-high
        mutation keeps drifting traits the judge doesn't select on, which
        inflates measured genome distance, which fragments speciation,
        which makes the population-wide picture look even more
        'stagnant').

        Fix: judge stagnation on POPULATION MEAN fitness, which has real
        headroom even after the single best genome caps out; use an
        epsilon relative to this generation's actual fitness spread instead
        of a near-machine-precision absolute one; and let the stagnation
        counter recover once mean fitness is moving again instead of only
        ever ratcheting upward. Also lowers the escalation ceiling from 0.5
        to 0.3 -- 0.5/0.5 is close to random search, which is far more
        disruptive than any bounded fitness landscape should ever trigger.
        """
        current_best = max(g.fitness for g in population)
        current_mean = float(np.mean([g.fitness for g in population]))
        self.generation += 1

        if not self.best_fitness_history:
            self.best_fitness_history.append(current_best)
            self._ad_mean_fitness_history.append(current_mean)
            return

        last_mean = self._ad_mean_fitness_history[-1]
        spread = max(current_best - min(g.fitness for g in population), 1e-6)
        eps = 0.01 * spread  # relative to this generation's actual fitness range, not a fixed absolute value

        if current_mean > last_mean + eps:
            self.stagnation_counter = max(0, self.stagnation_counter - 2)
            self.mutation_rate = float(np.clip(
                0.9 * self.mutation_rate + 0.1 * self.base_mutation_rate, 0.001, 0.2))
            self.architecture_mutation_rate = float(np.clip(
                0.9 * self.architecture_mutation_rate + 0.1 * self.base_architecture_mutation_rate, 0.001, 0.15))
        else:
            self.stagnation_counter += 1

        self.best_fitness_history.append(current_best)
        self._ad_mean_fitness_history.append(current_mean)

        if self.stagnation_counter > 12:
            _max_rate = 0.3
            self.architecture_mutation_rate = min(self.architecture_mutation_rate * 1.3, _max_rate)
            self.mutation_rate = min(self.mutation_rate * 1.3, _max_rate)
            print(f"[Ad-tuned] stagnation (gen {self.generation}): mean fitness flat, "
                  f"increasing mutation rates | weight={self.mutation_rate:.4f}, "
                  f"arch={self.architecture_mutation_rate:.4f}")
