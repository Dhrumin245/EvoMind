"""
Phase 5 -- live budget allocation via Thompson Sampling.

Everything through Phase 4 evaluated every genome with the same fixed
number of simulated impressions (AdMarketEnv.evaluate(), 250/segment for
everyone). That's fine for proving evolution works, but it's not how a real
ad campaign spends money: you don't give a probably-bad variant the same
budget as your best performer just to get a clean average. You want to
learn WHICH variant is winning while you're still spending, shifting real
budget toward it before the generation even ends -- and still protect
enough exploration budget that a genuinely great new variant (say, a fresh
child genome with a completely different headline) isn't starved just
because it hasn't proven itself yet.

This has no counterpart anywhere in EvoMind -- arena_core_env.py's agents
don't compete for a shared, finite resource across a population; every
genome just gets its own full episode. Budget allocation is a live-traffic,
shared-resource problem that a closed training simulation never has to
solve, so this module has no prior art to adapt -- it's new from here down.

Design:
  - Each genome_id is a "bandit arm" with a Beta(alpha, beta) posterior over
    its true click-through rate.
  - Within one generation, run several ROUNDS (not one shot): sample from
    each arm's current posterior, hand a sub-batch of impressions to the
    arms with the highest samples, observe realized clicks, update their
    posteriors, then repeat. Budget visibly shifts toward performers across
    rounds WITHIN a generation, not just across generations via selection.
  - Every genome gets a guaranteed minimum per round (real platforms do
    this too) so pure sampling variance can't fully starve a new arm.
  - genome.fitness is set from REALIZED empirical CTR/CVR -- the actual
    counts observed given how much budget it received -- not a theoretical
    expected value. A variant that only got a little budget has noisier,
    less certain fitness, exactly like a real live campaign.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from genomes.genome_ad import AdGenome
from environments.ad_market_env import AdMarketEnv


@dataclass
class BetaArm:
    alpha: float = 1.0
    beta: float = 1.0

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class ThompsonSamplingAllocator:
    """Beta-Bernoulli Thompson Sampling over a population of ad-genome arms."""

    def __init__(self, rng_seed: int = 42):
        self.arms: Dict[str, BetaArm] = {}
        self._rng = np.random.RandomState(rng_seed)

    def ensure_arm(self, genome_id: str) -> None:
        if genome_id not in self.arms:
            self.arms[genome_id] = BetaArm()

    def prune_to(self, genome_ids: List[str]) -> None:
        """Drop arms for genomes no longer in the population. Since
        AdGenome.crossover() always mints a brand-new random genome_id for
        children, this effectively means every generation's children start
        as cold-start arms -- realistic, since a genuinely new creative has
        no track record either."""
        keep = set(genome_ids)
        self.arms = {gid: arm for gid, arm in self.arms.items() if gid in keep}

    def allocate(self, genome_ids: List[str], round_budget: int, min_per_arm: int = 1) -> Dict[str, int]:
        """Vectorized Thompson Sampling: draw `round_budget - min_per_arm*n`
        samples from every arm's CURRENT posterior at once, and award each
        draw to whichever arm sampled highest. Vectorized (numpy, one call)
        rather than a per-draw Python loop -- with populations in the tens
        and thousands of draws per round, the naive loop is a real
        bottleneck; this isn't."""
        n = len(genome_ids)
        for gid in genome_ids:
            self.ensure_arm(gid)

        allocation = {gid: min_per_arm for gid in genome_ids}
        remaining = round_budget - min_per_arm * n
        if remaining <= 0:
            return allocation

        alphas = np.array([self.arms[gid].alpha for gid in genome_ids])
        betas = np.array([self.arms[gid].beta for gid in genome_ids])
        samples = self._rng.beta(alphas, betas, size=(remaining, n))
        winners = np.argmax(samples, axis=1)
        counts = np.bincount(winners, minlength=n)

        for i, gid in enumerate(genome_ids):
            allocation[gid] += int(counts[i])
        return allocation

    def update(self, genome_id: str, clicks: int, impressions: int) -> None:
        self.ensure_arm(genome_id)
        arm = self.arms[genome_id]
        arm.alpha += clicks
        arm.beta += max(impressions - clicks, 0)


class BudgetedMarketEvaluator:
    """Combines AdMarketEnv (the audience simulator) with
    ThompsonSamplingAllocator (the budget decision-maker) into one
    generation-scoring pipeline: allocate -> serve -> update -> repeat for
    several rounds -> set genome.fitness from what was actually observed.
    """

    def __init__(self, market_env: AdMarketEnv,
                 allocator: Optional[ThompsonSamplingAllocator] = None,
                 total_budget_per_generation: int = 6000,
                 rounds_per_generation: int = 15):
        self.env = market_env
        self.allocator = allocator or ThompsonSamplingAllocator()
        self.total_budget = total_budget_per_generation
        self.rounds = rounds_per_generation

    def run_generation(self, population: List[AdGenome]) -> List[Dict]:
        genome_ids = [g.genome_id for g in population]
        genome_by_id = {g.genome_id: g for g in population}
        self.allocator.prune_to(genome_ids)

        round_budget = max(self.total_budget // self.rounds, len(genome_ids))
        min_per_arm = max(1, round_budget // (5 * len(genome_ids)))

        cumulative = {gid: {"impressions": 0, "clicks": 0, "conversions": 0} for gid in genome_ids}
        allocation_by_round = []  # for diagnostics: how allocation shifts across rounds

        for _ in range(self.rounds):
            allocation = self.allocator.allocate(genome_ids, round_budget, min_per_arm)
            allocation_by_round.append(dict(allocation))
            for gid, n in allocation.items():
                if n <= 0:
                    continue
                outcome = self.env.serve_realized_impressions(genome_by_id[gid], n)
                self.allocator.update(gid, outcome["clicks"], outcome["impressions"])
                cumulative[gid]["impressions"] += outcome["impressions"]
                cumulative[gid]["clicks"] += outcome["clicks"]
                cumulative[gid]["conversions"] += outcome["conversions"]

        results = []
        for gid in genome_ids:
            c = cumulative[gid]
            empirical_ctr = c["clicks"] / c["impressions"] if c["impressions"] > 0 else 0.0
            empirical_cvr = c["conversions"] / c["clicks"] if c["clicks"] > 0 else 0.0
            genome = genome_by_id[gid]
            genome.fitness = float(100.0 * (0.6 * empirical_ctr + 0.4 * empirical_ctr * empirical_cvr * 10))
            results.append({
                "genome_id": gid,
                "impressions": c["impressions"],
                "clicks": c["clicks"],
                "conversions": c["conversions"],
                "empirical_ctr": empirical_ctr,
                "empirical_cvr": empirical_cvr,
                "posterior_mean_ctr": self.allocator.arms[gid].mean(),
            })

        raw = [g.fitness for g in population]
        lo, hi = min(raw), max(raw)
        for g in population:
            g.norm_fitness = (g.fitness - lo) / (hi - lo + 1e-9)

        self._last_allocation_by_round = allocation_by_round
        return results

    def budget_concentration(self, top_fraction: float = 0.2) -> float:
        """What share of THIS generation's total impressions went to the
        top `top_fraction` of genomes by impressions received. Rises over
        rounds/generations as the bandit learns who's actually winning."""
        if not self._last_allocation_by_round:
            return 0.0
        totals: Dict[str, int] = {}
        for round_alloc in self._last_allocation_by_round:
            for gid, n in round_alloc.items():
                totals[gid] = totals.get(gid, 0) + n
        sorted_totals = sorted(totals.values(), reverse=True)
        k = max(1, int(len(sorted_totals) * top_fraction))
        top_sum = sum(sorted_totals[:k])
        total_sum = sum(sorted_totals) or 1
        return top_sum / total_sum
