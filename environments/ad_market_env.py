"""
AdMarketEnv -- synthetic audience simulator that scores AdGenome fitness.

This is the piece that does NOT reuse EvoMind's arena_core_env.py shape.
arena_core_env.py is a step-based embodied-RL loop: reset() -> obs, then
step(actions) repeatedly across many timesteps per episode, driven by a
torch_brain forward pass every step. An ad doesn't act inside a world across
timesteps -- it gets shown to people and either lands or doesn't. So this
module is a genuinely new evaluation paradigm: single-shot exposure against
a population of simulated user segments, not a multi-step control loop.

What IS reused conceptually: the EvaluationConfig dataclass shape from
core/async_evaluator.py (base_seed, batch_size, num_workers) -- kept here
for stylistic and eventual-parallelization consistency, even though Phase 2
runs single-process vectorized numpy rather than true multiprocessing.

Model:
  impression -> segment sampled by audience share
             -> click probability = logistic(alignment(ad, segment) + noise)
             -> conversion probability (conditional on click) = logistic(...)
  fitness = blended CTR + conversion-rate signal, aggregated across segments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from genomes.genome_ad import AdGenome, EMBED_DIM, TRAIT_CANDIDATES, _stable_hash

NUM_TRAITS = len(TRAIT_CANDIDATES)
VECTOR_DIM = EMBED_DIM * NUM_TRAITS


def _segment_embedding(name: str, seed_salt: int = 1000) -> np.ndarray:
    """Deterministic pseudo-embedding for a user segment's creative
    preferences -- stand-in for real learned audience-preference vectors.
    Uses the same stable (non-randomized-per-process) hash as genome_ad.py."""
    rng = np.random.RandomState(_stable_hash(name, seed_salt) % (2**31))
    return rng.uniform(-1.0, 1.0, size=VECTOR_DIM).astype(np.float32)


@dataclass
class UserSegment:
    """A slice of the simulated audience with its own creative preferences,
    price sensitivity, and share of total impressions/budget."""
    name: str
    audience_share: float          # fraction of impressions this segment gets
    preference: np.ndarray          # preferred combined trait embedding
    price_sensitivity: float = 0.5  # 0 = ignores urgency/discount framing, 1 = very responsive
    base_ctr: float = 0.03          # baseline click-through before creative fit
    base_cvr: float = 0.10          # baseline click->conversion before creative fit
    noise_std: float = 0.35


def default_segments() -> List[UserSegment]:
    """Four archetypal segments, each with genuinely different preferences --
    this is what makes 'evolve toward the best ad' a non-trivial multi-modal
    optimization instead of converging on one universally-loved headline."""
    specs = [
        ("budget_conscious", 0.35, 0.7, 0.025, 0.09),
        ("premium_quality", 0.20, 0.2, 0.035, 0.14),
        ("trend_driven", 0.25, 0.5, 0.045, 0.08),
        ("skeptical_data_driven", 0.20, 0.3, 0.020, 0.16),
    ]
    segments = []
    for name, share, price_sens, base_ctr, base_cvr in specs:
        segments.append(UserSegment(
            name=name,
            audience_share=share,
            preference=_segment_embedding(name),
            price_sensitivity=price_sens,
            base_ctr=base_ctr,
            base_cvr=base_cvr,
        ))
    return segments


@dataclass
class MarketEvalConfig:
    """Mirrors the shape of core/async_evaluator.py's EvaluationConfig
    (base_seed, batch_size) for stylistic consistency with the rest of the
    evolved-from-EvoMind codebase."""
    base_seed: int = 42
    impressions_per_segment: int = 250
    batch_size: int = 8

    def __post_init__(self):
        assert self.impressions_per_segment > 0, "impressions_per_segment must be positive"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class AdMarketEnv:
    """Synthetic audience market. Call evaluate(genome) for one genome, or
    evaluate_population(genomes) to score an entire generation at once."""

    def __init__(self, segments: Optional[List[UserSegment]] = None,
                 config: Optional[MarketEvalConfig] = None):
        self.segments = segments or default_segments()
        self.config = config or MarketEvalConfig()
        assert abs(sum(s.audience_share for s in self.segments) - 1.0) < 1e-6, \
            "audience shares must sum to 1.0"
        self._rng = np.random.RandomState(self.config.base_seed)

    def _urgency_boost(self, genome: AdGenome, segment: UserSegment) -> float:
        """Optional trait modules interact with segment price sensitivity --
        an urgency badge helps budget-conscious segments more than premium
        ones, which is exactly the kind of interaction an evolved population
        needs to discover rather than being told."""
        boost = 0.0
        if genome.optional_traits.get("urgency_badge"):
            boost += 0.6 * segment.price_sensitivity
        if genome.optional_traits.get("social_proof_line"):
            boost += 0.3
        if genome.optional_traits.get("secondary_cta"):
            boost += 0.15
        return boost

    def _score_against_segment(self, genome: AdGenome, segment: UserSegment) -> Dict[str, float]:
        emb = genome.combined_embedding()
        # Cosine-style alignment, scaled -- how well this creative's traits
        # match what this segment responds to.
        alignment = float(np.dot(emb, segment.preference) /
                          (np.linalg.norm(emb) * np.linalg.norm(segment.preference) + 1e-9))
        urgency = self._urgency_boost(genome, segment)

        noise = self._rng.normal(0, segment.noise_std)
        ctr_logit = np.log(segment.base_ctr / (1 - segment.base_ctr)) + 2.2 * alignment + urgency + noise
        ctr = float(_sigmoid(np.array(ctr_logit)))

        cvr_logit = np.log(segment.base_cvr / (1 - segment.base_cvr)) + 1.4 * alignment + 0.5 * urgency
        cvr = float(_sigmoid(np.array(cvr_logit)))

        return {"segment": segment.name, "ctr": ctr, "cvr": cvr, "alignment": alignment}

    def evaluate(self, genome: AdGenome) -> Dict[str, float]:
        """Score one genome across all segments, weighted by audience share.
        Sets genome.fitness directly (matches the EvolutionEngine contract:
        fitness lives on the genome after evaluation) and returns the full
        breakdown for dashboards / bandit budget decisions later.
        """
        per_segment = [self._score_against_segment(genome, s) for s in self.segments]

        weighted_ctr = sum(r["ctr"] * s.audience_share for r, s in zip(per_segment, self.segments))
        weighted_cvr = sum(r["cvr"] * s.audience_share for r, s in zip(per_segment, self.segments))

        # Fitness blends CTR (reach) and CVR (quality of that reach) --
        # an ad that's clicky but never converts shouldn't dominate.
        fitness = 100.0 * (0.6 * weighted_ctr + 0.4 * weighted_ctr * weighted_cvr * 10)

        genome.fitness = float(fitness)
        return {
            "genome_id": genome.genome_id,
            "fitness": genome.fitness,
            "weighted_ctr": weighted_ctr,
            "weighted_cvr": weighted_cvr,
            "per_segment": per_segment,
        }

    def evaluate_population(self, genomes: List[AdGenome]) -> List[Dict[str, float]]:
        """Evaluate every genome, set .fitness and .norm_fitness on each
        (the two attributes EvolutionEngine's selection actually reads),
        and return per-genome breakdowns for logging/dashboards."""
        results = [self.evaluate(g) for g in genomes]

        raw = [g.fitness for g in genomes]
        lo, hi = min(raw), max(raw)
        for g in genomes:
            g.norm_fitness = (g.fitness - lo) / (hi - lo + 1e-9)

        return results

    def serve_realized_impressions(self, genome: AdGenome, n_impressions: int) -> Dict[str, float]:
        """Sample REALIZED click/conversion outcomes for a batch of n
        impressions, rather than returning an expected probability like
        evaluate() does. Needed by the Phase 5 bandit: Thompson Sampling
        learns from actual observed counts (you got 12 clicks out of 200
        impressions), not from a theoretical CTR nobody actually measured.

        One noise draw per segment establishes this batch's "true" CTR/CVR
        snapshot (representing market conditions during this serving
        window); each individual impression is then an independent
        Bernoulli trial against that snapshot -- vectorized with numpy
        since a live campaign can easily serve thousands of impressions
        per round.
        """
        if n_impressions <= 0:
            return {"genome_id": genome.genome_id, "impressions": 0, "clicks": 0,
                    "conversions": 0, "empirical_ctr": 0.0, "empirical_cvr": 0.0}

        per_segment = [self._score_against_segment(genome, s) for s in self.segments]
        shares = np.array([s.audience_share for s in self.segments])
        seg_assignment = self._rng.choice(len(self.segments), size=n_impressions, p=shares)

        ctr_lookup = np.array([r["ctr"] for r in per_segment])
        cvr_lookup = np.array([r["cvr"] for r in per_segment])
        ctr_per_impression = ctr_lookup[seg_assignment]
        cvr_per_impression = cvr_lookup[seg_assignment]

        clicked = self._rng.random(n_impressions) < ctr_per_impression
        converted = clicked & (self._rng.random(n_impressions) < cvr_per_impression)

        clicks = int(clicked.sum())
        conversions = int(converted.sum())
        return {
            "genome_id": genome.genome_id,
            "impressions": n_impressions,
            "clicks": clicks,
            "conversions": conversions,
            "empirical_ctr": clicks / n_impressions,
            "empirical_cvr": (conversions / clicks) if clicks > 0 else 0.0,
        }

    def best_segment_match(self, genome: AdGenome) -> str:
        """Which audience segment this creative currently resonates with
        most -- useful for the dashboard and later for bandit targeting."""
        scores = [(s.name, self._score_against_segment(genome, s)["ctr"]) for s in self.segments]
        return max(scores, key=lambda x: x[1])[0]
