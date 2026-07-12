"""
AdGenome -- evolvable representation of an ad creative.

Implements the exact duck-typed interface that EvolutionEngine (copied
verbatim from EvoMind's evolution/evolution.py) expects from a genome class:

    - staticmethod crossover(parent1, parent2) -> AdGenome
    - instance method mutate(weight_mutation_rate, weight_mutation_strength,
                              architecture_mutation_rate, layer_mutation_rate) -> AdGenome
    - instance method copy() -> AdGenome
    - instance method set_parents(parent_ids, generation) -> None
    - attributes: genome_id, fitness, norm_fitness, novelty_score,
                  novelty_score_norm, age, genes (list, for Population.validate())

Deliberately does NOT inherit from core.genome.EvolvableGenome -- that class's
__init__ hard-initializes neural-network-specific structures (modules, motif
library, learning-rule nets, torch brains) that have no meaning for an ad
creative. evolution.py never does isinstance() checks against EvolvableGenome
(confirmed by inspection of the real source), so a clean duck-typed class is
safer and far less code than fighting an NN-shaped constructor.
"""

import random
import numpy as np
import hashlib
from typing import List, Dict, Any, Optional


def _stable_hash(text: str, seed_salt: int = 0) -> int:
    """Deterministic across process runs, unlike Python's built-in hash()
    which is randomized per-process (PYTHONHASHSEED) for strings since
    Python 3.3. Using hash() here would have made every 'deterministic'
    pseudo-embedding actually change on every fresh run -- a real bug,
    caught by literally rerunning the same seeded script twice and noticing
    the numbers differed."""
    digest = hashlib.md5(f"{text}|{seed_salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


EMBED_DIM = 6

# Stands in for what becomes an LLM-backed content generator in Phase 4.
# Each trait has a pool of candidate values; each candidate gets a fixed
# pseudo-embedding so we can measure semantic distance and simulate audience
# response before any real LLM or text-embedding model is wired in.
TRAIT_CANDIDATES: Dict[str, List[str]] = {
    "headline": [
        "Stop overpaying for the same results",
        "The smarter way to get this done",
        "Join thousands who already switched",
        "Your competitors already found this",
        "Finally, software that respects your time",
        "Built for people who hate wasting money",
    ],
    "image_style": [
        "minimalist product shot on white",
        "lifestyle photo, warm natural light",
        "bold flat illustration, high contrast",
        "screenshot-driven UI mockup",
        "before/after comparison layout",
    ],
    "cta": [
        "Start free trial",
        "See pricing",
        "Get a demo",
        "Try it now",
        "Claim your spot",
    ],
    "tone": [
        "urgent and direct",
        "warm and reassuring",
        "playful and confident",
        "data-driven and precise",
    ],
    "color_scheme": [
        "navy and gold",
        "high-contrast black/white",
        "soft pastel",
        "brand primary plus neon accent",
    ],
}

OPTIONAL_TRAITS = ["urgency_badge", "social_proof_line", "secondary_cta"]


def _hash_embedding(text: str, dim: int = EMBED_DIM, seed_salt: int = 0) -> np.ndarray:
    """Deterministic pseudo-embedding: same candidate text always yields the
    same vector, across processes and machines. Stand-in for a real
    sentence-embedding model."""
    rng = np.random.RandomState(_stable_hash(text, seed_salt) % (2**31))
    return rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)


class AdTraitGene:
    """A single evolvable trait slot on an ad (headline, CTA, tone, ...)."""

    __slots__ = ("trait_name", "value", "embedding", "expression_offset")

    def __init__(self, trait_name: str, value: Optional[str] = None,
                 expression_offset: Optional[np.ndarray] = None):
        self.trait_name = trait_name
        self.value = value or random.choice(TRAIT_CANDIDATES[trait_name])
        self.embedding = _hash_embedding(self.value, seed_salt=_stable_hash(trait_name))
        self.expression_offset = (
            expression_offset if expression_offset is not None
            else np.zeros(EMBED_DIM, dtype=np.float32)
        )

    def effective_embedding(self) -> np.ndarray:
        return self.embedding + self.expression_offset

    def regenerate(self, generator=None):
        """Swap to a different candidate value entirely (big structural jump,
        the ad-genome equivalent of an architecture mutation).

        `generator` is any genomes.content_generator.ContentGenerator.
        Defaults to the original fixed-pool behavior (Phases 1-3) when None,
        so existing callers and tests are unaffected."""
        if generator is not None:
            new_value = generator.generate(self.trait_name, current_value=self.value)
            self.value = new_value
            self.embedding = _hash_embedding(new_value, seed_salt=_stable_hash(self.trait_name))
            self.expression_offset = np.zeros(EMBED_DIM, dtype=np.float32)
            return

        pool = [c for c in TRAIT_CANDIDATES[self.trait_name] if c != self.value]
        if not pool:
            return
        self.value = random.choice(pool)
        self.embedding = _hash_embedding(self.value, seed_salt=_stable_hash(self.trait_name))
        self.expression_offset = np.zeros(EMBED_DIM, dtype=np.float32)

    def perturb(self, strength: float):
        """Small continuous nudge -- a subtle rewording/tone shift within the
        same underlying idea, without swapping to a different candidate."""
        self.expression_offset = self.expression_offset + np.random.normal(
            0, strength, size=EMBED_DIM
        ).astype(np.float32)

    def copy(self) -> "AdTraitGene":
        return AdTraitGene(self.trait_name, self.value, self.expression_offset.copy())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trait_name": self.trait_name,
            "value": self.value,
            "expression_offset": self.expression_offset.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdTraitGene":
        return cls(data["trait_name"], data["value"],
                   np.array(data["expression_offset"], dtype=np.float32))


class AdGenome:
    """Evolvable ad creative. Duck-type compatible with EvoMind's real
    EvolutionEngine -- see module docstring for the exact contract."""

    # Class-level default content generator (see genomes/content_generator.py).
    # EvolutionEngine's reproduction loop calls child.mutate(weight_mutation_rate=...)
    # with no knowledge of content generation -- it never passes a
    # content_generator kwarg. Rather than change the engine's call site,
    # mutate() falls back to this class attribute when no per-call generator
    # is supplied, so `AdGenome.set_content_generator(llm_gen)` swaps
    # generation strategy for every future mutation without touching
    # evolution.py at all.
    _content_generator = None

    @classmethod
    def set_content_generator(cls, generator) -> None:
        cls._content_generator = generator

    def __init__(self, genome_id: Optional[str] = None,
                 genes: Optional[List[AdTraitGene]] = None,
                 optional_traits: Optional[Dict[str, bool]] = None):
        self.genome_id = genome_id or f"ad_{random.randint(0, 999999):06d}"
        self.genes: List[AdTraitGene] = genes or [
            AdTraitGene(name) for name in TRAIT_CANDIDATES.keys()
        ]
        self.optional_traits: Dict[str, bool] = optional_traits or {
            t: random.random() < 0.5 for t in OPTIONAL_TRAITS
        }

        # Required by EvolutionEngine / Population contract
        self.fitness: float = 0.0
        self.norm_fitness: float = 0.0
        self.novelty_score: float = 0.0
        self.novelty_score_norm: float = 0.0
        self.age: int = 0
        self.parent_ids: List[str] = []
        self.birth_generation: int = 0
        self.role: str = "ad_creative"

    def trait_map(self) -> Dict[str, AdTraitGene]:
        return {g.trait_name: g for g in self.genes}

    def combined_embedding(self) -> np.ndarray:
        """Concatenated effective embedding across all traits -- used for
        fitness scoring now, and for GenomeDistance/novelty in Phase 3."""
        return np.concatenate([g.effective_embedding() for g in self.genes])

    # ------------------------------------------------------------------
    # Core evolutionary operators -- this is the contract EvolutionEngine calls
    # ------------------------------------------------------------------

    @staticmethod
    def crossover(parent1: "AdGenome", parent2: "AdGenome") -> "AdGenome":
        """Uniform crossover: each trait independently inherited from one
        parent or the other. Simpler than NN crossover since traits don't
        need structural alignment the way network layers do."""
        child_genes = []
        for t1, t2 in zip(parent1.genes, parent2.genes):
            chosen = t1 if random.random() < 0.5 else t2
            child_genes.append(chosen.copy())

        child_optional = {}
        for trait in OPTIONAL_TRAITS:
            source = parent1.optional_traits if random.random() < 0.5 else parent2.optional_traits
            child_optional[trait] = source.get(trait, False)

        return AdGenome(genes=child_genes, optional_traits=child_optional)

    def mutate(self,
               weight_mutation_rate: float = 0.1,
               weight_mutation_strength: float = 0.1,
               architecture_mutation_rate: float = 0.05,
               layer_mutation_rate: float = 0.05,
               content_generator=None,
               **_ignored) -> "AdGenome":
        """
        Mirrors EvolvableGenome.mutate()'s four-knob interface:
          - weight_mutation_rate/strength -> continuous perturbation of a
            trait's expression embedding (subtle rewording/tone shift)
          - architecture_mutation_rate -> full regeneration of a trait to a
            different candidate value (big structural jump)
          - layer_mutation_rate -> toggle an optional trait module on/off
            (urgency badge, social proof line, secondary CTA)

        `content_generator` (genomes.content_generator.ContentGenerator) is
        optional dependency injection for what "regenerate" means -- if not
        passed explicitly, falls back to AdGenome._content_generator (set
        via AdGenome.set_content_generator()); None/unset keeps the original
        fixed-pool behavior from Phases 1-3.
        """
        generator = content_generator if content_generator is not None else type(self)._content_generator
        for gene in self.genes:
            if random.random() < architecture_mutation_rate:
                gene.regenerate(generator=generator)
            elif random.random() < weight_mutation_rate:
                gene.perturb(weight_mutation_strength)

        for trait in OPTIONAL_TRAITS:
            if random.random() < layer_mutation_rate:
                self.optional_traits[trait] = not self.optional_traits.get(trait, False)

        self.age += 1
        return self

    def copy(self) -> "AdGenome":
        clone = AdGenome(
            genes=[g.copy() for g in self.genes],
            optional_traits=dict(self.optional_traits),
        )
        clone.fitness = self.fitness
        clone.norm_fitness = self.norm_fitness
        clone.novelty_score = self.novelty_score
        clone.novelty_score_norm = self.novelty_score_norm
        clone.age = self.age
        clone.parent_ids = list(self.parent_ids)
        clone.birth_generation = self.birth_generation
        return clone

    def set_parents(self, parent_ids: List[str], generation: int):
        self.parent_ids = list(parent_ids)
        self.birth_generation = generation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "genes": [g.to_dict() for g in self.genes],
            "optional_traits": dict(self.optional_traits),
            "fitness": self.fitness,
            "norm_fitness": self.norm_fitness,
            "age": self.age,
            "parent_ids": list(self.parent_ids),
            "birth_generation": self.birth_generation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdGenome":
        genome = cls(
            genome_id=data["genome_id"],
            genes=[AdTraitGene.from_dict(g) for g in data["genes"]],
            optional_traits=dict(data["optional_traits"]),
        )
        genome.fitness = data.get("fitness", 0.0)
        genome.norm_fitness = data.get("norm_fitness", 0.0)
        genome.age = data.get("age", 0)
        genome.parent_ids = data.get("parent_ids", [])
        genome.birth_generation = data.get("birth_generation", 0)
        return genome

    def render_summary(self) -> str:
        t = self.trait_map()
        badges = [k for k, v in self.optional_traits.items() if v]
        return (f"[{self.genome_id}] \"{t['headline'].value}\" | "
               f"{t['image_style'].value} | CTA: {t['cta'].value} | "
               f"tone: {t['tone'].value} | {t['color_scheme'].value}"
               + (f" | +{','.join(badges)}" if badges else ""))
