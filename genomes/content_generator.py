"""
Phase 4 -- pluggable content generation.

Up through Phase 3, AdTraitGene.regenerate() could only resample from the
fixed TRAIT_CANDIDATES pool (5-6 hand-written strings per trait). That was
fine for proving the evolutionary machinery works, but it's a closed set --
the population can recombine and reweight those candidates, but it can never
discover a headline nobody wrote by hand.

This module defines a small interface so "how do we get a new trait value"
is a swappable dependency instead of a hardcoded random.choice():

  - ContentGenerator (ABC): generate(trait_name, current_value, exclude) -> str
  - TemplateContentGenerator: today's behavior (candidate pool), zero
    external dependencies, used as the default so Phase 1-3 tests keep
    passing unchanged.
  - LLMContentGenerator: real Anthropic API integration. Requires an
    ANTHROPIC_API_KEY in the environment -- NOT available in this sandbox,
    so it's exercised here via a mock client (see tests/test_phase4_*.py)
    rather than a live call. The integration code itself is real; only the
    credential is missing.

Note on embeddings: AdTraitGene's _hash_embedding() already hashes arbitrary
text into a vector -- it was never a lookup table keyed on the fixed pool.
That means novel LLM-generated strings get sensible embeddings for free,
with zero changes needed in genome_ad.py.
"""

import os
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from genomes.genome_ad import TRAIT_CANDIDATES


class ContentGenerator(ABC):
    """Interface for producing a new candidate value for a given trait."""

    @abstractmethod
    def generate(self, trait_name: str, current_value: Optional[str] = None,
                 exclude: Optional[List[str]] = None) -> str:
        """Return a new candidate string for `trait_name`, ideally different
        from `current_value` and anything in `exclude`."""
        raise NotImplementedError


class TemplateContentGenerator(ContentGenerator):
    """Default, dependency-free generator: samples from the hand-written
    TRAIT_CANDIDATES pool. This is exactly what AdTraitGene.regenerate() did
    in Phases 1-3, now expressed as a swappable strategy instead of inline
    logic, so the behavior is unchanged unless a different generator is
    explicitly injected."""

    def generate(self, trait_name: str, current_value: Optional[str] = None,
                 exclude: Optional[List[str]] = None) -> str:
        exclude_set = set(exclude or [])
        if current_value:
            exclude_set.add(current_value)
        pool = [c for c in TRAIT_CANDIDATES[trait_name] if c not in exclude_set]
        if not pool:
            pool = TRAIT_CANDIDATES[trait_name]
        return random.choice(pool)


_TRAIT_PROMPTS = {
    "headline": "a short, punchy ad headline (under 10 words)",
    "image_style": "a brief visual-style direction for an ad's hero image (5-8 words)",
    "cta": "a short call-to-action button label (2-4 words)",
    "tone": "a short description of an ad's tone/voice (2-4 words)",
    "color_scheme": "a short color palette description for an ad (2-5 words)",
}


class LLMContentGenerator(ContentGenerator):
    """Real Anthropic API integration. Generates genuinely novel trait
    content instead of resampling a fixed pool -- this is the piece that
    makes 'regenerate' mean something more than 'reshuffle the same six
    options'.

    Requires ANTHROPIC_API_KEY in the environment. Falls back to
    TemplateContentGenerator automatically on any API error (missing key,
    rate limit, network failure) so a live campaign never hard-crashes on
    an LLM hiccup -- it just temporarily reverts to template-based mutation.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", campaign_context: Optional[str] = None):
        self.model = model
        self.campaign_context = campaign_context or ""
        self._fallback = TemplateContentGenerator()
        self._client = None
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=api_key)

    def generate(self, trait_name: str, current_value: Optional[str] = None,
                 exclude: Optional[List[str]] = None) -> str:
        if self._client is None:
            return self._fallback.generate(trait_name, current_value, exclude)

        exclude_list = list(exclude or [])
        if current_value:
            exclude_list.append(current_value)

        prompt = self._build_prompt(trait_name, exclude_list)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=60,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            # Strip a leading/trailing quote if the model wrapped its answer.
            text = text.strip('"').strip()
            if not text:
                raise ValueError("Empty response from LLM")
            return text
        except Exception:
            # Any API failure (no key, rate limit, network, malformed
            # response) degrades to the template pool rather than crashing
            # a live evolutionary run.
            return self._fallback.generate(trait_name, current_value, exclude)

    def _build_prompt(self, trait_name: str, exclude_list: List[str]) -> str:
        description = _TRAIT_PROMPTS.get(trait_name, f"a short ad creative value for '{trait_name}'")
        lines = [f"Write {description} for a digital ad campaign."]
        if self.campaign_context:
            lines.append(f"Campaign context: {self.campaign_context}")
        if exclude_list:
            joined = "; ".join(exclude_list[:8])
            lines.append(f"Do not reuse any of these already-tried variants: {joined}")
        lines.append("Respond with ONLY the value itself, no explanation, no quotes.")
        return "\n".join(lines)


DEFAULT_GENERATOR: ContentGenerator = TemplateContentGenerator()
