"""
Phase 9 -- LLM-judged fitness, replacing AdMarketEnv's synthetic segment
math with real creative judgment for genomes that get selected/bred.

This is a genuinely different kind of fitness source than everything built
so far. AdMarketEnv scores a genome against fictional audience segments I
invented; BudgetedMarketEvaluator adds a bandit on top of that same fiction.
Neither of them is "wrong" as engineering, but neither actually evaluates
creative quality -- they evaluate fit against math I made up.
LLMJudgeFitnessBackend asks an actual model to judge the actual creative
content against a stated campaign brief. That's real judgment, though still
not the same as real measured performance (Phase 8's ad-platform
integration) -- see the module-level honesty notes below.

Design decisions, and why they're not the same as Phase 4's pattern:

1. BATCH judging, not one call per genome. An LLM judging 8 creatives
   side-by-side in one call gives more consistent, better-calibrated scores
   than 8 isolated calls (comparative judgment is a well-established
   strength of LLM evaluators over pointwise scoring) -- and it's ~8x
   cheaper and faster. Cost matters here in a way it didn't in Phase 4:
   judging a population of 40 across 20 generations at batch_size=8 is
   ~100 API calls total, not 800.

2. An explicit 1-10 rubric with anchor descriptions is embedded in every
   prompt, specifically because batch-relative scores from DIFFERENT
   batches aren't directly comparable otherwise -- batch A's "8/10" could
   mean something different from batch B's "8/10" if batch B happened to
   contain stronger competition. Anchoring to absolute rubric language
   (not just "rank these") is the mitigation; a further improvement would
   be seeding every batch with 1-2 fixed reference creatives for cross-batch
   calibration, not implemented here.

3. NO SILENT FALLBACK when the API is unavailable -- this is a deliberate
   reversal of Phase 4's design. LLMContentGenerator falling back to
   template text on API failure is a safe degradation (worst case, less
   creative variety). A fitness source silently reverting to something else
   would corrupt the evolutionary signal without anyone noticing --
   genomes would be selected/bred based on meaningless numbers while every
   log line looks normal. LLMJudgeUnavailableError is raised instead,
   forcing the caller (a real campaign runner) to explicitly decide what to
   do -- pause the campaign, alert someone -- rather than silently
   corrupting results.

4. Still text-only judgment. image_style is a style DIRECTION ("bold flat
   illustration, high contrast"), not a rendered image -- same gap flagged
   in Phase 8. The system prompt tells the judge this explicitly so it
   doesn't pretend to have seen an image it hasn't. A further extension
   would pass an actual rendered image to a vision-capable call once real
   image generation exists in this project (it doesn't yet).
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

from genomes.genome_ad import AdGenome


class LLMJudgeUnavailableError(RuntimeError):
    """Raised instead of silently degrading fitness -- see design note (3)
    in the module docstring."""


@dataclass
class CampaignBrief:
    """Judgment is meaningless without stated context -- 'which tagline is
    better' depends entirely on who's reading it. This is required, not
    optional, unlike AdMarketEnv which needed no external context because
    its 'audience' was hardcoded segment math."""
    audience: str
    brand_voice: str
    goal: str


class LLMJudgeFitnessBackend:
    def __init__(self, brief: CampaignBrief, model: str = "claude-sonnet-4-6",
                 batch_size: int = 8, api_key: Optional[str] = None):
        self.brief = brief
        self.model = model
        self.batch_size = batch_size
        self._client = None
        import os
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=key)

    def _system_prompt(self) -> str:
        return (
            "You are an expert creative strategist judging digital ad effectiveness. "
            "Be discriminating -- most creatives should NOT score above 7; reserve 9-10 "
            "for genuinely exceptional work. Note that 'image_style' describes a visual "
            "DIRECTION, not an actual rendered image -- judge it as a stated creative "
            "concept, not as if you had seen a photo. Respond with ONLY a JSON array, "
            "no markdown code fences, no text outside the JSON."
        )

    def _build_prompt(self, batch: List[AdGenome]) -> str:
        lines = [
            f"Campaign brief -- Audience: {self.brief.audience} | "
            f"Brand voice: {self.brief.brand_voice} | Goal: {self.brief.goal}",
            "",
            "Rubric (1-10):",
            "1-3: generic, forgettable, could belong to any brand",
            "4-6: decent but unremarkable, no standout hook",
            "7-8: strong, differentiated, clearly fits the brief",
            "9-10: exceptional, memorable, precisely targeted",
            "",
            "Creatives to judge:",
        ]
        for g in batch:
            t = g.trait_map()
            badges = [k for k, v in g.optional_traits.items() if v]
            lines.append(f"- id: {g.genome_id}")
            lines.append(f'  headline: "{t["headline"].value}"')
            lines.append(f"  image_style: {t['image_style'].value}")
            lines.append(f"  cta: {t['cta'].value}")
            lines.append(f"  tone: {t['tone'].value}")
            lines.append(f"  color_scheme: {t['color_scheme'].value}")
            if badges:
                lines.append(f"  extra_modules: {', '.join(badges)}")
        lines.append("")
        lines.append('JSON array format: [{"id": "...", "score": 7.5, "reasoning": "one sentence"}, ...]')
        return "\n".join(lines)

    def _judge_batch(self, batch: List[AdGenome]) -> Dict[str, Dict[str, Any]]:
        if self._client is None:
            raise LLMJudgeUnavailableError("No LLM client configured.")
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max(200, 120 * len(batch)),
                system=self._system_prompt(),
                messages=[{"role": "user", "content": self._build_prompt(batch)}],
            )
            text = "".join(
                b.text for b in response.content if getattr(b, "type", None) == "text"
            ).strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())

            results: Dict[str, Dict[str, Any]] = {}
            for item in parsed:
                results[item["id"]] = {
                    "score": max(1.0, min(10.0, float(item["score"]))),
                    "reasoning": item.get("reasoning", ""),
                }
            missing = [g.genome_id for g in batch if g.genome_id not in results]
            if missing:
                raise ValueError(f"Judge response missing scores for: {missing}")
            return results
        except LLMJudgeUnavailableError:
            raise
        except Exception as e:
            raise LLMJudgeUnavailableError(f"LLM judge call failed or returned unparseable output: {e}") from e

    def evaluate_population(self, population: List[AdGenome]) -> List[Dict[str, Any]]:
        """Judges the full population in batches, sets .fitness/.norm_fitness
        on every genome (the two attributes EvolutionEngine's selection
        actually reads -- same contract every other fitness source in this
        project has honored since Phase 1). Raises LLMJudgeUnavailableError
        rather than returning degraded results -- see design note (3)."""
        results: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(population), self.batch_size):
            batch = population[i:i + self.batch_size]
            results.update(self._judge_batch(batch))

        for genome in population:
            r = results.get(genome.genome_id)
            genome.fitness = float(r["score"]) if r else 0.0

        raw = [g.fitness for g in population]
        lo, hi = min(raw), max(raw)
        for g in population:
            g.norm_fitness = (g.fitness - lo) / (hi - lo + 1e-9)

        return [
            {"genome_id": g.genome_id, "fitness": g.fitness,
             "reasoning": results.get(g.genome_id, {}).get("reasoning", "")}
            for g in population
        ]
