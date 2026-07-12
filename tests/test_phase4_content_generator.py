"""
Phase 4 integration test.

No ANTHROPIC_API_KEY exists in this sandbox, so LLMContentGenerator is
exercised here through a mock Anthropic client rather than a live call.
The integration code itself (prompt construction, response parsing,
exception -> fallback path) is real and unit-testable regardless of whether
a live key is present; this test proves that code is correct so it can be
trusted the moment a real key is configured (see run_live_demo() at the
bottom, which is skipped unless ANTHROPIC_API_KEY is actually set).

What this proves:
  1. TemplateContentGenerator (default) behaves identically to the Phase
     1-3 hardcoded regenerate() -- backward compatibility, already
     confirmed by rerunning Phase 1/2 tests unchanged.
  2. LLMContentGenerator, given a mock client, builds a sensible prompt and
     correctly parses a real-shaped Anthropic API response.
  3. LLMContentGenerator falls back to the template pool -- doesn't crash --
     when the mock client raises (simulating a rate limit / network error).
  4. AdGenome.set_content_generator() lets an LLM-backed generator produce
     genuinely novel trait values (outside TRAIT_CANDIDATES) during a real
     mutate() call, with zero changes needed to evolution.py or genome_ad.py's
     embedding logic.
"""
import os
import random
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from genomes.genome_ad import AdGenome, TRAIT_CANDIDATES
from genomes.content_generator import (
    TemplateContentGenerator, LLMContentGenerator, ContentGenerator
)

random.seed(3)
np.random.seed(3)


def make_mock_anthropic_client(reply_text: str):
    """Builds a fake Anthropic client shaped like the real SDK's response
    object (response.content is a list of blocks with .type/.text)."""
    mock_client = MagicMock()
    mock_block = SimpleNamespace(type="text", text=reply_text)
    mock_response = SimpleNamespace(content=[mock_block])
    mock_client.messages.create.return_value = mock_response
    return mock_client


def test_template_generator_matches_pool():
    gen = TemplateContentGenerator()
    for _ in range(20):
        value = gen.generate("headline", current_value="Stop overpaying for the same results")
        assert value in TRAIT_CANDIDATES["headline"]
        assert value != "Stop overpaying for the same results"
    print("PASS: TemplateContentGenerator only returns pool values, never repeats current_value")


def test_llm_generator_prompt_and_parsing():
    llm_gen = LLMContentGenerator()
    llm_gen._client = make_mock_anthropic_client('"Ditch the guesswork. Start converting."')

    result = llm_gen.generate("headline", current_value="Old headline",
                              exclude=["Some other tried headline"])

    call_args = llm_gen._client.messages.create.call_args
    prompt = call_args.kwargs["messages"][0]["content"]
    assert "headline" in prompt.lower() or "punchy" in prompt.lower()
    assert "Old headline" in prompt or "Some other tried headline" in prompt

    # Quotes from a wrapped model response should be stripped.
    assert result == "Ditch the guesswork. Start converting."
    assert result not in TRAIT_CANDIDATES["headline"], (
        "This value should be genuinely novel, not from the fixed pool")
    print(f"PASS: LLMContentGenerator built a valid prompt and parsed a novel value: \"{result}\"")


def test_llm_generator_falls_back_on_api_error():
    llm_gen = LLMContentGenerator()
    broken_client = MagicMock()
    broken_client.messages.create.side_effect = RuntimeError("simulated rate limit error")
    llm_gen._client = broken_client

    result = llm_gen.generate("cta", current_value="Try it now")
    assert result in TRAIT_CANDIDATES["cta"], "Should have silently fallen back to the template pool"
    print(f"PASS: API failure fell back to template pool cleanly, got \"{result}\" (no crash)")


def test_genome_mutation_uses_injected_generator():
    novel_headline = "Your data deserves better tools"
    mock_client = make_mock_anthropic_client(novel_headline)
    llm_gen = LLMContentGenerator()
    llm_gen._client = mock_client

    AdGenome.set_content_generator(llm_gen)
    try:
        genome = AdGenome()
        # Force an architecture mutation (regenerate) on every gene to
        # guarantee the injected generator actually gets exercised.
        genome.mutate(weight_mutation_rate=0.0, architecture_mutation_rate=1.0,
                      layer_mutation_rate=0.0)
        values = [g.value for g in genome.genes]
        assert novel_headline in values, "Injected LLM generator's output should appear on the genome"

        # Embedding must exist and be finite for the novel string -- proves
        # _hash_embedding() needed zero changes to handle LLM-generated text.
        emb = genome.combined_embedding()
        assert np.all(np.isfinite(emb))
        print(f"PASS: genome adopted LLM-generated trait \"{novel_headline}\" with a valid embedding")
    finally:
        AdGenome.set_content_generator(None)  # don't leak state into other tests


def test_default_behavior_unchanged_when_no_generator_set():
    assert AdGenome._content_generator is None
    genome = AdGenome()
    genome.mutate(weight_mutation_rate=0.0, architecture_mutation_rate=1.0, layer_mutation_rate=0.0)
    for gene in genome.genes:
        assert gene.value in TRAIT_CANDIDATES[gene.trait_name]
    print("PASS: with no generator injected, mutation still only draws from the fixed pool (Phase 1-3 behavior)")


def run_live_demo():
    """Only runs against the real Anthropic API if ANTHROPIC_API_KEY is
    actually set in the environment -- not available in this sandbox."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIPPED live demo: no ANTHROPIC_API_KEY in this environment. "
              "Set one and rerun to see a real LLM generate ad trait content.")
        return
    llm_gen = LLMContentGenerator(campaign_context="B2B project management SaaS, mid-market")
    for trait in ["headline", "tone", "cta"]:
        print(f"{trait}: {llm_gen.generate(trait)}")


if __name__ == "__main__":
    test_template_generator_matches_pool()
    test_llm_generator_prompt_and_parsing()
    test_llm_generator_falls_back_on_api_error()
    test_genome_mutation_uses_injected_generator()
    test_default_behavior_unchanged_when_no_generator_set()
    run_live_demo()
    print("\nPASSED: content generator interface, LLM integration, and fallback path all verified.")
