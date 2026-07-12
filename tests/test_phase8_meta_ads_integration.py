"""
Phase 8 test -- Meta Ads integration logic, verified against a mocked HTTP
layer (no credentials or network access to graph.facebook.com exist in this
sandbox, same honesty constraint as Phase 4's LLMContentGenerator).

What this proves:
  1. create_ad_for_genome() builds a correctly-shaped Creative -> Ad request
     pair, in the right order, with the right CTA enum mapping.
  2. apply_budget_allocation() correctly converts the bandit's relative
     weights into real budget, clamped to Meta's minimum -- and that a
     starved arm still gets the platform's minimum rather than $0.
  3. poll_insights() correctly parses Meta's actual response shape
     (ad-id-keyed rows with an `actions` list) back into the
     {genome_id: {impressions, clicks, conversions}} shape
     ThompsonSamplingAllocator.update() expects.

What this does NOT prove: that any of this actually works against a live
ad account. That requires real credentials and is out of scope for what
can be verified here.
"""
from unittest.mock import MagicMock, patch

from genomes.genome_ad import AdGenome
from evolution.meta_ads_backend import MetaAdsBackend, MetaAdsConfig, CTA_TYPE_MAP


def make_config():
    return MetaAdsConfig(
        ad_account_id="act_999",
        access_token="fake_token",
        page_id="page_123",
        campaign_id="camp_123",
        adset_id="adset_123",
    )


def fake_image_resolver(genome):
    return f"https://cdn.example.com/{genome.genome_id}.png"


def test_creative_and_ad_payload_construction():
    genome = AdGenome()
    genome.genes[0].value = "Start free trial"  # force a known headline slot... actually set via trait

    # AdGenome doesn't index genes by trait name for direct set -- use trait_map
    t = genome.trait_map()
    t["cta"].value = "Start free trial"  # reuse a headline candidate to test unmapped CTA fallback path
    t["headline"].value = "Join thousands who already switched"

    backend = MetaAdsBackend(make_config(), image_asset_resolver=fake_image_resolver)

    mock_session = MagicMock()
    creative_response = MagicMock()
    creative_response.json.return_value = {"id": "creative_555"}
    creative_response.raise_for_status.return_value = None
    ad_response = MagicMock()
    ad_response.json.return_value = {"id": "ad_777"}
    ad_response.raise_for_status.return_value = None
    mock_session.post.side_effect = [creative_response, ad_response]
    backend.session = mock_session

    ad_id = backend.create_ad_for_genome(genome)

    assert ad_id == "ad_777"
    assert backend.genome_to_ad_id[genome.genome_id] == "ad_777"
    assert backend.genome_to_creative_id[genome.genome_id] == "creative_555"
    assert mock_session.post.call_count == 2, "Must create Creative BEFORE Ad -- Meta's required object order"

    creative_call, ad_call = mock_session.post.call_args_list
    creative_path = creative_call.args[0]
    creative_payload = creative_call.kwargs["data"]
    assert "adcreatives" in creative_path
    assert creative_payload["object_story_spec"]["link_data"]["message"] == "Join thousands who already switched"
    assert creative_payload["object_story_spec"]["link_data"]["image_url"] == f"https://cdn.example.com/{genome.genome_id}.png"

    ad_path = ad_call.args[0]
    ad_payload = ad_call.kwargs["data"]
    assert ad_path.endswith("/ads")
    assert ad_payload["status"] == "PAUSED", "New ads must start PAUSED pending review, never auto-live"
    assert ad_payload["creative"]["creative_id"] == "creative_555"
    print("PASS: Creative created before Ad, correct payload shape, new ads start PAUSED")


def test_cta_mapping_has_fallback_for_unmapped_text():
    # Simulates a Phase-4 LLM-generated CTA that isn't in the hand-mapped pool.
    genome = AdGenome()
    t = genome.trait_map()
    t["cta"].value = "Grab your discount before it's gone"  # not in CTA_TYPE_MAP

    backend = MetaAdsBackend(make_config(), image_asset_resolver=fake_image_resolver)
    payload = backend._creative_payload(genome)
    cta_type = payload["object_story_spec"]["link_data"]["call_to_action"]["type"]
    assert cta_type == "LEARN_MORE", "Unmapped CTA text must fall back to a safe default enum, not crash"
    print(f"PASS: unmapped CTA text falls back to default enum ({cta_type}) instead of raising")


def test_budget_allocation_respects_minimum_and_proportions():
    backend = MetaAdsBackend(make_config(), image_asset_resolver=fake_image_resolver)
    backend.genome_to_ad_id = {"g1": "ad_1", "g2": "ad_2", "g3": "ad_3"}
    backend.session = MagicMock()
    backend.session.post.return_value = MagicMock(json=lambda: {}, raise_for_status=lambda: None)

    # g3 barely won any bandit rounds -- its raw proportional share would be
    # below Meta's real-world minimum daily budget.
    weights = {"g1": 900.0, "g2": 95.0, "g3": 5.0}
    applied = backend.apply_budget_allocation(weights, total_daily_budget_cents=10000)

    assert applied["g1"] > applied["g2"] > applied["g3"] or applied["g3"] == backend.config.min_daily_budget_cents
    assert applied["g3"] >= backend.config.min_daily_budget_cents, \
        "A near-starved arm must still get the platform's real minimum, not a proportional near-zero"
    assert applied["g1"] > applied["g3"], "The clear winner should still get materially more budget"
    print(f"PASS: budget allocation g1={applied['g1']}c g2={applied['g2']}c g3={applied['g3']}c "
          f"(g3 clamped to platform minimum instead of getting ~50 cents)")


def test_poll_insights_parses_real_response_shape():
    backend = MetaAdsBackend(make_config(), image_asset_resolver=fake_image_resolver)
    backend.genome_to_ad_id = {"g1": "ad_111", "g2": "ad_222"}
    backend.session = MagicMock()

    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.json.return_value = {
        "data": [
            {"ad_id": "ad_111", "impressions": "4820", "clicks": "112", "spend": "18.40",
             "actions": [{"action_type": "link_click", "value": "97"},
                        {"action_type": "post_engagement", "value": "140"}]},
            {"ad_id": "ad_222", "impressions": "3110", "clicks": "40", "spend": "9.10",
             "actions": [{"action_type": "link_click", "value": "31"}]},
            {"ad_id": "ad_999_unknown", "impressions": "500", "clicks": "5", "spend": "1.00", "actions": []},
        ]
    }
    backend.session.get.return_value = fake_response

    results = backend.poll_insights(since_epoch=1000, until_epoch=2000)

    assert set(results.keys()) == {"g1", "g2"}, "Unknown ad_id (not in genome_to_ad_id) must be silently dropped"
    assert results["g1"]["impressions"] == 4820
    assert results["g1"]["clicks"] == 112
    assert results["g1"]["conversions"] == 97  # pulled from the actions list, not a top-level field
    assert results["g2"]["conversions"] == 31
    print(f"PASS: insights parsed correctly -- g1={results['g1']}, unknown ad_id correctly dropped")


if __name__ == "__main__":
    test_creative_and_ad_payload_construction()
    test_cta_mapping_has_fallback_for_unmapped_text()
    test_budget_allocation_respects_minimum_and_proportions()
    test_poll_insights_parses_real_response_shape()
    print("\nPASSED: Meta Ads integration logic verified against a mocked HTTP layer. "
          "NOT verified against a real ad account -- no credentials/network access in this sandbox.")
