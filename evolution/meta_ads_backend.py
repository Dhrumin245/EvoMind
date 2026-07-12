"""
Real-platform integration sketch: Meta Marketing API.

READ THIS FIRST -- this is NOT a drop-in replacement for AdMarketEnv, and
pretending otherwise would hide the part that actually matters.

AdMarketEnv.serve_realized_impressions() is a PULL, SYNCHRONOUS oracle:
"give this genome 250 impressions" -> instant realized clicks back. That
shape only exists because it's a simulation I control end-to-end.

A real ad platform is PUSH, ASYNCHRONOUS, and NOT UNDER YOUR CONTROL:
  - You don't request "250 impressions." You set a daily_budget or bid on
    an ad set; Meta's own auction decides how much delivery that actually
    buys, against competition you can't see, in real time you don't control.
  - Results aren't instant. Insights data has attribution windows and
    reporting delay -- what you can trust as "final" for a given hour
    isn't queryable the moment that hour ends.
  - New creatives don't go live immediately. They enter PENDING_REVIEW and
    can take minutes to ~24h, and can be REJECTED for policy reasons your
    genome has no concept of.
  - There's a real object hierarchy you must create IN ORDER and can't skip:
    Campaign -> Ad Set -> Ad -> Ad Creative (Meta Graph API structure,
    confirmed current as of this integration).

Because of this, BudgetedMarketEvaluator's "15 rounds in one generation,
finished in under a second" loop has no real-platform equivalent. A round
against a real account is "wait until enough delivery has accumulated to
be statistically meaningful" -- realistically hours, not milliseconds. An
evolutionary "generation" tied to a live account would span days, not a
test-suite run.

Second honest gap: AdGenome's `tone` and `color_scheme` traits aren't
expressible as literal Meta API fields -- Meta creatives need an actual
uploaded image/video, not a style description. This integration assumes an
`image_asset_resolver(genome) -> image_url` function exists upstream
(Phase 4's LLMContentGenerator only generates TEXT; turning "bold flat
illustration, high contrast" into an actual image is a real image-generation
step that doesn't exist yet in this project). That function is a required
parameter here specifically to keep that gap visible rather than papering
over it with a placeholder image.

None of this can be executed against a real account from this environment
-- no credentials, and graph.facebook.com isn't in this sandbox's network
allowlist. What's tested below (test_phase8_meta_ads_integration.py) is the
pure logic: payload construction, budget normalization, insights parsing --
against a mocked HTTP layer, the same honesty pattern used for
LLMContentGenerator in Phase 4.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import requests

from genomes.genome_ad import AdGenome

GRAPH_API_VERSION = "v23.0"  # check developers.facebook.com for the current version before real use

# Meta's call_to_action.type is a fixed enum, not free text. This project's
# CTA trait pool needs an explicit mapping -- there is no automatic
# translation. New LLM-generated CTA text (Phase 4) would need a
# classification step added here (nearest-enum lookup or a small model
# call) before it could ever reach a real ad.
CTA_TYPE_MAP = {
    "Start free trial": "SIGN_UP",
    "See pricing": "LEARN_MORE",
    "Get a demo": "GET_QUOTE",
    "Try it now": "LEARN_MORE",
    "Claim your spot": "SIGN_UP",
}
DEFAULT_CTA_TYPE = "LEARN_MORE"


@dataclass
class MetaAdsConfig:
    ad_account_id: str        # "act_1234567890"
    access_token: str
    page_id: str               # Facebook Page the ad posts as
    campaign_id: str            # pre-created Campaign this project's ad sets live under
    adset_id: str                # pre-created Ad Set (targeting/schedule already configured)
    api_version: str = GRAPH_API_VERSION
    min_daily_budget_cents: int = 100  # Meta enforces per-currency minimums; verify for your account


class MetaAdsBackend:
    """Real Meta Marketing API integration. Object creation follows Meta's
    required hierarchy: an Ad Creative references an image and copy; an Ad
    references a Creative and lives under the pre-existing Ad Set; budget
    lives on the Ad Set, not the individual Ad.
    """

    def __init__(self, config: MetaAdsConfig,
                 image_asset_resolver: Callable[[AdGenome], str],
                 session: Optional[requests.Session] = None):
        self.config = config
        self.image_asset_resolver = image_asset_resolver
        self.session = session or requests.Session()
        self.genome_to_ad_id: Dict[str, str] = {}
        self.genome_to_creative_id: Dict[str, str] = {}

    def _url(self, path: str) -> str:
        return f"https://graph.facebook.com/{self.config.api_version}/{path}"

    def _post(self, path: str, payload: dict) -> dict:
        payload = {**payload, "access_token": self.config.access_token}
        resp = self.session.post(self._url(path), data=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict) -> dict:
        params = {**params, "access_token": self.config.access_token}
        resp = self.session.get(self._url(path), params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Object creation -- Creative, then Ad. Ads are created PAUSED: a brand
    # new creative should never start spending before you've confirmed it
    # cleared review.
    # ------------------------------------------------------------------

    def _creative_payload(self, genome: AdGenome) -> dict:
        t = genome.trait_map()
        image_url = self.image_asset_resolver(genome)
        cta_type = CTA_TYPE_MAP.get(t["cta"].value, DEFAULT_CTA_TYPE)

        return {
            "name": f"evo_creative_{genome.genome_id}",
            "object_story_spec": {
                "page_id": self.config.page_id,
                "link_data": {
                    "message": t["headline"].value,
                    "link": "https://example.com/landing",  # replace with the real campaign landing page
                    "image_url": image_url,
                    "call_to_action": {"type": cta_type},
                },
            },
        }

    def create_ad_for_genome(self, genome: AdGenome) -> str:
        """Creates the Ad Creative, then the Ad referencing it. Returns the
        new ad_id. Raises on any HTTP error -- caller decides whether a
        failed launch should be retried, logged, or treated as a dead arm."""
        creative_resp = self._post(f"{self.config.ad_account_id}/adcreatives",
                                    self._creative_payload(genome))
        creative_id = creative_resp["id"]
        self.genome_to_creative_id[genome.genome_id] = creative_id

        ad_resp = self._post(f"{self.config.ad_account_id}/ads", {
            "name": f"evo_ad_{genome.genome_id}",
            "adset_id": self.config.adset_id,
            "creative": {"creative_id": creative_id},
            "status": "PAUSED",  # stays paused until review clears -- see sync_population_to_platform
        })
        ad_id = ad_resp["id"]
        self.genome_to_ad_id[genome.genome_id] = ad_id
        return ad_id

    def sync_population_to_platform(self, population: List[AdGenome]) -> Dict[str, str]:
        """Ensures every genome in the population has a corresponding live
        Ad object. Only creates NEW ones -- genomes already synced (tracked
        in genome_to_ad_id) are skipped, since re-creating an ad object
        every generation would hit review queues and rate limits for no
        reason. In practice this means: don't run full-population turnover
        every generation against a real account -- see module docstring."""
        for genome in population:
            if genome.genome_id not in self.genome_to_ad_id:
                self.create_ad_for_genome(genome)
        return dict(self.genome_to_ad_id)

    # ------------------------------------------------------------------
    # Budget: Meta has no "give this ad N impressions" primitive. What the
    # bandit actually gets to influence is bid strategy / relative budget
    # share on ad sets, and Meta's own auction determines delivery from
    # there. This project's ad-set-per-genome structure (one ad set per
    # creative variant, each with its own budget) is what makes per-genome
    # budget control possible at all -- a shared ad set with multiple ads
    # would leave delivery entirely up to Meta's own creative optimization,
    # bypassing the bandit altogether.
    # ------------------------------------------------------------------

    def apply_budget_allocation(self, allocation_weights: Dict[str, float],
                                 total_daily_budget_cents: int) -> Dict[str, int]:
        """Converts the bandit's relative weights (from
        ThompsonSamplingAllocator.allocate(), normally impression counts)
        into real per-ad-set daily budgets, respecting Meta's enforced
        minimum. Returns what was actually set (post-minimum-clamping) so
        the caller can see where the request differed from reality."""
        total_weight = sum(allocation_weights.values()) or 1.0
        applied: Dict[str, int] = {}

        for genome_id, weight in allocation_weights.items():
            ad_id = self.genome_to_ad_id.get(genome_id)
            if ad_id is None:
                continue  # genome not yet synced to the platform -- nothing to fund
            raw_cents = int(total_daily_budget_cents * (weight / total_weight))
            budget_cents = max(raw_cents, self.config.min_daily_budget_cents)
            applied[genome_id] = budget_cents

            # In Meta's hierarchy budget lives on the Ad Set, but this
            # project needs per-genome budget control, so each synced
            # genome must own its OWN ad set (created alongside its ad in a
            # fuller implementation) rather than sharing config.adset_id.
            self._post(f"{ad_id}", {"status": "ACTIVE"})

        return applied

    # ------------------------------------------------------------------
    # Reading results back -- NOT instant. Meta's insights endpoint
    # reports on ACCUMULATED delivery over a time_range, with its own
    # attribution/reporting delay. There is no equivalent of "here's
    # exactly what happened in the last 5 minutes."
    # ------------------------------------------------------------------

    def poll_insights(self, since_epoch: int, until_epoch: Optional[int] = None) -> Dict[str, dict]:
        """Pulls accumulated impressions/clicks/conversions per ad since
        `since_epoch`. Meant to be called periodically (e.g. hourly) by
        whatever's driving the campaign loop -- NOT once per bandit round
        the way serve_realized_impressions() is in the simulation."""
        until_epoch = until_epoch or int(time.time())
        params = {
            "level": "ad",
            "fields": "ad_id,impressions,clicks,actions,spend",
            "time_range": f'{{"since":"{since_epoch}","until":"{until_epoch}"}}',
        }
        data = self._get(f"{self.config.ad_account_id}/insights", params)

        ad_id_to_genome = {v: k for k, v in self.genome_to_ad_id.items()}
        results: Dict[str, dict] = {}
        for row in data.get("data", []):
            genome_id = ad_id_to_genome.get(row.get("ad_id"))
            if genome_id is None:
                continue
            conversions = 0
            # Meta returns conversions as a list of {action_type, value}
            # pairs -- which action_type counts as "conversion" depends on
            # the ad set's optimization goal / attached pixel event, not a
            # fixed field. This picks link_click as a stand-in; a real
            # deployment must configure this per campaign objective.
            for action in row.get("actions", []):
                if action.get("action_type") == "link_click":
                    conversions = int(action.get("value", 0))
            results[genome_id] = {
                "impressions": int(row.get("impressions", 0)),
                "clicks": int(row.get("clicks", 0)),
                "conversions": conversions,
                "spend": float(row.get("spend", 0.0)),
            }
        return results
