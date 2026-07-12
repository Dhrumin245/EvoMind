"""
Phase 6 -- campaign persistence and lifecycle, modeled on EvoMind's real
api/job_manager.py + api/trainer.py.

Scope decision, stated plainly: the real JobManager is a distributed,
multi-worker system -- SQLite-backed job records PLUS a command queue,
lease-based job control, and worker heartbeats, so multiple worker
processes across machines can coordinate who's running what. That machinery
exists to solve horizontal scaling, which is not what "prove a campaign can
be launched/paused/resumed" needs. What's reused here is the genuinely
domain-agnostic part: a status-enum-driven record persisted to SQLite
(JobRecord's tenant_id/job_id/status/generation shape), and a
save_checkpoint()/resume()/get_best_genome() trainer interface mirroring
api/trainer.py's EvoTrainer. The distributed lease/lock/heartbeat layer is
NOT reproduced -- this runs as one process, driven by explicit step() calls
(in production, a background loop or api/worker.py-style process would call
step() repeatedly; here, the API and the test call it directly).

Checkpointing is real, not simulated: every step serializes the full
population via AdGenome.to_dict() (built in Phase 1) plus the Thompson
Sampling allocator's per-arm posterior state, to a JSON file on disk. Resume
reloads via AdGenome.from_dict() -- proving that serialization path
actually round-trips, not just that the class has the methods.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from genomes.genome_ad import AdGenome
from environments.ad_market_env import AdMarketEnv, MarketEvalConfig
from evolution.ad_evolution_engine import AdEvolutionEngine
from evolution.budget_allocator import ThompsonSamplingAllocator, BudgetedMarketEvaluator, BetaArm
from api.ad_schemas import (
    CampaignConfig, CampaignStatusEnum, CampaignStatusResponse, CreativeSummary,
)


class CampaignNotRunningError(RuntimeError):
    """Raised when step() is called on a campaign that's paused/stopped --
    the API layer translates this into an HTTP 409."""


class CampaignNotFoundError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _CampaignRuntime:
    """In-memory live objects for a campaign. Deliberately NOT what's
    persisted -- genomes/engines aren't SQL-serializable, so the source of
    truth on disk is the checkpoint JSON + the SQLite status row. This is
    rebuildable from those two things at any time (that's the whole point
    of the resume path)."""
    population: List[AdGenome]
    engine: AdEvolutionEngine
    market_env: AdMarketEnv
    evaluator: BudgetedMarketEvaluator
    config: CampaignConfig


def _best_creative_summary(genome: AdGenome) -> CreativeSummary:
    t = genome.trait_map()
    return CreativeSummary(
        genome_id=genome.genome_id,
        headline=t["headline"].value,
        image_style=t["image_style"].value,
        cta=t["cta"].value,
        tone=t["tone"].value,
        color_scheme=t["color_scheme"].value,
        optional_traits=dict(genome.optional_traits),
        fitness=genome.fitness,
    )


class AdCampaignManager:
    def __init__(self, db_path: str = ":memory:", checkpoint_dir: str = "/tmp/ad_campaigns"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._runtimes: Dict[str, _CampaignRuntime] = {}

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS campaigns (
                tenant_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                generation INTEGER NOT NULL DEFAULT 0,
                total_generations INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                best_fitness REAL NOT NULL DEFAULT 0.0,
                best_creative_json TEXT,
                species_count INTEGER NOT NULL DEFAULT 0,
                total_impressions_served INTEGER NOT NULL DEFAULT 0,
                checkpoint_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (tenant_id, campaign_id)
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_history (
                tenant_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                generation INTEGER NOT NULL,
                best_fitness REAL NOT NULL,
                species_count INTEGER NOT NULL,
                impressions_this_generation INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, campaign_id, generation)
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, tenant_id: str, campaign_id: str) -> Path:
        return self.checkpoint_dir / f"{tenant_id}__{campaign_id}.json"

    def _save_checkpoint(self, tenant_id: str, campaign_id: str,
                          runtime: _CampaignRuntime) -> str:
        path = self._checkpoint_path(tenant_id, campaign_id)
        payload = {
            "population": [g.to_dict() for g in runtime.population],
            "allocator_arms": {
                gid: {"alpha": arm.alpha, "beta": arm.beta}
                for gid, arm in runtime.evaluator.allocator.arms.items()
            },
            "config": runtime.config.model_dump(),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def _load_checkpoint(self, tenant_id: str, campaign_id: str) -> _CampaignRuntime:
        path = self._checkpoint_path(tenant_id, campaign_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = CampaignConfig(**payload["config"])

        population = [AdGenome.from_dict(g) for g in payload["population"]]

        allocator = ThompsonSamplingAllocator()
        for gid, arm in payload["allocator_arms"].items():
            allocator.arms[gid] = BetaArm(alpha=arm["alpha"], beta=arm["beta"])

        market_env = AdMarketEnv(config=MarketEvalConfig())
        evaluator = BudgetedMarketEvaluator(
            market_env=market_env,
            allocator=allocator,
            total_budget_per_generation=config.budget_per_generation,
            rounds_per_generation=config.rounds_per_generation,
        )
        engine = AdEvolutionEngine(
            population_size=config.population_size,
            tournament_size=3,
            elite_count=2,
            mutation_rate=config.mutation_rate,
            mutation_strength=config.mutation_strength,
            architecture_mutation_rate=config.architecture_mutation_rate,
            genome_cls=AdGenome,
            speciation_enabled=True,
            novelty_archive_enabled=True,
            compatibility_threshold=config.compatibility_threshold,
            min_species_size=3,
        )
        return _CampaignRuntime(population=population, engine=engine,
                                 market_env=market_env, evaluator=evaluator, config=config)

    def _row_to_response(self, row: sqlite3.Row) -> CampaignStatusResponse:
        best_creative = json.loads(row["best_creative_json"]) if row["best_creative_json"] else None
        return CampaignStatusResponse(
            campaign_id=row["campaign_id"],
            tenant_id=row["tenant_id"],
            name=row["name"],
            status=CampaignStatusEnum(row["status"]),
            generation=row["generation"],
            total_generations=row["total_generations"],
            species_count=row["species_count"],
            best_fitness=row["best_fitness"],
            best_creative=CreativeSummary(**best_creative) if best_creative else None,
            total_impressions_served=row["total_impressions_served"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _get_row(self, tenant_id: str, campaign_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM campaigns WHERE tenant_id = ? AND campaign_id = ?",
            (tenant_id, campaign_id),
        ).fetchone()
        if row is None:
            raise CampaignNotFoundError(f"No campaign {campaign_id} for tenant {tenant_id}")
        return row

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def create_campaign(self, tenant_id: str, config: CampaignConfig) -> CampaignStatusResponse:
        campaign_id = f"camp_{uuid.uuid4().hex[:10]}"
        population = [AdGenome() for _ in range(config.population_size)]
        market_env = AdMarketEnv(config=MarketEvalConfig())
        evaluator = BudgetedMarketEvaluator(
            market_env=market_env,
            allocator=ThompsonSamplingAllocator(),
            total_budget_per_generation=config.budget_per_generation,
            rounds_per_generation=config.rounds_per_generation,
        )
        engine = AdEvolutionEngine(
            population_size=config.population_size,
            tournament_size=3,
            elite_count=2,
            mutation_rate=config.mutation_rate,
            mutation_strength=config.mutation_strength,
            architecture_mutation_rate=config.architecture_mutation_rate,
            genome_cls=AdGenome,
            speciation_enabled=True,
            novelty_archive_enabled=True,
            compatibility_threshold=config.compatibility_threshold,
            min_species_size=3,
        )
        runtime = _CampaignRuntime(population=population, engine=engine,
                                    market_env=market_env, evaluator=evaluator, config=config)
        self._runtimes[campaign_id] = runtime
        checkpoint_path = self._save_checkpoint(tenant_id, campaign_id, runtime)

        now = _now()
        self._conn.execute(
            """INSERT INTO campaigns
               (tenant_id, campaign_id, name, status, generation, total_generations,
                config_json, best_fitness, best_creative_json, species_count,
                total_impressions_served, checkpoint_path, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, ?, ?, 0.0, NULL, 0, 0, ?, ?, ?)""",
            (tenant_id, campaign_id, config.name, CampaignStatusEnum.QUEUED.value,
             config.total_generations, json.dumps(config.model_dump()), checkpoint_path, now, now),
        )
        self._conn.commit()
        return self.get_campaign(tenant_id, campaign_id)

    def get_campaign(self, tenant_id: str, campaign_id: str) -> CampaignStatusResponse:
        return self._row_to_response(self._get_row(tenant_id, campaign_id))

    def list_campaigns(self, tenant_id: str) -> List[CampaignStatusResponse]:
        rows = self._conn.execute(
            "SELECT * FROM campaigns WHERE tenant_id = ? ORDER BY created_at DESC", (tenant_id,)
        ).fetchall()
        return [self._row_to_response(r) for r in rows]

    def _set_status(self, tenant_id: str, campaign_id: str, status: CampaignStatusEnum) -> None:
        self._conn.execute(
            "UPDATE campaigns SET status = ?, updated_at = ? WHERE tenant_id = ? AND campaign_id = ?",
            (status.value, _now(), tenant_id, campaign_id),
        )
        self._conn.commit()

    def start_campaign(self, tenant_id: str, campaign_id: str) -> CampaignStatusResponse:
        self._get_row(tenant_id, campaign_id)  # 404s if missing
        self._set_status(tenant_id, campaign_id, CampaignStatusEnum.RUNNING)
        return self.get_campaign(tenant_id, campaign_id)

    def pause_campaign(self, tenant_id: str, campaign_id: str) -> CampaignStatusResponse:
        self._get_row(tenant_id, campaign_id)
        self._set_status(tenant_id, campaign_id, CampaignStatusEnum.PAUSED)
        return self.get_campaign(tenant_id, campaign_id)

    def resume_campaign(self, tenant_id: str, campaign_id: str) -> CampaignStatusResponse:
        row = self._get_row(tenant_id, campaign_id)
        if campaign_id not in self._runtimes:
            # Simulates recovering after a process restart: rebuild live
            # objects entirely from the persisted checkpoint, not from
            # whatever happens to still be in memory.
            self._runtimes[campaign_id] = self._load_checkpoint(tenant_id, campaign_id)
        self._set_status(tenant_id, campaign_id, CampaignStatusEnum.RUNNING)
        return self.get_campaign(tenant_id, campaign_id)

    def step(self, tenant_id: str, campaign_id: str) -> CampaignStatusResponse:
        """Advance exactly one generation. In production this is what a
        background loop / worker process calls repeatedly while status is
        RUNNING; here the API endpoint and the test call it directly."""
        row = self._get_row(tenant_id, campaign_id)
        if row["status"] != CampaignStatusEnum.RUNNING.value:
            raise CampaignNotRunningError(
                f"Campaign {campaign_id} is {row['status']}, not running -- call resume first")

        if campaign_id not in self._runtimes:
            self._runtimes[campaign_id] = self._load_checkpoint(tenant_id, campaign_id)
        runtime = self._runtimes[campaign_id]

        current_gen = row["generation"]
        results = runtime.evaluator.run_generation(runtime.population)
        impressions_this_gen = sum(r["impressions"] for r in results)

        best = max(runtime.population, key=lambda g: g.fitness)
        species_count = (len(runtime.engine.speciation_manager.species)
                          if runtime.engine.speciation_manager else 0)

        new_population = runtime.engine.create_next_generation(
            runtime.population, generation=current_gen, pop_name=campaign_id)
        if not isinstance(new_population, list):
            new_population = new_population.genomes
        runtime.population = new_population

        next_gen = current_gen + 1
        checkpoint_path = self._save_checkpoint(tenant_id, campaign_id, runtime)

        next_status = (CampaignStatusEnum.STOPPED if next_gen >= row["total_generations"]
                       else CampaignStatusEnum.RUNNING)

        self._conn.execute(
            """UPDATE campaigns SET generation = ?, status = ?, best_fitness = ?,
               best_creative_json = ?, species_count = ?,
               total_impressions_served = total_impressions_served + ?,
               checkpoint_path = ?, updated_at = ?
               WHERE tenant_id = ? AND campaign_id = ?""",
            (next_gen, next_status.value, best.fitness,
             _best_creative_summary(best).model_dump_json(), species_count,
             impressions_this_gen, checkpoint_path, _now(), tenant_id, campaign_id),
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO generation_history
               (tenant_id, campaign_id, generation, best_fitness, species_count, impressions_this_generation)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (tenant_id, campaign_id, current_gen, best.fitness, species_count, impressions_this_gen),
        )
        self._conn.commit()
        return self.get_campaign(tenant_id, campaign_id)

    def get_history(self, tenant_id: str, campaign_id: str) -> List[Dict[str, Any]]:
        self._get_row(tenant_id, campaign_id)  # 404s if missing
        rows = self._conn.execute(
            """SELECT generation, best_fitness, species_count, impressions_this_generation
               FROM generation_history WHERE tenant_id = ? AND campaign_id = ?
               ORDER BY generation ASC""",
            (tenant_id, campaign_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_creatives(self, tenant_id: str, campaign_id: str, limit: int = 8) -> List[CreativeSummary]:
        self._get_row(tenant_id, campaign_id)  # 404s if missing
        if campaign_id not in self._runtimes:
            self._runtimes[campaign_id] = self._load_checkpoint(tenant_id, campaign_id)
        population = self._runtimes[campaign_id].population
        ranked = sorted(population, key=lambda g: g.fitness, reverse=True)[:limit]
        return [_best_creative_summary(g) for g in ranked]
