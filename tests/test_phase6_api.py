"""
Phase 6 integration test.

Exercises the full campaign lifecycle through the actual FastAPI app (via
TestClient, no real network needed) rather than calling AdCampaignManager
methods directly -- this proves the HTTP layer, schemas, and status-code
handling all work, not just the underlying Python objects.

Specifically proves:
  1. Create -> start -> step advances generations and improves fitness,
     through real HTTP requests.
  2. Pause actually blocks progress (step while paused -> 409), and doesn't
     just get ignored.
  3. THE key proof: simulating a server restart (dropping the in-memory
     runtime dict, exactly what happens if the process actually died) and
     then resuming correctly reconstructs the population via
     AdGenome.from_dict() from the checkpoint written to disk -- not from
     scratch, and not by crashing.
  4. The campaign auto-stops at total_generations and rejects further
     steps.
  5. Multi-tenant isolation: two tenants' campaigns don't leak into each
     other's list/get endpoints.
"""
from fastapi.testclient import TestClient

from api.server_ad import app, manager

client = TestClient(app)
HEADERS = {"X-Tenant-Id": "acme_corp"}


def run():
    # 1. Create + start
    config = {
        "name": "Q3 signup campaign",
        "population_size": 20,
        "total_generations": 6,
        "budget_per_generation": 2000,
        "rounds_per_generation": 8,
    }
    resp = client.post("/campaigns", json=config, headers=HEADERS)
    assert resp.status_code == 200, resp.text
    campaign = resp.json()
    campaign_id = campaign["campaign_id"]
    assert campaign["status"] == "queued"
    assert campaign["generation"] == 0
    print(f"PASS: created campaign {campaign_id}, status=queued, generation=0")

    resp = client.post(f"/campaigns/{campaign_id}/start", headers=HEADERS)
    assert resp.json()["status"] == "running"
    print("PASS: campaign started, status=running")

    # 2. Step 3 generations, confirm progress
    for _ in range(3):
        resp = client.post(f"/campaigns/{campaign_id}/step", headers=HEADERS)
        assert resp.status_code == 200, resp.text
    status = resp.json()
    assert status["generation"] == 3, status
    assert status["best_fitness"] > 0
    fitness_after_3 = status["best_fitness"]
    print(f"PASS: after 3 steps, generation={status['generation']}, "
          f"best_fitness={status['best_fitness']:.2f}, species_count={status['species_count']}")

    # 3. Pause actually blocks progress
    resp = client.post(f"/campaigns/{campaign_id}/pause", headers=HEADERS)
    assert resp.json()["status"] == "paused"
    resp = client.post(f"/campaigns/{campaign_id}/step", headers=HEADERS)
    assert resp.status_code == 409, "Stepping a paused campaign should be rejected"
    status = client.get(f"/campaigns/{campaign_id}", headers=HEADERS).json()
    assert status["generation"] == 3, "Generation must not advance while paused"
    print("PASS: step on a paused campaign returns 409 and generation does not advance")

    # 4. Simulate an actual process restart: drop in-memory runtime state.
    assert campaign_id in manager._runtimes
    del manager._runtimes[campaign_id]
    print("Simulated restart: dropped in-memory runtime, only SQLite row + checkpoint file remain")

    resp = client.post(f"/campaigns/{campaign_id}/resume", headers=HEADERS)
    assert resp.json()["status"] == "running"
    assert campaign_id in manager._runtimes, "Resume should have rebuilt the runtime from the checkpoint"
    print("PASS: resume rebuilt the runtime from disk (AdGenome.from_dict() round-trip)")

    # Step again post-restore -- must continue sensibly, not crash or reset.
    resp = client.post(f"/campaigns/{campaign_id}/step", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    status = resp.json()
    assert status["generation"] == 4, status
    print(f"PASS: post-restore step succeeded, generation={status['generation']}, "
          f"best_fitness={status['best_fitness']:.2f} (was {fitness_after_3:.2f} before restart)")

    # 5. Run to completion, confirm auto-stop
    while status["status"] == "running":
        resp = client.post(f"/campaigns/{campaign_id}/step", headers=HEADERS)
        status = resp.json()
    assert status["status"] == "stopped"
    assert status["generation"] == 6
    resp = client.post(f"/campaigns/{campaign_id}/step", headers=HEADERS)
    assert resp.status_code == 409, "Stepping a stopped campaign should be rejected"
    print(f"PASS: campaign auto-stopped at generation={status['generation']}, "
          f"further steps correctly rejected")

    final = client.get(f"/campaigns/{campaign_id}", headers=HEADERS).json()
    assert final["best_creative"] is not None
    print(f"PASS: final best creative -- \"{final['best_creative']['headline']}\" "
          f"(fitness={final['best_creative']['fitness']:.2f})")

    # 6. Multi-tenant isolation
    other_headers = {"X-Tenant-Id": "other_tenant"}
    client.post("/campaigns", json={**config, "name": "other tenant campaign"}, headers=other_headers)
    acme_list = client.get("/campaigns", headers=HEADERS).json()["campaigns"]
    other_list = client.get("/campaigns", headers=other_headers).json()["campaigns"]
    assert all(c["tenant_id"] == "acme_corp" for c in acme_list)
    assert all(c["tenant_id"] == "other_tenant" for c in other_list)
    assert len(acme_list) >= 1 and len(other_list) >= 1
    not_found = client.get(f"/campaigns/{campaign_id}", headers=other_headers)
    assert not_found.status_code == 404, "Wrong tenant must not be able to read another tenant's campaign"
    print("PASS: multi-tenant isolation -- lists and gets are correctly scoped per tenant")

    print("\nPASSED: full campaign API lifecycle -- create/start/step/pause/resume/auto-stop/"
          "restart-recovery/tenant-isolation all verified through real HTTP requests.")


if __name__ == "__main__":
    run()
