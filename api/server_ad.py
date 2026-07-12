"""
Phase 6 -- FastAPI app exposing campaign lifecycle endpoints.

Tenant identification here is a simple X-Tenant-Id header. The real
EvoMind api/auth.py implements full API-key issuance/validation and billing
enforcement (96KB of it) -- reproducing that isn't needed to prove the
lifecycle works, but this header is exactly where that middleware would
plug in: swap this dependency for one that validates a real API key and
resolves it to a tenant_id.
"""

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.ad_schemas import CampaignConfig, CampaignStatusResponse, CampaignListResponse, GenerationHistoryEntry, CreativeSummary
from api.campaign_manager import AdCampaignManager, CampaignNotFoundError, CampaignNotRunningError

app = FastAPI(title="Ad Creative Evolution Engine")

# Demo-scope CORS: allows the dashboard (served from a different origin --
# an artifact preview, localhost:5173, wherever) to call this API directly
# from the browser. Tighten to specific origins before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = AdCampaignManager()


def _tenant(x_tenant_id: str = Header(default="default")) -> str:
    return x_tenant_id


@app.post("/campaigns", response_model=CampaignStatusResponse)
def create_campaign(config: CampaignConfig, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    return manager.create_campaign(tenant_id, config)


@app.get("/campaigns", response_model=CampaignListResponse)
def list_campaigns(tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    return CampaignListResponse(campaigns=manager.list_campaigns(tenant_id))


@app.get("/campaigns/{campaign_id}", response_model=CampaignStatusResponse)
def get_campaign(campaign_id: str, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    try:
        return manager.get_campaign(tenant_id, campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")


@app.post("/campaigns/{campaign_id}/start", response_model=CampaignStatusResponse)
def start_campaign(campaign_id: str, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    try:
        return manager.start_campaign(tenant_id, campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")


@app.post("/campaigns/{campaign_id}/pause", response_model=CampaignStatusResponse)
def pause_campaign(campaign_id: str, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    try:
        return manager.pause_campaign(tenant_id, campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")


@app.post("/campaigns/{campaign_id}/resume", response_model=CampaignStatusResponse)
def resume_campaign(campaign_id: str, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    try:
        return manager.resume_campaign(tenant_id, campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")


@app.post("/campaigns/{campaign_id}/step", response_model=CampaignStatusResponse)
def step_campaign(campaign_id: str, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    """Advance one generation. In production a background worker calls this
    repeatedly while status is RUNNING (see api/campaign_manager.py's
    docstring); exposed directly here so the API itself is testable without
    standing up a separate worker process."""
    try:
        return manager.step(tenant_id, campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")
    except CampaignNotRunningError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/campaigns/{campaign_id}/history")
def get_history(campaign_id: str, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    try:
        history = manager.get_history(tenant_id, campaign_id)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"history": [GenerationHistoryEntry(**h) for h in history]}


@app.get("/campaigns/{campaign_id}/creatives")
def get_creatives(campaign_id: str, limit: int = 8, tenant_id: str = Header(default="default", alias="X-Tenant-Id")):
    try:
        creatives = manager.get_top_creatives(tenant_id, campaign_id, limit=limit)
    except CampaignNotFoundError:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return {"creatives": creatives}
