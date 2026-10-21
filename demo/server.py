"""Unified FastAPI server for CAPACITYX.

Serves:
1. Starter Kit Demo endpoints:
   - GET / -> starter kit HTML UI (recorded by ./kit video)
   - GET /api/info -> agent info, model, and decider status
   - POST /api/chat -> chat interaction with HackathonAgent
   - POST /api/reset -> reset session
2. CapacityX Command Center endpoints:
   - GET /health -> health monitoring, decider connectivity, and Bedrock disclosure
   - GET /api/scenarios -> preconfigured incident scenarios for testing
   - POST /api/decide -> evaluates incident using canonical agent/decision.py
   - POST /api/mock-decision -> simulation evaluation for local UI development
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from decider_kit.config import load_dotenv, settings
from decider_kit.client import Decider
from agent.decision import evaluate_decision
from demo.models import (
    DecisionResponse,
    ShipmentIncidentRequest,
    SimulatedActionItem,
    SimulatedOutcome,
)

load_dotenv()
STATIC = Path(__file__).parent / "static"
app = FastAPI(
    title="CapacityX — Smart Logistics Decision Agent",
    description="Unified API combining Strands Decider 2B hackathon demo and CapacityX Logistics Command Center.",
    version="1.0.0",
)

# CORS middleware for local Vite frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Any = None


class Turn(BaseModel):
    message: str
    session_id: str = "demo"
    images: list[str] = []


def agent() -> Any:
    global _agent
    if _agent is None:
        if os.environ.get("DEMO_TARGET") == "agentcore":
            _agent = AgentCoreClient(os.environ["AGENTCORE_ARN"])
        else:
            from agent.agent import HackathonAgent

            _agent = HackathonAgent()
    return _agent


class AgentCoreClient:
    """The same chat() interface, served by the agent deployed on Amazon Bedrock AgentCore Runtime."""

    def __init__(self, arn: str):
        import boto3

        self.arn = arn
        self.client = boto3.client("bedrock-agentcore", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        self.sessions: dict[str, str] = {}

    def chat(self, message: str, session_id: str = "demo", images: list[str] | None = None) -> dict[str, Any]:
        sid = self.sessions.setdefault(session_id, f"demo-{session_id}-{uuid.uuid4().hex}")
        payload = {"prompt": message, **({"images": images} if images else {})}
        r = self.client.invoke_agent_runtime(agentRuntimeArn=self.arn, runtimeSessionId=sid, payload=json.dumps(payload).encode())
        body = r["response"].read() if hasattr(r["response"], "read") else r["response"]
        out = json.loads(body)
        out["served_by"] = "Amazon Bedrock AgentCore Runtime"
        return out


# ==============================================================================
# Starter Kit Demo Endpoints
# ==============================================================================

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info() -> dict[str, Any]:
    s = settings()
    name = "CapacityX: Autonomous Logistics Decision Agent"
    try:
        from agent import agent as mod

        name = getattr(mod, "NAME", name)
    except Exception:
        pass

    decider_mode = s.decider_mode
    if os.environ.get("DEMO_TARGET") == "agentcore":
        dep = Path(__file__).resolve().parent.parent / "submission" / "deployment.json"
        if dep.exists():
            decider_mode = json.loads(dep.read_text()).get("decider_mode", decider_mode)

    from decider_kit.llm import describe

    llm = describe()
    if os.environ.get("DEMO_TARGET") == "agentcore":
        llm = f"Amazon Bedrock ({s.bedrock_model})"

    return {
        "agent": name,
        "llm": llm,
        "bedrock_model": s.bedrock_model,
        "decider": decider_mode,
        "target": os.environ.get("DEMO_TARGET", "local"),
    }


@app.post("/api/chat")
def chat(t: Turn) -> JSONResponse:
    try:
        a = agent()
        if t.images:
            import inspect

            if "images" not in inspect.signature(a.chat).parameters:
                return JSONResponse({"error": "this agent does not take images: add images=None to HackathonAgent.chat"}, status_code=400)
            return JSONResponse(a.chat(t.message, t.session_id, images=t.images))
        return JSONResponse(a.chat(t.message, t.session_id))
    except Exception as e:
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/api/reset")
def reset() -> dict[str, bool]:
    global _agent
    _agent = None
    return {"ok": True}


# ==============================================================================
# CapacityX Command Center Endpoints
# ==============================================================================

@app.get("/health")
def health() -> dict[str, Any]:
    s = settings()
    decider_status = "configured"
    is_hosted = s.decider_mode == "ugmdu" or bool(s.decider_api_key)

    try:
        d = Decider()
        d.health()
        decider_status = "live_connected"
    except Exception as e:
        decider_status = f"unreachable_fallback ({type(e).__name__})"

    bedrock_enabled = os.environ.get("CAPACITYX_ENABLE_BEDROCK", "false").strip().lower() in ("true", "1", "yes")

    return {
        "status": "online",
        "service": "CapacityX Unified API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strands_decider": {
            "mode": s.decider_mode,
            "url": s.decider_url,
            "is_hosted": is_hosted,
            "status": decider_status,
            "auth_type": "Bearer" if is_hosted else "None/Basic",
        },
        "amazon_bedrock": {
            "enabled": bedrock_enabled,
            "provider": os.environ.get("LLM_PROVIDER", "bedrock"),
            "model_id": s.bedrock_model,
            "region": s.aws_region,
            "status": "live" if bedrock_enabled else "disabled_safe_default",
            "account_verification_blocker": "AWS Free tier ValidationException: Operation not allowed prevents unverified live calls without account upgrade.",
        },
    }


@app.get("/api/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "id": "scenario-1",
            "title": "Severe Engine Breakdown on Interstate (Backup Unit Available)",
            "description": "Primary vehicle disabled with hazardous cargo in transit. Standby unit in range.",
            "data": {
                "shipment_id": "SHP-7821-X",
                "delivery_deadline": "2026-10-20T16:30:00Z",
                "truck_breakdown": True,
                "breakdown_severity": "CRITICAL",
                "replacement_vehicle_available": True,
                "customer_notified": False,
                "cargo_type": "Cold-chain Perishables",
                "origin": "Chicago Depot North",
                "destination": "Milwaukee Logistics Park",
                "estimated_distance_remaining_km": 65.0,
            },
        },
        {
            "id": "scenario-2",
            "title": "Severe Highway Detour / Weather Closure (Truck Operational)",
            "description": "Major interstate corridor blocked by storm closure. Vehicle operational but reroute needed.",
            "data": {
                "shipment_id": "SHP-4402-A",
                "delivery_deadline": "2026-10-20T18:00:00Z",
                "truck_breakdown": False,
                "breakdown_severity": "NONE",
                "replacement_vehicle_available": False,
                "customer_notified": False,
                "cargo_type": "Electronics & Hardware",
                "origin": "Indianapolis Central Yard",
                "destination": "Louisville Freight Terminal",
                "estimated_distance_remaining_km": 115.0,
            },
        },
        {
            "id": "scenario-3",
            "title": "Critical Medical Delivery (On Schedule, Nominal Transit)",
            "description": "Routine transit with zero breakdown, monitored high-value payload.",
            "data": {
                "shipment_id": "SHP-1109-MED",
                "delivery_deadline": "2026-10-20T12:00:00Z",
                "truck_breakdown": False,
                "breakdown_severity": "NONE",
                "replacement_vehicle_available": True,
                "customer_notified": True,
                "cargo_type": "Pharmaceuticals & Vaccines",
                "origin": "Madison Biotech Depot",
                "destination": "Rockford Memorial Hospital",
                "estimated_distance_remaining_km": 30.5,
            },
        },
    ]


@app.post("/api/decide", response_model=DecisionResponse)
@app.post("/api/mock-decision", response_model=DecisionResponse)
def decide(payload: ShipmentIncidentRequest) -> DecisionResponse:
    now_iso = datetime.now(timezone.utc).isoformat()

    incident_dict = {
        "incident_id": payload.shipment_id,
        "vehicle_id": f"TRK-{payload.shipment_id[-4:]}",
        "status": "BREAKDOWN" if payload.truck_breakdown else "ACTIVE",
        "severity": payload.breakdown_severity or ("CRITICAL" if payload.truck_breakdown else "LOW"),
        "cargo_type": payload.cargo_type or "General Freight",
        "perishable": "perish" in (payload.cargo_type or "").lower() or "vaccin" in (payload.cargo_type or "").lower(),
        "backup_units_within_range": 1 if payload.replacement_vehicle_available else 0,
        "traffic_congestion": "HIGH" if not payload.truck_breakdown else "LOW",
        "weather_condition": "CLEAR",
        "estimated_delay_minutes": 85 if payload.truck_breakdown else 15,
        "sla_buffer_minutes": 45,
    }

    # Evaluate decision using the canonical Reusable Decision Pattern
    outcome = evaluate_decision(incident_dict)

    tool = outcome["action"]
    actions_preview: List[SimulatedActionItem] = []

    if tool == "dispatch_replacement":
        actions_preview = [
            SimulatedActionItem(
                system="Fleet Operations",
                action=f"Lock & assign backup unit STANDBY-42 to rendezvous with {payload.shipment_id}",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            ),
            SimulatedActionItem(
                system="Driver Telematics",
                action="Push rendezvous coordinates and freight transfer protocol to in-cab tablet",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            ),
        ]
    elif tool == "escalate_to_human":
        actions_preview = [
            SimulatedActionItem(
                system="Dispatch Console",
                action=f"Raise high-priority emergency ticket for dispatcher review on {payload.shipment_id}",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]
    elif tool == "reroute_shipment":
        actions_preview = [
            SimulatedActionItem(
                system="Navigation Optimization",
                action=f"Transmit weather/congestion bypass corridor waypoints to vehicle {payload.shipment_id}",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]
    elif tool == "crossdock":
        actions_preview = [
            SimulatedActionItem(
                system="Warehouse Management",
                action="Reserve cold-chain cross-dock bay for emergency temperature stabilization",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]
    else:
        actions_preview = [
            SimulatedActionItem(
                system="Automated Geofencing",
                action="Maintain routine 15-minute telemetry polling and SLA tracking",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]

    is_real = outcome["engine"] in ("STRANDS_DECIDER_2B_HOSTED", "STRANDS_DECIDER_2B_LOCAL")
    customer_rec = (
        "Customer already notified. Provide live tracking updates."
        if payload.customer_notified
        else "Mandatory customer alert recommended."
        if payload.truck_breakdown
        else "Standard scheduled milestone notices."
    )

    sim_outcome = SimulatedOutcome(
        selected_action=outcome["action"],
        action_type=outcome["action"].upper(),
        model_confidence=outcome["model_confidence"],
        alternatives=outcome["alternatives"],
        human_review_required=outcome["human_review_required"],
        human_review_reason=outcome["human_review_reason"],
        engine=outcome["engine"],
        is_real_model=is_real,
        fallback_reason="Safe Fallback applied" if not is_real else None,
        urgency=outcome["urgency"],
        headline=f"Recommended Action: {outcome['action'].replace('_', ' ').title()}",
        rationale=f"Selected {outcome['action']} via {outcome['engine']} with {outcome['model_confidence']:.1%} confidence.",
        actions_preview=actions_preview,
        estimated_delay_minutes=incident_dict["estimated_delay_minutes"],
        deadline_met_projected=not payload.truck_breakdown,
        sla_risk_level="HIGH" if payload.truck_breakdown else "LOW",
        customer_status_recommended=customer_rec,
        bedrock_validation={
            "status": "disabled",
            "disclaimer": "Amazon Bedrock operational check disabled by default.",
        },
    )

    disclaimer = (
        f"REAL MODEL INFERENCE: Evaluated with Strands Decider 2B ({outcome['engine']})."
        if is_real
        else "SAFE FALLBACK APPLIED: Inference offline or unconfigured."
    )

    return DecisionResponse(
        is_simulated=not is_real,
        real_action_taken=False,
        environment="HOSTED_REAL_MODEL" if outcome["engine"] == "STRANDS_DECIDER_2B_HOSTED" else "SAFE_FALLBACK",
        disclaimer=disclaimer,
        shipment_id=payload.shipment_id,
        delivery_deadline=payload.delivery_deadline,
        cargo_type=payload.cargo_type,
        origin=payload.origin,
        destination=payload.destination,
        timestamp=now_iso,
        outcome=sim_outcome,
    )
