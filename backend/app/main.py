"""FastAPI backend service for CAPACITYX — Smart Logistics Decision Agent.

Integrates Strands Decider 2B for typed agentic tool selection:
- dispatch_replacement
- reroute_shipment
- crossdock
- escalate_to_human
- continue_monitoring

Provides live health checks, model server connectivity status,
and deterministic safe fallback routing when inference is offline.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root and backend directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load environment variables from consolidated root .env if present
_root_env = ROOT_DIR / ".env"
if _root_env.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_root_env, override=False)
    except ImportError:
        pass

from app.models import (
    DecisionResponse,
    ShipmentIncidentRequest,
    SimulatedActionItem,
    SimulatedOutcome,
)
from patterns.strands_decider_2b import (
    LOGISTICS_TOOLS,
    IncidentContext,
    StrandsDecider2B,
)

app = FastAPI(
    title="CapacityX API",
    description=(
        "Logistics Incident Decision API integrating Strands Decider 2B typed tool selection "
        "and Amazon Bedrock Nova Micro with automated safe fallback policies."
    ),
    version="0.2.0",
)

# Enable CORS for local development origins
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

ENV_AI_PROVIDER = "CAPACITYX_AI_PROVIDER"
ENV_ENABLE_BEDROCK = "CAPACITYX_ENABLE_BEDROCK"
ENV_BEDROCK_MODE = "CAPACITYX_BEDROCK_MODE"


def get_configured_agent():
    """Instantiate CapacityX Orchestrator pairing Strands Decider 2B with optional Bedrock reasoning."""
    enable_bedrock = os.environ.get(ENV_ENABLE_BEDROCK, "false").strip().lower() in {
        "true", "1", "yes"
    }
    bedrock_mode = os.environ.get(
        ENV_BEDROCK_MODE,
        "mock" if enable_bedrock else "disabled",
    ).strip().lower()

    # Handle explicit provider toggle
    legacy_provider = os.environ.get(ENV_AI_PROVIDER, "strands").strip().lower()
    if legacy_provider in {"bedrock", "bedrock_nova_micro", "nova_micro"}:
        enable_bedrock = True
        if bedrock_mode == "disabled":
            bedrock_mode = "mock"

    from patterns.orchestrator import CapacityXOrchestrator
    return CapacityXOrchestrator(
        enable_bedrock=enable_bedrock,
        bedrock_mode=bedrock_mode,
    )


# Instantiate decision orchestrator (Strands Decider 2B primary, Bedrock disabled by default)
decider = get_configured_agent()


@app.get("/", tags=["Root"])
def root() -> Dict[str, Any]:
    """Root endpoint detailing API capabilities and model status."""
    return {
        "project": "CAPACITYX — Smart Logistics Decision Agent",
        "status": "online",
        "supported_tools": list(LOGISTICS_TOOLS.keys()),
        "endpoints": {
            "health": "/health",
            "api_health": "/api/health",
            "decide": "/api/decide",
            "mock_decision": "/api/mock-decision",
            "sample_scenarios": "/api/scenarios",
            "docs": "/docs",
        },
        "strands_decider_status": decider.get_health_status(),
    }


@app.get("/health", tags=["Health"])
@app.get("/api/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Backend and decision service health check endpoint."""
    model_health = decider.get_health_status()
    is_live = model_health.get("is_real_model_available", False)
    provider_name = getattr(decider, "provider_name", "strands_orchestrated")

    strands_info = model_health.get("strands_decider", model_health)
    bedrock_info = model_health.get("amazon_bedrock", {})

    server_url = getattr(decider, "server_url", strands_info.get("server_url", "http://127.0.0.1:8099"))
    if is_live:
        ai_engine = f"STRANDS_DECIDER_2B_REAL ({strands_info.get('checkpoint', 'active')})"
    else:
        ai_engine = f"SAFE_FALLBACK_POLICY (Live Strands Decider server is offline at {server_url})"

    if bedrock_info.get("status") in {"simulated_mock_ready", "simulated_mock"}:
        ai_engine += " + BEDROCK_OPERATIONAL_REASONING (Simulated Mock)"
    elif bedrock_info.get("status") == "validated":
        ai_engine += " + BEDROCK_OPERATIONAL_REASONING (Live Validated)"

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "capacityx-backend",
        "version": "0.2.0",
        "environment": "local-development",
        "ai_provider": provider_name,
        "strands_decider": strands_info,
        "amazon_bedrock": bedrock_info,
        "orchestration": model_health.get("orchestration"),
        "ai_engine": ai_engine,
    }


@app.get("/api/scenarios", tags=["Scenarios"])
def get_sample_scenarios() -> List[Dict[str, Any]]:
    """Return pre-configured sample logistics scenarios to test each tool selection path."""
    return [
        {
            "id": "scenario-1",
            "title": "Cold-Chain Truck Breakdown (Replacement Ready)",
            "description": "Refrigerated transport engine failure. Backup truck is stationed nearby.",
            "data": {
                "shipment_id": "SHP-8921-COLD",
                "delivery_deadline": "2026-10-20T14:30:00Z",
                "truck_breakdown": True,
                "breakdown_severity": "CRITICAL",
                "replacement_vehicle_available": True,
                "customer_notified": False,
                "cargo_type": "Cold-Chain Perishables (Frozen Seafood)",
                "origin": "Milwaukee Cold Depot",
                "destination": "Chicago Central Distribution Hub",
                "estimated_distance_remaining_km": 42.0,
            },
        },
        {
            "id": "scenario-2",
            "title": "Standard Freight Breakdown (No Replacement Vehicle)",
            "description": "Dry van transmission issue in rural corridor with no immediate replacement.",
            "data": {
                "shipment_id": "SHP-4012-DRY",
                "delivery_deadline": "2026-10-20T18:00:00Z",
                "truck_breakdown": True,
                "breakdown_severity": "MODERATE",
                "replacement_vehicle_available": False,
                "customer_notified": False,
                "cargo_type": "Standard Freight (Consumer Electronics)",
                "origin": "Indianapolis Hub",
                "destination": "Peoria Logistics Center",
                "estimated_distance_remaining_km": 115.0,
            },
        },
        {
            "id": "scenario-3",
            "title": "Critical Medical Delivery (On Schedule, Customer Notified)",
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


@app.post(
    "/api/decide",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Decision Agent"],
)
@app.post(
    "/api/mock-decision",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Decision Agent"],
)
def evaluate_decision(payload: ShipmentIncidentRequest) -> DecisionResponse:
    """Evaluate logistics incident using Strands Decider 2B typed tool selection.

    Selects among:
    - dispatch_replacement
    - reroute_shipment
    - crossdock
    - escalate_to_human
    - continue_monitoring

    If the real Decider 2B inference server is unreachable or confidence is insufficient,
    a safe deterministic fallback is executed and clearly labeled.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    incident = IncidentContext(
        shipment_id=payload.shipment_id,
        delivery_deadline=payload.delivery_deadline,
        truck_breakdown=payload.truck_breakdown,
        breakdown_severity=payload.breakdown_severity or "MODERATE",
        replacement_vehicle_available=payload.replacement_vehicle_available,
        customer_notified=payload.customer_notified,
        cargo_type=payload.cargo_type or "Standard Freight",
        origin=payload.origin or "Origin Depot",
        destination=payload.destination or "Destination Hub",
        estimated_distance_remaining_km=payload.estimated_distance_remaining_km or 50.0,
    )

    # Execute evaluation through the Strands Decider pattern
    decider_outcome = decider.evaluate(incident)

    # Construct execution previews based on the selected tool
    actions_preview: List[SimulatedActionItem] = []
    tool = decider_outcome.selected_action

    if tool == "dispatch_replacement":
        actions_preview = [
            SimulatedActionItem(
                system="Fleet Operations",
                action=f"Lock & assign backup vehicle to rendezvous with {payload.shipment_id}",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            ),
            SimulatedActionItem(
                system="Driver Telematics",
                action="Push cross-dock coordinate transfer instructions to driver",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            ),
            SimulatedActionItem(
                system="Customer Notification",
                action=f"Alert customer of replacement vehicle ETA for {payload.shipment_id}",
                status="PREVIEW (Withheld: Simulation Mode)" if not payload.customer_notified else "SKIPPED (Already alerted)",
                is_live_action=False,
            ),
        ]
    elif tool == "escalate_to_human":
        actions_preview = [
            SimulatedActionItem(
                system="Dispatch Console",
                action=f"Raise urgent high-priority ticket for human logistics dispatcher on {payload.shipment_id}",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            ),
            SimulatedActionItem(
                system="3PL Heavy Tow Broker",
                action="Prepare roadside assistance tow request to breakdown GPS coordinates",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            ),
        ]
    elif tool == "reroute_shipment":
        actions_preview = [
            SimulatedActionItem(
                system="Route Optimization",
                action=f"Push live traffic bypass waypoint sequence to truck navigation tablet",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]
    elif tool == "crossdock":
        actions_preview = [
            SimulatedActionItem(
                system="Warehouse Management",
                action=f"Book emergency cross-dock bay at nearest regional distribution center",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]
    else:  # continue_monitoring
        actions_preview = [
            SimulatedActionItem(
                system="Fleet Geofencing",
                action="Maintain standard 15-minute telemetry heartbeat monitoring",
                status="PREVIEW (Withheld: Simulation Mode)",
                is_live_action=False,
            )
        ]

    customer_rec = (
        "Customer already notified. Provide live tracking updates."
        if payload.customer_notified
        else "Mandatory customer alert recommended." if payload.truck_breakdown
        else "Standard scheduled milestone notices."
    )

    outcome_model = SimulatedOutcome(
        selected_action=decider_outcome.selected_action,
        action_type=decider_outcome.action_type,
        model_confidence=decider_outcome.model_confidence,
        alternatives=decider_outcome.alternatives,
        human_review_required=decider_outcome.human_review_required,
        human_review_reason=decider_outcome.human_review_reason,
        engine=decider_outcome.engine,
        is_real_model=decider_outcome.is_real_model,
        fallback_reason=decider_outcome.fallback_reason,
        model_id=decider_outcome.model_id,
        urgency=decider_outcome.urgency,
        headline=decider_outcome.headline,
        rationale=decider_outcome.rationale,
        actions_preview=actions_preview,
        estimated_delay_minutes=decider_outcome.estimated_delay_minutes,
        deadline_met_projected=decider_outcome.deadline_met_projected,
        sla_risk_level=decider_outcome.sla_risk_level,
        customer_status_recommended=customer_rec,
        bedrock_validation=decider_outcome.metadata.get("bedrock_validation"),
    )

    if decider_outcome.is_real_model:
        disclaimer = f"REAL MODEL INFERENCE: Evaluated with Strands Decider 2B ({decider_outcome.model_id})."
    else:
        disclaimer = f"SAFE FALLBACK APPLIED: {decider_outcome.fallback_reason}"

    bedrock_meta = decider_outcome.metadata.get("bedrock_validation", {})
    if bedrock_meta.get("disclaimer"):
        disclaimer += f" | {bedrock_meta['disclaimer']}"

    return DecisionResponse(
        is_simulated=not decider_outcome.is_real_model,
        real_action_taken=False,
        environment="LOCAL_REAL_MODEL" if decider_outcome.is_real_model else "LOCAL_SAFE_FALLBACK",
        disclaimer=disclaimer,
        shipment_id=payload.shipment_id,
        delivery_deadline=payload.delivery_deadline,
        cargo_type=payload.cargo_type,
        origin=payload.origin,
        destination=payload.destination,
        timestamp=now_iso,
        outcome=outcome_model,
    )
