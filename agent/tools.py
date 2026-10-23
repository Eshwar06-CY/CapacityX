"""CAPACITYX Logistics Decision Tools.

Canonical logistics tools decorated with @tool from Strands Agents SDK.
All operations operate strictly in PREVIEW/SIMULATION mode: no real-world fleet
dispatches, carrier locks, routing calls, or SMS/email notifications occur.
"""

from __future__ import annotations

from typing import Any
from strands import tool

SIMULATION_DISCLAIMER = "PREVIEW (Withheld: Simulation Mode)"

ACTIONS_LOG: list[dict[str, Any]] = []


@tool
def dispatch_replacement(
    incident_id: str,
    vehicle_id: str,
    backup_unit_id: str,
    eta_minutes: int,
    cargo_type: str = "general",
) -> dict[str, Any]:
    """Dispatch a nearby standby vehicle to rendezvous and transfer freight when primary vehicle is immobilized."""
    record = {
        "action": "dispatch_replacement",
        "incident_id": incident_id,
        "vehicle_id": vehicle_id,
        "backup_unit_id": backup_unit_id,
        "eta_minutes": eta_minutes,
        "cargo_type": cargo_type,
        "is_live_action": False,
        "status": SIMULATION_DISCLAIMER,
        "preview_summary": f"Standby unit {backup_unit_id} scheduled to rendezvous with {vehicle_id} in {eta_minutes} min (Simulation).",
    }
    ACTIONS_LOG.append(record)
    return record


@tool
def reroute_shipment(
    incident_id: str,
    vehicle_id: str,
    detour_reason: str,
    added_distance_miles: float = 0.0,
    added_minutes: int = 0,
) -> dict[str, Any]:
    """Reroute the vehicle along alternate corridors to bypass congestion, road closures, or hazardous weather."""
    record = {
        "action": "reroute_shipment",
        "incident_id": incident_id,
        "vehicle_id": vehicle_id,
        "detour_reason": detour_reason,
        "added_distance_miles": added_distance_miles,
        "added_minutes": added_minutes,
        "is_live_action": False,
        "status": SIMULATION_DISCLAIMER,
        "preview_summary": f"Alternate navigation corridor plotted for {vehicle_id} avoiding {detour_reason} (+{added_minutes}m, +{added_distance_miles}mi) (Simulation).",
    }
    ACTIONS_LOG.append(record)
    return record


@tool
def crossdock(
    incident_id: str,
    vehicle_id: str,
    facility_id: str,
    reason: str,
) -> dict[str, Any]:
    """Divert the vehicle to the nearest cross-dock facility for urgent cargo transshipment or cold-chain recovery."""
    record = {
        "action": "crossdock",
        "incident_id": incident_id,
        "vehicle_id": vehicle_id,
        "facility_id": facility_id,
        "reason": reason,
        "is_live_action": False,
        "status": SIMULATION_DISCLAIMER,
        "preview_summary": f"Diverted vehicle {vehicle_id} to cross-dock bay at {facility_id} due to {reason} (Simulation).",
    }
    ACTIONS_LOG.append(record)
    return record


@tool
def escalate_to_human(
    incident_id: str,
    vehicle_id: str,
    escalation_reason: str,
    dispatcher_queue: str = "CRITICAL_LOGISTICS",
) -> dict[str, Any]:
    """Escalate the incident immediately to a human logistics dispatcher when automation cannot guarantee SLA or safety."""
    record = {
        "action": "escalate_to_human",
        "incident_id": incident_id,
        "vehicle_id": vehicle_id,
        "escalation_reason": escalation_reason,
        "dispatcher_queue": dispatcher_queue,
        "is_live_action": False,
        "status": SIMULATION_DISCLAIMER,
        "preview_summary": f"Escalated incident {incident_id} ({vehicle_id}) to dispatcher queue {dispatcher_queue}: {escalation_reason} (Simulation).",
    }
    ACTIONS_LOG.append(record)
    return record


@tool
def continue_monitoring(
    incident_id: str,
    vehicle_id: str,
    interval_minutes: int = 15,
) -> dict[str, Any]:
    """Maintain automated telemetry heartbeat monitoring when vehicle status, ETA, and environmental conditions are nominal."""
    record = {
        "action": "continue_monitoring",
        "incident_id": incident_id,
        "vehicle_id": vehicle_id,
        "interval_minutes": interval_minutes,
        "is_live_action": False,
        "status": SIMULATION_DISCLAIMER,
        "preview_summary": f"Nominal conditions confirmed for {vehicle_id}. Routine automated telemetry polling active at {interval_minutes}m interval (Simulation).",
    }
    ACTIONS_LOG.append(record)
    return record


TOOLS = [
    dispatch_replacement,
    reroute_shipment,
    crossdock,
    escalate_to_human,
    continue_monitoring,
]

TOOL_BY_NAME = {t.__name__: t for t in TOOLS}
