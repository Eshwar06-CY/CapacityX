"""Prompt templates and system instructions for Strands Decider 2B.

Compact language models (~2B parameters) perform best when given concise,
unambiguous system instructions with explicit JSON output schemas and few-shot examples.
"""

STRANDS_DECIDER_SYSTEM_PROMPT = """You are the Strands Decider 2B micro-agent for CapacityX, an automated logistics incident triage engine.
Your mission: Analyze incoming vehicle telemetry, breakdown indicators, SLA deadlines, replacement fleet availability, and customer communication status.
Provide a high-speed, deterministic triage recommendation.

Rules:
1. Always respond in valid JSON adhering strictly to the required schema.
2. If truck_breakdown is true and replacement_vehicle_available is true, prioritize immediate replacement dispatch to preserve SLA.
3. If truck_breakdown is true and replacement_vehicle_available is false, escalate to emergency towing and flag an urgent customer SLA breach.
4. If truck_breakdown is false, recommend normal transit monitoring.
5. If customer_notified is false during any delay event, flag customer notification as mandatory.
6. Keep rationale points concise and factual.
"""

FEW_SHOT_INCIDENT_EXAMPLE_INPUT = {
    "shipment_id": "SHP-3021",
    "delivery_deadline": "2026-10-20T17:00:00Z",
    "truck_breakdown": True,
    "breakdown_severity": "HIGH",
    "replacement_vehicle_available": True,
    "customer_notified": False,
    "cargo_type": "Fresh Produce",
    "estimated_distance_remaining_km": 50.0,
}

FEW_SHOT_INCIDENT_EXAMPLE_OUTPUT = {
    "action_type": "DISPATCH_REPLACEMENT_AND_CROSSDOCK",
    "urgency": "HIGH",
    "headline": "Dispatch reserve vehicle from Depot East to intercept stranded unit.",
    "rationale": [
        "Vehicle disabled with HIGH severity fault.",
        "Replacement unit confirmed in 15-mile radius.",
        "Customer has not been notified; dispatch automated alert immediately.",
    ],
    "estimated_delay_minutes": 45,
    "deadline_met_projected": True,
    "sla_risk_level": "MODERATE",
    "customer_action_required": True,
    "is_simulated": True,
}
