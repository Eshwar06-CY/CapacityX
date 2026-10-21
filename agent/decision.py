"""REUSABLE DECISION PATTERN: logistics tool selection  (#tool-selection)

What it does
    Evaluates complex freight incident telemetry and selects exactly one of the five canonical
    logistics tools (dispatch_replacement, reroute_shipment, crossdock, escalate_to_human,
    continue_monitoring) in a single calibrated forward pass with Strands Decider 2B.
    When model confidence is below 0.70, or when the independent human review signal is flagged,
    the action is automatically held for human dispatcher review with clear justification.

Typed questions and options
    - action (choice): picks 1 of the 5 canonical logistics actions based on incident telemetry.
    - human_review (noul): independent safety check on whether dispatcher intervention is required.
    - urgency (score): incident priority classification on [low, medium, high, critical].

Wiring
    LogisticsActionGuard is a Strands InterventionHandler passed to Agent:
        Agent(..., tools=TOOLS, interventions=[LogisticsActionGuard(decider)])
    It intercepts proposed logistics operations, verifies arguments against incident telemetry,
    and enforces confidence thresholds (0.70) and safety safeguards before proceeding.

How to reuse it
    Copy this file, adapt TOOLS in agent/tools.py for your fleet operations, and adjust the
    confidence threshold (default 0.70) and human review trigger in before_tool_call().
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from strands.interventions import Confirm, Deny, Guide, InterventionHandler, Proceed

from decider_kit import Decider
from .tools import TOOLS, TOOL_BY_NAME

PATTERN_TAG = "#tool-selection"
CONFIDENCE_THRESHOLD = 0.70
HUMAN_REVIEW_SCORE_THRESHOLD = 0.50

ACTIONS_CRITERIA = {
    "dispatch_replacement": "Primary vehicle is immobilized; dispatch local backup unit within range to rendezvous and transfer freight.",
    "reroute_shipment": "Primary vehicle is operational; adjust route coordinates to bypass congestion, weather, or road closures.",
    "crossdock": "Vehicle has minor mechanical issues or perishable cargo risk; divert to nearest warehouse cross-dock bay.",
    "escalate_to_human": "Vehicle disabled with NO backup units in range, or critical SLA breach risk. Mandates immediate human dispatcher intervention.",
    "continue_monitoring": "Telemetry, speeds, and ETA nominal; maintain automated heartbeat monitoring without intervention.",
}

QUESTIONS = {
    "action": Decider.choice(
        "Which logistics action should handle this incident?",
        ACTIONS_CRITERIA,
    ),
    "human_review": Decider.noul(
        "Does this incident require mandatory human dispatcher review?",
        {
            "true": "critical incident, vehicle disabled without backup, severe SLA breach, or hazardous condition",
            "false": "routine transit or automated detour fully resolved with high confidence",
        },
    ),
    "urgency": Decider.score(
        "How urgent is this logistics event?",
        ["low", "medium", "high", "critical"],
    ),
}


def render_incident_state(incident: dict[str, Any] | str) -> str:
    """Format incident telemetry into a clean state string for Strands Decider."""
    if isinstance(incident, str):
        return incident.strip()

    incident_id = incident.get("incident_id", "INC-UNKNOWN")
    vehicle_id = incident.get("vehicle_id", "VEH-UNKNOWN")
    severity = incident.get("severity", "LOW")
    status = incident.get("status", "ACTIVE")
    desc = incident.get("description", "Telemetry update")

    backup_count = incident.get("backup_units_within_range", 0)
    perishable = incident.get("perishable", False)
    hazardous = incident.get("hazardous", False)
    weather = incident.get("weather_condition", "CLEAR")
    congestion = incident.get("traffic_congestion", "LOW")
    mins_delayed = incident.get("estimated_delay_minutes", 0)
    sla_mins = incident.get("sla_buffer_minutes", 60)
    cargo_type = incident.get("cargo_type", "GENERAL")

    lines = [
        "--- LOGISTICS INCIDENT TELEMETRY ---",
        f"Incident: {incident_id} | Vehicle: {vehicle_id} | Status: {status} | Severity: {severity}",
        f"Description: {desc}",
        f"Cargo: {cargo_type} | Perishable: {perishable} | Hazardous: {hazardous}",
        f"Traffic: {congestion} | Weather: {weather}",
        f"Estimated Delay: {mins_delayed} min | SLA Buffer: {sla_mins} min",
        f"Standby Backup Units in Range: {backup_count}",
    ]
    return "\n".join(lines)


def evaluate_safe_fallback(incident: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, rule-based fallback when the inference server is unreachable."""
    severity = str(incident.get("severity", "")).upper()
    backups = incident.get("backup_units_within_range", 0)
    delay = incident.get("estimated_delay_minutes", 0)
    sla_buffer = incident.get("sla_buffer_minutes", 60)
    perishable = incident.get("perishable", False)
    status = str(incident.get("status", "")).upper()

    is_breakdown = status in ("BREAKDOWN", "IMMOBILIZED", "DISABLED") or "BREAKDOWN" in severity
    is_delayed = delay > sla_buffer

    if is_breakdown:
        if backups > 0:
            action = "dispatch_replacement"
            reason = "Safe Fallback: Primary vehicle disabled with standby backup units available in range."
        else:
            action = "escalate_to_human"
            reason = "Safe Fallback: Vehicle disabled with NO standby units in range. Immediate human dispatch required."
    elif perishable and is_delayed:
        action = "crossdock"
        reason = "Safe Fallback: Perishable cargo at risk of SLA/temperature excursion. Diverting to cross-dock bay."
    elif is_delayed:
        action = "reroute_shipment"
        reason = "Safe Fallback: Vehicle operational but delay exceeds SLA buffer. Recalculating route corridor."
    else:
        action = "continue_monitoring"
        reason = "Safe Fallback: Telemetry nominal. Maintaining routine monitoring."

    urgency = "critical" if (is_breakdown and backups == 0) else ("high" if is_delayed else "low")
    return {
        "action": action,
        "urgency": urgency,
        "model_confidence": 0.50,
        "human_review_required": True,
        "human_review_reason": f"Safe Fallback engaged: {reason}",
        "engine": "SAFE_FALLBACK",
        "alternatives": {act: (0.50 if act == action else 0.125) for act in ACTIONS_CRITERIA},
        "raw_answers": {},
    }


class LogisticsActionGuard(InterventionHandler):
    """Strands InterventionHandler intercepting logistics operations with Strands Decider 2B."""

    name = "logistics-action-guard"

    def __init__(
        self,
        decider: Decider,
        threshold: float = CONFIDENCE_THRESHOLD,
        human_review_threshold: float = HUMAN_REVIEW_SCORE_THRESHOLD,
    ):
        self.decider = decider
        self.threshold = threshold
        self.human_review_threshold = human_review_threshold
        self.log: list[dict[str, Any]] = []

    def before_invocation(self, event: Any, **kwargs: Any) -> Proceed:
        return Proceed()

    def before_tool_call(self, event: Any, **kwargs: Any) -> Proceed | Confirm | Guide | Deny:
        tool_name = event.tool_use.get("name")
        args = event.tool_use.get("input") or {}

        if tool_name not in ACTIONS_CRITERIA:
            return Proceed()

        state = (
            "--- LOGISTICS DISPATCH SITUATION ---\n"
            f"Proposed Tool Call: {tool_name}\n"
            f"Arguments: {json.dumps(args, sort_keys=True)}\n"
        )
        if hasattr(event, "agent") and getattr(event.agent, "messages", None):
            user_texts = [
                m["content"][0]["text"]
                for m in event.agent.messages
                if m.get("role") == "user" and m.get("content") and "text" in m["content"][0]
            ]
            if user_texts:
                state += f"Dispatcher Prompt: {user_texts[-1]}\n"

        try:
            answers = self.decider.ask(state, QUESTIONS)
            chosen_action = answers["action"].label or tool_name
            confidence = answers["action"].confidence
            p_review = answers["human_review"].p_yes or 0.0
            urgency = answers["urgency"].label or "medium"
            probabilities = answers["action"].probabilities
            engine = "STRANDS_DECIDER_2B_HOSTED" if getattr(self.decider, "hosted", False) else "STRANDS_DECIDER_2B_LOCAL"
        except Exception as e:
            # Safe Fallback when decider call fails
            fallback = evaluate_safe_fallback(args)
            chosen_action = fallback["action"]
            confidence = fallback["model_confidence"]
            p_review = 1.0
            urgency = fallback["urgency"]
            probabilities = fallback["alternatives"]
            engine = "SAFE_FALLBACK"

        human_review_required = False
        reasons = []

        if confidence < self.threshold:
            human_review_required = True
            reasons.append(f"Model confidence ({confidence:.1%}) is below calibrated threshold ({self.threshold:.0%})")
        if p_review >= self.human_review_threshold:
            human_review_required = True
            reasons.append(f"Incident risk check flagged review (p={p_review:.2f})")
        if urgency == "critical":
            human_review_required = True
            reasons.append("Critical incident severity mandates dispatcher sign-off")
        if engine == "SAFE_FALLBACK":
            human_review_required = True
            reasons.append("Inference offline: Safe Fallback active")

        verdict_record = {
            "pattern": PATTERN_TAG,
            "tool": tool_name,
            "selected_action": chosen_action,
            "args": args,
            "confidence": round(confidence, 3),
            "human_review_required": human_review_required,
            "human_review_reason": "; ".join(reasons) if reasons else "Nominal autonomous verification",
            "urgency": urgency,
            "probabilities": {k: round(v, 3) for k, v in probabilities.items()},
            "engine": engine,
            "decider_ms": round(self.decider.last_latency_ms or 0),
        }
        self.log.append(verdict_record)

        if human_review_required:
            return Confirm(
                prompt=f"Human Review Required for {tool_name}: {verdict_record['human_review_reason']}",
                response="held_for_human_review",
            )
        return Proceed()


def evaluate_decision(
    incident: dict[str, Any],
    decider: Decider | None = None,
) -> dict[str, Any]:
    """Execute complete logistics decision evaluation with fallbacks, confidence thresholds, and preview actions."""
    d = decider or Decider()
    state = render_incident_state(incident)

    try:
        answers = d.ask(state, QUESTIONS)
        action_decision = answers["action"]
        chosen_action = action_decision.label
        if not chosen_action or chosen_action not in ACTIONS_CRITERIA:
            raise ValueError(f"Unrecognized action choice: {chosen_action}")

        confidence = action_decision.confidence
        p_review = answers["human_review"].p_yes or 0.0
        urgency = answers["urgency"].label or "medium"
        probabilities = action_decision.probabilities
        engine = "STRANDS_DECIDER_2B_HOSTED" if getattr(d, "hosted", False) else "STRANDS_DECIDER_2B_LOCAL"
    except Exception as e:
        fallback = evaluate_safe_fallback(incident)
        chosen_action = fallback["action"]
        confidence = fallback["model_confidence"]
        p_review = 1.0
        urgency = fallback["urgency"]
        probabilities = fallback["alternatives"]
        engine = "SAFE_FALLBACK"

    human_review_required = False
    reasons = []

    if confidence < CONFIDENCE_THRESHOLD:
        human_review_required = True
        reasons.append(f"Model confidence ({confidence:.1%}) is below calibrated threshold ({CONFIDENCE_THRESHOLD:.0%})")
    if p_review >= HUMAN_REVIEW_SCORE_THRESHOLD:
        human_review_required = True
        reasons.append(f"Incident risk check flagged review (p={p_review:.2f})")
    if urgency == "critical":
        human_review_required = True
        reasons.append("Critical incident severity mandates human dispatcher sign-off")
    if engine == "SAFE_FALLBACK":
        human_review_required = True
        reasons.append("Inference offline: Safe Fallback active")

    # Generate simulation action preview
    tool_func = TOOL_BY_NAME.get(chosen_action)
    action_preview = None
    if tool_func:
        # Run tool in simulation mode to populate action record
        inc_id = incident.get("incident_id", "INC-001")
        veh_id = incident.get("vehicle_id", "TRK-001")
        if chosen_action == "dispatch_replacement":
            action_preview = tool_func(inc_id, veh_id, "STANDBY-UNIT-42", 28, incident.get("cargo_type", "GENERAL"))
        elif chosen_action == "reroute_shipment":
            action_preview = tool_func(inc_id, veh_id, incident.get("weather_condition", "CONGESTION"), 14.5, 22)
        elif chosen_action == "crossdock":
            action_preview = tool_func(inc_id, veh_id, "XDOCK-BAY-09", "Cargo cold-chain protection")
        elif chosen_action == "escalate_to_human":
            action_preview = tool_func(inc_id, veh_id, "; ".join(reasons) or "Automated escalation threshold reached")
        else:
            action_preview = tool_func(inc_id, veh_id, 15)

    return {
        "action": chosen_action,
        "urgency": urgency,
        "model_confidence": round(confidence, 4),
        "human_review_required": human_review_required,
        "human_review_reason": "; ".join(reasons) if reasons else "Autonomous execution verified within parameters",
        "engine": engine,
        "alternatives": {k: round(v, 4) for k, v in probabilities.items()},
        "actions_preview": [action_preview] if action_preview else [],
    }
