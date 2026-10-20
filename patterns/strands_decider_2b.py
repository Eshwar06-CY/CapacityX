"""Strands Decider 2B Decision Pattern Implementation.

Integrates the official Strands Decider 2B model for agentic tool selection
following the upstream specification: https://github.com/strands-labs/strands-decider

Supported Decision Tools:
1. dispatch_replacement
2. reroute_shipment
3. crossdock
4. escalate_to_human
5. continue_monitoring
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

# Official upstream references
DEFAULT_MODEL_ID = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
ENV_DECIDER_URL = "STRANDS_DECIDER_URL"
ENV_DECIDER_USERNAME = "STRANDS_DECIDER_USERNAME"
ENV_DECIDER_PASSWORD = "STRANDS_DECIDER_PASSWORD"

DEFAULT_DECIDER_URL = os.environ.get(ENV_DECIDER_URL, "http://127.0.0.1:8099")
CONFIDENCE_THRESHOLD = 0.70  # Decisions below 70% confidence require human review

# The 5 typed logistics action tools
LOGISTICS_TOOLS: Dict[str, str] = {
    "dispatch_replacement": "Primary vehicle is immobilized or severely broken down, but a backup transport vehicle is available nearby. Dispatch replacement to rendezvous and transfer freight.",
    "reroute_shipment": "Vehicle is operational, but facing minor delays, road closures, or traffic; recompute path to avoid delay.",
    "crossdock": "Vehicle has partial defect or perishable freight that needs immediate intermediate warehousing at the closest crossdock facility.",
    "escalate_to_human": "Vehicle disabled with NO replacement units available, or high uncertainty / SLA breach that cannot be automated safely.",
    "continue_monitoring": "Vehicle and route are nominal; maintain standard telemetry tracking without intervention.",
}


@dataclass
class IncidentContext:
    """Standard context passed to any Strands Decider."""
    shipment_id: str
    delivery_deadline: str
    truck_breakdown: bool
    replacement_vehicle_available: bool
    customer_notified: bool
    breakdown_severity: str = "MODERATE"
    cargo_type: str = "Standard Freight"
    origin: str = "Origin Depot"
    destination: str = "Destination Hub"
    estimated_distance_remaining_km: float = 50.0

    def to_state_text(self) -> str:
        """Format incident context into a dense state representation for Strands Decider."""
        vehicle_status = (
            f"BREAKDOWN_IMMOBILIZED (Severity: {self.breakdown_severity})"
            if self.truck_breakdown
            else "NOMINAL_OPERATIONAL"
        )
        backup_status = "AVAILABLE_WITHIN_45MIN" if self.replacement_vehicle_available else "UNAVAILABLE_NONE_IN_RANGE"
        customer_status = "ALREADY_NOTIFIED" if self.customer_notified else "PENDING_ALERT"

        return (
            f"Shipment Identifier: {self.shipment_id}\n"
            f"Delivery Target Deadline: {self.delivery_deadline}\n"
            f"Freight Classification: {self.cargo_type}\n"
            f"Route: {self.origin} -> {self.destination} ({self.estimated_distance_remaining_km} km remaining)\n"
            f"Primary Vehicle Telemetry: {vehicle_status}\n"
            f"Backup Fleet Status: {backup_status}\n"
            f"Customer Contact State: {customer_status}"
        )


@dataclass
class DeciderOutcome:
    """Standard structured outcome returned by Strands Decider 2B with typed tool selection."""
    selected_action: str
    action_type: str
    urgency: str
    headline: str
    rationale: List[str]
    estimated_delay_minutes: int
    deadline_met_projected: bool
    sla_risk_level: str
    customer_action_required: bool
    human_review_required: bool
    engine: str  # "STRANDS_DECIDER_2B_REAL" or "SAFE_FALLBACK"
    is_real_model: bool
    model_confidence: Optional[float] = None
    alternatives: Dict[str, float] = field(default_factory=dict)
    fallback_reason: Optional[str] = None
    model_id: str = DEFAULT_MODEL_ID
    is_simulated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_action": self.selected_action,
            "action_type": self.action_type,
            "urgency": self.urgency,
            "headline": self.headline,
            "rationale": self.rationale,
            "estimated_delay_minutes": self.estimated_delay_minutes,
            "deadline_met_projected": self.deadline_met_projected,
            "sla_risk_level": self.sla_risk_level,
            "customer_action_required": self.customer_action_required,
            "human_review_required": self.human_review_required,
            "engine": self.engine,
            "is_real_model": self.is_real_model,
            "model_confidence": self.model_confidence,
            "alternatives": self.alternatives,
            "fallback_reason": self.fallback_reason,
            "model_id": self.model_id,
            "is_simulated": self.is_simulated,
            "metadata": self.metadata,
        }


class ServerUnavailableError(RuntimeError):
    """Raised when the official Strands Decider microservice is not responding."""


class ConfigurationError(ValueError):
    """Raised when environment variables or client settings are misconfigured."""


class StrandsDeciderClient:
    """Official synchronous HTTP client for a running Strands Decider 2B inference server.

    Adheres strictly to the upstream client contract in `examples/strands/_client.py`:
    - Endpoint: `POST {url}/v1/systemone`
    - Health: `GET {url}/health`

    Supports optional HTTP Basic Authentication (e.g. for authenticated ngrok tunnels)
    via STRANDS_DECIDER_USERNAME and STRANDS_DECIDER_PASSWORD.
    """

    def __init__(
        self,
        url: str = DEFAULT_DECIDER_URL,
        timeout: float = 30.0,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout

        # Read username/password from arguments or environment variables
        u = username if username is not None else os.environ.get(ENV_DECIDER_USERNAME)
        p = password if password is not None else os.environ.get(ENV_DECIDER_PASSWORD)

        # Normalize whitespace
        u = u.strip() if isinstance(u, str) and u.strip() else None
        p = p.strip() if isinstance(p, str) and p.strip() else None

        if u and p:
            # Safely encode credentials to Base64; never logged or exposed
            token = base64.b64encode(f"{u}:{p}".encode("utf-8")).decode("ascii")
            self._auth_header: Optional[str] = f"Basic {token}"
            self.is_authenticated: bool = True
        elif u or p:
            missing = ENV_DECIDER_PASSWORD if u else ENV_DECIDER_USERNAME
            raise ConfigurationError(
                f"Incomplete Strands Decider credentials configured: missing {missing}."
            )
        else:
            self._auth_header = None
            self.is_authenticated = False

    def _build_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct request headers including Basic Auth and ngrok tunnel compatibility headers."""
        headers: Dict[str, str] = {
            "User-Agent": "CapacityX-StrandsDeciderClient/0.2.0",
            # ngrok-skip-browser-warning prevents HTML interstitial warning page on free ngrok domains
            "ngrok-skip-browser-warning": "true",
        }
        if self._auth_header:
            headers["Authorization"] = self._auth_header
        if additional_headers:
            headers.update(additional_headers)
        return headers

    def health(self) -> Dict[str, Any]:
        """Query the loaded checkpoint, base model, and device from the running server."""
        headers = self._build_headers()
        req = urllib.request.Request(f"{self.url}/health", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as res:
                return json.loads(res.read().decode())
        except Exception as exc:
            raise ServerUnavailableError(
                f"Strands Decider server unreachable at {self.url} ({exc}). "
                f"Start with: strands-decider serve {DEFAULT_MODEL_ID} --port 8099"
            ) from exc

    def is_available(self) -> bool:
        """Non-raising health probe."""
        try:
            self.health()
            return True
        except Exception:
            return False

    def ask(
        self,
        state: str,
        questions: Mapping[str, Dict[str, Any]],
        model_id: str = DEFAULT_MODEL_ID,
    ) -> Dict[str, Any]:
        """Send state and typed questions to /v1/systemone."""
        payload = json.dumps({
            "state": state,
            "model": model_id,
            "questions": dict(questions),
        }).encode("utf-8")

        headers = self._build_headers({"Content-Type": "application/json"})
        req = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=payload,
            headers=headers,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                response_data = json.loads(res.read().decode())
                return response_data.get("answers", {})
        except urllib.error.URLError as exc:
            raise ServerUnavailableError(
                f"Strands Decider request failed at {self.url}/v1/systemone ({exc})"
            ) from exc

    def __repr__(self) -> str:
        auth_state = "authenticated" if self.is_authenticated else "unauthenticated"
        return f"<StrandsDeciderClient url='{self.url}' auth={auth_state}>"


class StrandsDecider2B:
    """Decision Agent orchestrator implementing the Strands Decider 2B pattern.

    Integrates the genuine Strands Decider 2B model via its official API,
    selecting among the 5 typed tools:
    - dispatch_replacement
    - reroute_shipment
    - crossdock
    - escalate_to_human
    - continue_monitoring

    If the model server is offline, credentials are misconfigured, or confidence is below threshold,
    it executes a deterministic safe fallback and clearly labels the outcome.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_DECIDER_URL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_id: str = DEFAULT_MODEL_ID,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.server_url = server_url
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        self._config_error: Optional[str] = None

        try:
            self.client: Optional[StrandsDeciderClient] = StrandsDeciderClient(
                url=server_url,
                username=username,
                password=password,
            )
        except ConfigurationError as exc:
            self.client = None
            self._config_error = str(exc)

    def get_health_status(self) -> Dict[str, Any]:
        """Check live model server availability."""
        if self._config_error:
            return {
                "status": "offline",
                "is_real_model_available": False,
                "server_url": self.server_url,
                "reason": f"Configuration error: {self._config_error}",
                "instruction": "Set both STRANDS_DECIDER_USERNAME and STRANDS_DECIDER_PASSWORD, or leave both unset for unauthenticated local servers.",
            }
        try:
            info = self.client.health()
            return {
                "status": "online",
                "is_real_model_available": True,
                "server_url": self.server_url,
                "is_authenticated": self.client.is_authenticated,
                "checkpoint": info.get("checkpoint"),
                "base_model": info.get("base_model"),
                "device": info.get("device"),
                "window": info.get("max_length"),
            }
        except Exception as exc:
            return {
                "status": "offline",
                "is_real_model_available": False,
                "server_url": self.server_url,
                "reason": str(exc),
                "instruction": f"Run 'strands-decider serve {self.model_id} --port 8099' on a GPU/WSL server to activate real inference.",
            }

    def evaluate(self, incident: IncidentContext) -> DeciderOutcome:
        """Evaluate logistics incident using real Strands Decider 2B, or safe fallback."""
        if self._config_error:
            return self._safe_rule_fallback(
                incident,
                reason=f"Credential configuration error: {self._config_error}",
            )

        state_text = incident.to_state_text()

        # Build official typed questions matching upstream specification
        questions = {
            "action": {
                "type": "choice",
                "instructions": "Select the optimal logistics tool/action for the current shipment state.",
                "criteria": LOGISTICS_TOOLS,
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this logistics incident convey high urgency or critical SLA risk?",
                "criteria": {
                    "true": "Primary vehicle breakdown, perishable cargo risk, or critical delivery deadline breach.",
                    "false": "Nominal operations, manageable minor delays, or routine monitoring.",
                },
            },
            "human_review": {
                "type": "noul",
                "instructions": "Does this situation require mandatory human dispatcher confirmation?",
                "criteria": {
                    "true": "Unresolvable asset shortage, hazardous condition, or significant uncertainty.",
                    "false": "Standard automated decision execution.",
                },
            },
            "urgency_tier": {
                "type": "choice",
                "instructions": "Classify the priority severity tier.",
                "criteria": {
                    "LOW": "Routine monitoring with nominal operations.",
                    "MODERATE": "Minor disruption with manageable delay.",
                    "HIGH": "Breakdown with immediate backup available.",
                    "CRITICAL": "Breakdown with no backup, or perishable cargo at immediate risk.",
                },
            },
        }

        # Attempt real inference
        try:
            answers = self.client.ask(state=state_text, questions=questions, model_id=self.model_id)
            return self._parse_real_model_response(incident, answers)
        except Exception as exc:
            # Safe, transparent fallback when model is not available
            return self._safe_rule_fallback(
                incident,
                reason=(
                    f"Official Strands Decider 2B service at {self.server_url} is unavailable ({exc}). "
                    "Safely routed to deterministic safety policy."
                ),
            )

    def _parse_real_model_response(
        self, incident: IncidentContext, answers: Dict[str, Any]
    ) -> DeciderOutcome:
        """Parse genuine model response with calibrated probabilities."""
        action_answer = answers.get("action", {})
        selected_action = action_answer.get("choice", "escalate_to_human")
        confidence = float(action_answer.get("confidence", 0.0))
        probabilities = action_answer.get("probabilities", {})

        is_urgent_score = float(answers.get("is_urgent", {}).get("noul", 0.5))
        human_review_score = float(answers.get("human_review", {}).get("noul", 0.5))
        urgency_tier = answers.get("urgency_tier", {}).get("choice", "HIGH" if is_urgent_score > 0.6 else "MODERATE")

        # Determine human review requirement based on model confidence and noul signal
        human_review_required = (confidence < self.confidence_threshold) or (human_review_score >= 0.5)

        # Map selected action to delay estimates and dashboard-friendly descriptions
        delay_map = {
            "dispatch_replacement": 45,
            "reroute_shipment": 25,
            "crossdock": 60,
            "escalate_to_human": 180,
            "continue_monitoring": 0,
        }
        estimated_delay = delay_map.get(selected_action, 60)
        deadline_met = estimated_delay <= 60

        headline = f"Strands Decider 2B selected '{selected_action}' with {confidence:.1%} calibrated confidence."
        rationale = [
            f"Real Decider 2B evaluated incident state across 5 typed tools.",
            f"Top tool: '{selected_action}' (calibrated confidence: {confidence:.3f}).",
            f"Confidence threshold ({self.confidence_threshold:.0%}): "
            + ("SATISFIED - Automated execution allowed." if not human_review_required else "LOW/FLAGGED - Human dispatcher review required."),
            f"Urgency assessment: {urgency_tier} (P(urgent) = {is_urgent_score:.2f}).",
        ]

        return DeciderOutcome(
            selected_action=selected_action,
            action_type=selected_action.upper(),
            urgency=urgency_tier,
            headline=headline,
            rationale=rationale,
            estimated_delay_minutes=estimated_delay,
            deadline_met_projected=deadline_met,
            sla_risk_level="LOW" if deadline_met and urgency_tier == "LOW" else "MODERATE" if deadline_met else "CRITICAL_BREACH",
            customer_action_required=not incident.customer_notified and incident.truck_breakdown,
            human_review_required=human_review_required,
            engine="STRANDS_DECIDER_2B_REAL",
            is_real_model=True,
            model_confidence=confidence,
            alternatives=probabilities,
            fallback_reason=None,
            model_id=self.model_id,
            is_simulated=False,
            metadata={"urgency_noul": is_urgent_score, "human_review_noul": human_review_score},
        )

    def _safe_rule_fallback(self, incident: IncidentContext, reason: str) -> DeciderOutcome:
        """Deterministic safety policy when model inference fails.

        Explicitly marked as SAFE_FALLBACK so it is never misrepresented as real model output.
        """
        is_perishable = any(
            kw in incident.cargo_type.lower()
            for kw in ["cold", "perish", "pharma", "food", "vaccine"]
        )

        if incident.truck_breakdown and incident.replacement_vehicle_available:
            selected = "dispatch_replacement"
            urgency = "CRITICAL" if is_perishable else "HIGH"
            headline = f"[Safe Fallback] Dispatch replacement vehicle to rendezvous with shipment {incident.shipment_id}"
            estimated_delay = 45
            deadline_met = True
            sla_risk = "MODERATE"
            rationale = [
                "SAFE FALLBACK APPLIED: Live model inference is offline or failed.",
                f"Breakdown reported ({incident.breakdown_severity}); replacement asset confirmed available.",
                f"Cargo sensitivity ({incident.cargo_type}) requires prioritized asset assignment.",
                "Mandatory human review flagged because inference ran in safe fallback mode.",
            ]
        elif incident.truck_breakdown and not incident.replacement_vehicle_available:
            selected = "escalate_to_human"
            urgency = "CRITICAL"
            headline = f"[Safe Fallback] Escalate to human dispatcher: Vehicle disabled with NO available replacement assets"
            estimated_delay = 180
            deadline_met = False
            sla_risk = "CRITICAL_BREACH"
            rationale = [
                "SAFE FALLBACK APPLIED: Live model inference is offline or failed.",
                "Vehicle immobilized with zero backup vehicles accessible within service range.",
                "Immediate manual dispatcher intervention required to contract 3PL recovery.",
            ]
        else:
            selected = "continue_monitoring"
            urgency = "LOW"
            headline = f"[Safe Fallback] Continue telemetry monitoring: Transit nominal toward {incident.destination}"
            estimated_delay = 0
            deadline_met = True
            sla_risk = "LOW"
            rationale = [
                "SAFE FALLBACK APPLIED: Live model inference is offline or failed.",
                "No mechanical breakdown detected; transit on schedule.",
            ]

        # In fallback mode, alternatives are empty or uncalibrated; model_confidence is None
        return DeciderOutcome(
            selected_action=selected,
            action_type=selected.upper(),
            urgency=urgency,
            headline=headline,
            rationale=rationale,
            estimated_delay_minutes=estimated_delay,
            deadline_met_projected=deadline_met,
            sla_risk_level=sla_risk,
            customer_action_required=not incident.customer_notified and incident.truck_breakdown,
            human_review_required=True,  # All safe fallbacks mandate human review
            engine="SAFE_FALLBACK",
            is_real_model=False,
            model_confidence=None,
            alternatives={},
            fallback_reason=reason,
            model_id="none (safe fallback rule policy)",
            is_simulated=True,
            metadata={"policy": "DETERMINISTIC_FAIL_SAFE"},
        )
