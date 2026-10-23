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
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Load environment variables from consolidated root .env if present
_root_env = Path(__file__).resolve().parent.parent / ".env"
if _root_env.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_root_env, override=False)
    except ImportError:
        pass

# Official upstream references
DEFAULT_MODEL_ID = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
DEFAULT_LOCAL_URL = "http://127.0.0.1:8099"
UGMDU_DECIDER_URL = "https://da25tnxy06gjk.cloudfront.net"

# Environment variable keys
ENV_DECIDER_API_KEY = "DECIDER_API_KEY"
ENV_DECIDER_URL = "DECIDER_URL"
ENV_DECIDER_MODE = "DECIDER_MODE"

ENV_STRANDS_DECIDER_URL = "STRANDS_DECIDER_URL"
ENV_DECIDER_USERNAME = "STRANDS_DECIDER_USERNAME"
ENV_DECIDER_PASSWORD = "STRANDS_DECIDER_PASSWORD"

DEFAULT_DECIDER_URL = (
    os.environ.get(ENV_DECIDER_URL)
    or os.environ.get(ENV_STRANDS_DECIDER_URL)
    or DEFAULT_LOCAL_URL
)
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
    human_review_reason: Optional[str] = None

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
            "human_review_reason": self.human_review_reason,
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
    """Official synchronous HTTP client for Strands Decider 2B.

    Supports both:
    1. Official Hosted CloudFront API (UG Madurai / team API key):
       - Base URL: https://da25tnxy06gjk.cloudfront.net
       - Endpoint: POST /v1/decide
       - Health: GET /v1/models
       - Auth: Authorization: Bearer <DECIDER_API_KEY>
       - Payload: {"state": state, "questions": dict(questions)} (strictly no model or generation parameters)
    2. Local server or authenticated ngrok tunnel:
       - Endpoint: POST /v1/systemone
       - Health: GET /health
       - Auth: Optional HTTP Basic (STRANDS_DECIDER_USERNAME & STRANDS_DECIDER_PASSWORD)
       - Payload: {"state": state, "model": model_id, "questions": dict(questions)}
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        # Resolve Basic Auth credentials (for local server or ngrok tunnel)
        u = username if username is not None else os.environ.get(ENV_DECIDER_USERNAME)
        p = password if password is not None else os.environ.get(ENV_DECIDER_PASSWORD)
        u = u.strip() if isinstance(u, str) and u.strip() else None
        p = p.strip() if isinstance(p, str) and p.strip() else None

        # Resolve API key from argument or environment variables
        # If explicit local/ngrok credentials are provided, preserve Basic Auth over ambient DECIDER_API_KEY
        if api_key is not None:
            key = api_key
        elif u and p:
            key = None
        else:
            key = os.environ.get(ENV_DECIDER_API_KEY) or os.environ.get("STRANDS_DECIDER_API_KEY")

        self.api_key = key.strip() if isinstance(key, str) and key.strip() else None

        # Resolve mode
        env_mode = os.environ.get(ENV_DECIDER_MODE, "").strip().lower()

        # Resolve URL
        resolved_url = url or os.environ.get(ENV_DECIDER_URL) or os.environ.get(ENV_STRANDS_DECIDER_URL)
        if not resolved_url:
            if self.api_key or env_mode == "ugmdu":
                resolved_url = UGMDU_DECIDER_URL
            else:
                resolved_url = DEFAULT_LOCAL_URL

        self.url = resolved_url.rstrip("/")
        self.timeout = timeout

        if self.api_key:
            self._auth_header: Optional[str] = f"Bearer {self.api_key}"
            self.is_authenticated: bool = True
            self.auth_type: str = "bearer"
        elif u and p:
            token = base64.b64encode(f"{u}:{p}".encode("utf-8")).decode("ascii")
            self._auth_header = f"Basic {token}"
            self.is_authenticated = True
            self.auth_type = "basic"
        elif u or p:
            missing = ENV_DECIDER_PASSWORD if u else ENV_DECIDER_USERNAME
            raise ConfigurationError(
                f"Incomplete Strands Decider credentials configured: missing {missing}."
            )
        else:
            self._auth_header = None
            self.is_authenticated = False
            self.auth_type = "none"

        # Hosted mode is active if Bearer API key is present or CloudFront endpoint is targeted (not Basic Auth)
        self.is_hosted: bool = (self.auth_type == "bearer") or (
            self.auth_type != "basic" and (env_mode == "ugmdu" or "da25tnxy06gjk.cloudfront.net" in self.url)
        )

    def _build_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Construct request headers including Bearer or Basic Auth and ngrok tunnel headers."""
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
        """Query server status: /v1/models for hosted API, /health for local server."""
        headers = self._build_headers()
        path = "/v1/models" if self.is_hosted else "/health"
        req = urllib.request.Request(f"{self.url}{path}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as res:
                data = json.loads(res.read().decode())
                if isinstance(data, dict):
                    return data
                elif isinstance(data, list):
                    return {"models": data, "status": "ok"}
                return {"raw": data}
        except Exception as exc:
            hint = (
                "Check DECIDER_API_KEY."
                if self.is_hosted
                else f"Start with: strands-decider serve {DEFAULT_MODEL_ID} --port 8099"
            )
            raise ServerUnavailableError(
                f"Strands Decider server unreachable at {self.url}{path} ({exc}). {hint}"
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
        """Send state and typed questions.

        In hosted mode (POST /v1/decide):
        - Sends ONLY 'state' and 'questions'.
        - Never sends 'model', 'max_tokens', 'temperature', or generation parameters.

        In local mode (POST /v1/systemone):
        - Sends 'state', 'model', and 'questions'.
        """
        if self.is_hosted:
            endpoint_path = "/v1/decide"
            payload_dict: Dict[str, Any] = {
                "state": state,
                "questions": dict(questions),
            }
        else:
            endpoint_path = "/v1/systemone"
            payload_dict = {
                "state": state,
                "model": model_id,
                "questions": dict(questions),
            }

        payload = json.dumps(payload_dict).encode("utf-8")
        headers = self._build_headers({"Content-Type": "application/json"})
        req = urllib.request.Request(
            f"{self.url}{endpoint_path}",
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                response_data = json.loads(res.read().decode())
                return response_data.get("answers", {})
        except urllib.error.URLError as exc:
            raise ServerUnavailableError(
                f"Strands Decider request failed at {self.url}{endpoint_path} ({exc})"
            ) from exc

    def __repr__(self) -> str:
        mode = "hosted" if self.is_hosted else "local"
        auth_state = f"authenticated ({self.auth_type})" if self.is_authenticated else "unauthenticated"
        return f"<StrandsDeciderClient url='{self.url}' mode={mode} auth={auth_state}>"


class StrandsDecider2B:
    """Decision Agent orchestrator implementing the Strands Decider 2B pattern.

    Integrates the genuine Strands Decider 2B model via its official API
    (hosted CloudFront or local inference), selecting among the 5 typed tools:
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
        server_url: Optional[str] = None,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        model_id: str = DEFAULT_MODEL_ID,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.model_id = model_id
        self._config_error: Optional[str] = None

        try:
            self.client: Optional[StrandsDeciderClient] = StrandsDeciderClient(
                url=server_url,
                api_key=api_key,
                username=username,
                password=password,
            )
            self.server_url = self.client.url
        except ConfigurationError as exc:
            self.client = None
            self.server_url = server_url or DEFAULT_LOCAL_URL
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
            mode = "hosted" if self.client.is_hosted else "local"
            models_data = info.get("data")
            first_model_id = (
                models_data[0].get("id")
                if isinstance(models_data, list) and models_data and isinstance(models_data[0], dict)
                else None
            )
            checkpoint = info.get("checkpoint") or first_model_id or (f"hosted ({mode})" if self.client.is_hosted else None)
            return {
                "status": "online",
                "is_real_model_available": True,
                "server_url": self.server_url,
                "mode": mode,
                "is_authenticated": self.client.is_authenticated,
                "checkpoint": checkpoint,
                "base_model": info.get("base_model"),
                "device": info.get("device"),
                "window": info.get("max_length"),
            }
        except Exception as exc:
            instruction = (
                "Ensure DECIDER_API_KEY is valid and has remaining quota."
                if getattr(self.client, "is_hosted", False)
                else f"Run 'strands-decider serve {self.model_id} --port 8099' on a GPU/WSL server to activate real inference."
            )
            return {
                "status": "offline",
                "is_real_model_available": False,
                "server_url": self.server_url,
                "mode": "hosted" if getattr(self.client, "is_hosted", False) else "local",
                "reason": str(exc),
                "instruction": instruction,
            }

    def evaluate(self, incident: IncidentContext) -> DeciderOutcome:
        """Evaluate logistics incident using real Strands Decider 2B, or safe fallback."""
        if self._config_error:
            return self._safe_rule_fallback(
                incident,
                reason=f"Credential configuration error: {self._config_error}",
            )

        state_text = incident.to_state_text()

        # Build official typed questions matching upstream specification (strictly <= 4 questions)
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
            # Safe, transparent fallback when model is not available or response invalid
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
        if not answers or not isinstance(answers, dict):
            raise ValueError("Model returned empty or non-dictionary answers payload.")

        action_answer = answers.get("action")
        if not action_answer or not isinstance(action_answer, dict):
            raise ValueError("Model response missing or empty 'action' answer.")

        selected_action = action_answer.get("choice") or action_answer.get("label")
        if not selected_action or selected_action not in LOGISTICS_TOOLS:
            raise ValueError(
                f"Invalid action choice '{selected_action}' returned by model; not in permitted action schema: {list(LOGISTICS_TOOLS.keys())}"
            )

        probabilities = {
            k: float(v) for k, v in action_answer.get("probabilities", {}).items()
        }
        if probabilities:
            confidence = max(probabilities.values())
        else:
            confidence = float(action_answer.get("confidence", 0.0))

        is_urgent_score = float(answers.get("is_urgent", {}).get("noul", 0.5))
        human_review_score = float(answers.get("human_review", {}).get("noul", 0.5))
        urgency_tier = answers.get("urgency_tier", {}).get("choice", "HIGH" if is_urgent_score > 0.6 else "MODERATE")

        # Determine human review requirement based on model confidence and noul signal
        confidence_below_threshold = confidence < self.confidence_threshold
        model_flagged_review = human_review_score >= 0.5

        review_reasons: List[str] = []
        if confidence_below_threshold:
            review_reasons.append(
                f"Model confidence ({confidence:.1%}) is below safety threshold ({self.confidence_threshold:.0%})"
            )
        if model_flagged_review:
            review_reasons.append(
                f"Model human-review safety safeguard flagged (P(review) = {human_review_score:.2f} >= 0.50)"
            )

        human_review_required = confidence_below_threshold or model_flagged_review
        human_review_reason = " | ".join(review_reasons) if review_reasons else None

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

        # Rationale bullets accurately distinguishing confidence threshold vs human review signal
        confidence_bullet = (
            f"Confidence threshold ({self.confidence_threshold:.0%}): SATISFIED ({confidence:.1%} >= {self.confidence_threshold:.0%}) - Statistical confidence criteria met."
            if not confidence_below_threshold
            else f"Confidence threshold ({self.confidence_threshold:.0%}): BELOW THRESHOLD ({confidence:.1%} < {self.confidence_threshold:.0%}) - Low model confidence requires review."
        )

        review_bullet = (
            f"Human review safeguard: FLAGGED ({human_review_reason}) - Dispatcher confirmation required."
            if human_review_required
            else f"Human review safeguard: CLEAR (P(review) = {human_review_score:.2f} < 0.50) - Automated execution authorized."
        )

        rationale = [
            f"Real Decider 2B evaluated incident state across 5 typed tools.",
            f"Top tool: '{selected_action}' (calibrated confidence: {confidence:.3f}).",
            confidence_bullet,
            review_bullet,
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
            human_review_reason=human_review_reason,
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
            human_review_reason="Safe fallback policy engaged; automated dispatch is withheld pending human confirmation.",
            engine="SAFE_FALLBACK",
            is_real_model=False,
            model_confidence=None,
            alternatives={},
            fallback_reason=reason,
            model_id="none (safe fallback rule policy)",
            is_simulated=True,
            metadata={"policy": "DETERMINISTIC_FAIL_SAFE"},
        )
