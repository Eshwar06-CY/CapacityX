"""Amazon Bedrock Nova Micro Provider for CAPACITYX.

Integrates Amazon Bedrock's Converse API for model `amazon.nova-micro-v1:0` in `us-east-1`
as an optional decision engine alongside the default Strands Decider 2B client.

Official AWS References:
- Model Parameters: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-nova.html
- Converse API & Nova Code Examples: https://docs.aws.amazon.com/bedrock/latest/userguide/service_code_examples_bedrock-runtime_amazon_nova.html

Safeguards:
- Strict output-token limit (300 tokens) and finite timeouts (5s connect, 15s read).
- Automatic application-level retries disabled (max_attempts = 0).
- No hardcoded credentials or logged secrets/payloads.
- No calibrated probability hallucination: model_confidence=None, alternatives={}.
- Robust deterministic safe fallback on any timeout, permission error, or malformed response.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

# Attempt boto3 import; graceful degradation when boto3 is not yet installed
try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None
    Config = None
    ClientError = None
    ConnectTimeoutError = None
    ReadTimeoutError = None
    BOTO3_AVAILABLE = False

from patterns.strands_decider_2b import (
    LOGISTICS_TOOLS,
    DeciderOutcome,
    IncidentContext,
)

logger = logging.getLogger("capacityx.bedrock")

# Official model and environment defaults
DEFAULT_BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"
DEFAULT_BEDROCK_REGION = "us-east-1"
ENV_BEDROCK_MODEL_ID = "BEDROCK_MODEL_ID"
ENV_BEDROCK_REGION = "AWS_REGION"

# Guardrail timeouts and token limits
DEFAULT_MAX_TOKENS = 300
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 15.0

ALLOWED_ACTIONS = set(LOGISTICS_TOOLS.keys())
ALLOWED_URGENCIES = {"LOW", "MODERATE", "HIGH", "CRITICAL"}


class BedrockConfigurationError(ValueError):
    """Raised when Bedrock configuration parameters or credentials are invalid."""


class BedrockProviderUnavailableError(RuntimeError):
    """Raised when boto3 is missing or Bedrock client initialization fails."""


def build_nova_system_prompt() -> str:
    """Build standardized system prompt defining the logistics tool-selection task."""
    tools_doc = "\n".join(
        f"- {tool}: {desc}" for tool, desc in LOGISTICS_TOOLS.items()
    )
    return (
        "You are an expert logistics incident decision agent for CAPACITYX.\n"
        "Your task is to analyze telemetry from a logistics incident and select exactly ONE "
        "mitigation tool from the following five allowed actions:\n\n"
        f"{tools_doc}\n\n"
        "RESPONSE INSTRUCTIONS:\n"
        "Respond ONLY with a single valid JSON object. Do not include markdown formatting, "
        "code blocks, preambles, or additional text. Your JSON must strictly follow this schema:\n"
        "{\n"
        '  "selected_action": "one of [dispatch_replacement, reroute_shipment, crossdock, escalate_to_human, continue_monitoring]",\n'
        '  "urgency": "one of [LOW, MODERATE, HIGH, CRITICAL]",\n'
        '  "headline": "concise 1-sentence decision summary",\n'
        '  "rationale": ["bullet 1", "bullet 2"],\n'
        '  "estimated_delay_minutes": <integer>,\n'
        '  "human_review_required": <boolean>\n'
        "}"
    )


def build_nova_user_message(incident: IncidentContext) -> str:
    """Format incident context for Nova Micro inference without leaking sensitive data."""
    return (
        "Logistics Incident Telemetry State:\n"
        f"{incident.to_state_text()}\n\n"
        "Select the single best mitigation action and respond with the requested JSON object."
    )


class BedrockNovaMicroClient:
    """Low-level Converse API client for Amazon Bedrock Nova Micro.

    Adheres to the official AWS Nova Converse API format with strict timeout
    and zero-retry configurations.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_BEDROCK_MODEL_ID,
        region_name: Optional[str] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        client: Optional[Any] = None,
    ) -> None:
        self.model_id = model_id or os.environ.get(ENV_BEDROCK_MODEL_ID, DEFAULT_BEDROCK_MODEL_ID)
        self.region_name = region_name or os.environ.get(ENV_BEDROCK_REGION, DEFAULT_BEDROCK_REGION)
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        if client is not None:
            self._client = client
        elif BOTO3_AVAILABLE and boto3 is not None and Config is not None:
            config = Config(
                connect_timeout=self.connect_timeout,
                read_timeout=self.read_timeout,
                retries={"max_attempts": 0},  # Disable automated application-level retries
            )
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region_name,
                config=config,
            )
        else:
            self._client = None

    def converse(
        self,
        user_text: str,
        system_text: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """Invoke Amazon Bedrock Converse API for Nova Micro."""
        if self._client is None:
            raise BedrockProviderUnavailableError(
                "boto3 is not available or bedrock-runtime client could not be initialized."
            )

        messages = [
            {
                "role": "user",
                "content": [{"text": user_text}],
            }
        ]

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": 0.0,
                "topP": 0.9,
            },
        }

        if system_text:
            kwargs["system"] = [{"text": system_text}]

        return self._client.converse(**kwargs)

    def __repr__(self) -> str:
        # Sanitized string representation: never prints credentials
        return (
            f"<BedrockNovaMicroClient model='{self.model_id}' "
            f"region='{self.region_name}' timeouts=({self.connect_timeout}s/{self.read_timeout}s)>"
        )


class BedrockNovaMicroAgent:
    """Decision Agent orchestrator implementing the Bedrock Nova Micro decision pattern.

    Integrates amazon.nova-micro-v1:0 via the Bedrock Converse API. Validates tool selections
    against the 5 canonical logistics actions, enforces zero-probability-hallucination policies,
    and falls back deterministically to safe rule policies on errors.
    """

    provider_name = "bedrock_nova_micro"

    def __init__(
        self,
        model_id: str = DEFAULT_BEDROCK_MODEL_ID,
        region_name: Optional[str] = None,
        client: Optional[BedrockNovaMicroClient] = None,
    ) -> None:
        self.model_id = model_id or os.environ.get(ENV_BEDROCK_MODEL_ID, DEFAULT_BEDROCK_MODEL_ID)
        self.region_name = region_name or os.environ.get(ENV_BEDROCK_REGION, DEFAULT_BEDROCK_REGION)

        try:
            self.client = client or BedrockNovaMicroClient(
                model_id=self.model_id,
                region_name=self.region_name,
            )
            self._init_error: Optional[str] = None
        except Exception as exc:
            self.client = None
            self._init_error = str(exc)

    def get_health_status(self) -> Dict[str, Any]:
        """Return provider configuration status without making live network calls."""
        if not BOTO3_AVAILABLE:
            return {
                "status": "offline",
                "provider": self.provider_name,
                "is_real_model_available": False,
                "model_id": self.model_id,
                "region": self.region_name,
                "reason": "boto3 library is not installed in the environment.",
                "instruction": "Run 'pip install boto3' in your virtual environment to enable the Bedrock provider.",
            }

        if self._init_error:
            return {
                "status": "offline",
                "provider": self.provider_name,
                "is_real_model_available": False,
                "model_id": self.model_id,
                "region": self.region_name,
                "reason": f"Client initialization error: {self._init_error}",
                "instruction": "Verify AWS credentials and environment variables.",
            }

        return {
            "status": "configured_offline_ready",
            "provider": self.provider_name,
            "is_real_model_available": False,  # Disclose honestly until verified live
            "live_inference_tested": False,
            "model_id": self.model_id,
            "region": self.region_name,
            "notice": (
                "Amazon Bedrock Nova Micro adapter configured with Converse API. "
                "Offline unit tests verified. Live AWS calls require valid IAM permissions "
                "(bedrock:InvokeModel) and approved model access in AWS Console."
            ),
        }

    def evaluate(self, incident: IncidentContext) -> DeciderOutcome:
        """Evaluate a logistics incident with Amazon Bedrock Nova Micro or safe fallback."""
        if self._init_error or self.client is None:
            return self._safe_rule_fallback(
                incident,
                reason=f"Bedrock provider unavailable: {self._init_error or 'client uninitialized'}",
            )

        system_text = build_nova_system_prompt()
        user_text = build_nova_user_message(incident)

        try:
            response = self.client.converse(
                user_text=user_text,
                system_text=system_text,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
            return self._parse_and_validate_response(incident, response)

        except Exception as exc:
            # Handle client timeouts, permission errors, and other SDK exceptions safely
            err_msg = str(exc)
            if ClientError and isinstance(exc, ClientError):
                code = exc.response.get("Error", {}).get("Code", "ClientError")
                msg = exc.response.get("Error", {}).get("Message", err_msg)
                reason = f"Amazon Bedrock API error ({code}): {msg}"
            elif (ReadTimeoutError and isinstance(exc, ReadTimeoutError)) or (
                ConnectTimeoutError and isinstance(exc, ConnectTimeoutError)
            ):
                reason = f"Amazon Bedrock request timed out: {err_msg}"
            elif "timeout" in err_msg.lower():
                reason = f"Amazon Bedrock request timed out: {err_msg}"
            else:
                reason = f"Amazon Bedrock invocation failed: {err_msg}"

            return self._safe_rule_fallback(incident, reason=reason)

    def _parse_and_validate_response(
        self, incident: IncidentContext, response: Dict[str, Any]
    ) -> DeciderOutcome:
        """Strictly parse and validate structured Nova Micro response."""
        try:
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])

            if not content or "text" not in content[0]:
                return self._safe_rule_fallback(
                    incident,
                    reason="Amazon Bedrock response contained empty or non-text message content.",
                )

            raw_text = content[0]["text"].strip()
            # Clean possible markdown formatting (e.g. ```json ... ```)
            cleaned_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r"\s*```$", "", cleaned_text).strip()

            data = json.loads(cleaned_text)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            return self._safe_rule_fallback(
                incident,
                reason=f"Amazon Bedrock returned malformed/unparseable JSON response ({exc}).",
            )

        if not isinstance(data, dict):
            return self._safe_rule_fallback(
                incident,
                reason="Amazon Bedrock JSON response was not a structured object.",
            )

        selected_action = str(data.get("selected_action", "")).strip()

        # Strict validation: selected_action must be one of the 5 allowed logistics tools
        if selected_action not in ALLOWED_ACTIONS:
            return self._safe_rule_fallback(
                incident,
                reason=(
                    f"Amazon Bedrock selected unrecognized action '{selected_action}'. "
                    f"Must be one of: {list(ALLOWED_ACTIONS)}."
                ),
            )

        raw_urgency = str(data.get("urgency", "HIGH")).strip().upper()
        urgency = raw_urgency if raw_urgency in ALLOWED_URGENCIES else "HIGH"

        headline = str(data.get("headline", "")).strip() or (
            f"Amazon Bedrock Nova Micro selected '{selected_action}'."
        )

        raw_rationale = data.get("rationale", [])
        if isinstance(raw_rationale, list) and raw_rationale:
            rationale = [str(r) for r in raw_rationale]
        else:
            rationale = [f"Decision selected by Amazon Bedrock Nova Micro ({selected_action})."]

        # Parse estimated delay minutes
        try:
            estimated_delay = int(data.get("estimated_delay_minutes", 45))
        except (ValueError, TypeError):
            estimated_delay = 45

        human_review_required = bool(data.get("human_review_required", False))
        deadline_met = estimated_delay <= 60

        sla_risk = (
            "CRITICAL_BREACH"
            if not deadline_met and incident.truck_breakdown
            else "HIGH"
            if not deadline_met
            else "MODERATE"
            if incident.truck_breakdown
            else "LOW"
        )

        return DeciderOutcome(
            selected_action=selected_action,
            action_type=selected_action.upper(),
            urgency=urgency,
            headline=headline,
            rationale=rationale,
            estimated_delay_minutes=estimated_delay,
            deadline_met_projected=deadline_met,
            sla_risk_level=sla_risk,
            customer_action_required=(not incident.customer_notified) and incident.truck_breakdown,
            human_review_required=human_review_required,
            engine="AMAZON_BEDROCK_NOVA_MICRO",
            is_real_model=True,
            model_confidence=None,  # Do not trust generative LLM for calibrated probabilities
            alternatives={},
            fallback_reason=None,
            model_id=self.model_id,
            is_simulated=False,
            metadata={
                "provider": self.provider_name,
                "region": self.region_name,
                "stop_reason": response.get("stopReason"),
                "usage": response.get("usage", {}),
            },
        )

    def _safe_rule_fallback(self, incident: IncidentContext, reason: str) -> DeciderOutcome:
        """Deterministic safety policy preserving human-review safeguards."""
        is_perishable = bool(
            re.search(r"cold|perish|pharma|food|vaccine|seafood", incident.cargo_type, re.IGNORECASE)
        )

        if incident.truck_breakdown and incident.replacement_vehicle_available:
            selected_action = "dispatch_replacement"
            urgency = "CRITICAL" if is_perishable else "HIGH"
            delay = 35 if is_perishable else 45
            headline = f"[Safe Fallback] Dispatch replacement vehicle for {incident.shipment_id}"
            sla_risk = "MODERATE"
            deadline_met = True
            rationale = [
                f"SAFE FALLBACK APPLIED ({self.provider_name}): {reason}",
                f"Vehicle immobilized ({incident.breakdown_severity}); backup unit available.",
                "Mandatory human review required in fallback mode.",
            ]
        elif incident.truck_breakdown and not incident.replacement_vehicle_available:
            selected_action = "escalate_to_human"
            urgency = "CRITICAL"
            delay = 180
            headline = f"[Safe Fallback] Escalate to human logistics dispatcher for {incident.shipment_id}"
            sla_risk = "CRITICAL_BREACH"
            deadline_met = False
            rationale = [
                f"SAFE FALLBACK APPLIED ({self.provider_name}): {reason}",
                "Vehicle immobilized with NO replacement units within service range.",
                "High probability of delivery SLA breach. Human dispatcher intervention required.",
            ]
        else:
            selected_action = "continue_monitoring"
            urgency = "LOW"
            delay = 0
            headline = f"[Safe Fallback] Continue monitoring: Nominal transit for {incident.shipment_id}"
            sla_risk = "LOW"
            deadline_met = True
            rationale = [
                f"SAFE FALLBACK APPLIED ({self.provider_name}): {reason}",
                "Vehicle telemetry nominal; proceeding to destination.",
            ]

        return DeciderOutcome(
            selected_action=selected_action,
            action_type=selected_action.upper(),
            urgency=urgency,
            headline=headline,
            rationale=rationale,
            estimated_delay_minutes=delay,
            deadline_met_projected=deadline_met,
            sla_risk_level=sla_risk,
            customer_action_required=(not incident.customer_notified) and incident.truck_breakdown,
            human_review_required=True,
            engine="SAFE_FALLBACK",
            is_real_model=False,
            model_confidence=None,
            alternatives={},
            fallback_reason=reason,
            model_id=self.model_id,
            is_simulated=True,
            metadata={"fallback_rule": True, "provider": self.provider_name},
        )

    def __repr__(self) -> str:
        return f"<BedrockNovaMicroAgent model='{self.model_id}' region='{self.region_name}'>"
