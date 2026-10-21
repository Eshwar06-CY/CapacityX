"""CAPACITYX Decision Orchestration Layer.

Pairs Strands Decider 2B (System 1 Decision Engine) with Amazon Bedrock Nova Micro
(System 2 Operational Reasoning & Validation Component).

Architecture:
1. Strands Decider 2B (Primary Decision Engine):
   - Evaluates logistics incident telemetry.
   - Selects one typed action among the 5 canonical tools:
     (dispatch_replacement, reroute_shipment, crossdock, escalate_to_human, continue_monitoring).
   - Generates calibrated confidence probabilities and alternative distribution.
   - Enforces deterministic safe fallback if offline or confidence < 0.70.

2. Amazon Bedrock Nova Micro (Operational Reasoning & Validation Component):
   - Receives the incident context alongside the Strands Decider's selected tool.
   - Performs complementary operational reasoning: evaluates operational viability,
     recommends logistical precautions, and audits safety constraints.
   - DISABLED BY DEFAULT (CAPACITYX_ENABLE_BEDROCK="false").
   - If AWS account access is unverified/blocked (e.g. Free plan ValidationException:
     "Operation not allowed"), it reports the status honestly without halting the decision.
   - In offline mock/test mode, results are explicitly labeled as simulated (is_simulated=True).
   - Never fabricates model confidence or live AWS inference.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from patterns.bedrock_nova_micro import (
    DEFAULT_BEDROCK_MODEL_ID,
    DEFAULT_BEDROCK_REGION,
    BedrockNovaMicroClient,
)
from patterns.strands_decider_2b import (
    LOGISTICS_TOOLS,
    DeciderOutcome,
    IncidentContext,
    StrandsDecider2B,
)

logger = logging.getLogger("capacityx.orchestrator")

ENV_ENABLE_BEDROCK = "CAPACITYX_ENABLE_BEDROCK"
ENV_BEDROCK_MODE = "CAPACITYX_BEDROCK_MODE"  # "disabled", "mock", "live"


@dataclass
class BedrockValidationResult:
    """Structured operational reasoning outcome from Amazon Bedrock Nova Micro."""

    status: str  # "disabled", "simulated_mock", "blocked_unverified", "validated", "failed"
    is_simulated: bool
    operational_validation: bool
    operational_notes: List[str] = field(default_factory=list)
    recommended_precautions: List[str] = field(default_factory=list)
    model_id: str = DEFAULT_BEDROCK_MODEL_ID
    disclaimer: str = ""
    error_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BedrockOperationalReasoner:
    """Operational reasoning and validation component powered by Bedrock Nova Micro.

    Accepts the incident context AND the Strands Decider's selected tool, providing
    complementary logistical reasoning while preserving the primary decision.
    """

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "disabled",
        model_id: str = DEFAULT_BEDROCK_MODEL_ID,
        region_name: str = DEFAULT_BEDROCK_REGION,
        client: Optional[Any] = None,
    ) -> None:
        self.enabled = enabled
        self.mode = mode.lower() if mode else "disabled"
        self.model_id = model_id
        self.region_name = region_name
        self._client = client

    def get_health_status(self) -> Dict[str, Any]:
        """Report Bedrock reasoning component availability without live calls."""
        if not self.enabled or self.mode == "disabled":
            return {
                "status": "disabled",
                "enabled": False,
                "mode": "disabled",
                "model_id": self.model_id,
                "region": self.region_name,
                "is_real_model_available": False,
                "live_inference_tested": False,
                "notice": (
                    "Amazon Bedrock reasoning is disabled by default. "
                    "Set CAPACITYX_ENABLE_BEDROCK=true to activate."
                ),
            }

        if self.mode == "mock":
            return {
                "status": "simulated_mock_ready",
                "enabled": True,
                "mode": "mock",
                "model_id": f"{self.model_id} (offline mock)",
                "region": self.region_name,
                "is_real_model_available": False,
                "live_inference_tested": False,
                "notice": "Mock Bedrock reasoning active for offline testing.",
            }

        return {
            "status": "live_configured_unverified",
            "enabled": True,
            "mode": "live",
            "model_id": self.model_id,
            "region": self.region_name,
            "is_real_model_available": False,
            "live_inference_tested": False,
            "notice": (
                "Live Bedrock configured. Requires verified AWS account, "
                "bedrock:InvokeModel IAM permission, and model access grant."
            ),
        }

    def validate_and_reason(
        self,
        incident: IncidentContext,
        strands_outcome: DeciderOutcome,
    ) -> BedrockValidationResult:
        """Execute complementary operational reasoning on the Strands decision."""
        if not self.enabled or self.mode == "disabled":
            return BedrockValidationResult(
                status="disabled",
                is_simulated=True,
                operational_validation=True,
                operational_notes=["Bedrock operational reasoning disabled by default."],
                recommended_precautions=[],
                model_id="none",
                disclaimer="Bedrock validation disabled by default. Decision governed by Strands Decider 2B.",
            )

        if self.mode == "mock":
            return self._generate_simulated_mock_reasoning(incident, strands_outcome)

        # Mode == "live": Attempt Bedrock Converse API invocation safely
        return self._invoke_live_bedrock_reasoning(incident, strands_outcome)

    def _generate_simulated_mock_reasoning(
        self,
        incident: IncidentContext,
        strands_outcome: DeciderOutcome,
    ) -> BedrockValidationResult:
        """Deterministic mock operational reasoning for offline testing."""
        tool = strands_outcome.selected_action
        is_perishable = bool(
            re.search(r"cold|perish|pharma|food|vaccine|seafood", incident.cargo_type, re.IGNORECASE)
        )

        notes: List[str] = []
        precautions: List[str] = []

        if tool == "dispatch_replacement":
            notes = [
                f"Bedrock operational check confirms replacement vehicle intercept for {incident.shipment_id}.",
                "Backup unit ETA (45 min) falls safely within projected delivery window.",
            ]
            if is_perishable:
                notes.append(f"Cold-chain safeguard validated: {incident.cargo_type} priority transfer required.")
                precautions = [
                    "Driver must log cargo core temperature before and after transfer.",
                    "Notify receiving warehouse dock of updated transport vehicle plate.",
                ]
            else:
                precautions = [
                    "Transfer standard bill of lading and verify seal integrity.",
                ]
        elif tool == "reroute_shipment":
            notes = [
                f"Route optimization validated: dynamic bypass around reported corridor congestion.",
                f"Estimated delay (+{strands_outcome.estimated_delay_minutes} min) preserves delivery SLA.",
            ]
            precautions = [
                "Transmit updated turn-by-turn navigation telemetry to driver cab tablet.",
            ]
        elif tool == "crossdock":
            notes = [
                f"Intermediate cross-dock bay reservation recommended for inspection at nearest logistics hub.",
            ]
            precautions = [
                "Confirm bay availability with warehouse management system prior to arrival.",
            ]
        elif tool == "escalate_to_human":
            notes = [
                f"High-risk incident confirmed: vehicle disabled with no local backup assets in range.",
                "Automated recovery infeasible without SLA breach risk. Human dispatch intervention validated.",
            ]
            precautions = [
                "Engage regional 3PL towing partner for emergency roadside recovery.",
                "Issue proactive client service advisory regarding delivery delay.",
            ]
        else:  # continue_monitoring
            notes = [
                f"Nominal telemetry confirmed: speed, heading, and milestone progression are on schedule.",
            ]
            precautions = [
                "Maintain standard 15-minute automated telemetry heartbeat.",
            ]

        return BedrockValidationResult(
            status="simulated_mock",
            is_simulated=True,
            operational_validation=True,
            operational_notes=notes,
            recommended_precautions=precautions,
            model_id=f"{self.model_id} (offline mock)",
            disclaimer="SIMULATED: Mock Bedrock operational reasoning for offline testing only. Not live AWS inference.",
        )

    def _invoke_live_bedrock_reasoning(
        self,
        incident: IncidentContext,
        strands_outcome: DeciderOutcome,
    ) -> BedrockValidationResult:
        """Call live Bedrock Converse API with safe handling for AWS verification blocks."""
        try:
            client = self._client or BedrockNovaMicroClient(
                model_id=self.model_id,
                region_name=self.region_name,
            )

            system_prompt = (
                "You are an operational reasoning agent for CAPACITYX smart logistics.\n"
                "Strands Decider 2B (System 1) has selected a primary logistics tool.\n"
                "Your role (System 2) is to review operational viability and provide specific "
                "risk mitigation notes and driver precautions.\n\n"
                "Respond ONLY with a valid JSON object matching this schema:\n"
                "{\n"
                '  "validated": <boolean>,\n'
                '  "operational_notes": ["note 1", "note 2"],\n'
                '  "recommended_precautions": ["precaution 1", "precaution 2"]\n'
                "}"
            )

            user_prompt = (
                f"Incident Context:\n{incident.to_state_text()}\n\n"
                f"Strands Decider 2B Decision:\n"
                f"- Selected Tool: {strands_outcome.selected_action}\n"
                f"- Action Type: {strands_outcome.action_type}\n"
                f"- Model Confidence: {strands_outcome.model_confidence or 'N/A (Safe Fallback)'}\n"
                f"- Urgency Tier: {strands_outcome.urgency}\n\n"
                "Provide complementary operational validation and precautions."
            )

            response = client.converse(
                user_text=user_prompt,
                system_text=system_prompt,
                max_tokens=300,
            )

            return self._parse_live_response(response)

        except Exception as exc:
            err_msg = str(exc)
            logger.warning("Bedrock operational reasoning unavailable: %s", err_msg)

            # Detect known AWS Free tier verification block vs standard errors
            is_blocked = (
                "operation not allowed" in err_msg.lower()
                or "validationexception" in err_msg.lower()
                or "accessdenied" in err_msg.lower()
            )
            status_label = "blocked_unverified" if is_blocked else "failed"

            return BedrockValidationResult(
                status=status_label,
                is_simulated=False,
                operational_validation=True,  # Preserve Strands decision as valid
                operational_notes=[
                    f"Live Bedrock reasoning unavailable ({err_msg}). Primary Strands decision preserved.",
                ],
                recommended_precautions=[
                    "Proceed with Strands Decider automated execution under standard monitoring.",
                ],
                model_id=self.model_id,
                error_reason=err_msg,
                disclaimer=(
                    "UNVERIFIED: AWS account access verification required for live Bedrock calls. "
                    "Decision evaluated solely with Strands Decider 2B."
                ),
            )

    def _parse_live_response(self, response: Dict[str, Any]) -> BedrockValidationResult:
        """Parse structured JSON from live Bedrock Converse API response."""
        try:
            output = response.get("output", {})
            message = output.get("message", {})
            content = message.get("content", [])
            raw_text = content[0]["text"].strip() if content else ""

            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

            data = json.loads(cleaned)
            notes = [str(n) for n in data.get("operational_notes", [])]
            precautions = [str(p) for p in data.get("recommended_precautions", [])]
            validated = bool(data.get("validated", True))

            return BedrockValidationResult(
                status="validated",
                is_simulated=False,
                operational_validation=validated,
                operational_notes=notes or ["Live Bedrock operational reasoning confirmed."],
                recommended_precautions=precautions,
                model_id=self.model_id,
                disclaimer="LIVE INFERENCE: Operational reasoning verified with Amazon Bedrock Nova Micro.",
            )
        except Exception as exc:
            return BedrockValidationResult(
                status="failed",
                is_simulated=False,
                operational_validation=True,
                operational_notes=[f"Failed to parse Bedrock response ({exc})."],
                recommended_precautions=[],
                model_id=self.model_id,
                error_reason=str(exc),
                disclaimer="Live Bedrock response parsing failed. Strands Decider decision preserved.",
            )


class CapacityXOrchestrator:
    """Agent Orchestrator coordinating Strands Decider 2B with Amazon Bedrock Nova Micro.

    Enforces the hybrid System 1 / System 2 workflow:
    - Strands Decider 2B selects the primary tool and provides calibrated confidence.
    - Bedrock Nova Micro validates operational viability and generates logistical precautions.
    - Preserves Strands decisions seamlessly if Bedrock is disabled, unverified, or offline.
    """

    provider_name = "strands_orchestrated"

    def __init__(
        self,
        decider: Optional[StrandsDecider2B] = None,
        bedrock_reasoner: Optional[BedrockOperationalReasoner] = None,
        enable_bedrock: bool = False,
        bedrock_mode: str = "disabled",
    ) -> None:
        self.decider = decider or StrandsDecider2B()

        # Read environment configuration if not explicitly provided
        if not enable_bedrock:
            enable_bedrock = os.environ.get(ENV_ENABLE_BEDROCK, "false").strip().lower() in {
                "true", "1", "yes"
            }

        if bedrock_mode == "disabled":
            bedrock_mode = os.environ.get(
                ENV_BEDROCK_MODE,
                "mock" if enable_bedrock else "disabled",
            ).strip().lower()

        self.enable_bedrock = enable_bedrock
        self.bedrock_enabled = enable_bedrock
        self.bedrock_mode = bedrock_mode

        self.bedrock_reasoner = bedrock_reasoner or BedrockOperationalReasoner(
            enabled=self.enable_bedrock,
            mode=self.bedrock_mode,
        )

    @property
    def server_url(self) -> str:
        """Expose Strands Decider server URL for backward compatibility."""
        return getattr(self.decider, "server_url", "http://127.0.0.1:8099")

    def get_health_status(self) -> Dict[str, Any]:
        """Aggregate health across Strands Decider 2B and Amazon Bedrock."""
        strands_health = self.decider.get_health_status()
        bedrock_health = self.bedrock_reasoner.get_health_status()

        is_strands_live = strands_health.get("is_real_model_available", False)

        return {
            "status": "online" if is_strands_live else strands_health.get("status", "healthy"),
            "provider": self.provider_name,
            "orchestration": {
                "decision_selection": "Strands Decider 2B (System 1)",
                "operational_reasoning": "Amazon Bedrock Nova Micro (System 2)",
                "bedrock_enabled": self.enable_bedrock,
                "bedrock_mode": self.bedrock_mode,
            },
            "strands_decider": strands_health,
            "amazon_bedrock": bedrock_health,
            "is_real_model_available": is_strands_live,
            "checkpoint": strands_health.get("checkpoint"),
            "server_url": self.server_url,
        }

    def evaluate(self, incident: IncidentContext) -> DeciderOutcome:
        """Orchestrate evaluation: Strands Decider selects tool, Bedrock reasons/validates."""
        # 1. Primary Tool Selection: Strands Decider 2B
        outcome = self.decider.evaluate(incident)

        # 2. Complementary Operational Reasoning: Bedrock Nova Micro
        validation_result = self.bedrock_reasoner.validate_and_reason(incident, outcome)

        # 3. Attach Bedrock validation into metadata
        outcome.metadata["bedrock_validation"] = validation_result.to_dict()

        # 4. Enrich rationale with Bedrock operational advisory (if active)
        if validation_result.status in {"simulated_mock", "validated"}:
            for note in validation_result.operational_notes:
                outcome.rationale.append(f"[Bedrock Reasoning] {note}")

            # If Bedrock operational audit flags an issue, require human confirmation
            if not validation_result.operational_validation:
                outcome.human_review_required = True
                bedrock_reason = "Bedrock operational safety check flagged logistical risk; human confirmation required."
                outcome.human_review_reason = (
                    f"{outcome.human_review_reason} | {bedrock_reason}"
                    if outcome.human_review_reason
                    else bedrock_reason
                )
                outcome.rationale.append(f"[Bedrock Safety Flag] {bedrock_reason}")
        elif validation_result.status == "blocked_unverified":
            outcome.rationale.append(
                f"[Bedrock Notice] Live inference blocked: {validation_result.error_reason}. Strands decision preserved."
            )

        return outcome

    def __repr__(self) -> str:
        return (
            f"<CapacityXOrchestrator decider={self.decider!r} "
            f"bedrock_enabled={self.enable_bedrock} mode='{self.bedrock_mode}'>"
        )
