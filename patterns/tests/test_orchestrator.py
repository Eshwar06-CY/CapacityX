"""Unit Tests for CapacityX Decision Orchestration Layer.

Verifies:
1. Default configuration: Strands Decider 2B is primary; Bedrock reasoning is disabled.
2. Mock Bedrock operational reasoning successfully enriches Strands decision when enabled.
3. Bedrock operational safety flags enforce human review confirmation.
4. AWS account restriction (ValidationException: Operation not allowed) is handled
   gracefully without failing or altering the Strands decision.
5. Bedrock request timeout or malformed JSON handles safely with transparent fallback.
6. Strands safe fallback behavior is strictly preserved when model server is offline.
7. Only the 5 canonical logistics tools are allowed in any outcome.
8. Zero fabricated confidence: Bedrock never generates or alters model_confidence.
9. Health status aggregation discloses Strands availability and Bedrock mode honestly.

NOTE: All tests are strictly offline using unittest.mock. No real AWS calls occur.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root and backend are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from patterns.bedrock_nova_micro import BedrockNovaMicroClient
from patterns.orchestrator import (
    BedrockOperationalReasoner,
    BedrockValidationResult,
    CapacityXOrchestrator,
)
from patterns.strands_decider_2b import (
    LOGISTICS_TOOLS,
    DeciderOutcome,
    IncidentContext,
    ServerUnavailableError,
    StrandsDecider2B,
    StrandsDeciderClient,
)


class TestCapacityXOrchestrator(unittest.TestCase):
    """Test suite for CapacityX hybrid orchestration layer."""

    def setUp(self):
        self.incident_breakdown_ready = IncidentContext(
            shipment_id="SHP-ORCH-100",
            delivery_deadline="2026-10-20T16:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Perishables (Seafood)",
            origin="Chicago Depot",
            destination="Milwaukee Retail",
            estimated_distance_remaining_km=45.0,
        )

        self.incident_breakdown_no_backup = IncidentContext(
            shipment_id="SHP-ORCH-200",
            delivery_deadline="2026-10-20T18:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=False,
            customer_notified=False,
            cargo_type="Dry Freight",
            origin="Indianapolis",
            destination="Peoria",
            estimated_distance_remaining_km=110.0,
        )

        self.incident_nominal = IncidentContext(
            shipment_id="SHP-ORCH-300",
            delivery_deadline="2026-10-20T12:00:00Z",
            truck_breakdown=False,
            breakdown_severity="NONE",
            replacement_vehicle_available=True,
            customer_notified=True,
            cargo_type="Standard Goods",
            origin="Madison",
            destination="Rockford",
            estimated_distance_remaining_km=30.0,
        )

    def test_default_configuration_strands_primary_bedrock_disabled(self):
        """By default, Strands Decider runs and Bedrock reasoning is completely disabled."""
        orchestrator = CapacityXOrchestrator(
            enable_bedrock=False,
            bedrock_mode="disabled",
        )
        self.assertFalse(orchestrator.enable_bedrock)
        self.assertEqual(orchestrator.bedrock_mode, "disabled")

        # Mock Strands client to simulate a real model answer
        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.return_value = {
                "action": {
                    "type": "choice",
                    "choice": "dispatch_replacement",
                    "confidence": 0.89,
                    "probabilities": {"dispatch_replacement": 0.89, "crossdock": 0.08},
                },
                "is_urgent": {"type": "noul", "noul": 0.90},
                "human_review": {"type": "noul", "noul": 0.12},
                "urgency_tier": {"type": "choice", "choice": "HIGH"},
            }

            outcome = orchestrator.evaluate(self.incident_breakdown_ready)

            # Strands decision is returned as primary
            self.assertEqual(outcome.selected_action, "dispatch_replacement")
            self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
            self.assertTrue(outcome.is_real_model)
            self.assertAlmostEqual(outcome.model_confidence, 0.89)

            # Bedrock validation status is recorded as disabled
            bedrock_meta = outcome.metadata.get("bedrock_validation", {})
            self.assertEqual(bedrock_meta.get("status"), "disabled")
            self.assertTrue(bedrock_meta.get("is_simulated"))
            self.assertIn("disabled by default", bedrock_meta.get("disclaimer", ""))

    def test_orchestration_with_mocked_bedrock_validation_success(self):
        """When Bedrock mock mode is enabled, it provides simulated operational reasoning."""
        orchestrator = CapacityXOrchestrator(
            enable_bedrock=True,
            bedrock_mode="mock",
        )

        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.return_value = {
                "action": {
                    "type": "choice",
                    "choice": "dispatch_replacement",
                    "confidence": 0.85,
                    "probabilities": {"dispatch_replacement": 0.85, "reroute_shipment": 0.10},
                },
                "is_urgent": {"type": "noul", "noul": 0.80},
                "human_review": {"type": "noul", "noul": 0.10},
                "urgency_tier": {"type": "choice", "choice": "HIGH"},
            }

            outcome = orchestrator.evaluate(self.incident_breakdown_ready)

            # Strands primary selection preserved
            self.assertEqual(outcome.selected_action, "dispatch_replacement")
            self.assertAlmostEqual(outcome.model_confidence, 0.85)

            # Bedrock operational reasoning attached
            bedrock_meta = outcome.metadata.get("bedrock_validation", {})
            self.assertEqual(bedrock_meta.get("status"), "simulated_mock")
            self.assertTrue(bedrock_meta.get("is_simulated"))
            self.assertTrue(bedrock_meta.get("operational_validation"))
            self.assertTrue(len(bedrock_meta.get("operational_notes", [])) > 0)
            self.assertIn("SIMULATED", bedrock_meta.get("disclaimer", ""))

            # Rationale contains Bedrock reasoning bullet
            has_bedrock_note = any("[Bedrock Reasoning]" in r for r in outcome.rationale)
            self.assertTrue(has_bedrock_note)

    def test_bedrock_flags_operational_safety_risk_enforces_human_review(self):
        """If Bedrock operational check flags a safety issue, human_review_required is set True."""
        mock_reasoner = MagicMock(spec=BedrockOperationalReasoner)
        mock_reasoner.validate_and_reason.return_value = BedrockValidationResult(
            status="simulated_mock",
            is_simulated=True,
            operational_validation=False,  # Fails operational validation
            operational_notes=["Hazardous road closure reported on rendezvous route."],
            recommended_precautions=["Hold automated dispatch until human confirms bypass."],
            model_id="amazon.nova-micro-v1:0 (offline mock)",
            disclaimer="SIMULATED: Mock Bedrock operational reasoning.",
        )

        orchestrator = CapacityXOrchestrator(
            bedrock_reasoner=mock_reasoner,
            enable_bedrock=True,
            bedrock_mode="mock",
        )

        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.return_value = {
                "action": {
                    "type": "choice",
                    "choice": "dispatch_replacement",
                    "confidence": 0.95,  # High confidence
                    "probabilities": {"dispatch_replacement": 0.95},
                },
                "is_urgent": {"type": "noul", "noul": 0.80},
                "human_review": {"type": "noul", "noul": 0.05},  # Strands says low review needed
                "urgency_tier": {"type": "choice", "choice": "HIGH"},
            }

            outcome = orchestrator.evaluate(self.incident_breakdown_ready)

            # Even though Strands confidence was 0.95, Bedrock safety flag forces human review
            self.assertTrue(outcome.human_review_required)
            has_safety_flag = any("[Bedrock Safety Flag]" in r for r in outcome.rationale)
            self.assertTrue(has_safety_flag)

    def test_aws_validation_exception_operation_not_allowed_handled_safely(self):
        """Simulate AWS Free plan ValidationException (Operation not allowed).

        Orchestrator must handle the block gracefully, log the block reason, and
        preserve the primary Strands Decider decision without crashing.
        """
        raw_boto_client = MagicMock()
        try:
            from botocore.exceptions import ClientError
            exc = ClientError(
                {"Error": {"Code": "ValidationException", "Message": "Operation not allowed"}},
                "Converse",
            )
        except ImportError:
            exc = RuntimeError("ValidationException: Operation not allowed")

        raw_boto_client.converse.side_effect = exc

        live_reasoner = BedrockOperationalReasoner(
            enabled=True,
            mode="live",
            client=raw_boto_client,
        )

        orchestrator = CapacityXOrchestrator(
            bedrock_reasoner=live_reasoner,
            enable_bedrock=True,
            bedrock_mode="live",
        )

        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.return_value = {
                "action": {
                    "type": "choice",
                    "choice": "dispatch_replacement",
                    "confidence": 0.88,
                    "probabilities": {"dispatch_replacement": 0.88},
                },
                "is_urgent": {"type": "noul", "noul": 0.85},
                "human_review": {"type": "noul", "noul": 0.15},
                "urgency_tier": {"type": "choice", "choice": "HIGH"},
            }

            outcome = orchestrator.evaluate(self.incident_breakdown_ready)

            # Strands decision is preserved
            self.assertEqual(outcome.selected_action, "dispatch_replacement")
            self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")

            # Bedrock status transparently reflects blocked_unverified
            bedrock_meta = outcome.metadata.get("bedrock_validation", {})
            self.assertEqual(bedrock_meta.get("status"), "blocked_unverified")
            self.assertFalse(bedrock_meta.get("is_simulated"))
            self.assertIn("Operation not allowed", bedrock_meta.get("error_reason", ""))
            self.assertIn("UNVERIFIED", bedrock_meta.get("disclaimer", ""))

    def test_bedrock_timeout_handled_safely(self):
        """Simulate Bedrock timeout: Strands decision is preserved safely."""
        raw_boto_client = MagicMock()
        try:
            from botocore.exceptions import ReadTimeoutError
            exc = ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
        except ImportError:
            exc = TimeoutError("Request timed out")

        raw_boto_client.converse.side_effect = exc

        live_reasoner = BedrockOperationalReasoner(
            enabled=True,
            mode="live",
            client=raw_boto_client,
        )

        orchestrator = CapacityXOrchestrator(
            bedrock_reasoner=live_reasoner,
            enable_bedrock=True,
            bedrock_mode="live",
        )

        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.return_value = {
                "action": {
                    "type": "choice",
                    "choice": "reroute_shipment",
                    "confidence": 0.75,
                    "probabilities": {"reroute_shipment": 0.75},
                },
                "is_urgent": {"type": "noul", "noul": 0.50},
                "human_review": {"type": "noul", "noul": 0.20},
                "urgency_tier": {"type": "choice", "choice": "MODERATE"},
            }

            outcome = orchestrator.evaluate(self.incident_nominal)

            self.assertEqual(outcome.selected_action, "reroute_shipment")
            bedrock_meta = outcome.metadata.get("bedrock_validation", {})
            self.assertEqual(bedrock_meta.get("status"), "failed")

    def test_strands_offline_safe_fallback_preserved_with_orchestration(self):
        """When Strands server is offline, safe fallback engages deterministically."""
        orchestrator = CapacityXOrchestrator(
            enable_bedrock=True,
            bedrock_mode="mock",
        )

        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.side_effect = ServerUnavailableError("Connection refused on port 8099")

            outcome = orchestrator.evaluate(self.incident_breakdown_no_backup)

            self.assertEqual(outcome.engine, "SAFE_FALLBACK")
            self.assertFalse(outcome.is_real_model)
            self.assertEqual(outcome.selected_action, "escalate_to_human")
            self.assertTrue(outcome.human_review_required)
            self.assertFalse(outcome.deadline_met_projected)

    def test_orchestrator_strictly_retains_5_allowed_actions(self):
        """All orchestration flows must select from the 5 canonical logistics actions."""
        orchestrator = CapacityXOrchestrator(
            enable_bedrock=True,
            bedrock_mode="mock",
        )

        for incident in [
            self.incident_breakdown_ready,
            self.incident_breakdown_no_backup,
            self.incident_nominal,
        ]:
            outcome = orchestrator.evaluate(incident)
            self.assertIn(outcome.selected_action, set(LOGISTICS_TOOLS.keys()))

    def test_zero_confidence_fabrication(self):
        """Bedrock must never fabricate or overwrite Strands model_confidence."""
        orchestrator = CapacityXOrchestrator(
            enable_bedrock=True,
            bedrock_mode="mock",
        )

        with patch.object(StrandsDeciderClient, "ask") as mock_ask:
            mock_ask.return_value = {
                "action": {
                    "type": "choice",
                    "choice": "crossdock",
                    "confidence": 0.777,
                    "probabilities": {"crossdock": 0.777, "dispatch_replacement": 0.15},
                },
                "is_urgent": {"type": "noul", "noul": 0.70},
                "human_review": {"type": "noul", "noul": 0.10},
                "urgency_tier": {"type": "choice", "choice": "MODERATE"},
            }

            outcome = orchestrator.evaluate(self.incident_breakdown_ready)

            # Confidence is preserved exactly from Strands
            self.assertAlmostEqual(outcome.model_confidence, 0.777)

    def test_health_status_aggregation_and_disclosure(self):
        """Health check aggregate must disclose Strands health and Bedrock mode honestly."""
        orchestrator = CapacityXOrchestrator(
            enable_bedrock=False,
            bedrock_mode="disabled",
        )
        health = orchestrator.get_health_status()

        self.assertIn("strands_decider", health)
        self.assertIn("amazon_bedrock", health)
        self.assertIn("orchestration", health)
        self.assertEqual(health["amazon_bedrock"]["status"], "disabled")


if __name__ == "__main__":
    unittest.main()
