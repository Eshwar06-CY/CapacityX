"""Offline Unit Tests for Amazon Bedrock Nova Micro Provider.

Verifies:
1. Valid Converse API response parsing and schema conformance.
2. Rejection of invalid/unrecognized action choices triggering safe fallback.
3. Malformed or non-JSON model output triggering safe fallback.
4. AWS SDK permission errors (e.g. AccessDeniedException) handled safely.
5. Finite timeout handling without application-level retries.
6. Safeguards: uncalibrated probabilities (model_confidence=None, alternatives={}).
7. Credential privacy and honest health check reporting.

NOTE: All tests are strictly offline using unittest.mock. No real AWS network calls
are made or permitted.
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

from patterns.bedrock_nova_micro import (
    BedrockNovaMicroAgent,
    BedrockNovaMicroClient,
    build_nova_system_prompt,
    build_nova_user_message,
)
from patterns.strands_decider_2b import IncidentContext


class TestBedrockNovaMicroOffline(unittest.TestCase):
    """Test suite for Bedrock Nova Micro provider using mocked SDK responses."""

    def setUp(self):
        self.mock_client = MagicMock(spec=BedrockNovaMicroClient)
        self.mock_client.model_id = "amazon.nova-micro-v1:0"
        self.mock_client.region_name = "us-east-1"
        self.agent = BedrockNovaMicroAgent(
            model_id="amazon.nova-micro-v1:0",
            region_name="us-east-1",
            client=self.mock_client,
        )

        self.incident_breakdown_ready = IncidentContext(
            shipment_id="SHP-TEST-BEDROCK-1",
            delivery_deadline="2026-10-20T16:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Perishables",
            origin="Chicago Depot",
            destination="Milwaukee Hub",
            estimated_distance_remaining_km=40.0,
        )

        self.incident_breakdown_no_backup = IncidentContext(
            shipment_id="SHP-TEST-BEDROCK-2",
            delivery_deadline="2026-10-20T18:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=False,
            customer_notified=False,
            cargo_type="Dry Freight",
            origin="Indianapolis",
            destination="Peoria",
            estimated_distance_remaining_km=90.0,
        )

        self.incident_nominal = IncidentContext(
            shipment_id="SHP-TEST-BEDROCK-3",
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

    def _make_mock_converse_response(self, text_content: str, stop_reason: str = "end_turn") -> dict:
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": text_content}],
                }
            },
            "stopReason": stop_reason,
            "usage": {"inputTokens": 140, "outputTokens": 45, "totalTokens": 185},
        }

    def test_valid_output_dispatch_replacement(self):
        """Test valid structured Nova Micro output selecting dispatch_replacement."""
        model_payload = {
            "selected_action": "dispatch_replacement",
            "urgency": "HIGH",
            "headline": "Dispatching backup truck to rendezvous and transfer cold-chain cargo.",
            "rationale": [
                "Primary vehicle is immobilized with critical severity.",
                "Backup vehicle is available within 45 minutes.",
                "Cold-chain goods require immediate transfer to prevent spoilage.",
            ],
            "estimated_delay_minutes": 40,
            "human_review_required": False,
        }
        self.mock_client.converse.return_value = self._make_mock_converse_response(
            json.dumps(model_payload)
        )

        outcome = self.agent.evaluate(self.incident_breakdown_ready)

        self.assertEqual(outcome.engine, "AMAZON_BEDROCK_NOVA_MICRO")
        self.assertTrue(outcome.is_real_model)
        self.assertFalse(outcome.is_simulated)
        self.assertEqual(outcome.selected_action, "dispatch_replacement")
        self.assertEqual(outcome.action_type, "DISPATCH_REPLACEMENT")
        self.assertEqual(outcome.urgency, "HIGH")
        self.assertIsNone(outcome.fallback_reason)
        self.assertEqual(outcome.model_id, "amazon.nova-micro-v1:0")

        # Safeguard verification: uncalibrated probabilities must remain None/empty
        self.assertIsNone(outcome.model_confidence)
        self.assertEqual(outcome.alternatives, {})
        self.assertFalse(outcome.human_review_required)
        self.assertEqual(outcome.estimated_delay_minutes, 40)
        self.assertTrue(outcome.deadline_met_projected)

    def test_valid_output_wrapped_in_markdown_codeblock(self):
        """Test parser handles Nova responses enclosed in ```json ... ``` markdown blocks."""
        model_payload = {
            "selected_action": "reroute_shipment",
            "urgency": "MODERATE",
            "headline": "Reroute shipment around highway congestion.",
            "rationale": ["Vehicle operational but facing heavy traffic."],
            "estimated_delay_minutes": 25,
            "human_review_required": False,
        }
        raw_text = f"```json\n{json.dumps(model_payload)}\n```"
        self.mock_client.converse.return_value = self._make_mock_converse_response(raw_text)

        outcome = self.agent.evaluate(self.incident_nominal)

        self.assertEqual(outcome.engine, "AMAZON_BEDROCK_NOVA_MICRO")
        self.assertEqual(outcome.selected_action, "reroute_shipment")
        self.assertFalse(outcome.human_review_required)

    def test_invalid_action_triggers_safe_fallback(self):
        """When Nova selects an action outside the 5 allowed tools, safe fallback must engage."""
        model_payload = {
            "selected_action": "order_private_helicopter",  # Invalid action
            "urgency": "CRITICAL",
            "headline": "Airlift the cargo.",
            "rationale": ["Immediate delivery needed."],
            "estimated_delay_minutes": 10,
        }
        self.mock_client.converse.return_value = self._make_mock_converse_response(
            json.dumps(model_payload)
        )

        outcome = self.agent.evaluate(self.incident_breakdown_ready)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIsNotNone(outcome.fallback_reason)
        self.assertIn("unrecognized action 'order_private_helicopter'", outcome.fallback_reason)
        # Safe rule assigned dispatch_replacement because backup vehicle is available
        self.assertEqual(outcome.selected_action, "dispatch_replacement")

    def test_malformed_response_triggers_safe_fallback(self):
        """When Nova outputs non-JSON plain text, safe fallback must engage."""
        self.mock_client.converse.return_value = self._make_mock_converse_response(
            "I apologize, but as an AI I am unable to make this decision."
        )

        outcome = self.agent.evaluate(self.incident_breakdown_ready)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIn("malformed/unparseable JSON", outcome.fallback_reason)

    def test_permission_error_triggers_safe_fallback(self):
        """Simulate AWS AccessDeniedException; agent must fail safely without crashing."""
        error_response = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "User is not authorized to perform: bedrock:InvokeModel on resource",
            }
        }
        try:
            from botocore.exceptions import ClientError
            exc = ClientError(error_response, "Converse")
        except ImportError:
            exc = RuntimeError("AccessDeniedException: User is not authorized")

        self.mock_client.converse.side_effect = exc

        outcome = self.agent.evaluate(self.incident_breakdown_ready)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIn("AccessDeniedException", outcome.fallback_reason)

    def test_timeout_triggers_safe_fallback(self):
        """Simulate request timeout; agent must engage safe fallback with clear notice."""
        try:
            from botocore.exceptions import ReadTimeoutError
            exc = ReadTimeoutError(endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com")
        except ImportError:
            exc = TimeoutError("Request timed out")

        self.mock_client.converse.side_effect = exc

        outcome = self.agent.evaluate(self.incident_breakdown_ready)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIn("timed out", outcome.fallback_reason.lower())

    def test_safe_fallback_no_backup_escalates_to_human(self):
        """In fallback mode, breakdown with no backup assets must escalate to human dispatcher."""
        self.mock_client.converse.side_effect = RuntimeError("Offline mock test error")

        outcome = self.agent.evaluate(self.incident_breakdown_no_backup)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertEqual(outcome.selected_action, "escalate_to_human")
        self.assertEqual(outcome.sla_risk_level, "CRITICAL_BREACH")
        self.assertFalse(outcome.deadline_met_projected)
        self.assertTrue(outcome.human_review_required)

    def test_safe_fallback_nominal_transit(self):
        """In fallback mode, nominal transit must continue monitoring."""
        self.mock_client.converse.side_effect = RuntimeError("Offline mock test error")

        outcome = self.agent.evaluate(self.incident_nominal)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertEqual(outcome.selected_action, "continue_monitoring")
        self.assertEqual(outcome.urgency, "LOW")
        self.assertTrue(outcome.deadline_met_projected)

    def test_boto_converse_call_parameters(self):
        """Verify client constructs standard Converse API arguments with zero retries."""
        raw_boto_client = MagicMock()
        raw_boto_client.converse.return_value = self._make_mock_converse_response(
            json.dumps({
                "selected_action": "continue_monitoring",
                "urgency": "LOW",
                "headline": "Nominal tracking.",
                "rationale": ["Route is clear."],
                "estimated_delay_minutes": 0,
            })
        )

        client = BedrockNovaMicroClient(
            model_id="amazon.nova-micro-v1:0",
            region_name="us-east-1",
            client=raw_boto_client,
        )
        agent = BedrockNovaMicroAgent(client=client)
        agent.evaluate(self.incident_nominal)

        self.assertEqual(raw_boto_client.converse.call_count, 1)
        call_kwargs = raw_boto_client.converse.call_args[1]
        self.assertEqual(call_kwargs["modelId"], "amazon.nova-micro-v1:0")
        self.assertEqual(call_kwargs["inferenceConfig"]["maxTokens"], 300)
        self.assertEqual(call_kwargs["inferenceConfig"]["temperature"], 0.0)
        self.assertIn("system", call_kwargs)
        self.assertIn("dispatch_replacement", call_kwargs["system"][0]["text"])

    def test_honest_health_status_disclosure(self):
        """Health status must honestly declare unverified/offline status until live verification."""
        health = self.agent.get_health_status()
        self.assertFalse(health["is_real_model_available"])
        self.assertFalse(health.get("live_inference_tested", False))
        self.assertEqual(health["provider"], "bedrock_nova_micro")
        self.assertEqual(health["model_id"], "amazon.nova-micro-v1:0")

    def test_sanitized_repr_never_leaks_secrets(self):
        """String representation must never expose secrets or credentials."""
        real_client = BedrockNovaMicroClient(
            model_id="amazon.nova-micro-v1:0",
            region_name="us-east-1",
            client=MagicMock(),
        )
        agent_repr = repr(self.agent)
        client_repr = repr(real_client)
        for s in [agent_repr, client_repr]:
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", s)
            self.assertNotIn("session_token", s)
            self.assertIn("amazon.nova-micro-v1:0", s)

    def test_prompt_builders_contain_all_5_tools(self):
        """Ensure system prompt specifies all 5 tools and user message includes state."""
        sys_prompt = build_nova_system_prompt()
        required_tools = [
            "dispatch_replacement",
            "reroute_shipment",
            "crossdock",
            "escalate_to_human",
            "continue_monitoring",
        ]
        for tool in required_tools:
            self.assertIn(tool, sys_prompt)

        user_msg = build_nova_user_message(self.incident_breakdown_ready)
        self.assertIn("SHP-TEST-BEDROCK-1", user_msg)
        self.assertIn("BREAKDOWN_IMMOBILIZED", user_msg)


class TestProviderSelection(unittest.TestCase):
    """Test environment variable provider and orchestration configuration logic in backend."""

    def test_default_provider_is_strands(self):
        """When default environment is used, CapacityXOrchestrator is instantiated with Strands primary."""
        with patch.dict(os.environ, {}, clear=True):
            from backend.app.main import get_configured_agent
            from patterns.orchestrator import CapacityXOrchestrator
            from patterns.strands_decider_2b import StrandsDecider2B
            agent = get_configured_agent()
            self.assertIsInstance(agent, CapacityXOrchestrator)
            self.assertIsInstance(agent.decider, StrandsDecider2B)
            self.assertFalse(agent.bedrock_enabled)

    def test_explicit_bedrock_provider_selection(self):
        """When Bedrock is enabled via environment variable, orchestrator activates Bedrock component."""
        with patch.dict(os.environ, {"CAPACITYX_ENABLE_BEDROCK": "true", "CAPACITYX_BEDROCK_MODE": "mock"}):
            from backend.app.main import get_configured_agent
            from patterns.orchestrator import CapacityXOrchestrator
            agent = get_configured_agent()
            self.assertIsInstance(agent, CapacityXOrchestrator)
            self.assertTrue(agent.bedrock_enabled)
            self.assertEqual(agent.bedrock_mode, "mock")


if __name__ == "__main__":
    unittest.main()
