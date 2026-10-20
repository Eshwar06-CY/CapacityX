"""Unit tests for the Strands Decider 2B Decision Pattern and Tool Selection.

Verifies:
1. Question and state formatting for the 5 typed tools.
2. Real model response parsing, confidence scores, and probability distribution.
3. Confidence thresholding and human review triggers.
4. Deterministic safe fallback behavior when the inference server is unreachable.
"""

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from patterns.strands_decider_2b import (
    ENV_DECIDER_PASSWORD,
    ENV_DECIDER_USERNAME,
    LOGISTICS_TOOLS,
    ConfigurationError,
    IncidentContext,
    ServerUnavailableError,
    StrandsDecider2B,
    StrandsDeciderClient,
)


class TestStrandsDecider2B(unittest.TestCase):
    """Test suite for Strands Decider 2B pattern."""

    def setUp(self):
        self.decider = StrandsDecider2B(
            server_url="http://127.0.0.1:8099",
            confidence_threshold=0.70,
        )

    def test_incident_state_text_generation(self):
        """Verify that IncidentContext produces a dense, structured state string."""
        incident = IncidentContext(
            shipment_id="SHP-TEST-100",
            delivery_deadline="2026-10-20T16:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Perishables",
            origin="Chicago Hub",
            destination="Milwaukee Retail",
            estimated_distance_remaining_km=45.0,
        )
        state_text = incident.to_state_text()
        self.assertIn("Shipment Identifier: SHP-TEST-100", state_text)
        self.assertIn("BREAKDOWN_IMMOBILIZED", state_text)
        self.assertIn("AVAILABLE_WITHIN_45MIN", state_text)
        self.assertIn("PENDING_ALERT", state_text)
        self.assertIn("Cold-Chain Perishables", state_text)

    def test_logistics_tools_definition(self):
        """Ensure all 5 required typed tools are declared with descriptions."""
        required_tools = {
            "dispatch_replacement",
            "reroute_shipment",
            "crossdock",
            "escalate_to_human",
            "continue_monitoring",
        }
        self.assertEqual(set(LOGISTICS_TOOLS.keys()), required_tools)
        for tool, description in LOGISTICS_TOOLS.items():
            self.assertTrue(len(description) > 10, f"Description for {tool} should be informative.")

    @patch.object(StrandsDeciderClient, "ask")
    def test_real_model_response_high_confidence(self, mock_ask):
        """Test parsing of a real Strands Decider response with confidence above threshold."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "dispatch_replacement",
                "confidence": 0.884,
                "probabilities": {
                    "dispatch_replacement": 0.884,
                    "crossdock": 0.052,
                    "reroute_shipment": 0.034,
                    "escalate_to_human": 0.021,
                    "continue_monitoring": 0.009,
                },
            },
            "is_urgent": {"type": "noul", "noul": 0.92},
            "human_review": {"type": "noul", "noul": 0.15},
            "urgency_tier": {"type": "choice", "choice": "HIGH"},
        }

        incident = IncidentContext(
            shipment_id="SHP-8921-COLD",
            delivery_deadline="2026-10-20T14:30:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Seafood",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
        self.assertTrue(outcome.is_real_model)
        self.assertFalse(outcome.is_simulated)
        self.assertEqual(outcome.selected_action, "dispatch_replacement")
        self.assertEqual(outcome.action_type, "DISPATCH_REPLACEMENT")
        self.assertAlmostEqual(outcome.model_confidence, 0.884)
        self.assertIn("dispatch_replacement", outcome.alternatives)
        self.assertFalse(outcome.human_review_required)
        self.assertIsNone(outcome.fallback_reason)

    @patch.object(StrandsDeciderClient, "ask")
    def test_real_model_response_low_confidence_triggers_human_review(self, mock_ask):
        """When model confidence is below 70%, human_review_required must be flagged True."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "reroute_shipment",
                "confidence": 0.54,  # Below 0.70 threshold
                "probabilities": {
                    "reroute_shipment": 0.54,
                    "crossdock": 0.32,
                    "dispatch_replacement": 0.10,
                    "escalate_to_human": 0.03,
                    "continue_monitoring": 0.01,
                },
            },
            "is_urgent": {"type": "noul", "noul": 0.65},
            "human_review": {"type": "noul", "noul": 0.40},
            "urgency_tier": {"type": "choice", "choice": "MODERATE"},
        }

        incident = IncidentContext(
            shipment_id="SHP-302-UNCERTAIN",
            delivery_deadline="2026-10-20T18:00:00Z",
            truck_breakdown=False,
            replacement_vehicle_available=True,
            customer_notified=True,
            cargo_type="Standard Freight",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
        self.assertTrue(outcome.is_real_model)
        self.assertEqual(outcome.selected_action, "reroute_shipment")
        self.assertTrue(outcome.human_review_required, "Low confidence must flag human review.")

    @patch.object(StrandsDeciderClient, "ask")
    def test_safe_fallback_when_inference_server_unreachable(self, mock_ask):
        """When server is offline, safe fallback must engage, clearly labeled as SAFE_FALLBACK."""
        mock_ask.side_effect = ServerUnavailableError("Connection refused on port 8099")

        incident = IncidentContext(
            shipment_id="SHP-FALLBACK-1",
            delivery_deadline="2026-10-20T14:30:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Produce",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.is_simulated)
        self.assertTrue(outcome.human_review_required)
        self.assertIsNotNone(outcome.fallback_reason)
        self.assertIn("Connection refused", outcome.fallback_reason)
        self.assertEqual(outcome.selected_action, "dispatch_replacement")
        self.assertIsNone(outcome.model_confidence)

    @patch.object(StrandsDeciderClient, "ask")
    def test_safe_fallback_no_backup_escalates_to_human(self, mock_ask):
        """Breakdown with no backup assets must escalate to human dispatcher in fallback mode."""
        mock_ask.side_effect = ServerUnavailableError("Server offline")

        incident = IncidentContext(
            shipment_id="SHP-FALLBACK-2",
            delivery_deadline="2026-10-20T17:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=False,
            customer_notified=False,
            cargo_type="Electronics",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.selected_action, "escalate_to_human")
        self.assertEqual(outcome.sla_risk_level, "CRITICAL_BREACH")
        self.assertFalse(outcome.deadline_met_projected)
        self.assertTrue(outcome.human_review_required)

    @patch.object(StrandsDeciderClient, "ask")
    def test_safe_fallback_nominal_transit(self, mock_ask):
        """Nominal transit with no breakdown must continue monitoring in fallback mode."""
        mock_ask.side_effect = ServerUnavailableError("Server offline")

        incident = IncidentContext(
            shipment_id="SHP-FALLBACK-3",
            delivery_deadline="2026-10-20T12:00:00Z",
            truck_breakdown=False,
            replacement_vehicle_available=True,
            customer_notified=True,
            cargo_type="Standard Goods",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.selected_action, "continue_monitoring")
        self.assertEqual(outcome.urgency, "LOW")
        self.assertTrue(outcome.deadline_met_projected)


class TestStrandsDeciderAuthentication(unittest.TestCase):
    """Test suite for HTTP Basic Authentication and ngrok tunnel compatibility."""

    @patch("urllib.request.urlopen")
    def test_authenticated_client_headers_on_health_and_ask(self, mock_urlopen):
        """Verify that HTTP Basic Auth and ngrok compatibility headers are sent on both /health and /v1/systemone."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "checkpoint": "model-2b",
            "answers": {"action": {"type": "choice", "choice": "dispatch_replacement"}},
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        client = StrandsDeciderClient(
            url="https://demo-tunnel.ngrok-free.app",
            username="ngrok_user",
            password="secret_pass_123",
        )

        self.assertTrue(client.is_authenticated)

        # 1. Test GET /health
        client.health()
        self.assertEqual(mock_urlopen.call_count, 1)
        req_health = mock_urlopen.call_args[0][0]
        self.assertEqual(req_health.full_url, "https://demo-tunnel.ngrok-free.app/health")
        expected_auth = "Basic " + base64.b64encode(b"ngrok_user:secret_pass_123").decode("ascii")
        self.assertEqual(req_health.get_header("Authorization"), expected_auth)
        self.assertEqual(req_health.get_header("Ngrok-skip-browser-warning"), "true")
        self.assertNotIn("secret_pass_123", str(req_health.headers))

        # 2. Test POST /v1/systemone
        client.ask(state="Incident State", questions={"action": {"type": "choice"}})
        self.assertEqual(mock_urlopen.call_count, 2)
        req_ask = mock_urlopen.call_args[0][0]
        self.assertEqual(req_ask.full_url, "https://demo-tunnel.ngrok-free.app/v1/systemone")
        self.assertEqual(req_ask.get_header("Authorization"), expected_auth)
        self.assertEqual(req_ask.get_header("Content-type"), "application/json")
        self.assertEqual(req_ask.get_header("Ngrok-skip-browser-warning"), "true")
        self.assertNotIn("secret_pass_123", str(req_ask.headers))

    @patch.dict(os.environ, {}, clear=True)
    @patch("urllib.request.urlopen")
    def test_unauthenticated_client_local_requests(self, mock_urlopen):
        """When credentials are not supplied, no Authorization header should be attached."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "checkpoint": "model-2b",
            "answers": {"action": {"type": "choice", "choice": "dispatch_replacement"}},
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        client = StrandsDeciderClient(url="http://127.0.0.1:8099")
        self.assertFalse(client.is_authenticated)

        # GET /health
        client.health()
        req_health = mock_urlopen.call_args[0][0]
        self.assertIsNone(req_health.get_header("Authorization"))

        # POST /v1/systemone
        client.ask(state="Incident State", questions={"action": {"type": "choice"}})
        req_ask = mock_urlopen.call_args[0][0]
        self.assertIsNone(req_ask.get_header("Authorization"))

    @patch.dict(os.environ, {}, clear=True)
    def test_incomplete_credentials_raises_configuration_error(self):
        """Supplying only username or only password must raise ConfigurationError with a clean message."""
        # Username only via kwargs
        with self.assertRaises(ConfigurationError) as ctx1:
            StrandsDeciderClient(username="testuser", password=None)
        self.assertIn("missing STRANDS_DECIDER_PASSWORD", str(ctx1.exception))
        self.assertNotIn("testuser", str(ctx1.exception))

        # Password only via kwargs
        with self.assertRaises(ConfigurationError) as ctx2:
            StrandsDeciderClient(username=None, password="secretpassword")
        self.assertIn("missing STRANDS_DECIDER_USERNAME", str(ctx2.exception))
        self.assertNotIn("secretpassword", str(ctx2.exception))

    @patch("urllib.request.urlopen")
    def test_environment_variables_handling(self, mock_urlopen):
        """Test picking up credentials from environment variables."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"checkpoint": "model-2b"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        # Both env vars set
        with patch.dict(os.environ, {
            ENV_DECIDER_USERNAME: "env_admin",
            ENV_DECIDER_PASSWORD: "env_secret_password",
        }):
            client = StrandsDeciderClient()
            self.assertTrue(client.is_authenticated)
            client.health()
            req = mock_urlopen.call_args[0][0]
            expected_auth = "Basic " + base64.b64encode(b"env_admin:env_secret_password").decode("ascii")
            self.assertEqual(req.get_header("Authorization"), expected_auth)

        # Only username set in env
        with patch.dict(os.environ, {ENV_DECIDER_USERNAME: "env_admin"}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                StrandsDeciderClient()
            self.assertIn("missing STRANDS_DECIDER_PASSWORD", str(ctx.exception))

        # Only password set in env
        with patch.dict(os.environ, {ENV_DECIDER_PASSWORD: "env_secret_password"}, clear=True):
            with self.assertRaises(ConfigurationError) as ctx:
                StrandsDeciderClient()
            self.assertIn("missing STRANDS_DECIDER_USERNAME", str(ctx.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_strands_decider_2b_fails_safely_on_incomplete_credentials(self):
        """StrandsDecider2B should not crash if credentials are incomplete; it must fall back safely."""
        decider = StrandsDecider2B(username="partial_user", password=None)

        # Health check reflects offline configuration error
        status = decider.get_health_status()
        self.assertEqual(status["status"], "offline")
        self.assertFalse(status["is_real_model_available"])
        self.assertIn("missing STRANDS_DECIDER_PASSWORD", status["reason"])

        # Decision evaluation executes safe fallback safely
        incident = IncidentContext(
            shipment_id="SHP-CONFIG-ERR",
            delivery_deadline="2026-10-20T16:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
        )
        outcome = decider.evaluate(incident)
        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIn("missing STRANDS_DECIDER_PASSWORD", outcome.fallback_reason)

    def test_repr_never_leaks_secrets(self):
        """repr and str must never expose passwords or base64 tokens."""
        client = StrandsDeciderClient(
            url="https://demo-tunnel.ngrok-free.app",
            username="secret_agent",
            password="super_sensitive_token_999",
        )
        repr_str = repr(client)
        str_str = str(client)

        self.assertNotIn("super_sensitive_token_999", repr_str)
        self.assertNotIn("super_sensitive_token_999", str_str)
        token = base64.b64encode(b"secret_agent:super_sensitive_token_999").decode("ascii")
        self.assertNotIn(token, repr_str)
        self.assertNotIn(token, str_str)
        self.assertIn("authenticated", repr_str)


if __name__ == "__main__":
    unittest.main()
