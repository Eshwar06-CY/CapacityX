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
    ENV_DECIDER_API_KEY,
    ENV_DECIDER_MODE,
    ENV_DECIDER_PASSWORD,
    ENV_DECIDER_URL,
    ENV_DECIDER_USERNAME,
    LOGISTICS_TOOLS,
    UGMDU_DECIDER_URL,
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
        self.assertIsNone(outcome.human_review_reason)
        self.assertIsNone(outcome.fallback_reason)
        self.assertTrue(any("SATISFIED" in r for r in outcome.rationale))
        self.assertTrue(any("CLEAR" in r for r in outcome.rationale))

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
        self.assertIsNotNone(outcome.human_review_reason)
        self.assertIn("54.0%", outcome.human_review_reason)
        self.assertTrue(any("BELOW THRESHOLD" in r for r in outcome.rationale))

    @patch.object(StrandsDeciderClient, "ask")
    def test_real_model_exact_confidence_threshold_allows_automated_action(self, mock_ask):
        """When action confidence is exactly 0.70 and human_review is low (0.10), automated action is allowed."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "dispatch_replacement",
                "confidence": 0.70,
                "probabilities": {
                    "dispatch_replacement": 0.70,
                    "crossdock": 0.15,
                    "reroute_shipment": 0.10,
                    "escalate_to_human": 0.04,
                    "continue_monitoring": 0.01,
                },
            },
            "is_urgent": {"type": "noul", "noul": 0.60},
            "human_review": {"type": "noul", "noul": 0.10},
            "urgency_tier": {"type": "choice", "choice": "HIGH"},
        }

        incident = IncidentContext(
            shipment_id="SHP-THRESHOLD-EXACT",
            delivery_deadline="2026-10-20T15:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Dairy",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
        self.assertTrue(outcome.is_real_model)
        self.assertEqual(outcome.selected_action, "dispatch_replacement")
        self.assertAlmostEqual(outcome.model_confidence, 0.70)
        self.assertFalse(
            outcome.human_review_required,
            "Exact confidence threshold of 0.70 with human_review score < 0.5 must not require human review.",
        )
        self.assertIsNone(outcome.human_review_reason)
        self.assertTrue(any("SATISFIED (70.0% >= 70%)" in r for r in outcome.rationale))

    @patch.object(StrandsDeciderClient, "ask")
    def test_real_model_human_review_signal_independently_triggers_review(self, mock_ask):
        """Human-review signal (noul >= 0.5) must independently trigger human review even with high confidence."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "dispatch_replacement",
                "confidence": 0.90,
                "probabilities": {
                    "dispatch_replacement": 0.90,
                    "crossdock": 0.05,
                    "reroute_shipment": 0.03,
                    "escalate_to_human": 0.01,
                    "continue_monitoring": 0.01,
                },
            },
            "is_urgent": {"type": "noul", "noul": 0.85},
            "human_review": {"type": "noul", "noul": 0.80},
            "urgency_tier": {"type": "choice", "choice": "HIGH"},
        }

        incident = IncidentContext(
            shipment_id="SHP-HIGH-CONF-REVIEW",
            delivery_deadline="2026-10-20T14:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Pharmaceuticals",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
        self.assertTrue(outcome.is_real_model)
        self.assertEqual(outcome.selected_action, "dispatch_replacement")
        self.assertAlmostEqual(outcome.model_confidence, 0.90)
        self.assertTrue(
            outcome.human_review_required,
            "High human_review noul (0.80 >= 0.5) must independently trigger review even when confidence is 0.90.",
        )
        self.assertIsNotNone(outcome.human_review_reason)
        self.assertIn("P(review) = 0.80", outcome.human_review_reason)
        # Verify threshold is SATISFIED and NOT falsely labeled as LOW/FLAGGED
        self.assertTrue(
            any("SATISFIED (90.0% >= 70%)" in r for r in outcome.rationale),
            "Confidence threshold must report SATISFIED when confidence exceeds 70%.",
        )
        self.assertTrue(
            any("FLAGGED" in r for r in outcome.rationale),
            "Human review safeguard bullet must report FLAGGED.",
        )

    @patch.object(StrandsDeciderClient, "ask")
    def test_no_contradictory_confidence_threshold_warning_when_confidence_is_high(self, mock_ask):
        """When confidence is ~93% and human review is flagged, ensure no contradictory 'low threshold' warning is shown."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "dispatch_replacement",
                "confidence": 0.930,
                "probabilities": {
                    "dispatch_replacement": 0.930,
                    "crossdock": 0.040,
                    "reroute_shipment": 0.020,
                    "escalate_to_human": 0.007,
                    "continue_monitoring": 0.003,
                },
            },
            "is_urgent": {"type": "noul", "noul": 0.95},
            "human_review": {"type": "noul", "noul": 0.85},
            "urgency_tier": {"type": "choice", "choice": "CRITICAL"},
        }

        incident = IncidentContext(
            shipment_id="SHP-8921-COLD",
            delivery_deadline="2026-10-21T18:00:00Z",
            truck_breakdown=True,
            breakdown_severity="CRITICAL",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Cold-Chain Perishables (Frozen Seafood)",
        )

        outcome = self.decider.evaluate(incident)

        self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
        self.assertTrue(outcome.is_real_model)
        self.assertEqual(outcome.selected_action, "dispatch_replacement")
        self.assertAlmostEqual(outcome.model_confidence, 0.930, places=3)
        self.assertTrue(outcome.human_review_required)

        # The reason must explicitly cite the human-review safeguard, not a low confidence threshold
        self.assertIn("P(review) = 0.85 >= 0.50", outcome.human_review_reason)
        self.assertNotIn("below safety threshold", outcome.human_review_reason)

        # Rationale must state confidence threshold is SATISFIED (93.0% >= 70%)
        rationale_text = " ".join(outcome.rationale)
        self.assertIn("Confidence threshold (70%): SATISFIED (93.0% >= 70%)", rationale_text)
        self.assertNotIn("Confidence threshold (70%): LOW/FLAGGED", rationale_text)
        self.assertIn("Human review safeguard: FLAGGED", rationale_text)

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


class TestStrandsDeciderHostedAPI(unittest.TestCase):
    """Test suite for the official hosted Strands Decider CloudFront API."""

    def setUp(self):
        self.incident = IncidentContext(
            shipment_id="SHP-HOSTED-01",
            delivery_deadline="2026-10-20T17:00:00Z",
            truck_breakdown=True,
            breakdown_severity="MODERATE",
            replacement_vehicle_available=True,
            customer_notified=False,
            cargo_type="Standard Logistics Freight",
        )

    def test_hosted_mode_env_activation_and_url_default(self):
        """When DECIDER_API_KEY is configured, client defaults to CloudFront URL with Bearer auth."""
        with patch.dict(os.environ, {ENV_DECIDER_API_KEY: "dcd_live_test_key_123"}, clear=True):
            client = StrandsDeciderClient()
            self.assertTrue(client.is_hosted)
            self.assertTrue(client.is_authenticated)
            self.assertEqual(client.url, UGMDU_DECIDER_URL)
            self.assertEqual(client.auth_type, "bearer")
            self.assertEqual(client._auth_header, "Bearer dcd_live_test_key_123")

    @patch("urllib.request.urlopen")
    def test_hosted_request_body_schema_strictness(self, mock_urlopen):
        """Hosted POST /v1/decide must contain ONLY 'state' and 'questions' (strictly no model or generation params)."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "answers": {
                "action": {"type": "choice", "choice": "reroute_shipment", "confidence": 0.85, "probabilities": {"reroute_shipment": 0.85}},
                "is_urgent": {"type": "noul", "noul": 0.40},
                "human_review": {"type": "noul", "noul": 0.10},
                "urgency_tier": {"type": "choice", "choice": "MODERATE"},
            },
            "usage": {"input_tokens": 55},
        }).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        client = StrandsDeciderClient(
            url="https://da25tnxy06gjk.cloudfront.net",
            api_key="dcd_live_fake_key",
        )
        questions = {
            "action": {"type": "choice", "instructions": "Select tool", "criteria": LOGISTICS_TOOLS},
            "is_urgent": {"type": "noul", "instructions": "Urgent?"},
        }
        client.ask(state="Shipment SHP-1 incident", questions=questions)

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://da25tnxy06gjk.cloudfront.net/v1/decide")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer dcd_live_fake_key")
        self.assertEqual(req.get_header("Content-type"), "application/json")

        # Inspect request body JSON
        sent_body = json.loads(req.data.decode("utf-8"))
        self.assertIn("state", sent_body)
        self.assertIn("questions", sent_body)
        self.assertEqual(len(sent_body.keys()), 2, "Body must contain strictly 'state' and 'questions'.")
        self.assertNotIn("model", sent_body)
        self.assertNotIn("max_tokens", sent_body)
        self.assertNotIn("temperature", sent_body)
        self.assertNotIn("messages", sent_body)

    @patch.object(StrandsDeciderClient, "ask")
    def test_hosted_successful_choice_and_probability_parsing(self, mock_ask):
        """Verify successful parsing of genuine choice, probability distribution, and confidence."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "crossdock",
                "probabilities": {
                    "crossdock": 0.842,
                    "dispatch_replacement": 0.088,
                    "reroute_shipment": 0.040,
                    "escalate_to_human": 0.020,
                    "continue_monitoring": 0.010,
                },
                "confidence": 0.842,
            },
            "is_urgent": {"type": "noul", "noul": 0.55},
            "human_review": {"type": "noul", "noul": 0.15},
            "urgency_tier": {"type": "choice", "choice": "MODERATE"},
        }

        decider = StrandsDecider2B(api_key="dcd_live_test")
        outcome = decider.evaluate(self.incident)

        self.assertEqual(outcome.engine, "STRANDS_DECIDER_2B_REAL")
        self.assertTrue(outcome.is_real_model)
        self.assertEqual(outcome.selected_action, "crossdock")
        self.assertAlmostEqual(outcome.model_confidence, 0.842, places=3)
        self.assertEqual(outcome.alternatives["crossdock"], 0.842)
        self.assertFalse(outcome.human_review_required)

    @patch.object(StrandsDeciderClient, "ask")
    def test_hosted_empty_action_answer_triggers_safe_fallback(self, mock_ask):
        """Simulate user's bug: empty action answer or empty answers payload safely engages fallback."""
        # 1. Empty answers dict
        mock_ask.return_value = {}
        decider = StrandsDecider2B(api_key="dcd_live_test")
        outcome = decider.evaluate(self.incident)
        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIn("empty or non-dictionary answers payload", outcome.fallback_reason)

        # 2. Empty action dict inside answers
        mock_ask.return_value = {"action": {}}
        outcome2 = decider.evaluate(self.incident)
        self.assertEqual(outcome2.engine, "SAFE_FALLBACK")
        self.assertTrue(outcome2.human_review_required)
        self.assertIn("missing or empty 'action' answer", outcome2.fallback_reason)

    @patch.object(StrandsDeciderClient, "ask")
    def test_hosted_invalid_action_choice_rejected(self, mock_ask):
        """Invalid action outside permitted schema is rejected and triggers safe fallback."""
        mock_ask.return_value = {
            "action": {
                "type": "choice",
                "choice": "unauthorized_fly_drone_action",
                "confidence": 0.99,
            },
            "is_urgent": {"type": "noul", "noul": 0.1},
            "human_review": {"type": "noul", "noul": 0.1},
        }

        decider = StrandsDecider2B(api_key="dcd_live_test")
        outcome = decider.evaluate(self.incident)
        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertTrue(outcome.human_review_required)
        self.assertIn("Invalid action choice", outcome.fallback_reason)

    @patch("urllib.request.urlopen")
    def test_hosted_health_check_queries_v1_models(self, mock_urlopen):
        """Hosted health check must query GET /v1/models with Bearer auth."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"models": ["hobson-v19"], "status": "ok"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = None
        mock_urlopen.return_value = mock_response

        client = StrandsDeciderClient(api_key="dcd_live_health_test")
        info = client.health()
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.full_url, "https://da25tnxy06gjk.cloudfront.net/v1/models")
        self.assertEqual(req.get_header("Authorization"), "Bearer dcd_live_health_test")
        self.assertEqual(info.get("status"), "ok")

    @patch.object(StrandsDeciderClient, "ask")
    def test_hosted_network_timeout_triggers_safe_fallback(self, mock_ask):
        """When hosted CloudFront API times out, safe fallback deterministically engages."""
        mock_ask.side_effect = ServerUnavailableError("HTTP Error 429: Rate limited; retry")
        decider = StrandsDecider2B(api_key="dcd_live_test")
        outcome = decider.evaluate(self.incident)
        self.assertEqual(outcome.engine, "SAFE_FALLBACK")
        self.assertFalse(outcome.is_real_model)
        self.assertTrue(outcome.human_review_required)
        self.assertIn("Rate limited", outcome.fallback_reason)

    def test_hosted_repr_masks_api_key(self):
        """Hosted client repr must mask API key and never leak credentials."""
        client = StrandsDeciderClient(api_key="dcd_live_confidential_key_99999")
        repr_str = repr(client)
        str_str = str(client)
        self.assertNotIn("confidential", repr_str)
        self.assertNotIn("confidential", str_str)
        self.assertNotIn("99999", repr_str)
        self.assertNotIn("99999", str_str)
        self.assertIn("mode=hosted", repr_str)
        self.assertIn("bearer", repr_str)


if __name__ == "__main__":
    unittest.main()
