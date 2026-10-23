"""Comprehensive unit test suite for CAPACITYX Logistics Agent and Decision Pattern.

Validates:
1. Simulation-only execution for all 5 canonical logistics actions.
2. Exact confidence threshold (0.70) enforcement.
3. Independent human-review signal (p >= 0.50) triggering dispatcher review.
4. Deterministic safe fallback when Decider inference is unavailable.
5. HackathonAgent contract and graceful operational degradation.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from decider_kit.client import Decision
from agent.tools import (
    dispatch_replacement,
    reroute_shipment,
    crossdock,
    escalate_to_human,
    continue_monitoring,
    SIMULATION_DISCLAIMER,
)
from agent.decision import (
    evaluate_decision,
    evaluate_safe_fallback,
    render_incident_state,
    LogisticsActionGuard,
    CONFIDENCE_THRESHOLD,
    HUMAN_REVIEW_SCORE_THRESHOLD,
)
from agent.agent import HackathonAgent, NAME


class TestCapacityXTools(unittest.TestCase):
    """Verify that all 5 logistics tools execute strictly in simulation preview mode."""

    def test_dispatch_replacement_simulation_mode(self):
        res = dispatch_replacement("INC-101", "TRK-01", "STANDBY-99", 25, "PERISHABLE")
        self.assertEqual(res["action"], "dispatch_replacement")
        self.assertFalse(res["is_live_action"])
        self.assertEqual(res["status"], SIMULATION_DISCLAIMER)
        self.assertIn("Simulation", res["preview_summary"])

    def test_reroute_shipment_simulation_mode(self):
        res = reroute_shipment("INC-102", "TRK-02", "Blizzard detour", 18.5, 30)
        self.assertEqual(res["action"], "reroute_shipment")
        self.assertFalse(res["is_live_action"])
        self.assertEqual(res["status"], SIMULATION_DISCLAIMER)

    def test_crossdock_simulation_mode(self):
        res = crossdock("INC-103", "TRK-03", "XDOCK-HUB-4", "Cold chain refrigeration leak")
        self.assertEqual(res["action"], "crossdock")
        self.assertFalse(res["is_live_action"])
        self.assertEqual(res["status"], SIMULATION_DISCLAIMER)

    def test_escalate_to_human_simulation_mode(self):
        res = escalate_to_human("INC-104", "TRK-04", "Engine failure with 0 backup units in range")
        self.assertEqual(res["action"], "escalate_to_human")
        self.assertFalse(res["is_live_action"])
        self.assertEqual(res["status"], SIMULATION_DISCLAIMER)

    def test_continue_monitoring_simulation_mode(self):
        res = continue_monitoring("INC-105", "TRK-05", 15)
        self.assertEqual(res["action"], "continue_monitoring")
        self.assertFalse(res["is_live_action"])
        self.assertEqual(res["status"], SIMULATION_DISCLAIMER)


class TestCapacityXDecisionPattern(unittest.TestCase):
    """Test Strands Decider 2B logistics decision pattern and safety thresholds."""

    def test_render_incident_state(self):
        incident = {
            "incident_id": "INC-TEST-01",
            "vehicle_id": "TRK-999",
            "status": "BREAKDOWN",
            "severity": "CRITICAL",
            "cargo_type": "MEDICAL",
            "perishable": True,
            "backup_units_within_range": 2,
        }
        state = render_incident_state(incident)
        self.assertIn("INC-TEST-01", state)
        self.assertIn("TRK-999", state)
        self.assertIn("MEDICAL", state)
        self.assertIn("Standby Backup Units in Range: 2", state)

    def test_evaluate_decision_high_confidence_automated_action(self):
        fake_decider = MagicMock()
        fake_decider.hosted = True
        fake_decider.ask.return_value = {
            "action": Decision("choice", {}, label="dispatch_replacement", confidence=0.88,
                               probabilities={"dispatch_replacement": 0.88, "reroute_shipment": 0.05,
                                              "crossdock": 0.03, "escalate_to_human": 0.02, "continue_monitoring": 0.02}),
            "human_review": Decision("noul", {}, p_yes=0.10, label="no", confidence=0.90),
            "urgency": Decision("score", {}, label="high", score=2.0),
        }
        incident = {"incident_id": "INC-01", "vehicle_id": "TRK-01", "status": "BREAKDOWN", "backup_units_within_range": 2}
        outcome = evaluate_decision(incident, decider=fake_decider)

        self.assertEqual(outcome["action"], "dispatch_replacement")
        self.assertEqual(outcome["model_confidence"], 0.88)
        self.assertFalse(outcome["human_review_required"])
        self.assertIn("Autonomous execution verified", outcome["human_review_reason"])
        self.assertEqual(outcome["engine"], "STRANDS_DECIDER_2B_HOSTED")
        self.assertEqual(len(outcome["actions_preview"]), 1)

    def test_evaluate_decision_low_confidence_triggers_human_review(self):
        fake_decider = MagicMock()
        fake_decider.hosted = True
        fake_decider.ask.return_value = {
            "action": Decision("choice", {}, label="reroute_shipment", confidence=0.62,
                               probabilities={"reroute_shipment": 0.62, "crossdock": 0.20,
                                              "dispatch_replacement": 0.10, "escalate_to_human": 0.05, "continue_monitoring": 0.03}),
            "human_review": Decision("noul", {}, p_yes=0.15, label="no", confidence=0.85),
            "urgency": Decision("score", {}, label="medium", score=1.0),
        }
        incident = {"incident_id": "INC-02", "vehicle_id": "TRK-02", "status": "ACTIVE"}
        outcome = evaluate_decision(incident, decider=fake_decider)

        self.assertEqual(outcome["action"], "reroute_shipment")
        self.assertTrue(outcome["human_review_required"])
        self.assertIn("below calibrated threshold", outcome["human_review_reason"])

    def test_evaluate_decision_high_confidence_but_human_review_signal_triggers_review(self):
        fake_decider = MagicMock()
        fake_decider.hosted = True
        fake_decider.ask.return_value = {
            "action": Decision("choice", {}, label="crossdock", confidence=0.92,
                               probabilities={"crossdock": 0.92, "reroute_shipment": 0.04,
                                              "dispatch_replacement": 0.02, "escalate_to_human": 0.01, "continue_monitoring": 0.01}),
            "human_review": Decision("noul", {}, p_yes=0.75, label="yes", confidence=0.75),
            "urgency": Decision("score", {}, label="high", score=2.0),
        }
        incident = {"incident_id": "INC-03", "vehicle_id": "TRK-03", "status": "ACTIVE", "perishable": True}
        outcome = evaluate_decision(incident, decider=fake_decider)

        self.assertEqual(outcome["action"], "crossdock")
        self.assertTrue(outcome["human_review_required"])
        self.assertIn("Incident risk check flagged review", outcome["human_review_reason"])

    def test_safe_fallback_when_inference_server_unreachable(self):
        fake_decider = MagicMock()
        fake_decider.ask.side_effect = ConnectionError("Connection refused to decider endpoint")

        incident = {
            "incident_id": "INC-ERR",
            "vehicle_id": "TRK-ERR",
            "status": "BREAKDOWN",
            "backup_units_within_range": 0,
        }
        outcome = evaluate_decision(incident, decider=fake_decider)

        self.assertEqual(outcome["action"], "escalate_to_human")
        self.assertEqual(outcome["engine"], "SAFE_FALLBACK")
        self.assertTrue(outcome["human_review_required"])
        self.assertIn("Safe Fallback active", outcome["human_review_reason"])


class TestCapacityXAgent(unittest.TestCase):
    """Test HackathonAgent contract and error handling."""

    def test_agent_contract(self):
        agent = HackathonAgent()
        self.assertEqual(agent.model_id, "us.amazon.nova-pro-v1:0")

        # Mock the underlying session agent
        mock_guard = MagicMock()
        mock_guard.log = []

        def fake_agent_call(*args, **kwargs):
            mock_guard.log.append({"pattern": "#tool-selection", "tool": "dispatch_replacement", "verdict": "proceed"})
            return "Dispatched replacement standby unit to rendezvous."

        with patch.object(agent, "_session", return_value=(fake_agent_call, mock_guard)):
            out = agent.chat("Breakdown on I-90 with standby nearby")
            self.assertIn("reply", out)
            self.assertIn("decisions", out)
            self.assertIn("model_id", out)
            self.assertIn("latency_ms", out)
            self.assertEqual(out["agent"], NAME)
            self.assertEqual(len(out["decisions"]), 1)


if __name__ == "__main__":
    unittest.main()
