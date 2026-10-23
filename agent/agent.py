"""CapacityX: Autonomous Logistics Decision Agent.

Strands Agent with calibrated Strands Decider 2B logistics tool selection
and Amazon Bedrock (via Strands Agents SDK) operational viability reasoning.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from strands import Agent

from decider_kit import Decider
from decider_kit.llm import agent_model, default_model_id, image_blocks, strip_thinking

from .decision import LogisticsActionGuard, render_incident_state
from .tools import TOOLS

NAME = "CapacityX: Autonomous Logistics Decision Agent"

SYSTEM_PROMPT = """You are CapacityX, an AI Autonomous Logistics Dispatch Agent.
You triage real-time freight and fleet disruptions (vehicle breakdowns, hazardous weather,
traffic congestion, perishable cold-chain risks, and critical SLA breaches).

You have 5 canonical logistics tools available:
1. dispatch_replacement: when the primary vehicle is disabled and standby units are in range.
2. reroute_shipment: when the vehicle is operational but encounters route delay or weather hazards.
3. crossdock: when perishable freight or minor vehicle faults require warehouse transshipment.
4. escalate_to_human: when an immobilized vehicle has no backups or critical SLA breach is imminent.
5. continue_monitoring: when vehicle telemetry, speed, and ETA are nominal.

All tools execute in preview/simulation mode: no real vehicles are dispatched.
Be concise, authoritative, and provide clear logistical rationale for all operational actions."""

EAGER_PROMPT = """You are CapacityX, an AI Autonomous Logistics Dispatch Agent.
Act immediately on all fleet telemetry signals without requesting clarification.
Call the most appropriate logistics tool immediately."""


class HackathonAgent:
    """One Strands agent per session with the logistics decision pattern wired in."""

    def __init__(
        self,
        decider: Decider | None = None,
        model_id: str | None = None,
        eager: bool | None = None,
    ):
        self.decider = decider or Decider()
        self.eager = eager if eager is not None else os.environ.get("DEMO_EAGER") == "1"
        self.model_id = model_id or default_model_id()
        self.sessions: dict[str, tuple[Agent, LogisticsActionGuard]] = {}

    def _session(self, session_id: str) -> tuple[Agent, LogisticsActionGuard]:
        if session_id not in self.sessions:
            guard = LogisticsActionGuard(self.decider)
            model = agent_model(self.model_id)
            system = EAGER_PROMPT if self.eager else SYSTEM_PROMPT
            agent = Agent(
                model=model,
                system_prompt=system,
                tools=TOOLS,
                interventions=[guard],
                callback_handler=None,
            )
            self.sessions[session_id] = (agent, guard)
        return self.sessions[session_id]

    def chat(
        self,
        message: str,
        session_id: str = "default",
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Process dispatcher message or incident telemetry, invoking the decision guard and Bedrock agent."""
        agent, guard = self._session(session_id)
        before = len(guard.log)
        t0 = time.perf_counter()

        prompt_input: Any = message
        if images:
            prompt_input = [*image_blocks(images), {"text": message}]

        try:
            result = agent(prompt_input, invocation_state={"images": list(images or [])})
            reply = strip_thinking(str(result))
        except Exception as e:
            # When Bedrock or cloud LLM is blocked/unverified, provide a clear, honest operational reply
            # while preserving all Strands Decider decisions made during the turn.
            err_msg = str(e)
            reply = (
                f"[CapacityX Operational Notice] Telemetry evaluated. "
                f"Conversational generation offline ({type(e).__name__}: {err_msg[:120]}). "
                f"All calibrated decision safeguards and tool previews are logged in the decision panel."
            )

        latency_ms = round((time.perf_counter() - t0) * 1000)
        return {
            "reply": reply,
            "decisions": guard.log[before:],
            "model_id": self.model_id,
            "latency_ms": latency_ms,
            "agent": NAME,
            "eager": self.eager,
        }


if __name__ == "__main__":
    import sys

    a = HackathonAgent()
    sample = (
        "INC-809: Vehicle TRK-104 has engine breakdown on I-90. "
        "Backup unit STANDBY-22 is 18 minutes away. Refrigerated dairy cargo. What action?"
    )
    res = a.chat(sys.argv[1] if len(sys.argv) > 1 else sample)
    print(json.dumps(res, indent=2))
