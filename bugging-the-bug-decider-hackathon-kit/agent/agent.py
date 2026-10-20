"""The agent: an LLM does the talking (Amazon Bedrock by default), Strands Decider 2B makes the decision.

KIT EXAMPLE (an online-store support agent). /new-agent rewrites this file, tools.py,
decision.py, eval_cases.jsonl and demo_script.yaml for your own use case; the interface below
(`HackathonAgent.chat`) stays the same, so the demo UI, the benchmark, the video and the AWS
deployment keep working unchanged.

    from agent.agent import HackathonAgent
    out = HackathonAgent().chat("Please refund order A1001, it arrived damaged.")
    out["reply"], out["decisions"]
"""

from __future__ import annotations

import time
from typing import Any

from strands import Agent

from decider_kit import Decider
from decider_kit.llm import agent_model, default_model_id, image_blocks, strip_thinking

from .decision import OrderActionGuard
from .tools import TOOLS

NAME = "Kit example: store support agent"
SYSTEM_PROMPT = """You are a support agent for an online store. Help customers with their orders.
You can look orders up, refund them and cancel them. Be brief and friendly.
If you need an order id, ask for it. Never make up order ids or amounts."""


# Demos only: an eager prompt makes the agent act without asking, so the decision has mistakes to
# catch on camera (upstream's examples do the same). Say so in your video if you use it.
EAGER_PROMPT = """You are a support agent for an online store. You can look orders up, refund them and cancel them.
Act immediately and never ask the customer questions. If an order id is missing, use the most likely order
(customers usually mean A1001, A1002 or A1003) and call the tool straight away. Refund the full amount by default."""


class HackathonAgent:
    """One Strands agent per session, with the decision pattern wired in."""

    def __init__(self, decider: Decider | None = None, model_id: str | None = None, eager: bool | None = None):
        import os

        self.decider = decider or Decider()
        self.eager = eager if eager is not None else os.environ.get("DEMO_EAGER") == "1"
        self.model_id = model_id or default_model_id()  # LLM_PROVIDER in .env: bedrock, local, ugmdu
        self.sessions: dict[str, tuple[Agent, OrderActionGuard]] = {}

    def _session(self, session_id: str) -> tuple[Agent, OrderActionGuard]:
        if session_id not in self.sessions:
            guard = OrderActionGuard(self.decider)
            agent = Agent(model=agent_model(self.model_id), system_prompt=EAGER_PROMPT if self.eager else SYSTEM_PROMPT, tools=TOOLS,
                          interventions=[guard], callback_handler=None)
            self.sessions[session_id] = (agent, guard)
        return self.sessions[session_id]

    def chat(self, message: str, session_id: str = "default", images: list[Any] | None = None) -> dict[str, Any]:
        """`images` (optional): photos the user attached (paths, bytes or base64); the LLM sees them
        as Strands image blocks. A decision that looks at them passes the same list to Decider.ask."""
        agent, guard = self._session(session_id)
        before = len(guard.log)
        t0 = time.perf_counter()
        # the LLM sees the photos as image blocks; tools get them via decider_kit.llm.attached_images
        result = agent([*image_blocks(images), {"text": message}] if images else message,
                       invocation_state={"images": list(images or [])})
        reply = strip_thinking(str(result))  # Nova's <thinking>, Qwen's <think>
        return {"reply": reply, "decisions": guard.log[before:], "model_id": self.model_id,
                "latency_ms": round((time.perf_counter() - t0) * 1000), "agent": NAME, "eager": self.eager}


if __name__ == "__main__":
    import json
    import sys

    a = HackathonAgent()
    for msg in sys.argv[1:] or ["Hi, can you refund my order? It was the headphones.", "It's A1002, refund the full amount please."]:
        print(json.dumps(a.chat(msg), indent=1))
