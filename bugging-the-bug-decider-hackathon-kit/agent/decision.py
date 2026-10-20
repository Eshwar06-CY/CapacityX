"""REUSABLE DECISION PATTERN: order-action guardrail  (#guardrails)

KIT EXAMPLE. This file belongs to the kit's example agent. Run /new-agent in Kiro (or follow
docs/BUILD.md) to replace it with the decision your own agent hands to Strands Decider 2B.

What it does
    Before the support agent runs a tool that changes an order (refund, cancel), Strands
    Decider 2B checks the call against the conversation:
    - are the arguments grounded in what the customer said (no invented order ids or amounts)?
    - does the customer clearly ask for this action, rather than asking about it?
    - is it premature (should the agent confirm first)?
    Grounded and requested calls run; otherwise the agent is sent back to ask (`Guide`), or a
    human confirms (`Confirm`) when the decider is unsure.

Typed questions and options
    args_grounded (noul), explicit_request (noul), premature (noul); see QUESTIONS below.

Wiring
    `OrderActionGuard` is a Strands `InterventionHandler`; agent/agent.py passes it to the Agent:
        Agent(..., interventions=[OrderActionGuard(decider)])
    It only inspects tools in GUARDED_TOOLS; read-only tools run untouched. If the agent retries
    a call that was just sent back, unchanged, the retry is denied (Strands requires guide
    handlers to converge).

How to reuse it
    Copy this file, change GUARDED_TOOLS to your irreversible tools, and reword the three
    questions for your domain. Thresholds are in `before_tool_call`.
"""

from __future__ import annotations

import json
from typing import Any

from strands.interventions import Confirm, Deny, Guide, InterventionHandler, Proceed

from decider_kit import Decider
from decider_kit.patterns.guardrails import render_messages

IS_KIT_EXAMPLE = True  # /new-agent removes this; the submission check fails while it is True

PATTERN_TAG = "#guardrails"
GUARDED_TOOLS = {"refund_order", "cancel_order"}

QUESTIONS = {
    "args_grounded": Decider.noul(
        "Are the order id and any amount in the proposed action stated by the customer or returned by an earlier lookup?",
        {"true": "every argument comes from the customer's words or an earlier tool result",
         "false": "an argument was guessed or invented"}),
    "explicit_request": Decider.noul(
        "Is the customer's latest message an instruction to perform this action now, rather than a question about it?",
        {"true": "an instruction such as 'refund it' or 'please cancel A1003'",
         "false": "a question ('can I...?', 'what happens if...?', 'what are my options?'), a complaint, or a status check"}),
    "premature": Decider.noul(
        "Should the assistant confirm details with the customer before taking this action?",
        {"true": "something is ambiguous or unconfirmed; ask first", "false": "everything needed is confirmed"}),
}


class OrderActionGuard(InterventionHandler):
    name = "order-action-guard"

    def __init__(self, decider: Decider, yes: float = 0.5, unsure_band: tuple[float, float] = (0.4, 0.6)):
        self.decider, self.yes, self.unsure_band = decider, yes, unsure_band
        self.log: list[dict[str, Any]] = []
        self._guided: set[str] = set()  # calls already sent back; a retry of the same call is denied

    def before_invocation(self, event, **kwargs):
        self._guided.clear()  # a new customer message: guided calls may now be justified
        return Proceed()

    def before_tool_call(self, event, **kwargs):
        name = event.tool_use["name"]
        if name not in GUARDED_TOOLS:
            return Proceed()
        args = event.tool_use.get("input") or {}
        sig = f"{name}:{json.dumps(args, sort_keys=True)}"
        if sig in self._guided:  # Guide must converge: the same call again, unchanged, is refused
            self.log.append({"pattern": PATTERN_TAG, "tool": name, "args": args, "verdict": "deny", "reason": "retried a guided call"})
            return Deny(reason="This exact action was already rejected. Ask the customer for the missing details instead.")
        state = ("A customer-support conversation, then an order action the assistant wants to take.\n\n"
                 f"--- CONVERSATION ---\n{render_messages(event.agent.messages)}\n\n"
                 f"--- PROPOSED ACTION ---\ntool: {name}\narguments: {json.dumps(args)}")
        a = self.decider.ask(state, QUESTIONS)
        g, e, p = (a[k].p_yes or 0.0 for k in ("args_grounded", "explicit_request", "premature"))
        rec = {"pattern": PATTERN_TAG, "tool": name, "args": args, "args_grounded": round(g, 3),
               "explicit_request": round(e, 3), "premature": round(p, 3), "decider_ms": round(self.decider.last_latency_ms or 0)}
        lo, hi = self.unsure_band
        if g < self.yes:
            rec["verdict"], action = "guide", Guide(feedback="Do not guess order ids or amounts. Look the order up or ask the customer.")
        elif e < self.yes:
            rec["verdict"], action = "guide", Guide(feedback="The customer has not asked for this action. Answer their question instead, and offer the action.")
        elif lo <= p <= hi:
            rec["verdict"], action = "confirm", Confirm(prompt=f"Approve {name} {json.dumps(args)}?", response="yes")
        elif p > hi:
            rec["verdict"], action = "guide", Guide(feedback="Confirm the details with the customer before taking this action.")
        else:
            rec["verdict"], action = "proceed", Proceed()
        if rec["verdict"] == "guide":
            self._guided.add(sig)
        self.log.append(rec)
        return action
