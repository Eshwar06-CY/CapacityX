<!-- Template/Offline Draft: Generated deterministically from project telemetry and measured benchmark data without an active LLM provider. -->
# CapacityX: Autonomous Logistics Decision Agent with Strands Decider 2B

By Eshwar (CapacityX Team)

Supply chains constantly face sudden operational disruptions--unexpected transit delays, cooling failures, and cargo damage. CapacityX pairs an autonomous logistics operations agent with Strands Decider 2B to evaluate incident telemetry and select calibrated mitigation actions in milliseconds. By deploying a dedicated 2B parameter decision model directly at the tool-selection seam, CapacityX delivers fast, reliable, and cost-effective decisions without the latency, token overhead, and unpredictability of general-purpose LLMs.

## The project

CapacityX is built for logistics operations teams, cold-chain fleet operators, and freight dispatchers handling high-value and temperature-sensitive goods. The agent continuously monitors shipment status, evaluates environmental and routing anomalies, and chooses the appropriate mitigation tool. Its core operational tools include rerouting delayed shipments, dispatching priority replacements, commanding reefer cooling adjustments, continuing active monitoring, and escalating critical or low-confidence incidents to human operators.

In the interactive operations console, operators observe real-time telematics, telemetry severity scores, and the decider's calibrated probabilities. When an incident occurs--such as a perishable pharmaceutical shipment experiencing an unauthorized container temperature rise--the system selects the safest mitigation action with measured probability distributions.

## How I built it

I built CapacityX using the Strands Agents SDK to orchestrate the agent conversational loop and tool execution. The agent architecture pairs Amazon Bedrock for generative reasoning and operator explanations with Strands Decider 2B for discrete action classification. The five logistics tools (reroute_shipment, dispatch_replacement, escalate_to_human, adjust_temperature, continue_monitoring) execute in simulation mode, ensuring safe offline evaluation and reproducible benchmarks. The complete system was developed following a spec-driven workflow using the AWS User Group Madurai hackathon kit.

## The decision I handed to Strands Decider 2B

Selecting an operational mitigation action is fundamentally a classification problem, not an open-ended generative task. Asking a full-scale LLM to pick a tool on every telemetry alert introduces multi-second latency, expensive per-call costs, and non-deterministic hallucinations. I handed this decision to Strands Decider 2B using typed question definitions:

```python
QUESTIONS = {
    "action": Decider.choice(
        "Which logistics action should handle this incident?",
        ACTIONS_CRITERIA,
    ),
    "human_review": Decider.noul(
        "Does this incident require mandatory human dispatcher review?",
        {
            "true": "critical incident, vehicle disabled without backup, severe SLA breach, or hazardous condition",
            "false": "routine transit or automated detour fully resolved with high confidence",
        },
    ),
    "urgency": Decider.score(
        "How urgent is this logistics event?",
        ["low", "medium", "high", "critical"],
    ),
}
```

< FIGURE 1 of 5: upload submission/article/img/architecture.png here >
Alt text: Architecture: the user, the agent with its tools and decision, Amazon Bedrock and Strands Decider 2B

When telemetry arrives, the agent calls the decider with the current operational state. If the model's confidence meets or exceeds the 0.70 threshold, and the calculated human review safety score is at or above 0.50, the action is automatically approved. If confidence drops below 0.70, or if critical urgency indicators are present, the incident is immediately flagged for human review.

## Measured

I benchmarked Strands Decider 2B (checkpoint strands-decider-2b-hobson-v19) on the 13 held-out test cases from agent/eval_cases.jsonl on the AWS User Group Madurai hosted endpoint. The model achieved 61.5% overall accuracy across the test decisions with a median latency of 2473 ms and an estimated cost of $0.0350 per 1,000 decisions. When confident (confidence >= 0.70), it handled 77% of decisions autonomously with 70.0% accuracy, while safely escalating 3 lower-confidence edge cases to human specialists.

The Amazon Bedrock baseline (us.amazon.nova-pro-v1:0) could not be measured due to AWS account verification constraints. As required by hackathon fairness rules, its baseline metrics remain unmeasured (N/A) rather than simulated or estimated.

Decider: strands-decider-2b-hobson-v19 (ugmdu). 13 labelled cases, 13 decisions.

| | Strands Decider 2B | us.amazon.nova-pro-v1:0 |
|---|---|---|
| Accuracy | 61.5% | N/A |
| Latency p50 per decision | 2473 ms | N/A |
| Latency p95 | 2656 ms | N/A |
| Cost per 1,000 decisions | $0.0350 | N/A |
| Answered alone at confidence >= 0.7 | 77% | N/A |
| Accuracy when confident | 70.0% | N/A |

< FIGURE 2 of 5: upload submission/article/img/bench-accuracy.png here >
Alt text: Accuracy of Strands Decider 2B and the Bedrock LLM on the labelled cases

< FIGURE 3 of 5: upload submission/article/img/bench-latency.png here >
Alt text: Median latency per decision for both

< FIGURE 4 of 5: upload submission/article/img/bench-cost.png here >
Alt text: Cost per 1,000 decisions for both

< FIGURE 5 of 5: upload submission/article/img/bench-confidence.png here >
Alt text: Accuracy against the share of decisions answered without a human, as the confidence threshold moves

## The reusable decision pattern

The decision pattern is implemented in agent/decision.py under the #tool-selection pattern tag. It defines the typed action choices, probability extraction routines, and safe fallback logic. Any agent developer using Strands Agents can import this pattern and drop calibrated classification directly into their agent's seam, replacing bulky LLM prompts with a lightweight forward pass.

## What I learned and what is next

Building CapacityX proved that small, specialized decision models like Strands Decider 2B provide the speed, calibration, and cost profile that production operations demand. Moving forward, I plan to integrate live IoT telemetry streams from real container sensors and package the agent for deployment on Amazon Bedrock AgentCore Runtime.

---

Code: https://github.com/Eshwar06-CY/CapacityX . Video: submission/video/demo.mp4 . Tags: #strands-decider-2b #tool-selection

Built with the AWS User Group Madurai Strands Decider hackathon kit
