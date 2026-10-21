# Use case: CapacityX Smart Logistics Decision Agent

- **Pitch:** An autonomous AI logistics dispatch agent that pairs Strands Decider 2B calibrated tool selection with Amazon Bedrock operational reasoning to resolve freight disruptions safely.
- **Who it is for:** Fleet managers, dispatch operations centers, third-party logistics (3PL) providers, and autonomous freight operations.
- **Features:**
  - Real-time telemetry evaluation (vehicle breakdowns, weather closures, congestion, and cold-chain temperature spikes);
  - Instant tool selection across 5 canonical actions (`dispatch_replacement`, `reroute_shipment`, `crossdock`, `escalate_to_human`, `continue_monitoring`);
  - Strict human-review safeguard triggered when model confidence is `< 0.70` or risk is high (`p >= 0.50`);
  - All operational actions held in simulation preview mode (zero unauthorized fleet dispatches);
  - Conversational operational viability and logistics precaution reasoning via Amazon Bedrock;
  - Deployable to Amazon Bedrock AgentCore Runtime.
- **The decision handed to Strands Decider 2B:** Which of the five canonical logistics tools should handle this incident? (`#tool-selection`).
- **Why a decision model:** Modern freight networks process continuous vehicle heartbeats and disruption signals. Strands Decider 2B evaluates dense structured telemetry in a single forward pass, returning calibrated probabilities and confidence scores that deterministic code can branch on, with sub-100ms latency and zero hallucinated actions.
- **Amazon Bedrock:** Conversational reasoning, dispatcher interaction, and deep viability analysis (Amazon Nova).
