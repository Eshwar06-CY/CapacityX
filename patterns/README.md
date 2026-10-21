# CAPACITYX Agent Decision & Orchestration Patterns

This directory contains the core decision and reasoning components for **CAPACITYX**:
1. **Strands Decider 2B** (`patterns/strands_decider_2b.py`): System 1 fast, typed logistics tool selector with calibrated probabilities.
2. **Amazon Bedrock Nova Micro** (`patterns/bedrock_nova_micro.py`): Converse API adapter for `amazon.nova-micro-v1:0` in `us-east-1`.
3. **Agent Orchestrator** (`patterns/orchestrator.py`): Orchestrates Strands tool selection with Bedrock complementary operational reasoning.
4. **Offline Test Suite** (`patterns/tests/`): 38 deterministic, 100% offline unit tests.

---

## 1. System 1 vs. System 2 Decision Pattern

| Role | Component | Responsibility | Output Characteristics |
|---|---|---|---|
| **System 1 (Primary)** | **Strands Decider 2B** | Evaluates incident state and selects 1 of 5 canonical tools | Calibrated confidence (0.0 to 1.0), probability distribution over all 5 tools |
| **System 2 (Complementary)** | **Amazon Bedrock Nova Micro** | Operational reasoning & viability analysis | Logistical precautions, warehouse routing notes, risk detection |

### Workflow:
1. `StrandsDecider2B.evaluate_decision()` receives structured incident telemetry (`IncidentContext`).
2. If the inference server is online, it evaluates choices and returns the selected action, urgency, confidence score, and alternative tool distribution. If offline, the deterministic safe fallback engages.
3. The selected action and context are forwarded to `BedrockOperationalReasoner`.
4. If Bedrock is disabled (default), the Strands decision passes through with `bedrock_validation.status: "disabled"`.
5. If Bedrock mock mode is enabled for tests, simulated operational precautions are generated with `is_simulated: true`.
6. If live Bedrock inference is enabled, the Converse API is invoked. If AWS returns an account verification error (`ValidationException: Operation not allowed`), it is trapped safely and recorded as `status: "blocked_unverified"` without disrupting the primary Strands decision.

---

## 2. The 5 Canonical Logistics Decision Tools

All orchestration pathways strictly map to the 5 canonical logistics actions:

| Tool Name | Type | Selection Criteria |
|---|---|---|
| `dispatch_replacement` | Action | Primary vehicle immobilized; replacement asset available within 45 min. Dispatches backup unit to rendezvous. |
| `reroute_shipment` | Action | Vehicle operational; recomputes route coordinates to bypass congestion, road closures, or minor delays. |
| `crossdock` | Action | Vehicle has partial mechanical issue or perishable cargo; diverts to closest warehouse cross-dock bay. |
| `escalate_to_human` | Escalation | Vehicle disabled with **NO** backup units within range, or severe SLA breach predicted. Mandates human dispatcher intervention. |
| `continue_monitoring` | Nominal | Telemetry and route are nominal; maintain automated heartbeat monitoring without intervention. |

---

## 3. Configuration & Modes

| Variable | Values | Description |
|---|---|---|
| `DECIDER_API_KEY` | string | Official hosted Strands Decider API key (Bearer token). Activates hosted mode automatically. |
| `DECIDER_MODE` | `ugmdu` | Optional hosted mode selector |
| `DECIDER_URL` | URL string | Optional decider URL override (defaults to `https://da25tnxy06gjk.cloudfront.net` in hosted mode) |
| `STRANDS_DECIDER_URL` | URL string | Optional local or ngrok decider URL (defaults to `http://127.0.0.1:8099`) |
| `STRANDS_DECIDER_USERNAME` | string | Optional HTTP Basic Auth username for local server |
| `STRANDS_DECIDER_PASSWORD` | string | Optional HTTP Basic Auth password for local server |
| `CAPACITYX_ENABLE_BEDROCK` | `false` (default) / `true` | Enable or disable Bedrock operational reasoning |
| `CAPACITYX_BEDROCK_MODE` | `disabled` (default) / `mock` / `live` | Operational mode for the Bedrock component |
| `BEDROCK_MODEL_ID` | `amazon.nova-micro-v1:0` | Target Amazon Bedrock foundation model |
| `AWS_REGION` | `us-east-1` | Target AWS region |

---

## 4. Amazon Bedrock Free Tier Restriction & Mock Transparency

- **Live Bedrock Blocker**: AWS accounts on the Free plan receive:
  ```text
  ValidationException: Operation not allowed
  ```
  when attempting to invoke foundation models via the Converse API.
- **Safety Guarantee**: The orchestrator handles this exception gracefully. The application does not crash, no live AWS calls are made during normal operation or test execution, and the primary Strands Decider decision is preserved.
- **Simulation Transparency**: When using the offline mock for tests, outputs are clearly stamped:
  ```json
  {
    "status": "simulated_mock",
    "is_simulated": true,
    "disclaimer": "SIMULATED: Mock Bedrock operational reasoning for offline testing only. Not live AWS inference."
  }
  ```
  **Under no circumstances is simulated mock output represented as real Bedrock inference.**

---

## 5. Offline Test Execution

Run the complete 47-test offline suite:
```powershell
python -m unittest discover -s patterns/tests -p "test_*.py" -v
```

Expected output:
```text
Ran 47 tests in ~8.8s
OK
```

Individual test suites:
- `python -m unittest -v patterns/tests/test_strands_decider.py` (24 tests: core decider, thresholds, auth, hosted CloudFront API)
- `python -m unittest -v patterns/tests/test_bedrock_nova_micro.py` (14 tests)
- `python -m unittest -v patterns/tests/test_orchestrator.py` (9 tests)
