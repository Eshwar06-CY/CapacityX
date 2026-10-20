# Strands Decider 2B — Typed Logistics Decision Pattern

This module implements the **Strands Decider 2B** decision pattern for **CAPACITYX**, adhering to the official upstream specification from [Strands Labs (AWS)](https://github.com/strands-labs/strands-decider).

---

## 1. What is Strands Decider 2B?

Strands Decider is an open-source **"System 1" decision model** designed specifically for agentic AI workflows. Unlike conventional generative LLMs that produce free-form text, Strands Decider:
- Evaluates a contextual **state** against **typed questions** (`choice`, `noul`, `score`).
- Selects among bounded options in milliseconds.
- Outputs **calibrated confidence probabilities** (allowing automated actions when confidence is high, and triggering human review when confidence is low).
- Requires no prompt engineering for schema compliance; decision boundaries are learned and deterministic.

---

## 2. The 5 Typed Logistics Decision Tools

In CapacityX, the Decider selects from 5 discrete tools:

| Tool Name | Type | Selection Criteria |
|---|---|---|
| `dispatch_replacement` | Action | Primary vehicle is immobilized; backup transport asset is available within range. Dispatches backup unit to rendezvous and cross-dock freight. |
| `reroute_shipment` | Action | Vehicle is operational; recomputes route coordinates to bypass congestion, road closures, or minor delays. |
| `crossdock` | Action | Vehicle has partial mechanical issue or perishable cargo; diverts to closest warehouse cross-dock bay. |
| `escalate_to_human` | Escalation | Vehicle disabled with **NO** backup units within range, or severe SLA breach predicted. Mandates human dispatcher intervention. |
| `continue_monitoring` | Nominal | Telemetry and route are nominal; maintain automated heartbeat monitoring without intervention. |

---

## 3. Architecture & Fail-Safe Fallback

```
[ Shipment Incident Payload ]
              │
              ▼
    [ IncidentContext.to_state_text() ]
              │
              ▼
   [ StrandsDecider2B.evaluate() ]
              │
     Is Decider Server online?
          /             \
        YES              NO / Inference Error
        /                 \
       ▼                   ▼
[ StrandsDeciderClient ]  [ _safe_rule_fallback() ]
POST /v1/systemone        - Deterministic safety policy
- Action: choice          - Selected action assigned safely
- Is Urgent: noul         - human_review_required = True
- Human Review: noul      - confidence = None
- Urgency Tier: choice    - engine = "SAFE_FALLBACK"
       │                   - is_real_model = False
       ▼                   │
Outcome:                   │
- engine = REAL_MODEL      │
- confidence parsed        │
- alternatives mapped      │
       \                  /
        ▼                ▼
     [ Standard DeciderOutcome ]
                 │
                 ▼
     [ FastAPI Backend /api/decide ]
```

---

## 4. Hardware Requirements & Windows Environment Blocker

### The Machine Profile (Inspected)
- **CPU**: Intel Core Ultra 5 125H
- **GPU**: Integrated Intel Arc Graphics (No discrete NVIDIA GPU / No CUDA)
- **Total System RAM**: 16 GB
- **Currently Available Free RAM**: **~2.3 GB**
- **OS**: Windows 11 (Python 3.13)

### Exact Local Execution Blockers:
1. **Memory Exhaustion**: The `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` model weights require **~4.5 GB to 5 GB** in RAM. Loading these weights locally into Python on this machine with only ~2.3 GB of free physical memory will instantly exhaust RAM and trigger Windows pagefile thrashing or crash with an `OutOfMemoryError`.
2. **Platform & Accelerator Support**: The official repository (`docs/inference.md`) explicitly specifies:
   - Linux with NVIDIA CUDA GPU
   - Windows with WSL2 and NVIDIA GPU passthrough
   - Apple Silicon Mac with MPS / MLX
   - CPU-only execution on Windows is unaccelerated and takes several seconds per forward pass.

---

## 5. How to Run the Real Decider 2B Server (When Hardware is Ready)

When running on an instance with sufficient memory (e.g. an external server, GPU instance, or WSL2 with at least 8 GB allocated):

### 1. Install Upstream Dependencies
```bash
pip install strands-decider
```

### 2. Start the Inference Server
```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
```

### 3. Point CapacityX to the Server
In PowerShell for local inference:
```powershell
$env:STRANDS_DECIDER_URL = "http://127.0.0.1:8099"
```

For an authenticated ngrok HTTP Basic Auth tunnel:
```powershell
$env:STRANDS_DECIDER_URL = "https://your-tunnel-subdomain.ngrok-free.app"
$env:STRANDS_DECIDER_USERNAME = "your_ngrok_username"
$env:STRANDS_DECIDER_PASSWORD = "your_ngrok_password"
```

Once the server answers `GET /health`, CapacityX will automatically switch from `SAFE_FALLBACK` to `STRANDS_DECIDER_2B_REAL` and display real model probabilities!

---

## 6. Running Unit Tests

To verify the decision pattern, request serialization, tool definitions, confidence thresholding, safe fallback logic, and HTTP Basic Auth tunnel security:

```powershell
python -m unittest -v patterns/tests/test_strands_decider.py
```
Expected output:
```text
Ran 13 tests in 0.007s
OK
```
