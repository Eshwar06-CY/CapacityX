# CAPACITYX — Smart Logistics Decision Agent

> **Hackathon Prototype** | Real-time incident triage and automated logistics decision-making with multi-agent orchestration: Strands Decider 2B (System 1) paired with Amazon Bedrock Nova Micro (System 2).  
> *Built with the AWS User Group Madurai Strands Decider hackathon kit*

Welcome to **CapacityX**! This project provides an intelligent decision-making system for real-time logistics disruptions (e.g., vehicle breakdowns, strict delivery deadlines, temperature-sensitive cold chain freight, and replacement dispatching), keeping customers informed while safeguarding service level agreements (SLAs).

---

## 🏛️ Multi-Agent Orchestration Architecture

CapacityX pairs **Strands Decider 2B** with **Amazon Bedrock Nova Micro** in a complementary two-tier decision workflow:

```
                      [ Shipment Incident Telemetry ]
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │  SYSTEM 1: Strands Decider 2B (Primary Decision Engine) │
        │  - Evaluates typed choice & noul questions             │
        │  - Selects 1 of 5 canonical logistics actions          │
        │  - Provides calibrated confidence & alternative probs  │
        └──────────────────────────┬─────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │  Typed Logistics Decision   │
                    │  + Urgency Tier             │
                    │  + Calibrated Confidence    │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │  SYSTEM 2: Amazon Bedrock Nova Micro (Validation Layer)│
        │  - Complementary operational reasoning & validation    │
        │  - Contextual precautions & warehouse routing notes    │
        │  - Flags human review if operational risks detected    │
        │  - Does NOT alter tool selection or fake confidence   │
        └──────────────────────────┬─────────────────────────────┘
                                   │
                                   ▼
        ┌────────────────────────────────────────────────────────┐
        │  EXECUTION SAFEGUARDS & ACTION PREVIEW                 │
        │  - Zero external dispatches (is_live_action: False)    │
        │  - Human review mandatory if confidence < 70% or risk  │
        │  - Transparent audit trail & disclaimers               │
        └────────────────────────────────────────────────────────┘
```

### The 5 Canonical Logistics Tools
Strands Decider 2B strictly selects among the five predefined logistics actions:
1. `dispatch_replacement`: Vehicle immobilized; dispatch local backup unit within range to rendezvous and transfer freight.
2. `reroute_shipment`: Vehicle operational; adjust route coordinates to bypass congestion, weather, or road closures.
3. `crossdock`: Vehicle has minor mechanical issue or perishable cargo; divert to nearest logistics warehouse cross-dock bay.
4. `escalate_to_human`: Vehicle disabled with NO backup units in range, or critical SLA breach risk. Mandates immediate human dispatcher intervention.
5. `continue_monitoring`: Telemetry, speeds, and ETA nominal; maintain automated heartbeat monitoring without intervention.

---

## ⚠️ Important Disclosure: Amazon Bedrock & AWS Free Tier Status

- **AWS Account Free Tier Blocker**: The developer AWS account is on the Free tier. When invoking Amazon Bedrock Nova Micro, AWS returns:
  ```text
  ValidationException: Operation not allowed
  ```
  This is an AWS account verification restriction. Do **not** upgrade this account or incur cloud charges.
- **Bedrock Disabled by Default**: To guarantee zero unexpected AWS requests and zero application crashes, Amazon Bedrock reasoning is **disabled by default** (`CAPACITYX_ENABLE_BEDROCK=false`).
- **Offline Mocking for Tests Only**: The codebase includes an offline mock mode (`CAPACITYX_BEDROCK_MODE=mock`) used **strictly for local development and offline unit testing**. All mocked outputs are explicitly labeled:
  ```json
  "is_simulated": true,
  "disclaimer": "SIMULATED: Mock Bedrock operational reasoning for offline testing only. Not live AWS inference."
  ```
  **A mock is never represented as live Bedrock inference.**
- **Safe Fallback & Zero Fabrication**: When Bedrock is unavailable, blocked by AWS account limits, or times out, the system safely records `status: "blocked_unverified"` or `status: "disabled"`, completely preserves the Strands Decider 2B primary decision, and never fabricates confidence scores.

---

## 📁 Project Directory Structure

```text
CapacityX/
├── .gitignore                   # Ignores venvs, node_modules, build artifacts & secrets (.env)
├── .env.example                 # Committed configuration template (placeholders only)
├── .env                         # Local private environment configuration (git-ignored)
├── README.md                    # Root project documentation (this file)
│
├── frontend/                    # React + Vite Web Application
│   ├── index.html               # Main HTML entry with typography & metadata
│   ├── package.json             # NPM scripts and dependencies
│   ├── vite.config.js           # Vite dev server configuration (loads root .env safely)
│   └── src/
│       ├── App.jsx              # Main Logistics Dashboard & Outcome Panel
│       ├── index.css            # Custom Vanilla CSS design system & command center theme
│       └── main.jsx             # React DOM root render
│
├── backend/                     # Python FastAPI Backend
│   ├── requirements.txt         # FastAPI, Uvicorn, Boto3, Pydantic dependencies
│   ├── README.md                # Backend specific guidance
│   └── app/
│       ├── __init__.py          # Package initializer
│       ├── main.py              # FastAPI app with /health, /api/decide, /api/scenarios
│       └── models.py            # Pydantic data schemas for incidents, outcomes, and Bedrock validation
│
└── patterns/                    # Reusable Decision & Orchestration Patterns
    ├── README.md                # Orchestration pattern guide & Bedrock setup instructions
    ├── orchestrator.py          # CapacityXOrchestrator pairing Strands Decider 2B & Bedrock Nova Micro
    ├── strands_decider_2b.py    # Strands Decider 2B HTTP client, 5 tools, & safe fallback
    ├── bedrock_nova_micro.py    # Amazon Bedrock Converse API adapter for amazon.nova-micro-v1:0
    ├── prompt_templates.py      # Few-shot prompts for compact models
    ├── decision_schema.json     # Standardized JSON Schema contract
    └── tests/                   # Comprehensive offline unit test suite (47 tests)
        ├── test_strands_decider.py      # 24 tests: Strands Decider 2B, thresholds, auth, hosted CloudFront API
        ├── test_bedrock_nova_micro.py   # 14 tests: Bedrock adapter & error handling
        └── test_orchestrator.py         # 9 tests: Orchestrator, mock, safety flags, & fallbacks
```

---

## 🧪 Comprehensive Offline Test Suite (47 Tests)

All tests are completely offline and make zero live cloud or external network calls.

### Run All 47 Tests:
```powershell
.\backend\venv\Scripts\python.exe -m unittest discover -s patterns/tests -p "test_*.py" -v
```

### Expected Result:
```text
Ran 47 tests in ~8.50s
OK
```

### Test Suites Breakdown:
1. **`test_strands_decider.py` (24 tests)**:
   - Tool definitions and question schema matching Strands Decider specification.
   - Exact 0.70 confidence threshold verification (calibrated boundary).
   - Independent `human_review` signal trigger (`noul >= 0.5`).
   - Hosted CloudFront API (`https://da25tnxy06gjk.cloudfront.net`) integration and Bearer auth.
   - Authenticated ngrok HTTP Basic Auth tunnel headers (`ngrok-skip-browser-warning`).
   - Safe fallback when inference server is offline.
   - Secret hygiene: `repr` and `str` never expose passwords, tokens, or API keys.
2. **`test_bedrock_nova_micro.py` (14 tests)**:
   - Bedrock Converse API request formatting (`messages`, `system`, `inferenceConfig`).
   - Rejection of invalid/unrecognized action choices triggering safe fallback.
   - Malformed/non-JSON output handling.
   - AWS SDK permission errors (e.g., `AccessDeniedException`) handled safely.
   - Finite timeout handling with retries disabled (`max_attempts: 0`).
   - Zero confidence fabrication (`model_confidence: null`, `alternatives: {}`).
3. **`test_orchestrator.py` (9 tests)**:
   - Strands primary + Bedrock disabled by default.
   - Mock Bedrock operational reasoning when mock mode is enabled.
   - Bedrock operational risk flags independently enforcing `human_review_required: true`.
   - AWS Free tier `ValidationException: Operation not allowed` handled without crashing.
   - Read timeout handled safely with Strands decision preserved.
   - Retention of all 5 canonical logistics actions across all execution paths.
   - Zero confidence fabrication and honest health status disclosure.

---

## 🚀 How to Run the Project Locally

### Step 0: Initialize Root Configuration
CAPACITYX uses a single consolidated `.env` file at the project root for all local configuration:
```powershell
# In project root: D:\Antigravity_Projects\CapacityX
Copy-Item .env.example .env
```
Open `.env` in your editor and optionally set your `DECIDER_API_KEY` for hosted Strands Decider mode.

### Terminal 1: Run the FastAPI Backend
```powershell
cd d:\Antigravity_Projects\CapacityX\backend
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8081 --reload
```
- **Health Check**: [http://127.0.0.1:8081/health](http://127.0.0.1:8081/health)
- **Interactive Swagger Docs**: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)
- **Decision Endpoint**: `POST http://127.0.0.1:8081/api/decide`

### Terminal 2: Run the React + Vite Frontend
```powershell
cd d:\Antigravity_Projects\CapacityX\frontend
npm run dev
```
Open your browser at **`http://localhost:5173`**.

---

## ⚙️ Configuration & Environment Variables

All settings are configured in the single root `.env` file. Vite and FastAPI load from this file automatically.

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://127.0.0.1:8081` | Public API base URL used by the React/Vite dashboard to reach FastAPI |
| `DECIDER_API_KEY` | *(optional)* | Private team API key for hosted Strands Decider API. Kept strictly on backend. |
| `DECIDER_MODE` | `ugmdu` | Mode selector (e.g., `ugmdu` for hosted CloudFront API) |
| `DECIDER_URL` | `https://da25tnxy06gjk.cloudfront.net` | Base URL for hosted Strands Decider API |
| `STRANDS_DECIDER_URL` | `http://127.0.0.1:8099` | URL for local or ngrok Strands Decider server (used when `DECIDER_API_KEY` is unset) |
| `STRANDS_DECIDER_USERNAME` | *(optional)* | HTTP Basic Auth username for local/ngrok server |
| `STRANDS_DECIDER_PASSWORD` | *(optional)* | HTTP Basic Auth password for local/ngrok server |
| `CAPACITYX_ENABLE_BEDROCK` | `false` | Enable or disable Bedrock operational reasoning |
| `CAPACITYX_BEDROCK_MODE` | `disabled` | Bedrock operational mode: `disabled`, `mock`, or `live` |
| `BEDROCK_MODEL_ID` | `amazon.nova-micro-v1:0` | Amazon Bedrock model ID |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock Runtime |
| `AWS_ACCESS_KEY_ID` | *(optional)* | AWS access key (only used in live mode) |
| `AWS_SECRET_ACCESS_KEY` | *(optional)* | AWS secret access key (only used in live mode) |

---

## ☁️ How to Enable Live Bedrock (Only After Account Verification)

Once your AWS account has completed verification and Amazon Bedrock access is granted:

### 1. Prerequisites in AWS Console
1. Navigate to **Amazon Bedrock** in the `us-east-1` region.
2. Under **Bedrock Configurations** &gt; **Model access**, ensure **Amazon Nova Micro** (`amazon.nova-micro-v1:0`) is marked as **Access granted**.
3. Ensure your IAM user/role has permission:
   ```json
   {
     "Effect": "Allow",
     "Action": "bedrock:InvokeModel",
     "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"
   }
   ```

### 2. Configure Environment Variables in PowerShell
```powershell
$env:CAPACITYX_ENABLE_BEDROCK = "true"
$env:CAPACITYX_BEDROCK_MODE = "live"
$env:AWS_ACCESS_KEY_ID = "YOUR_AWS_ACCESS_KEY"
$env:AWS_SECRET_ACCESS_KEY = "YOUR_AWS_SECRET_KEY"
$env:AWS_REGION = "us-east-1"
$env:BEDROCK_MODEL_ID = "amazon.nova-micro-v1:0"
```

### 3. Restart the Backend
Restart the FastAPI backend in Terminal 1. The `/health` endpoint will reflect `amazon_bedrock.status: "live"`.

> **Security Warning**: Never commit AWS credentials to Git or include them in `.env` files tracked by source control.

---

## 🛡️ Human-Review Safeguards & Operational Safety

1. **Simulation Mode Only**: All execution items in `outcome.actions_preview` are flagged with `is_live_action: false` and `status: "PREVIEW (Withheld: Simulation Mode)"`.
2. **Mandatory Human Review**:
   - Triggered when Strands Decider confidence is `< 0.70`.
   - Triggered when Strands human-review signal `noul >= 0.50`.
   - Triggered when operating in Safe Fallback mode.
   - Triggered when Bedrock operational check detects logistical hazards (e.g., severe SLA penalty or multi-vehicle collisions).
3. **Zero Automated Cloud Actions**: No real-world fleet dispatches, carrier locks, or SMS/email notifications occur.
