# CAPACITYX — Smart Logistics Decision Agent

> **Hackathon Prototype** | Real-time incident triage and automated logistics decision-making with SLA protection.

Welcome to **CapacityX**! This project provides an intelligent decision system for logistics disruptions (e.g. vehicle breakdowns, tight delivery deadlines, and replacement dispatching), keeping customers informed while safeguarding SLAs.

---

## ⚠️ Important Note on AI Decision Architecture & Local Hardware

- **Strands Decider 2B Integration**: Implemented according to the official [Strands Labs specification](https://github.com/strands-labs/strands-decider) using 5 typed tools (`dispatch_replacement`, `reroute_shipment`, `crossdock`, `escalate_to_human`, `continue_monitoring`).
- **Real Model vs Safe Fallback**: When the official `strands-decider` server is running (at `STRANDS_DECIDER_URL` or `http://127.0.0.1:8099`), real calibrated probabilities and choices are evaluated. When the model server is offline or confidence is below threshold (70%), a transparent, deterministic **Safe Fallback Policy** engages and flags `human_review_required: true`.
- **Local Windows Hardware Constraint**: The full 2B model weights (~4.5 GB) cannot be loaded in-process into RAM on this machine without risk of crashing due to memory limits (~2.3 GB free RAM) and absence of a CUDA GPU. An external GPU server or WSL2 environment is recommended for live inference serving.
- **Zero Cloud Deployment**: No AWS credentials, Bedrock APIs, or cloud resources are invoked in this task.

---

## 📁 Project Directory Structure

```text
CapacityX/
├── .gitignore                   # Ignores venvs, node_modules, build artifacts & secrets
├── README.md                    # Root project documentation (this file)
│
├── frontend/                    # React + Vite Web Application
│   ├── index.html               # Main HTML entry with typography & metadata
│   ├── package.json             # NPM scripts and dependencies
│   ├── vite.config.js           # Vite dev server configuration
│   └── src/
│       ├── App.jsx              # Main Logistics Dashboard & Outcome Panel
│       ├── index.css            # Custom Vanilla CSS design system & command center theme
│       └── main.jsx             # React DOM root render
│
├── backend/                     # Python FastAPI Backend
│   ├── requirements.txt         # FastAPI & Uvicorn dependencies
│   ├── README.md                # Backend specific guidance
│   └── app/
│       ├── __init__.py          # Package initializer
│       ├── main.py              # FastAPI app with /health, /api/mock-decision, /api/scenarios
│       └── models.py            # Pydantic data schemas for incidents and outcomes
│
└── patterns/                    # Reusable Decision Patterns
    ├── README.md                # Strands Decider 2B pattern guide & AWS integration path
    ├── strands_decider_2b.py    # Python implementation & Bedrock template
    ├── prompt_templates.py      # Few-shot prompts optimized for 2B compact models
    └── decision_schema.json     # Standardized JSON Schema contract
```

---

## 🚀 How to Run the Project Locally (Exact PowerShell Commands)

You will need two PowerShell terminal windows: one for the backend and one for the frontend.

### Terminal 1: Run the FastAPI Backend

Open a PowerShell window in the project root:

```powershell
# 1. Navigate to the backend directory
cd d:\Antigravity_Projects\CapacityX\backend

# 2. (Optional but recommended) Create and activate a Python virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install required Python packages (FastAPI, Uvicorn, Pydantic)
pip install -r requirements.txt

# 4. Start the FastAPI backend server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8081
```

Once running, access:
- **Health Check**: [http://127.0.0.1:8081/health](http://127.0.0.1:8081/health)
- **Interactive API Swagger Docs**: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)
- **Temporary Mock Decision Endpoint**: `POST http://127.0.0.1:8081/api/mock-decision`

---

### Terminal 2: Run the React + Vite Frontend

Open a second PowerShell window:

```powershell
# 1. Navigate to the frontend directory
cd d:\Antigravity_Projects\CapacityX\frontend

# 2. Start the Vite development server
npm run dev
```

Open your browser at the local Vite address (usually **`http://localhost:5173`**).

---

### Terminal 3 (Optional): Run Decision Pattern Unit Tests

```powershell
cd d:\Antigravity_Projects\CapacityX
python -m unittest patterns/tests/test_strands_decider.py
```
*Executes all 7 test cases validating question schema, typed tool selection, confidence thresholds, and safe fallback handling.*

---

## 🖥️ Dashboard Features

1. **Incident Telemetry Input**:
   - **Shipment ID** and **Target Delivery Deadline**
   - **Truck Breakdown Switch** with severity toggle (Low / Moderate / Critical)
   - **Replacement Vehicle Availability Switch** (Within 45-minute response radius)
   - **Customer Notified Switch** (Checks whether the customer has received prior notice)
   - Cargo classification (e.g. Cold-Chain, Pharmaceuticals, Standard Freight) and Route details
2. **Quick Scenario Presets**:
   - *Cold-Chain Breakdown (Replacement Ready)*: Simulates immediate dispatch to prevent temperature breach.
   - *Dry Freight Breakdown (No Replacement)*: Demonstrates emergency towing escalation and SLA breach alert.
   - *Urgent Medical (Normal Transit, Notified)*: Shows standard nominal monitoring.
3. **Outcome & Action Panel**:
   - Prominently labeled as **⚡ SIMULATION MODE**.
   - Clear banner stating the temporary mock status of the calculation.
   - **Simulated Results vs Real Actions Breakdown Table**: Clearly illustrates what a live agent would execute (e.g. AWS EventBridge, Amazon SNS, Fleet Dispatch) versus the fact that actions are withheld during simulation.
   - Live backend health status indicator with an on-demand re-ping button.
   - Expandable **Raw JSON Inspector** for contract verification.

---

## 🔮 Future AWS Integration Roadmap (When Ready)

When you are ready to expand this hackathon project into AWS cloud services:
1. **Amazon Bedrock**: Connect `patterns/strands_decider_2b.py` to an Amazon Bedrock foundation model (e.g., Nova Micro or lightweight SLMs) using `boto3`.
2. **AWS Lambda**: Wrap the decision pattern into serverless functions triggered by Amazon API Gateway or IoT Core telemetry.
3. **Amazon EventBridge**: Publish structured decision events to coordinate fleet management and customer notification microservices.
4. **Amazon SNS / Amazon SES**: Dispatch real-world SMS/Email notifications to customers when delivery delays are predicted.
