# CapacityX — Backend Service (FastAPI)

Welcome! This is the backend API service for **CAPACITYX — Smart Logistics Decision Agent**.

## Overview
- Framework: **FastAPI** + **Uvicorn**
- Purpose: Provides health check monitoring and temporary mock decision endpoints for local UI dashboard development.
- **Note on AI & Cloud**: The current mock endpoints (`/api/mock-decision`) use lightweight local Python heuristics for testing. They **do not** invoke Amazon Bedrock or deploy cloud resources.

---

## Quickstart (Local PowerShell)

### 1. Open PowerShell and navigate to `backend/`
```powershell
cd d:\Antigravity_Projects\CapacityX\backend
```

### 2. Create and activate a Python virtual environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If your PowerShell execution policy restricts scripts, you can run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Start the FastAPI development server
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8081
```

The backend will be live at:
- Interactive Swagger API Docs: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)
- Health check: [http://127.0.0.1:8081/health](http://127.0.0.1:8081/health)
- Mock Decision Endpoint: `POST http://127.0.0.1:8081/api/mock-decision`

---

## Endpoints Summary

| Method | Endpoint | Purpose | AI / Cloud Status |
|---|---|---|---|
| `GET` | `/health` or `/api/health` | Service health status check | Local status check |
| `GET` | `/api/scenarios` | Predefined test scenarios | Static scenario data |
| `POST` | `/api/mock-decision` | Triage decision simulation for UI | **Temporary Local Mock** (No Bedrock / Cloud) |
| `GET` | `/docs` | Interactive OpenAPI documentation | Auto-generated Swagger |
