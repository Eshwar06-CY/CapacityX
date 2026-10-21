import subprocess
import sys
from pathlib import Path

from fakes import FakeDecider

from decider_kit import bench

ROOT = Path(__file__).resolve().parent.parent


def test_bench_without_llm():
    qs = {"q": {"type": "noul", "instructions": "x?"}}
    cases = [{"state": "a", "expected": {"q": "yes"}}, {"state": "b", "expected": {"q": "no"}}]
    r = bench.run(cases, qs, None, decider=FakeDecider(lambda s, n, q: 0.9))
    s = r["summary"]["decider"]
    assert s["accuracy"] == 0.5 and s["decisions"] == 2 and "llm" not in r["summary"]
    assert "Strands Decider 2B" in bench.markdown(r)




def test_status_runs():
    out = subprocess.run([sys.executable, "scripts/status.py"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "progress" in out.stdout


def test_check_flags_kit_example():
    out = subprocess.run([sys.executable, "scripts/check_submission.py", "--only", "pattern"], cwd=ROOT, capture_output=True, text=True)
    src = (ROOT / "agent" / "decision.py").read_text()
    if "IS_KIT_EXAMPLE" in src:
        assert out.returncode == 1 and "kit example" in out.stdout


def test_demo_passes_attached_images_to_chat(monkeypatch):
    from fastapi.testclient import TestClient

    import demo.server as srv

    seen = {}

    class A:
        def chat(self, message, session_id="demo", images=None):
            seen.update(message=message, images=images)
            return {"reply": "ok", "decisions": [], "model_id": "m", "latency_ms": 1}

    class TextOnly:
        def chat(self, message, session_id="demo"):
            return {"reply": "ok", "decisions": [], "model_id": "m", "latency_ms": 1}

    c = TestClient(srv.app)
    monkeypatch.setattr(srv, "_agent", A())
    assert c.post("/api/chat", json={"message": "list this", "images": ["data:image/png;base64,AAAA"]}).status_code == 200
    assert seen == {"message": "list this", "images": ["data:image/png;base64,AAAA"]}
    monkeypatch.setattr(srv, "_agent", TextOnly())
    assert c.post("/api/chat", json={"message": "x", "images": ["AAAA"]}).status_code == 400
    assert c.post("/api/chat", json={"message": "x"}).status_code == 200


def test_bench_markdown_formats_missing_llm_as_na():
    r = {
        "summary": {
            "decider": {
                "accuracy": 0.8,
                "decisions": 10,
                "latency_p50_ms": 120.0,
                "latency_p95_ms": 250.0,
                "usd_per_1k_decisions": 0.035,
                "confident_share": 0.9,
                "accuracy_when_confident": 0.85,
            },
            "llm": {
                "accuracy": None,
                "decisions": 0,
                "latency_p50_ms": None,
                "latency_p95_ms": None,
                "usd_per_1k_decisions": None,
                "confident_share": None,
                "accuracy_when_confident": None,
            },
        },
        "threshold": 0.7,
        "rows": [{"expected": {"q": "yes"}}],
        "llm_model": "test-model",
    }
    md = bench.markdown(r)
    assert "| Accuracy | 80.0% | N/A |" in md
    assert "| Cost per 1,000 decisions | $0.0350 | N/A |" in md
    assert "$0.0000" not in md.split("| Cost per 1,000 decisions |")[1].splitlines()[0]


def test_bench_markdown_formats_genuine_zero_cost():
    r = {
        "summary": {
            "decider": {
                "accuracy": 0.8,
                "decisions": 10,
                "latency_p50_ms": 100.0,
                "latency_p95_ms": 200.0,
                "usd_per_1k_decisions": 0.0,
                "confident_share": 1.0,
                "accuracy_when_confident": 0.8,
            },
            "llm": {
                "accuracy": 0.75,
                "decisions": 10,
                "latency_p50_ms": 500.0,
                "latency_p95_ms": 800.0,
                "usd_per_1k_decisions": 0.0,
                "confident_share": 0.8,
                "accuracy_when_confident": 0.75,
            },
        },
        "threshold": 0.7,
        "rows": [{"expected": {"q": "yes"}}],
        "llm_model": "local-llm",
    }
    md = bench.markdown(r)
    assert "| Cost per 1,000 decisions | $0.0000 | $0.0000 |" in md


def test_check_measured_status_on_unmeasured_llm():
    out = subprocess.run([sys.executable, "scripts/check_submission.py", "--only", "measured"], cwd=ROOT, capture_output=True, text=True)
    assert "WARN" in out.stdout and "measured comparison" in out.stdout
    assert "LLM baseline could not be measured" in out.stdout
    assert out.returncode == 0

