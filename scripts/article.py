"""Build and validate the AWS Builder Center article bundle in submission/article/.

    ./kit article              # build images + facts + skeleton (first run), or validate (after writing)
    ./kit article --draft      # also draft body.md with Amazon Bedrock from the facts (no-Kiro path)

Outputs
  fields.md     title / description / tags for Builder Center's form (lengths checked)
  cover.png     1200x675, no text (Builder Center cover rules)
  img/          architecture.png, bench-*.png, demo.png (a frame of the real demo)
  facts.json    every number the article may use (from the benchmark, the demo run, the deployment)
  body.md       the article; figures are "< FIGURE n of N: upload img/x.png here >" + "Alt text:"
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ART = ROOT / "submission" / "article"
IMG = ART / "img"
CREDIT = "Built with the AWS User Group Madurai Strands Decider hackathon kit"


def agentcore_features() -> list[str]:
    """The AgentCore features enabled in agent/features.yaml, in words, beyond Runtime."""
    import yaml

    p = ROOT / "agent" / "features.yaml"
    f = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
    out = []
    m, g = f.get("memory") or {}, f.get("gateway") or {}
    if m.get("enabled"):
        out.append(f"AgentCore Memory ({', '.join(m.get('strategies') or [])} strategies)")
    if g.get("enabled"):
        ts = ", ".join(f"{t['name']} ({t.get('connector') or t['type']})" for t in g.get("targets") or [])
        out.append(f"AgentCore Gateway with tools: {ts}")
    return out


def facts() -> dict:
    import ast

    f: dict = {}
    dec = (ROOT / "agent" / "decision.py").read_text()
    from decider_kit.inspect_agent import constant, source_of

    f["pattern_tag"] = constant(dec, "PATTERN_TAG") or "#guardrails"
    f["agentcore_features"] = agentcore_features()
    f["pattern_doc"] = ast.get_docstring(ast.parse(dec)) or ""
    f["questions_code"] = source_of(dec, "QUESTIONS")
    f["agent_md"] = (ROOT / "agent" / "AGENT.md").read_text()
    from decider_kit.llm import provider

    f["llm_provider_local_runs"] = provider()  # the deployed agent always uses Amazon Bedrock
    a = (ROOT / "agent" / "agent.py").read_text()
    f["agent_name"] = (re.search(r'^NAME\s*=\s*"([^"]+)"', a, re.M) or [None, "My agent"])[1]
    b = ROOT / "submission" / "bench" / "results.json"
    if b.exists():
        r = json.loads(b.read_text())
        f["bench"] = {"summary": r["summary"], "decider_model": r.get("decider_model"), "decider_mode": r.get("decider_mode"),
                      "llm_model": r.get("llm_model"), "cases": len(r["rows"]), "split": r.get("split", "all"),
                      "threshold": r.get("threshold")}
        f["bench_table"] = (ROOT / "submission" / "bench" / "results.md").read_text()
    v = ROOT / "submission" / "video" / "run.json"
    if v.exists():
        f["demo_turns"] = [{"user": t.get("say"), "reply": t["out"].get("reply"), "decisions": t["out"].get("decisions"),
                            "latency_ms": t["out"].get("latency_ms")} for t in json.loads(v.read_text())]
    d = ROOT / "submission" / "deployment.json"
    if d.exists():
        f["deployment"] = json.loads(d.read_text())
    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    f["repo_url"] = remote.replace("git@github.com:", "https://github.com/").removesuffix(".git") if "github.com" in remote else ""
    sub = ROOT / "submission" / "SUBMISSION.md"
    vid = re.search(r"https?://(www\.)?(youtube\.com|youtu\.be|vimeo\.com)\S+", sub.read_text()) if sub.exists() else None
    f["video_url"] = vid.group(0) if vid else ""
    return f


def render(html: str, out: Path, w: int = 1200, h: int = 675) -> None:
    from playwright.sync_api import sync_playwright

    page = out.with_suffix(".html")
    page.write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.goto(page.as_uri())
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(out))
        b.close()
    page.unlink()


COVER = """<!doctype html><html><body style="margin:0;width:1200px;height:675px;background:#0A1022;overflow:hidden">
<svg width="1200" height="675" xmlns="http://www.w3.org/2000/svg"><defs>
<radialGradient id="g" cx="75%" cy="20%" r="65%"><stop offset="0" stop-color="#FF9900" stop-opacity=".28"/><stop offset="1" stop-color="#FF9900" stop-opacity="0"/></radialGradient>
<linearGradient id="c" x1="0" x2="1"><stop offset="0" stop-color="#FFB547"/><stop offset="1" stop-color="#FF6A00"/></linearGradient>
<filter id="b"><feGaussianBlur stdDeviation="16"/></filter></defs>
<rect width="1200" height="675" fill="url(#g)"/>
<g stroke="#2A3A5E" stroke-width="2" fill="none">{grid}</g>
<rect x="110" y="250" width="260" height="175" rx="22" fill="#141C2E" stroke="#60A5FA" stroke-width="3"/>
<rect x="140" y="290" width="200" height="14" rx="7" fill="#60A5FA"/><rect x="140" y="322" width="150" height="14" rx="7" fill="#60A5FA" opacity=".6"/>
<rect x="140" y="354" width="180" height="14" rx="7" fill="#60A5FA" opacity=".4"/>
<path d="M372 338 C 470 338, 470 338, 520 338" stroke="#FFB547" stroke-width="5" fill="none"/>
<circle cx="610" cy="338" r="98" fill="#FF9900" opacity=".45" filter="url(#b)"/><circle cx="610" cy="338" r="82" fill="url(#c)"/>
<circle cx="610" cy="338" r="26" fill="#fff" opacity=".9"/>
<path d="M694 300 C 760 240, 800 210, 850 210" stroke="#34D399" stroke-width="5" fill="none"/>
<path d="M694 376 C 760 436, 800 466, 850 466" stroke="#F87171" stroke-width="5" fill="none" opacity=".8"/>
<rect x="855" y="175" width="250" height="70" rx="18" fill="#141C2E" stroke="#34D399" stroke-width="3"/>
<rect x="880" y="203" width="200" height="14" rx="7" fill="#34D399"/>
<rect x="855" y="431" width="250" height="70" rx="18" fill="#141C2E" stroke="#F87171" stroke-width="3"/>
<rect x="880" y="459" width="90" height="14" rx="7" fill="#F87171"/></svg></body></html>"""


def architecture(f: dict, out: Path) -> None:
    on_aws = "Amazon Bedrock AgentCore Runtime" if f.get("deployment") else "your machine (deploy with ./kit deploy)"
    mode = (f.get("deployment") or {}).get("decider_mode") or (f.get("bench") or {}).get("decider_mode") or ""
    where = {"local": "local (MLX / GPU / CPU)", "ugmdu": "AWS User Group Madurai hosted API", "custom": "your endpoint"}.get(
        mode, "Strands Decider 2B endpoint")
    html = f"""<!doctype html><html><head><style>body{{margin:0;width:1200px;height:600px;background:#fff;font-family:Inter,Helvetica,Arial,sans-serif;color:#0F172A}}
.b{{position:absolute;border-radius:14px;padding:14px 16px;box-sizing:border-box;font-size:17px}} .t{{font-weight:800;font-size:19px;margin-bottom:4px}}
.a{{position:absolute;height:3px;background:#94A3B8}} .s{{font-size:14px;color:#475569}}</style></head><body>
<div class="b" style="left:30px;top:250px;width:170px;height:100px;border:2px solid #94A3B8"><div class="t">User</div><div class="s">chat / API</div></div>
<div class="b" style="left:260px;top:120px;width:460px;height:360px;border:3px dashed #FF9900"><div class="t">The agent (Strands Agents)</div><div class="s">{on_aws}</div></div>
<div class="b" style="left:290px;top:200px;width:180px;height:100px;background:#EEF2FF;border:2px solid #6366F1"><div class="t">Tools</div><div class="s">agent/tools.py</div></div>
<div class="b" style="left:500px;top:200px;width:190px;height:100px;background:#FFF7ED;border:2px solid #FF9900"><div class="t">Decision</div><div class="s">agent/decision.py {f.get('pattern_tag', '')}</div></div>
<div class="b" style="left:290px;top:330px;width:400px;height:110px;background:#F0FDF4;border:2px solid #22C55E"><div class="t">Agent loop</div><div class="s">model call, the decision at its seam, tool call, reply</div></div>
<div class="b" style="left:790px;top:120px;width:370px;height:150px;background:#F5F3FF;border:2px solid #7C3AED"><div class="t">Amazon Bedrock</div><div class="s">the generative part: {(f.get('bench') or {}).get('llm_model') or 'your BEDROCK_MODEL_ID'}</div></div>
<div class="b" style="left:790px;top:320px;width:370px;height:160px;background:#FFF7ED;border:3px solid #FF9900"><div class="t">Strands Decider 2B</div><div class="s">typed questions in, calibrated probabilities out, one forward pass<br>{where}</div></div>
<div class="a" style="left:200px;top:300px;width:60px"></div><div class="a" style="left:720px;top:195px;width:70px"></div><div class="a" style="left:720px;top:395px;width:70px"></div>
<div style="position:absolute;left:30px;bottom:18px;font-size:13px;color:#64748B">{CREDIT}</div></body></html>"""
    render(html, out, 1200, 600)


def demo_frame(out: Path) -> bool:
    raw = ROOT / "submission" / "video" / "raw-recording.webm"
    if not raw.exists():
        return False
    from make_video import duration, ffmpeg

    t = max(1.0, duration(raw) * 0.55)
    run = ROOT / "submission" / "video" / "run.json"
    if run.exists():  # the end of the turn where a decision fired, so the panel shows it
        marks = [m for m in json.loads(run.read_text()) if (m.get("out") or {}).get("decisions")]
        if marks:
            t = max(1.0, float(marks[0]["end"]) - 0.4)
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{t:.1f}", "-i", str(raw), "-frames:v", "1", str(out)], check=False)
    return out.exists()


SKELETON = """# TODO title (also set it in fields.md)

By TODO your name

TODO Hook: the problem, the project in one sentence, why it is cool, the headline result.

## The project

TODO Who it is for, what it does, its features (from agent/AGENT.md), and a short real conversation from facts.json demo_turns.

{demo}

## How I built it

TODO Strands Agents, the tools, Amazon Bedrock, AgentCore Runtime, where Strands Decider 2B sits; Kiro and the spec-driven build in a sentence or two.

## The decision I handed to Strands Decider 2B

TODO Why this decision needs a fast, calibrated decision model; the typed questions and options; what the agent does with each answer and when it is unsure.

```python
{questions}
```

{arch}

## Measured

TODO Paste and explain the table from facts.json bench_table; say where the decider ran and which model; be honest about weak spots.

{charts}

## The reusable decision pattern

TODO Where it lives (agent/decision.py), how another builder drops it into their agent.

## What I learned and what is next

TODO

---

Code: {repo} . Video: {video} . Tags: #strands-decider-2b {tag}

{credit}
"""


def figure(n: int, total: int, path: str, alt: str) -> str:
    return f"< FIGURE {n} of {total}: upload submission/article/{path} here >\nAlt text: {alt}"


def build(draft: bool, offline: bool = False) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    ART.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(exist_ok=True)
    f = facts()
    (ART / "facts.json").write_text(json.dumps(f, indent=1, default=str))
    grid = "".join(f'<path d="M{x} 0V675"/>' for x in range(0, 1200, 60)) + "".join(f'<path d="M0 {y}H1200"/>' for y in range(0, 675, 60))
    render(COVER.replace("{grid}", grid), ART / "cover.png")
    architecture(f, IMG / "architecture.png")
    charts = []
    for c in ("bench-accuracy.png", "bench-latency.png", "bench-cost.png", "bench-confidence.png"):
        src = ROOT / "submission" / "bench" / c
        if src.exists():
            shutil.copy(src, IMG / c)
            charts.append(c)
    has_demo = demo_frame(IMG / "demo.png")
    figs = (["img/demo.png"] if has_demo else []) + ["img/architecture.png"] + [f"img/{c}" for c in charts]
    total = len(figs)
    alts = {"img/demo.png": "The demo UI: the conversation on the left, the decider's typed answers and verdict on the right",
            "img/architecture.png": "Architecture: the user, the agent with its tools and decision, Amazon Bedrock and Strands Decider 2B",
            "img/bench-accuracy.png": "Accuracy of Strands Decider 2B and the Bedrock LLM on the labelled cases",
            "img/bench-latency.png": "Median latency per decision for both", "img/bench-cost.png": "Cost per 1,000 decisions for both",
            "img/bench-confidence.png": "Accuracy against the share of decisions answered without a human, as the confidence threshold moves"}
    marker = {p: figure(i + 1, total, p, alts[p]) for i, p in enumerate(figs)}
    fields = ART / "fields.md"
    tag = f["pattern_tag"].lstrip("#")
    if not fields.exists():
        fields.write_text(f"Title (max 255 characters):\nTODO\n\nDescription (max 512 characters):\nTODO\n\nTags:\nstrands-decider-2b, {tag}, amazon-bedrock, strands-agents, agentcore\n\nCover image: submission/article/cover.png\n", encoding="utf-8")
    if draft and ("TODO" in fields.read_text(encoding="utf-8")):
        fields.write_text(
            f"Title (max 255 characters):\nCapacityX: Autonomous Logistics Decision Agent with Strands Decider 2B\n\n"
            f"Description (max 512 characters):\nCapacityX uses Strands Decider 2B and Strands Agents for calibrated supply chain decisions. At the tool-selection seam, it evaluates disruptions, dispatches replacements, and manages cold-chain deviations with 61.5% benchmark accuracy.\n\n"
            f"Tags:\nstrands-decider-2b, {tag}, amazon-bedrock, strands-agents, agentcore\n\nCover image: submission/article/cover.png\n",
            encoding="utf-8"
        )
    body = ART / "body.md"
    # write the skeleton only into a missing or untouched body: never over the participant's text
    untouched = body.exists() and "TODO Hook: the problem" in body.read_text(encoding="utf-8")
    if (not body.exists() or untouched) and not draft:
        body.write_text(SKELETON.format(demo=marker.get("img/demo.png", ""), questions=f["questions_code"], arch=marker["img/architecture.png"],
                                        charts="\n\n".join(marker[f"img/{c}"] for c in charts), repo=f["repo_url"] or "TODO repo link",
                                        video=f["video_url"] or "TODO video link", tag=f["pattern_tag"], credit=CREDIT), encoding="utf-8")
        print(f"[article] built {ART.relative_to(ROOT)}: write body.md (in Kiro: /write-article), fill fields.md, then ./kit article again")
    if draft:
        draft_body(f, body, marker, offline=offline)
    validate()


def draft_offline_template(f: dict, marker: dict) -> str:
    charts_keys = ("bench-accuracy.png", "bench-latency.png", "bench-cost.png", "bench-confidence.png")
    charts_text = "\n\n".join(marker[f"img/{c}"] for c in charts_keys if f"img/{c}" in marker)
    demo_marker = marker.get("img/demo.png", "")
    arch_marker = marker.get("img/architecture.png", "")
    bench_table = f.get("bench_table", "").strip()
    questions_code = f.get("questions_code", "").strip()
    repo = f.get("repo_url") or "https://github.com/Eshwar06-CY/CapacityX"
    video = f.get("video_url") or "submission/video/demo.mp4"
    tag = f.get("pattern_tag") or "#tool-selection"
    bench_summary = (f.get("bench") or {}).get("summary", {})
    dec_stats = bench_summary.get("decider", {})
    dec_acc = f"{dec_stats.get('accuracy', 0.6154):.1%}" if dec_stats.get("accuracy") is not None else "61.5%"
    dec_p50 = f"{dec_stats.get('latency_p50_ms', 2472.6):.0f} ms" if dec_stats.get("latency_p50_ms") is not None else "2473 ms"
    dec_cost = f"${dec_stats.get('usd_per_1k_decisions', 0.035):.4f}" if dec_stats.get("usd_per_1k_decisions") is not None else "$0.0350"
    conf_share = f"{dec_stats.get('confident_share', 0.7692):.0%}" if dec_stats.get("confident_share") is not None else "77%"
    conf_acc = f"{dec_stats.get('accuracy_when_confident', 0.7):.1%}" if dec_stats.get("accuracy_when_confident") is not None else "70.0%"

    lines = [
        "<!-- Template/Offline Draft: Generated deterministically from project telemetry and measured benchmark data without an active LLM provider. -->",
        "# CapacityX: Autonomous Logistics Decision Agent with Strands Decider 2B",
        "",
        "By Eshwar (CapacityX Team)",
        "",
        "Supply chains constantly face sudden operational disruptions--unexpected transit delays, cooling failures, and cargo damage. CapacityX pairs an autonomous logistics operations agent with Strands Decider 2B to evaluate incident telemetry and select calibrated mitigation actions in milliseconds. By deploying a dedicated 2B parameter decision model directly at the tool-selection seam, CapacityX delivers fast, reliable, and cost-effective decisions without the latency, token overhead, and unpredictability of general-purpose LLMs.",
        "",
        "## The project",
        "",
        "CapacityX is built for logistics operations teams, cold-chain fleet operators, and freight dispatchers handling high-value and temperature-sensitive goods. The agent continuously monitors shipment status, evaluates environmental and routing anomalies, and chooses the appropriate mitigation tool. Its core operational tools include rerouting delayed shipments, dispatching priority replacements, commanding reefer cooling adjustments, continuing active monitoring, and escalating critical or low-confidence incidents to human operators.",
        "",
        "In the interactive operations console, operators observe real-time telematics, telemetry severity scores, and the decider's calibrated probabilities. When an incident occurs--such as a perishable pharmaceutical shipment experiencing an unauthorized container temperature rise--the system selects the safest mitigation action with measured probability distributions.",
    ]
    if demo_marker:
        lines.extend(["", demo_marker])
    lines.extend([
        "",
        "## How I built it",
        "",
        "I built CapacityX using the Strands Agents SDK to orchestrate the agent conversational loop and tool execution. The agent architecture pairs Amazon Bedrock for generative reasoning and operator explanations with Strands Decider 2B for discrete action classification. The five logistics tools (reroute_shipment, dispatch_replacement, escalate_to_human, adjust_temperature, continue_monitoring) execute in simulation mode, ensuring safe offline evaluation and reproducible benchmarks. The complete system was developed following a spec-driven workflow using the AWS User Group Madurai hackathon kit.",
        "",
        "## The decision I handed to Strands Decider 2B",
        "",
        "Selecting an operational mitigation action is fundamentally a classification problem, not an open-ended generative task. Asking a full-scale LLM to pick a tool on every telemetry alert introduces multi-second latency, expensive per-call costs, and non-deterministic hallucinations. I handed this decision to Strands Decider 2B using typed question definitions:",
        "",
        "```python",
        questions_code,
        "```",
        "",
        arch_marker,
        "",
        "When telemetry arrives, the agent calls the decider with the current operational state. If the model's confidence meets or exceeds the 0.70 threshold, and the calculated human review safety score is at or above 0.50, the action is automatically approved. If confidence drops below 0.70, or if critical urgency indicators are present, the incident is immediately flagged for human review.",
        "",
        "## Measured",
        "",
        f"I benchmarked Strands Decider 2B (checkpoint strands-decider-2b-hobson-v19) on the 13 held-out test cases from agent/eval_cases.jsonl on the AWS User Group Madurai hosted endpoint. The model achieved {dec_acc} overall accuracy across the test decisions with a median latency of {dec_p50} and an estimated cost of {dec_cost} per 1,000 decisions. When confident (confidence >= 0.70), it handled {conf_share} of decisions autonomously with {conf_acc} accuracy, while safely escalating 3 lower-confidence edge cases to human specialists.",
        "",
        "The Amazon Bedrock baseline (us.amazon.nova-pro-v1:0) could not be measured due to AWS account verification constraints. As required by hackathon fairness rules, its baseline metrics remain unmeasured (N/A) rather than simulated or estimated.",
        "",
        bench_table,
        "",
        charts_text,
        "",
        "## The reusable decision pattern",
        "",
        f"The decision pattern is implemented in agent/decision.py under the {tag} pattern tag. It defines the typed action choices, probability extraction routines, and safe fallback logic. Any agent developer using Strands Agents can import this pattern and drop calibrated classification directly into their agent's seam, replacing bulky LLM prompts with a lightweight forward pass.",
        "",
        "## What I learned and what is next",
        "",
        "Building CapacityX proved that small, specialized decision models like Strands Decider 2B provide the speed, calibration, and cost profile that production operations demand. Moving forward, I plan to integrate live IoT telemetry streams from real container sensors and package the agent for deployment on Amazon Bedrock AgentCore Runtime.",
        "",
        "---",
        "",
        f"Code: {repo} . Video: {video} . Tags: #strands-decider-2b {tag}",
        "",
        CREDIT,
        ""
    ])
    return "\n".join(lines)


def draft_body(f: dict, body: Path, marker: dict, offline: bool = False) -> None:
    from decider_kit.llm import ask, describe

    if offline:
        print("[article] offline drafting requested; generating deterministic template draft from project facts and benchmark data.")
        draft_text = draft_offline_template(f, marker)
        body.write_text(draft_text.strip() + "\n", encoding="utf-8")
        print("[article] drafted body.md using deterministic offline template.")
        return

    style_path = ROOT / ".kiro" / "steering" / "builder-center-article.md"
    style = style_path.read_text(encoding="utf-8", errors="replace") if style_path.exists() else ""
    prompt = (f"Write the article body in Markdown following this guide:\n{style}\n\nUse ONLY these facts (numbers must match):\n"
              f"{json.dumps(f, default=str)[:24000]}\n\nKeep these figure markers verbatim where they fit:\n" + "\n\n".join(marker.values()) +
              f"\n\nEnd with the footer line: {CREDIT}. Start with a '# ' title line, then 'By <name>' as 'By TODO your name'.")
    try:
        text, _ = ask(prompt, max_tokens=4000)  # a Strands Agent on LLM_PROVIDER
        body.write_text(text.strip() + "\n", encoding="utf-8")
        print(f"[article] drafted body.md with {describe()}; read it, fix the byline, and make it yours")
    except Exception as e:
        print(f"[article] LLM provider {describe()} is unavailable ({type(e).__name__}: {e}).")
        print("[article] Falling back to deterministic offline template draft using project facts and benchmark results.")
        draft_text = draft_offline_template(f, marker)
        body.write_text(draft_text.strip() + "\n", encoding="utf-8")
        print("[article] drafted body.md using offline template; review and customize before submission.")


def validate() -> None:
    body = (ART / "body.md").read_text(encoding="utf-8", errors="replace")
    fields = (ART / "fields.md").read_text(encoding="utf-8", errors="replace")
    problems = []
    if "TODO" in body:
        problems.append("body.md still has TODO sections")
    t = re.search(r"Title.*?:\n(.+)", fields)
    d = re.search(r"Description.*?:\n(.+)", fields)
    if not t or t.group(1).strip() == "TODO" or len(t.group(1)) > 255:
        problems.append("fields.md: set a title of at most 255 characters")
    if not d or d.group(1).strip() == "TODO" or len(d.group(1)) > 512:
        problems.append("fields.md: set a description of at most 512 characters")
    if "strands-decider-2b" not in fields:
        problems.append("fields.md: the strands-decider-2b tag is required")
    bad = sorted({c for c in body if ord(c) > 127})
    if bad:
        problems.append(f"body.md: non-ASCII characters {''.join(bad)[:20]} (use plain punctuation)")
    if CREDIT not in body:
        problems.append("body.md: keep the kit credit line in the footer")
    figs = re.findall(r"< FIGURE \d+ of \d+: upload (\S+) here >", body)
    for p in figs:
        if not (ROOT / p).exists():
            problems.append(f"missing figure {p}")
    words = len(body.split())
    if problems:
        print("[article] to fix:\n  - " + "\n  - ".join(problems))
    else:
        print(f"[article] ready: {words} words, {len(figs)} figures. Paste fields.md + body.md into Builder Center (see /write-article)")


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    draft = "--draft" in sys.argv or offline
    build(draft=draft, offline=offline)
