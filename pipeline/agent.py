"""Make a character move like a video, with a model choosing the steps and Claude Code for hard cases.

Usage:
  python pipeline/agent.py --video <clip> --character <rigged.blend> --run <name> [options]
Options:
  --mode plan       run the standard steps in order, no model needed (default)
  --mode llm        a model picks each step, reads the scores and may retry with other settings.
                    Any OpenAI-compatible chat API works (DeepSeek, for example). Set, in your own shell:
                      LLM_API_KEY (your key), LLM_BASE_URL (default https://api.deepseek.com),
                      LLM_MODEL (default deepseek-chat)
  --escalate dry|on|off   hand stuck steps to Claude Code: dry only writes the report (default)
  --claude-budget 2       dollar cap per hand-off;  --max-steps 25;  --skip-render
Everything the run does is logged to pipeline/runs/<name>/log.jsonl, with a summary.json at the end.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import escalate as handoff  # noqa: E402
import tools  # noqa: E402

PLAN = ["track_video", "inspect_clip", "clean_tracking", "retarget", "clear_arms", "render_motion", "score_motion"]
STOPPERS = {"track_video", "inspect_clip", "retarget", "render_motion"}  # later steps need these to have worked
MEASURES = {"inspect_clip", "score_motion", "render_compare"}  # same input gives the same score: no point retrying
TRIES_BEFORE_HANDOFF = 2

parser = argparse.ArgumentParser()
parser.add_argument("--video", required=True)
parser.add_argument("--character", required=True)
parser.add_argument("--run", required=True)
parser.add_argument("--mode", choices=["plan", "llm"], default="plan")
parser.add_argument("--escalate", choices=["off", "dry", "on"], default="dry")
parser.add_argument("--claude-budget", type=float, default=2.0)
parser.add_argument("--max-steps", type=int, default=25)
parser.add_argument("--skip-render", action="store_true")
opt = parser.parse_args()
GOAL = (f"Animate the character {opt.character} with the motion of the performer in {opt.video}, "
        f"meeting the targets in pipeline/targets.json.")
run_dir = tools.run_dir(opt.run)
log_path = run_dir / "log.jsonl"
history = []


def log(entry):
    entry["time"] = time.strftime("%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def call(tool, args):
    print(f"-> {tool} {json.dumps(args)}", flush=True)
    result = tools.TOOLS[tool](**args)
    history.append(result)
    log({"event": "tool", **{k: v for k, v in result.items() if k != "log_tail"}})
    status = "PASS" if result["passed"] else ("ran" if result["passed"] is None and result["ok"] else "FAIL")
    print(f"   {status} in {result['seconds']} s  {json.dumps(result['metrics'])}  {result['failed_targets'] or ''}", flush=True)
    return result


def maybe_handoff(result, force=False):
    tries = sum(1 for h in history if h["tool"] == result["tool"])
    if result["passed"] is not False or (tries < TRIES_BEFORE_HANDOFF and not force) or opt.escalate == "off":
        return None
    out = handoff.escalate(GOAL, result, history, run_dir, opt.escalate, opt.claude_budget)
    log({"event": "handoff", **out})
    print(f"   handed to Claude Code ({out['mode']}): {out.get('verdict') or out.get('command')}", flush=True)
    return out


def default_args(tool):
    if tool == "track_video":
        return {"run": opt.run, "video": opt.video}
    if tool == "retarget":
        return {"run": opt.run, "character": opt.character}
    return {"run": opt.run}


# ---------------------------------------------------------------- plan mode: fixed order, no model
def run_plan():
    for tool in PLAN:
        if opt.skip_render and tool in ("render_motion", "score_motion"):
            continue
        result = call(tool, default_args(tool))
        if result["passed"] is False:
            if tool not in MEASURES:
                result = call(tool, default_args(tool))  # one plain retry (rendering and tracking can fail transiently)
            out = maybe_handoff(result, force=tool in MEASURES)
            if out and out.get("verdict", {}).get("fixed"):
                result = call(tool, default_args(tool))
        if not result["ok"] and tool in STOPPERS:
            print(f"Stopping: {tool} did not run, and later steps need it.")
            return


# ---------------------------------------------------------------- model mode
SYSTEM = f"""You run a 3D animation pipeline by calling tools. Goal: {GOAL}
Each tool returns JSON with "ok", "passed" and "failed_targets". Work like this:
1. The usual order is: {', '.join(PLAN)}. Use run name "{opt.run}".
2. If a step fails its targets, you may retry it once with different settings when the tool takes any
   (clear_arms margin_mm, track_video max_seconds). If it still fails, call handoff with what you think is wrong:
   a stronger assistant that can fix code will look at it. Don't call handoff for steps that passed.
3. Don't repeat a step that passed. Rendering is slow: never render twice without a reason.
4. When the targets are met, or nothing more can be done, call finish with a short summary of the scores."""
MODEL_TOOLS = [{"type": "function", "function": s} for s in tools.SCHEMAS] + [
    {"type": "function", "function": {"name": "handoff", "description": "Ask a stronger assistant to fix the scripts behind a failing step.",
                                      "parameters": {"type": "object", "properties": {"tool": {"type": "string"},
                                                                                      "why": {"type": "string"}}, "required": ["tool", "why"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End the run.",
                                      "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def chat(messages):
    base = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
    body = json.dumps({"model": os.environ.get("LLM_MODEL", "deepseek-chat"), "messages": messages,
                       "tools": MODEL_TOOLS, "tool_choice": "auto", "temperature": 0}).encode()
    request = urllib.request.Request(f"{base}/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {os.environ.get('LLM_API_KEY', '')}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["choices"][0]["message"]


def run_llm():
    if not os.environ.get("LLM_API_KEY"):
        sys.exit("Set LLM_API_KEY in your own shell first (it is never written to any file).")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Start."}]
    for step in range(opt.max_steps):
        reply = chat(messages)
        messages.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")})
        calls = reply.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content": "Call a tool, or finish."})
            continue
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"].get("arguments") or "{}")
            if name == "finish":
                log({"event": "finish", "summary": args.get("summary")})
                print("Model finished:", args.get("summary"))
                return
            if name == "handoff":
                last = next((h for h in reversed(history) if h["tool"] == args.get("tool")), history[-1] if history else None)
                content = handoff.escalate(GOAL + " Model's note: " + args.get("why", ""), last, history, run_dir,
                                           opt.escalate, opt.claude_budget) if last else {"error": "no such step has run"}
                log({"event": "handoff", **content})
            elif name in tools.TOOLS:
                if opt.skip_render and name in ("render_motion", "score_motion"):
                    content = {"tool": name, "skipped": "rendering is turned off for this run (--skip-render)"}
                else:
                    result = call(name, args)
                    content = {k: v for k, v in result.items() if k != "log_tail"}
                    if not result["ok"]:
                        content["log_tail"] = result["log_tail"]
                    out = maybe_handoff(result)
                    if out:
                        content["automatic_handoff"] = out
            else:
                content = {"error": f"unknown tool {name}"}
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(content)})
    print("Stopped: step limit reached.")


log({"event": "start", "goal": GOAL, "mode": opt.mode, "escalate": opt.escalate})
run_plan() if opt.mode == "plan" else run_llm()
summary = {"goal": GOAL, "steps": [{"tool": h["tool"], "ok": h["ok"], "passed": h["passed"], "metrics": h["metrics"],
                                     "failed_targets": h["failed_targets"]} for h in history],
           "all_passed": all(h["passed"] is not False for h in history) and bool(history)}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("SUMMARY", "all targets met" if summary["all_passed"] else "some targets not met", "->", run_dir / "summary.json")
