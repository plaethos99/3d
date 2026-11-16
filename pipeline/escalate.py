"""Hand a step the pipeline couldn't get right to Claude Code, which can read and fix the scripts.

The report says what was being made, which tool failed, its scores against its targets, the log,
and what's expected back. Claude Code runs headless in the project folder ("claude -p"), with a
spending cap and only file and command tools. One session is kept across hand-offs (its id is
saved), so Claude Code remembers earlier fixes.

Modes: "off" never calls it; "dry" writes the report and the command but doesn't run it;
"on" runs it.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SESSION_FILE = Path(__file__).parent / "claude_session.txt"

INSTRUCTIONS = """You are fixing a step of an automated 3D character pipeline in this project folder.
Rules:
- Find the cause in the scripts and fix it so it works for any character or video, not only this one.
  Don't just loosen targets.json to make a number pass, unless the target itself is clearly wrong; if so, say why.
- Confirm by re-running the failing tool: <python> pipeline/run_tool.py <tool> '<json args>'
  (python is tools/pose-env/Scripts/python.exe).
- Keep changes small. Don't download anything, don't touch files outside this folder, and don't change the
  BookShorts folder.
- Finish with one line of JSON: {"fixed": true|false, "summary": "...", "changed_files": [...], "rerun_passed": true|false}"""


def report(goal, result, history):
    lines = [
        f"Goal: {goal}",
        f"Failing tool: {result['tool']} with arguments {json.dumps(result['args'])}",
        f"Ran without error: {result['ok']}. Passed targets: {result['passed']}.",
        f"Failed targets: {result['failed_targets']}",
        f"Scores: {json.dumps(result['metrics'])}",
        f"Files: {json.dumps(result['files'])}",
        f"Tries so far on this tool: {sum(1 for h in history if h['tool'] == result['tool'])}",
        ("Note: this tool only measures the result. If its scores miss, the cause is in an earlier step "
         "(clean_tracking, retarget, clear_arms) or in the target itself." if result["tool"] in
         ("score_motion", "inspect_clip", "render_compare") else ""),
        "Log (last lines):", result["log_tail"],
        "Earlier steps this run: " + json.dumps([{"tool": h["tool"], "passed": h["passed"]} for h in history[-10:]]),
    ]
    return "\n".join(lines)


def escalate(goal, result, history, run_dir, mode="dry", budget_usd=2.0, model=None):
    text = report(goal, result, history)
    path = Path(run_dir) / f"handoff_{len(list(Path(run_dir).glob('handoff_*.md'))) + 1}.md"
    path.write_text(text, encoding="utf-8")
    cmd = ["claude", "-p", text, "--output-format", "json", "--max-budget-usd", str(budget_usd),
           "--permission-mode", "acceptEdits", "--allowedTools", "Read Edit Write Glob Grep Bash",
           "--append-system-prompt", INSTRUCTIONS]
    if model:
        cmd += ["--model", model]
    if SESSION_FILE.exists():
        cmd += ["--resume", SESSION_FILE.read_text(encoding="utf-8").strip()]
    if mode != "on":
        shown = " ".join(c if " " not in c else '"<...>"' for c in cmd)
        return {"handed_off": False, "mode": mode, "report": str(path), "command": shown}
    done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, encoding="utf-8", errors="replace",
                          timeout=3600, shell=False)
    answer = {}
    try:
        answer = json.loads(done.stdout)
    except json.JSONDecodeError:
        answer = {"result": done.stdout[-2000:], "error": done.stderr[-2000:]}
    if answer.get("session_id"):
        SESSION_FILE.write_text(answer["session_id"], encoding="utf-8")
    verdict = {}
    for line in reversed(str(answer.get("result", "")).splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                verdict = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    return {"handed_off": True, "mode": mode, "report": str(path), "cost_usd": answer.get("total_cost_usd"),
            "verdict": verdict, "raw_tail": str(answer.get("result", ""))[-800:]}
