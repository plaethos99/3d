# Video-to-animation pipeline

Makes a rigged character move like a person in a video, checks the result against targets, and
hands any step it can't get right to Claude Code, which can fix the scripts.

## How it works

1. **Tools** (`tools.py`): each wraps one project script and returns scores plus pass or fail
   against `targets.json`:
   `track_video`, `inspect_clip`, `clean_tracking`, `retarget`, `clear_arms`, `render_motion`,
   `score_motion`, `render_compare`.
2. **The driver** (`agent.py`) runs those tools, in one of two modes:
   - `--mode plan`: the standard order, no model needed. A step that fails is retried once, and then handed off.
   - `--mode llm`: a cheap model chooses each step and its settings, reads the scores, and
     decides to retry, hand off, or finish. Any OpenAI-compatible chat API works, DeepSeek for example.
3. **Hand-off** (`escalate.py`): when a step fails twice, the driver writes a report (goal, scores,
   failed targets, log) and runs Claude Code headless in this folder:
   `claude -p <report> --output-format json --max-budget-usd 2 --permission-mode acceptEdits`.
   Claude Code fixes the script, re-runs the step with `run_tool.py` to confirm, and replies in JSON.
   One Claude Code session is kept (its id is in `claude_session.txt`), so it remembers earlier fixes.

## Running it

```
tools\pose-env\Scripts\python.exe -I pipeline\agent.py --video <clip.mp4> --character <rigged.blend> --run <name>
```

Options: `--mode plan|llm`, `--escalate dry|on|off` (default `dry`: only writes the hand-off report),
`--claude-budget 2` (dollars per hand-off), `--max-steps 25`, `--skip-render`.

To use a model, set these in your own shell. The key is never written to a file.

```
set LLM_API_KEY=<your key>
set LLM_BASE_URL=https://api.deepseek.com
set LLM_MODEL=<model name>
```

Run one tool on its own: `python pipeline\run_tool.py inspect_clip "{\"run\": \"dance\"}"`.

## Output

Everything goes to `pipeline/runs/<name>/`: `log.jsonl` (every step and hand-off), `summary.json`,
`motion.blend` and `motion.glb` (the animated character), `side_by_side.mp4`, and any `handoff_N.md` reports.

## Limits

- Rendering is the slow step: about 2–3 s per rendered frame for the current Joyce on this laptop's CPU.
  Long clips render every Nth frame.
- Only motion from video is wrapped so far. Building a character (head fit, eyes, hair, clothes,
  hands, face rig) is still a sequence of separate scripts.
- Scores come from a tracker that reads cartoon bodies less precisely than real people, so arm
  scores are rough.
