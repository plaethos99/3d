"""A stand-in for an OpenAI-compatible chat API, for testing agent.py's model mode without a key.

Usage: python pipeline/tests/fake_model_server.py <port>
It answers like a careful but simple model: it calls the next step of the standard plan, retries a
failed step once, then asks for a hand-off, and finishes at the end. It records every request it
receives in requests.jsonl next to this file, so a test can check what the agent sent.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PLAN = ["track_video", "inspect_clip", "clean_tracking", "retarget", "clear_arms", "render_motion", "score_motion"]
LOG = Path(__file__).parent / "requests.jsonl"


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"path": self.path, "auth_set": self.headers.get("Authorization", "") != "Bearer ",
                                "tools": [t["function"]["name"] for t in body.get("tools", [])],
                                "messages": len(body["messages"])}) + "\n")
        results = [json.loads(m["content"]) for m in body["messages"] if m["role"] == "tool"]
        done = {r["tool"] for r in results if "tool" in r and r.get("passed") is not False}
        failed = [r for r in results if r.get("passed") is False]
        run = "fake_model_test"
        if failed and failed[-1]["tool"] not in done:
            last = failed[-1]
            tries = sum(1 for r in failed if r["tool"] == last["tool"])
            call = ("handoff", {"tool": last["tool"], "why": "it failed twice"}) if tries >= 2 else (last["tool"], last["args"])
            if tries >= 3:
                call = ("finish", {"summary": f"stuck on {last['tool']}"})
        else:
            remaining = [t for t in PLAN if t not in done]
            if not remaining:
                call = ("finish", {"summary": f"{len(done)} steps done"})
            else:
                tool = remaining[0]
                args = {"run": run}
                if tool == "track_video":
                    args["video"] = "clips/pexels-5995594/real_walk_a.mp4"
                if tool == "retarget":
                    args["character"] = "tests/joyce_v2/output/joyce_v2_face.blend"
                call = (tool, args)
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{len(body['messages'])}", "type": "function",
            "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}
        data = json.dumps({"choices": [{"message": message}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
