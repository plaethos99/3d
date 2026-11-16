"""Run one pipeline tool from the command line and print its result as JSON.

Usage: python pipeline/run_tool.py <tool> '<json args>'
e.g.   python pipeline/run_tool.py inspect_clip '{"run": "dance"}'
Used by people, and by Claude Code when it has fixed a script and wants to confirm the step passes.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tools  # noqa: E402

if len(sys.argv) < 2 or sys.argv[1] not in tools.TOOLS:
    print("tools:", ", ".join(tools.TOOLS))
    sys.exit(2)
result = tools.TOOLS[sys.argv[1]](**(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
print(json.dumps(result, indent=1))
sys.exit(0 if result["ok"] else 1)
