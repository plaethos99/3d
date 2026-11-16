# AI-driven 3D character building and animation

Experiments toward a 3D tool where an AI does the work an artist would, inside real 3D software:
building, texturing, rigging and animating characters from reference images and videos, step by
step in Blender. Nothing is made with a mesh-generating AI model. Every shape comes from scripted
modelling, fitting to measurements, sculpted base meshes, and simulation, and every step is
checked against the references with numbers.

This repository holds code only. Tools, models, video clips, reference images and outputs aren't
included (see Setup).

## What's here

| Folder | What it does |
|---|---|
| `pipeline/` | Video-to-animation pipeline: tools with pass/fail targets, a driver that a cheap model (any OpenAI-compatible API) can run, and an automatic hand-off to Claude Code for steps it can't fix. See `pipeline/README.md`. |
| `tests/joyce/scripts/` | First character, built from code: modelling, textures, 22-bone rig, skin weights, poses, a walk cycle, and motion from video (tracking, label repair, retargeting, foot contact, rendering, scoring). |
| `tests/realhead/scripts/` | General tools: fit a sculpted base head to photos or frames, project skin colour, eyes and eyelids, ears and hair outline fitting, strand hair grooming, wrinkle depth, cloth simulation, garment shape fitting, hands with finger bones, opening the mouth, a face rig with expression shapes, measurement tools. |
| `tests/joyce_v2/scripts/` | Joining those tools on Joyce: new head on the old body, refitted hair, clothes detail. |

## Setup

The scripts expect this layout under the repository folder (paths are set at the top of each script
and in `pipeline/tools.py`):

| Path | What | Where to get it |
|---|---|---|
| `tools/blender-5.2.3-windows-x64/` | Blender 5.2.3, portable | https://www.blender.org/download/ (the Windows portable zip) |
| `tools/pose-env/` | Python with `mediapipe` 1.2.2, `opencv-contrib-python`, `numpy` | `python -m venv tools/pose-env`, then `tools/pose-env/Scripts/pip install mediapipe==1.2.2 opencv-contrib-python numpy` |
| `tools/pose-model/pose_landmarker_heavy.task` | MediaPipe body tracker | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task |
| `tools/face-model/face_landmarker.task` | MediaPipe face tracker | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task |
| `tools/human-base-meshes/` | Blender Studio Human Base Meshes v1.4.1 (CC0) | https://download.blender.org/demo/asset-bundles/human-base-meshes/ |

Optional, to let Claude Code work in an open Blender window (Blender Lab's official MCP server, v1.0.3):

| Path | What | Where to get it |
|---|---|---|
| Blender add-on | `mcp-1.0.3.zip` | https://projects.blender.org/lab/blender_mcp/releases. Install with `blender -b -c extension install-file -r user_default -e mcp-1.0.3.zip` |
| `tools/blender-mcp-env/` | the server | `git clone --depth 1 --branch v1.0.3 https://projects.blender.org/lab/blender_mcp.git`, then `python -m venv tools/blender-mcp-env` and `tools/blender-mcp-env/Scripts/pip install blender_mcp/mcp` |
| Claude Code | register it | `claude mcp add blender --scope user --env BLENDER_PATH=<blender.exe> -- <tools\blender-mcp-env\Scripts\blender-mcp.exe>` |

Start Blender with `pipeline\start_blender_mcp.ps1 [file.blend]`. The server lets an AI run any Python in
Blender, so only run it while you use it.

Reference images and video clips aren't included; use your own, or ones you have the rights to.

## Quick start

```
tools\pose-env\Scripts\python.exe -I pipeline\agent.py --video <clip.mp4> --character <rigged.blend> --run <name>
```

## Status

Research code, tested on one Windows laptop (CPU rendering). It works end to end for motion from
video, but building a character is still a sequence of separate scripts with settings chosen per
character. Many scripts still have this machine's folder paths written in them.
