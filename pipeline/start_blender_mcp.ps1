# Open Blender with its MCP bridge running, so Claude Code (or any MCP client) can work in the open window.
# The bridge listens on localhost:9876 only. --online-mode allows it for this session without changing
# Blender's saved preferences. Close Blender to stop it.
# Usage: pipeline\start_blender_mcp.ps1 [file.blend]
param([string]$Blend = "")
$root = Split-Path -Parent $PSScriptRoot
$blender = Join-Path $root "tools\blender-5.2.3-windows-x64\blender.exe"
$args = @("--online-mode")
if ($Blend) { $args += (Resolve-Path $Blend).Path }
Start-Process -FilePath $blender -ArgumentList $args
