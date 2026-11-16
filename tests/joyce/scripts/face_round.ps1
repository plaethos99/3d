# Rebuild Joyce, rig her, render the standing pose and print the face scores.
# Usage: face_round.ps1 <round name>
param([string]$name)
$bl = "C:\Users\Admin\Downloads\3d\tools\blender-5.2.3-windows-x64\blender.exe"
$r = "C:\Users\Admin\Downloads\3d\tests\joyce"
$sw = [Diagnostics.Stopwatch]::StartNew()
$o = & $bl --background --factory-startup --python-exit-code 1 --python "$r\scripts\build_joyce.py" -- full "$r\output\joyce_model.blend"
"build exit: $LASTEXITCODE seconds: $([int]$sw.Elapsed.TotalSeconds)"
$o | Select-String 'BUILD_OK|HAIR_LOCKS|Traceback|Error|line \d' | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { $o | Select-Object -Last 15; exit 1 }
$o1 = & $bl --background "$r\output\joyce_model.blend" --python-exit-code 1 --python "$r\scripts\rig_joyce.py" -- "$r\output\joyce_rigged.blend"
"rig exit: $LASTEXITCODE"
$o1 | Select-String 'RIG_OK|Traceback|Error' | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { $o1 | Select-Object -Last 15; exit 1 }
$o2 = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$r\scripts\render_compare.py" -- "$r\refs\stand.png" "$r\renders\face\$name" 38 --samples 48 --height 1248 --pose stand --face-spec "$r\scripts\face_spec_v2.json"
"render exit: $LASTEXITCODE"
$o2 | Select-String 'IOU|LANDMARK|Traceback|Error:' | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { $o2 | Select-Object -Last 15 }
