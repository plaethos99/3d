# Video to animated Joyce: tracks the video, retargets it, renders it and compares the two.
# Usage: video_to_motion.ps1 -Video <path> -Name <run name> [-Truth <truth.json>] [-MaxSeconds 30] [-Character <rigged.blend>]
param(
    [Parameter(Mandatory = $true)][string]$Video,
    [Parameter(Mandatory = $true)][string]$Name,
    [string]$Truth = "",
    [double]$MaxSeconds = 30,
    [string]$Character = ""
)
$ErrorActionPreference = "Continue"
$root = "C:\Users\Admin\Downloads\3d"
$bl = "$root\tools\blender-5.2.3-windows-x64\blender.exe"
$py = "$root\tools\pose-env\Scripts\python.exe"
$model = "$root\tools\pose-model\pose_landmarker_heavy.task"
$r = "$root\tests\joyce"
$s = "$r\scripts"
$out = "$r\motion\$Name"
if (-not $Character) { $Character = "$r\output\joyce_rigged.blend" }
New-Item -ItemType Directory -Force $out | Out-Null

function RunPython([string]$arguments) {
    # Native libraries print harmless start-up notes to stderr, so it is sent to nul.
    $text = & cmd /c "`"$py`" -I $arguments 2>nul"
    return @{ code = $LASTEXITCODE; text = $text }
}

$sw = [Diagnostics.Stopwatch]::StartNew()
$t = RunPython "`"$s\track_video.py`" `"$model`" `"$Video`" `"$out\source_pose.json`" $MaxSeconds"
"track exit: $($t.code) seconds: $([int]$sw.Elapsed.TotalSeconds)"
$t.text | Select-String 'TRACK_OK|Could not' | ForEach-Object { $_.Line }
if ($t.code -ne 0) { exit 1 }
$t = RunPython "`"$s\clean_tracking.py`" `"$out\source_pose.json`" `"$out\clean_pose.json`""
"clean exit: $($t.code)"
$t.text | Select-String 'CLEAN_REPORT' | ForEach-Object { $_.Line }
if ($t.code -ne 0) { exit 1 }

$sw.Restart()
$extra = @()
if ($Truth) { $extra = @("--truth", $Truth) }
$o = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$s\retarget_video.py" -- "$out\clean_pose.json" "$out\joyce_motion.blend" "$out\joyce_motion.glb" "$out\report.json" @extra
"retarget exit: $LASTEXITCODE seconds: $([int]$sw.Elapsed.TotalSeconds)"
$o | Select-String 'RETARGET_REPORT|Traceback|Error:|line \d' | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { $o | Select-Object -Last 15; exit 1 }

$sw.Restart()
$o = & $bl --background "$out\joyce_motion.blend" --python-exit-code 1 --python "$s\render_motion.py" -- "$out\frames" "$out\joyce_result.mp4"
"render exit: $LASTEXITCODE seconds: $([int]$sw.Elapsed.TotalSeconds)"
$o | Select-String 'RENDER_MOTION_OK|Traceback|Error:' | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { $o | Select-Object -Last 15; exit 1 }

$sw.Restart()
$t = RunPython "`"$s\track_video.py`" `"$model`" `"$out\joyce_result.mp4`" `"$out\result_pose.json`" $MaxSeconds"
"re-track exit: $($t.code)"
$t.text | Select-String 'TRACK_OK' | ForEach-Object { $_.Line }
$t = RunPython "`"$s\compare_motion.py`" `"$out\source_pose.json`" `"$out\result_pose.json`" `"$Video`" `"$out\joyce_result.mp4`" `"$out\side_by_side.mp4`" `"$out\side_by_side.png`""
"compare exit: $($t.code) seconds: $([int]$sw.Elapsed.TotalSeconds)"
$t.text | Select-String 'POSE_ERROR|COMPARE_OK' | ForEach-Object { $_.Line }
$t = RunPython "`"$s\score_motion.py`" `"$out\source_pose.json`" `"$out\result_pose.json`" `"$Name`""
$t.text | Select-String 'LIMB_DIRECTION' | ForEach-Object { $_.Line }
