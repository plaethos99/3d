# Re-run every pose, the walk, the export and the before-and-after face scores.
$bl = "C:\Users\Admin\Downloads\3d\tools\blender-5.2.3-windows-x64\blender.exe"
$r = "C:\Users\Admin\Downloads\3d\tests\joyce"
"--- baseline: the model before this round, re-measured with the corrected chin check"
$o = & $bl --background "$r\output\joyce_v1_before_face_work.blend" --python-exit-code 1 --python "$r\scripts\render_compare.py" -- "$r\refs\stand.png" "$r\renders\face\baseline_stand" 38 --samples 48 --height 1248 --pose stand --face-spec "$r\scripts\face_spec_v1.json"
$o | Select-String 'IOU|LANDMARK|Traceback' | ForEach-Object { $_.Line }
"--- current"
$o = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$r\scripts\render_compare.py" -- "$r\refs\stand.png" "$r\renders\face\final_stand" 38 --samples 64 --height 1248 --pose stand --face-spec "$r\scripts\face_spec_v2.json"
$o | Select-String 'IOU|LANDMARK|Traceback' | ForEach-Object { $_.Line }
foreach ($n in 'sit','wave','surprised') {
  $o = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$r\scripts\render_compare.py" -- "$r\refs\$n.png" "$r\renders\face\final_$n" 38 --samples 32 --height 936 --pose $n
  "$n exit: $LASTEXITCODE"
  $o | Select-String 'IOU |Traceback' | ForEach-Object { $_.Line }
}
$o = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$r\scripts\render_compare.py" -- "$r\refs\back.png" "$r\renders\face\final_back" 150 --samples 32 --height 1248 --pose rest
"back exit: $LASTEXITCODE"
$o | Select-String 'IOU |Traceback' | ForEach-Object { $_.Line }
$o = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$r\scripts\animate_walk.py" -- "$r\output\joyce_final.blend" "$r\output\joyce_final.glb"
"animate exit: $LASTEXITCODE"
$o | Select-String 'GROUND|LOWEST|STANCE|GLB_MB|ANIMATE_OK|Traceback' | ForEach-Object { $_.Line }
if ($LASTEXITCODE -ne 0) { $o | Select-Object -Last 15; exit 1 }
foreach ($job in @(@('walk-a',1),@('walk-pass',7),@('walk-b',13))) {
  $n=$job[0]; $f=$job[1]
  $o = & $bl --background "$r\output\joyce_final.blend" --python-exit-code 1 --python "$r\scripts\render_compare.py" -- "$r\refs\$n.png" "$r\renders\face\final_$n" 38 --samples 24 --height 936 --frame $f
  "$n exit: $LASTEXITCODE"
  $o | Select-String 'IOU |Traceback' | ForEach-Object { $_.Line }
}
$o = & $bl --background --factory-startup --python-exit-code 1 --python "$r\scripts\verify_glb.py" -- "$r\output\joyce_final.glb"
"verify exit: $LASTEXITCODE"
$o | Select-String 'GLB_MESHES|GLB_ARM|GLB_ANIM|VERIFY_OK|Traceback' | ForEach-Object { $_.Line }
