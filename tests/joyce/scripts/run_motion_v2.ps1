# Re-run the known walk and both real stretches through the repaired pipeline, as new runs,
# so the earlier results stay available to compare against. Also refreshes the final walk export.
$root = "C:\Users\Admin\Downloads\3d"
$r = "$root\tests\joyce"
$s = "$r\scripts"
$bl = "$root\tools\blender-5.2.3-windows-x64\blender.exe"
$o = & $bl --background "$r\output\joyce_rigged.blend" --python-exit-code 1 --python "$s\animate_walk.py" -- "$r\output\joyce_final.blend" "$r\output\joyce_final.glb"
"animate exit: $LASTEXITCODE"
$o | Select-String 'GROUND|LOWEST|STANCE|GLB_MB|ANIMATE_OK|Traceback' | ForEach-Object { $_.Line }
$o = & $bl --background --factory-startup --python-exit-code 1 --python "$s\verify_glb.py" -- "$r\output\joyce_final.glb"
"verify exit: $LASTEXITCODE"
$o | Select-String 'GLB_MESHES|GLB_ARM|GLB_ANIM|VERIFY_OK|Traceback' | ForEach-Object { $_.Line }
"=== real_walk_a_v2"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$s\video_to_motion.ps1" -Video "$root\clips\pexels-5995594\real_walk_a.mp4" -Name real_walk_a_v2
"=== real_walk_b_v2"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$s\video_to_motion.ps1" -Video "$root\clips\pexels-5995594\real_walk_b.mp4" -Name real_walk_b_v2
"=== known_walk_v2"
& powershell -NoProfile -ExecutionPolicy Bypass -File "$s\video_to_motion.ps1" -Video "$r\renders\joyce_walk.mp4" -Name known_walk_v2 -Truth "$r\motion\walk_truth.json"
"ALL_DONE"
