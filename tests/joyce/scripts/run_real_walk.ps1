# Run the video pipeline on the two longest uninterrupted stretches of the Pexels clip.
$root = "C:\Users\Admin\Downloads\3d"
$py = "$root\tools\pose-env\Scripts\python.exe"
$s = "$root\tests\joyce\scripts"
$source = "$root\clips\pexels-5995594\5995594-hd_720_1280_24fps.mp4"
foreach ($job in @(@("real_walk_a", 240, 291), @("real_walk_b", 157, 207))) {
    $name, $first, $last = $job
    $segment = "$root\clips\pexels-5995594\$name.mp4"
    & cmd /c "`"$py`" -I `"$s\cut_segment.py`" `"$source`" `"$segment`" $first $last 2>nul"
    "=== $name frames $first to $last"
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$s\video_to_motion.ps1" -Video $segment -Name $name
}
"ALL_DONE"
