"""Build Human Generator child livekeys for every age and both sexes (runs Blender for each step).

Usage: python pipeline/humgen/child/build_ages.py [--ages 2,4,6,9,12,15] [--genders female,male] [--work dir]
Writes <content>/livekeys/special/age_child/<gender>_child_<age>.npz, which HG lists as livekeys "child_<age>"
(HG strips the gender prefix), and keeps the intermediate npz files in --work (default tests/humgen/child).
Heights: typical (about the 50th percentile of WHO / CDC growth charts) for each age, since MakeHuman's
children run short. MPFB's age macro: 1 year = 0, 11 = 0.1875, 25 = 0.5 (linear between).
Prints BUILD_AGES_OK.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BLENDER = os.path.join(ROOT, "tools", "blender-5.2.3-windows-x64", "blender.exe")
HERE = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "tools", "Human", "content")

# typical height in metres by age (about the median of the WHO / CDC growth charts)
HEIGHT = {
    "female": {2: 0.86, 3: 0.95, 4: 1.02, 5: 1.09, 6: 1.15, 7: 1.21, 8: 1.27, 9: 1.33, 10: 1.38, 11: 1.44,
               12: 1.51, 13: 1.57, 14: 1.60, 15: 1.62, 16: 1.63, 17: 1.63},
    "male": {2: 0.87, 3: 0.96, 4: 1.03, 5: 1.10, 6: 1.16, 7: 1.22, 8: 1.28, 9: 1.33, 10: 1.39, 11: 1.44,
             12: 1.49, 13: 1.56, 14: 1.64, 15: 1.70, 16: 1.74, 17: 1.76},
}


def mpfb_age(years):
    if years <= 11:
        return (years - 1) / 10 * 0.1875
    return 0.1875 + (years - 11) / 14 * 0.3125


def run(script, *args):
    cmd = [BLENDER, "-b", "-P", os.path.join(HERE, script), "--", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = [ln for ln in p.stdout.splitlines() if ln.endswith("_OK") or "_OK " in ln]
    info = [ln for ln in p.stdout.splitlines() if ln.startswith(("CHILD", "EDGE_FIT", "WRAP 29", "BODY"))]
    if not ok:
        print(p.stdout[-3000:], p.stderr[-2000:])
        raise SystemExit(f"BUILD_AGES_FAILED {script} {args}")
    for ln in info:
        print("   ", ln)


def opt(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


ages = [int(a) for a in opt("--ages", "2,4,6,9,12,15").split(",")]
genders = opt("--genders", "female,male").split(",")
work = os.path.abspath(opt("--work", os.path.join(ROOT, "tests", "humgen", "child")))
os.makedirs(work, exist_ok=True)
t0 = time.time()
for g in genders:
    hg = os.path.join(work, f"hg_{g}.npz")
    if not os.path.exists(hg):
        run("export_hg.py", hg, "--gender", g)
    adult = os.path.join(work, f"mpfb_{g}_0.5.npz")
    if not os.path.exists(adult):
        run("export_mpfb.py", hg, adult, "--gender", g, "--age", "0.5")
    for a in ages:
        m = f"{mpfb_age(a):.4f}"
        child = os.path.join(work, f"mpfb_{g}_{a}y.npz")
        print(f"{g} {a}y: MPFB age {m}, height {HEIGHT[g][a]} m ({time.time() - t0:.0f}s)")
        run("export_mpfb.py", hg, child, "--gender", g, "--age", m)
        lk = os.path.join(CONTENT, "livekeys", "special", "age_child", f"{g}_child_{a}.npz")
        run("transfer.py", hg, adult, child, os.path.join(work, f"child_{a}_{g}.npz"),
            "--height", str(HEIGHT[g][a]), "--livekey", lk)
print(f"BUILD_AGES_OK ({time.time() - t0:.0f}s)")
