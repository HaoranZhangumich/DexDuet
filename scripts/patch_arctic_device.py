#!/usr/bin/env python3
"""Make ARCTIC sequence processing honor the caller-selected torch device."""

from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
processing = repo_root / "arctic" / "src" / "arctic" / "processing.py"
text = processing.read_text()
old = "    quat_world2cam = rot.matrix_to_quaternion(world2cam_batch).cuda()\n"
new = (
    "    quat_world2cam = "
    "rot.matrix_to_quaternion(world2cam_batch).to(rot_r_world.device)\n"
)

if new in text:
    print(f"ARCTIC device patch already present: {processing}")
elif old in text:
    processing.write_text(text.replace(old, new, 1))
    print(f"Applied ARCTIC device patch: {processing}")
else:
    raise SystemExit(
        f"Could not find the expected transform_mano_rot_cam line in {processing}; "
        "the checked-out ARCTIC revision may be incompatible"
    )
