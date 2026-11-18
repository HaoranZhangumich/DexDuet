#!/usr/bin/env python3
"""Apply the MANO joint-selector patch required by the ARCTIC setup guide."""

from importlib.util import find_spec
from pathlib import Path


spec = find_spec("smplx")
if spec is None or spec.origin is None:
    raise SystemExit("smplx is not installed in the active environment")

body_models = Path(spec.origin).parent / "body_models.py"
text = body_models.read_text()
marker = "        # # Add pre-selected extra joints that might be needed\n"
commented = marker + "        # joints = self.vertex_joint_selector(vertices, joints)\n"
active = marker + "        joints = self.vertex_joint_selector(vertices, joints)\n"

if active in text:
    print(f"SMPL-X MANO patch already present: {body_models}")
elif commented in text:
    body_models.write_text(text.replace(commented, active, 1))
    print(f"Applied SMPL-X MANO patch: {body_models}")
else:
    raise SystemExit(
        f"Could not find the expected MANO block in {body_models}; "
        "the installed smplx version may be incompatible"
    )
