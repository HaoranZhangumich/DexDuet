"""Reference check of the hand graph per hand: the retargeted reference keypoints of Box-200 (box-30-230-s01-u01) scored against
the human hand with the reward's own nodes and palm frame (scripts/reference_check.py). Catches a mirrored finger map, a
flipped palm normal (about 180 degrees) or a wrong wrist (a degenerate palm) before training. Runs for every hand whose
retargeted clip exists (examples/retarget_missing.sh); skipped otherwise.

Thresholds are loose on purpose: the retargeting matches fingertips, not directions, so even Inspire's reference sits at a
median palm-normal angle of 12 / 21 degrees (left / right), finger edges up to 43 degrees and a mean gap error of 1.8 / 2.0 cm.
Allegro's reference: palm 8 / 18, forward 7 / 9, edges 11-21 degrees, gap error 0.7 / 0.5 cm.
Every hand, Dex3 included, has the same 45 degree palm limit. Dex3 has three fingers, so its palm normal is the calibrated
wrist-link axis against the human index x little-base normal (finger_spec.palm_normal_mode 'calibrated_axis'): 8 / 15
degrees median. With the finger rule (index x middle base on both hands) it was 65 / 46 degrees, which needed a 90 degree
limit before.

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m pytest -q tests/test_reference_check.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests' / 'golden'))
sys.path.insert(0, str(ROOT / 'scripts'))
from dexmachina.asset_utils import get_asset_path  # noqa: E402

HANDS = ('inspire_hand', 'allegro_hand', 'dex3_hand', 'xhand', 'schunk_hand')
CLIP = 'box-30-230-s01-u01'
MAX_PALM_DEG, MAX_FORWARD_DEG, MAX_FINGER_DEG, MAX_GAP_CM = 45.0, 45.0, 60.0, 3.0


def have_clip(hand):
    return (Path(get_asset_path(f'retargeted/{hand}/s01/box_use_01_vector_para.pt')).exists()
            and Path(get_asset_path('arctic/processed/s01/box_use_01.npy')).exists())


@pytest.mark.parametrize('hand', HANDS)
def test_retargeted_reference_follows_the_human_hand(hand):
    if not have_clip(hand):
        pytest.skip(f'no retargeted Box-200 clip for {hand}')
    import make_inspire_golden as mg
    import reference_check
    mg.cpu_torch_load()
    m, demo, ret = reference_check.load(hand, CLIP)
    for side in ('left', 'right'):
        try:
            x = reference_check.reference_angles(m, demo, ret, side)
        except ValueError as e:
            pytest.skip(str(e))
        palm_deg, forward_deg = x['palm'].median(), x['forward'].median()
        finger_deg, gap = x['fingers'].median(dim=0).values, x['gap_cm'].mean()
        msg = (f'{hand} {side} ({m.finger_specs[side].palm_normal_mode}): palm {palm_deg:.1f} forward {forward_deg:.1f} '
               f'fingers {finger_deg.tolist()} gap {gap:.2f} cm')
        assert palm_deg < MAX_PALM_DEG and forward_deg < MAX_FORWARD_DEG, msg
        assert (finger_deg < MAX_FINGER_DEG).all() and gap < MAX_GAP_CM, msg
