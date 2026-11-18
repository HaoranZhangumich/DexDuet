"""Finger correspondence between a robot hand and the human (MANO) hand, for the hand-graph and hand-object contact rewards.

The retargeting config (dexmachina/assets/<hand>/retarget_config.yaml) already says which human finger each robot finger
copies: target_finger_links[i] is retargeted to MANO joint human_finger_idxs[i], and the entries with an index 16..20 are the
fingertips (16 thumb, 17 index, 18 middle, 19 ring, 20 pinky). Human fingers that no robot fingertip copies are missing on
this hand; they are left out of both rewards (edges, fingertip gaps, contact regions and the normalisations).

Everything else is resolved from the hand cfg (dexmachina/envs/hand_cfgs/<hand>.py: urdf_path, wrist_link_name,
collision_palm_name) and the URDF:
  - wrist: the cfg wrist_link_name (also the retargeting origin), paired with MANO joint 0;
  - finger chain: the links from the wrist out to the fingertip; a finger's own chain is the part no other finger shares,
    and its first link is the finger base, paired with MANO joint 13 / 1 / 4 / 10 / 7 (thumb / index / middle / ring / pinky);
  - palm normal, by palm_normal_mode (PALM_NORMAL_MIN_FINGERS):
      'fingers' (four or more present fingers): (index base - wrist) x (outermost finger base - wrist), the outermost finger
        being the last present finger after the thumb, with the same nodes on the human and on the robot;
      'calibrated_axis' (fewer than four, e.g. Dex3): human (index base - wrist) x (little base - wrist), MANO 1 and 7 (the
        human hand always has five fingers); robot R_wrist(t) a, a fixed unit axis a of the wrist link, calibrated once per
        clip to the human normal over the retargeted reference (rewards.calibrate_palm_axis);
    palm forward: wrist to the middle base, or to the midpoint of the index and middle bases when no finger lies beyond the
    middle (e.g. Dex3), the same nodes on the human and on the robot;
  - fingertip gaps: the thumb with every other present finger, then each pair of neighbouring present fingers;
  - gap scale: the retargeting scaling_factor;
  - contact regions: the palm (collision_palm_name, plus the extra palm links of PALM_LINKS) and, per present finger, the
    last k collision links of its own chain (CONTACT_DEPTH), which cover the MANO phalanges of ho_contact.MANO_ANCHOR_JOINTS.

Errors name the hand, the side and the link that is missing.
"""
import importlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import yaml

FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
MANO_WRIST = 0
MANO_TIPS = {'thumb': 16, 'index': 17, 'middle': 18, 'ring': 19, 'pinky': 20}
MANO_BASES = {'thumb': 13, 'index': 1, 'middle': 4, 'ring': 10, 'pinky': 7}
REQUIRED_FINGERS = ('thumb', 'index', 'middle')
# Palm normal rule, the one place that decides it. With four or more present fingers the knuckles span the palm, and the
# normal comes from the finger bases on both hands ('fingers'; Inspire, XHand, Schunk, Allegro, Ability, DexRobot). With
# fewer (Dex3: thumb, index, middle) the only knuckles beside the wrist are the human index and middle ones, ~2.2-2.5 cm
# apart, which leave the palm plane ill-conditioned, and the retargeting, which matches three fingertips, rolls the robot
# palm 30-50 degrees relative to the human's; there the robot normal is a calibrated wrist-link axis ('calibrated_axis').
# On the retargeted Box-200 reference the median human-robot palm-normal angle drops from 65 / 46 to 8 / 15 degrees
# (left / right); Inspire's own reference is at 12 / 21 with the finger rule.
PALM_NORMAL_MIN_FINGERS = 4


def palm_normal_mode(n_fingers):
    """The palm-normal rule of a hand with n_fingers present fingers (thumb included)."""
    return 'fingers' if n_fingers >= PALM_NORMAL_MIN_FINGERS else 'calibrated_axis'


# Contact regions: how many collision links at the distal end of a finger's own chain form its region (thumb, other fingers).
# They cover the MANO phalanges of ho_contact.MANO_ANCHOR_JOINTS: the middle and distal phalanges of the fingers and the
# proximal and distal phalanges of the thumb. Inspire 2/1 gives TH_L3+TH_L4 and *_L2, as in the Inspire-only code.
CONTACT_DEPTH = {
    'inspire_hand': (2, 1),     # TH_L3+L4; FF/MF/RF/LF_L2
    'xhand': (2, 1),            # thumb_rota_link1+2; index_rota_link2, mid/ring/pinky_link2
    'schunk_hand': (2, 2),      # b+c; p+t, o+s, n+r, m+q
    'allegro_hand': (3, 3),     # link_14+15+15_tip; the last two segments of each finger plus its tip
    'dex3_hand': (2, 1),        # thumb_1 (box) + thumb_2; index_1, middle_1
    'ability_hand': (2, 1),     # thumb_L1+L2; *_L2
    'dexrobot_hand': (3, 3),    # f_link1_3+1_4+tip; f_linkk_3+k_4+tip
}
# palm links when the palm is more than the cfg's collision_palm_name ('{side}' is left/right)
PALM_LINKS = {'schunk_hand': ('{side}_hand_e1', '{side}_hand_e2')}


def asset_hand_name(hand):
    """Asset folder of a hand name (as dexmachina/retargeting/parallel_retarget.py: 'xhand' stays 'xhand')."""
    return hand if 'hand' in hand else hand + '_hand'


def load_robot_cfg(hand, side):
    """The hand cfg of one side (dexmachina/envs/hand_cfgs/<hand>.py), without importing the simulator."""
    if side not in ('left', 'right'):
        raise ValueError(f'{hand}: side must be left or right, got {side!r}')
    prefix = hand.replace('_hand', '')
    try:
        module = importlib.import_module(f'dexmachina.envs.hand_cfgs.{prefix}')
    except ImportError as e:
        raise ValueError(f'Unknown hand {hand!r}: no hand cfg dexmachina/envs/hand_cfgs/{prefix}.py ({e})') from None
    cfgs = [getattr(module, a) for a in dir(module) if a.upper().endswith('_CFGS') and isinstance(getattr(module, a), dict)]
    if len(cfgs) != 1 or side not in cfgs[0]:
        raise ValueError(f'Hand {hand!r}: dexmachina/envs/hand_cfgs/{prefix}.py has no {side} cfg')
    return cfgs[0][side]


def retarget_config_path(hand):
    from dexmachina.asset_utils import get_asset_path
    return str(get_asset_path(f'{asset_hand_name(hand)}/retarget_config.yaml'))


def load_retarget_config(hand):
    path = retarget_config_path(hand)
    if not os.path.exists(path):
        raise ValueError(f'Hand {hand!r}: no retargeting config {path}; the finger correspondence comes from it')
    with open(path) as f:
        return yaml.safe_load(f)


def retarget_scale(hand):
    """The hand's retargeting scaling_factor (human-to-robot length scale), the hand-graph fingertip-gap scale."""
    cfg = load_retarget_config(hand)
    if 'scaling_factor' not in cfg:
        raise ValueError(f'Hand {hand!r}: {retarget_config_path(hand)} has no top-level scaling_factor')
    return float(cfg['scaling_factor'])


def resolve_hand_graph_scale(hand, value=None, needed=True):
    """The hand-graph gap scale as a number: an explicit value wins, else the hand's retargeting scale. When the hand graph
    is off and the hand has no retargeting config, the legacy default 1.2 is returned (the value is then unused)."""
    if value is not None:
        return float(value)
    try:
        return retarget_scale(hand)
    except ValueError:
        if needed:
            raise
        return 1.2


@dataclass(frozen=True)
class FingerSpec:
    hand: str
    side: str
    urdf_path: str
    scale: float                  # retargeting scaling_factor
    fingers: tuple                # present fingers, human order (thumb first)
    missing: tuple                # human fingers with no robot finger
    wrist: str                    # wrist link (MANO 0)
    bases: tuple                  # finger base link per present finger
    tips: tuple                   # fingertip link per present finger
    chains: dict                  # finger -> its own chain, palm to tip
    gap_pairs: tuple              # fingertip-gap pairs, indices into fingers
    outermost: str                # last present finger after the thumb (palm normal)
    forward_to_middle: bool       # palm forward: wrist -> middle base (True) or -> midpoint of index and middle bases
    contact_links: dict           # region -> links, palm first (None for a hand without a CONTACT_DEPTH entry)
    collision_links: tuple        # URDF links with a collision element, URDF order

    @property
    def human_bases(self):
        return tuple(MANO_BASES[f] for f in self.fingers)

    @property
    def human_tips(self):
        return tuple(MANO_TIPS[f] for f in self.fingers)

    @property
    def palm_normal_mode(self):
        """'fingers' or 'calibrated_axis' (see PALM_NORMAL_MIN_FINGERS)."""
        return palm_normal_mode(len(self.fingers))

    @property
    def palm_nodes(self):
        """Robot links of the palm frame: wrist, index base, middle base, outermost base. With 'calibrated_axis' the robot
        normal is the wrist link's calibrated axis and the outermost base is not used."""
        base = dict(zip(self.fingers, self.bases))
        return (self.wrist, base['index'], base['middle'], base[self.outermost])

    @property
    def human_palm_nodes(self):
        """MANO joints of the human palm frame: wrist, index base, middle base, and the base that spans the normal with the
        index base: the outermost present finger's ('fingers') or the little finger's ('calibrated_axis')."""
        outer = 'pinky' if self.palm_normal_mode == 'calibrated_axis' else self.outermost
        return (MANO_WRIST, MANO_BASES['index'], MANO_BASES['middle'], MANO_BASES[outer])

    @property
    def regions(self):
        return ('palm',) + self.fingers

    def finger_weights(self, thumb_weight):
        """Per present finger: thumb_weight for the thumb, 1 for the others (not normalised)."""
        return [thumb_weight if f == 'thumb' else 1.0 for f in self.fingers]

    def contact_regions(self):
        """[(region, links)] in region order, for ho_contact.robot_anchor_points."""
        if self.contact_links is None:
            raise ValueError(f'Hand {self.hand!r}: no contact-region depth defined (finger_spec.CONTACT_DEPTH)')
        return [(r, self.contact_links[r]) for r in self.regions]


def _parse_urdf(path, hand, side):
    if not os.path.exists(path):
        raise ValueError(f'Hand {hand!r} {side}: URDF {path} not found')
    root = ET.parse(path).getroot()
    links = [l.get('name') for l in root.findall('link')]
    collision = tuple(l.get('name') for l in root.findall('link') if l.find('collision') is not None)
    parent = dict()
    for j in root.findall('joint'):
        parent[j.find('child').get('link')] = j.find('parent').get('link')
    return links, collision, parent


def load_finger_spec(hand, side, robot_cfg=None):
    """Resolve the finger correspondence of one hand side (see the module docstring)."""
    robot_cfg = load_robot_cfg(hand, side) if robot_cfg is None else robot_cfg
    urdf = str(robot_cfg['urdf_path'])
    wrist = robot_cfg.get('wrist_link_name', 'base_link')
    where = f'hand {hand!r} {side}'
    rc = load_retarget_config(hand)
    if side not in rc:
        raise ValueError(f'{where}: {retarget_config_path(hand)} has no {side} section')
    side_cfg = rc[side]
    origin = side_cfg.get('target_origin_link')
    if origin is not None and origin != wrist:
        raise ValueError(f'{where}: retargeting origin {origin!r} differs from the cfg wrist_link_name {wrist!r}')
    if 'scaling_factor' not in rc:
        raise ValueError(f'{where}: {retarget_config_path(hand)} has no top-level scaling_factor')
    tip_of = dict()
    for link, idx in zip(side_cfg['target_finger_links'], side_cfg['human_finger_idxs']):
        for f, t in MANO_TIPS.items():
            if int(idx) == t:
                if f in tip_of:
                    raise ValueError(f'{where}: two fingertips are retargeted to MANO joint {t} ({tip_of[f]!r}, {link!r})')
                tip_of[f] = link
    fingers = tuple(f for f in FINGERS if f in tip_of)
    missing = tuple(f for f in FINGERS if f not in tip_of)
    lacking = [f for f in REQUIRED_FINGERS if f not in tip_of]
    if lacking:
        raise ValueError(f'{where}: the rewards need a thumb, index and middle finger; the retargeting has no {lacking} '
                         f'fingertip (human_finger_idxs 16..18)')
    links, collision, parent = _parse_urdf(urdf, hand, side)
    for name in [wrist] + [tip_of[f] for f in fingers]:
        if name not in links:
            raise ValueError(f'{where}: link {name!r} is not in {urdf}')
    full = dict()                                                       # finger -> links distal to the wrist, palm to tip
    for f in fingers:
        chain, cur = [], tip_of[f]
        while cur != wrist:
            chain.append(cur)
            if cur not in parent:
                raise ValueError(f'{where}: fingertip {tip_of[f]!r} does not hang below the wrist {wrist!r} in {urdf}')
            cur = parent[cur]
        full[f] = chain[::-1]
    shared = {l for f in fingers for l in full[f] if sum(l in full[g] for g in fingers) > 1}
    chains = dict()
    for f in fingers:
        own = [l for l in full[f] if l not in shared]
        if not own or own != full[f][len(full[f]) - len(own):]:
            raise ValueError(f'{where}: the {f} chain {full[f]} has no own distal part')
        chains[f] = tuple(own)
    n = len(fingers)
    gap_pairs = tuple((0, j) for j in range(1, n)) + tuple((j, j + 1) for j in range(1, n - 1))
    outermost = fingers[-1]
    contact_links = None
    depth = CONTACT_DEPTH.get(asset_hand_name(hand))
    if depth is not None:
        palm = tuple(p.format(side=side) for p in PALM_LINKS.get(asset_hand_name(hand), (robot_cfg.get('collision_palm_name'),)))
        contact_links = {'palm': palm}
        for f in fingers:
            k = depth[0] if f == 'thumb' else depth[1]
            own_coll = [l for l in chains[f] if l in collision]
            if len(own_coll) < k:
                raise ValueError(f'{where}: the {f} chain {list(chains[f])} has {len(own_coll)} collision links, its contact '
                                 f'region needs {k}')
            contact_links[f] = tuple(own_coll[-k:])
        for region, names in contact_links.items():
            for name in names:
                if name is None or name not in collision:
                    raise ValueError(f'{where}: contact link {name!r} of the {region} region is not a collision link of {urdf}')
    return FingerSpec(hand=hand, side=side, urdf_path=urdf, scale=float(rc['scaling_factor']), fingers=fingers,
                      missing=missing, wrist=wrist, bases=tuple(chains[f][0] for f in fingers),
                      tips=tuple(tip_of[f] for f in fingers), chains=chains, gap_pairs=gap_pairs, outermost=outermost,
                      forward_to_middle=any(f in fingers for f in ('ring', 'pinky')), contact_links=contact_links,
                      collision_links=collision)


def hand_graph_nodes(spec, kpt_names):
    """Indices of the hand-graph nodes in the robot keypoints. Nodes that are not keypoints (e.g. a wrist without collision,
    or fingertips the retargeting did not record) become extra nodes, numbered after the keypoints; the wrist comes first
    among them. Returns (index dict of lists: src, dst, tips, palm; extra link names)."""
    names = [str(n) for n in kpt_names]
    extra = []

    def find(link):
        if link in names:
            return names.index(link)
        if link not in extra:
            extra.append(link)
        return len(names) + extra.index(link)
    find(spec.wrist)                                                   # the wrist first among the extras
    index = dict(src=[find(b) for b in spec.bases], dst=[find(t) for t in spec.tips], tips=[find(t) for t in spec.tips],
                 palm=[find(n) for n in spec.palm_nodes])
    return index, extra
