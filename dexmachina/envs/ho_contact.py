"""Hand-object contact reward, "where and how far" (opt-in: --ho_contact_weight > 0). DexMachina's contact reward is untouched.

Per hand the anchors (contact regions) are the palm and the outer part of each robot finger; the robot fingers and the human
fingers they copy come from the retargeting's finger correspondence (dexmachina/envs/finger_spec.py). A five-fingered hand has
all six regions; a human finger with no robot counterpart (Allegro's pinky; Dex3's ring and pinky) has no region, so it leaves
both the sum and the normalisation below. For every region and object part the human hand (ARCTIC MANO, precomputed per frame
by dexmachina/hand_proc/ho_contact_assets.py, always all six regions) gives two targets: its distance to the part surface and
the nearest surface spot, in the part's own frame. The robot gets the same two numbers every step from
points on its anchor links, looked up in a distance grid precomputed for each part; when the simulator reports one of the
anchor's links touching the part, the distance is 0 and the spot is the contact point. Per anchor and part:

    error  = |min(d_robot, cap) - min(d_human, cap)| + |spot_robot - spot_human| * [d_human < cap]
    score  = exp(-max(error - tol, 0) / scale)
    weight = 1 if the robot or the human touches (< touch), near if either is within cap, 0 otherwise
    hand   = sum(weight * score) / sum(weight), 1 when every weight is 0
    reward = ho_contact_weight * mean(hand_left, hand_right)
"""
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F

from dexmachina.envs.math_utils import matrix_from_quat

ANCHORS = ('palm', 'thumb', 'index', 'middle', 'ring', 'pinky')     # the human target files' regions, in file order
PARTS = ('bottom', 'top')            # the object's link order, also used by the contact buffers
# MANO joints (16-joint kinematic order) whose skinning dominates each anchor's vertices: the palm; the middle and distal
# phalanges of the fingers and the proximal and distal phalanges of the thumb (the Inspire L2 / TH_L3+L4 links span the same)
MANO_ANCHOR_JOINTS = {'palm': (0,), 'thumb': (14, 15), 'index': (2, 3), 'middle': (5, 6), 'ring': (11, 12), 'pinky': (8, 9)}
# robot links of each region: per hand and side from dexmachina/envs/finger_spec.py (Inspire: base_link | TH_L3+TH_L4 | *_L2)
ANCHOR_POINTS = {'palm': 256, 'thumb': 64, 'index': 48, 'middle': 48, 'ring': 48, 'pinky': 48}   # surface points per region
DEFAULTS = dict(cap=0.05, tol=0.01, scale=0.02, touch=0.01, near=0.5)


def object_mesh_path(assets_dir, obj_name, part):
    """The part mesh the environment uses (object.py: stl preferred, else obj)."""
    stl = os.path.join(assets_dir, 'arctic', obj_name, f'{part}_watertight_tiny.stl')
    return stl if os.path.exists(stl) else stl[:-4] + '.obj'


def grid_path(assets_dir, obj_name, part):
    """3 mm grids with a 6 cm margin (the cap must stay below the margin)."""
    return os.path.join(assets_dir, 'ho_contact', 'grids', f'{obj_name}_{part}.npz')


def human_target_path(assets_dir, subject_name, obj_name, use_clip):
    return os.path.join(assets_dir, 'ho_contact', 'human', subject_name, f'{obj_name}_use_{use_clip}.npz')


def load_human_targets(assets_dir, subject_name, obj_name, use_clip, frame_start, frame_end):
    """Per-frame human targets for this clip: ho_dist_{side} (T, anchors, parts), ho_spot_{side} (T, anchors, parts, 3)."""
    path = human_target_path(assets_dir, subject_name, obj_name, use_clip)
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found; build it with dexmachina/hand_proc/ho_contact_assets.py human')
    z = np.load(path)
    if tuple(str(a) for a in z['anchors']) != ANCHORS or tuple(str(p) for p in z['parts']) != PARTS:
        raise ValueError(f'{path}: anchor or part order differs from {ANCHORS} / {PARTS}')
    if frame_end > z['dist'].shape[0]:
        raise ValueError(f'{path} has {z["dist"].shape[0]} frames, clip needs {frame_end}')
    out = dict()
    for i, side in enumerate(('left', 'right')):
        out[f'ho_dist_{side}'] = z['dist'][frame_start:frame_end, i].astype(np.float32)
        out[f'ho_spot_{side}'] = z['spot'][frame_start:frame_end, i].astype(np.float32)
    return out


class PartDistanceGrid:
    """Distance to a part's surface and the nearest surface point on a regular grid in the part frame. Queries are trilinear;
    points outside the grid take the border value, which lies farther from the part than the reward's cap."""

    def __init__(self, path, device):
        z = np.load(path)
        dist, near = z['dist'].astype(np.float32), z['nearest'].astype(np.float32)          # (X, Y, Z), (X, Y, Z, 3)
        vol = np.concatenate([dist[None], np.moveaxis(near, -1, 0)], 0)                      # (4, X, Y, Z)
        self.vol = torch.tensor(vol, device=device).permute(0, 3, 2, 1).unsqueeze(0).contiguous()   # (1, 4, Z, Y, X)
        self.origin = torch.tensor(np.asarray(z['origin']), dtype=torch.float32, device=device)
        self.extent = torch.tensor((np.array(dist.shape) - 1) * float(z['spacing']), dtype=torch.float32, device=device)
        self.margin = float(z['margin'])

    def query(self, pts):
        """pts (..., 3) in the part frame -> distance (...), nearest surface point (..., 3)."""
        shape = pts.shape[:-1]
        g = (pts.reshape(1, 1, 1, -1, 3) - self.origin) / self.extent * 2.0 - 1.0
        out = F.grid_sample(self.vol, g, mode='bilinear', padding_mode='border', align_corners=True).view(4, -1)
        return out[0].view(shape), out[1:].t().reshape(*shape, 3)


def _floats(text, default):
    return tuple(float(v) for v in text.split()) if text is not None else default


def rpy_matrix(rpy):
    """URDF origin rpy: fixed-axis roll (x), pitch (y), yaw (z), i.e. R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def collision_elements(urdf_path):
    """Every <collision> element of every link: {link: [dict(kind, ..., xyz, rpy)]}, kind mesh (path, scale), box (size),
    cylinder (radius, length; along z, centred) or sphere (radius)."""
    root, base = ET.parse(urdf_path).getroot(), os.path.dirname(urdf_path)
    out = dict()
    for link in root.findall('link'):
        items = []
        for col in link.findall('collision'):
            org = col.find('origin')
            xyz = _floats(org.get('xyz') if org is not None else None, (0.0, 0.0, 0.0))
            rpy = _floats(org.get('rpy') if org is not None else None, (0.0, 0.0, 0.0))
            geom = col.find('geometry')
            shapes = [] if geom is None else list(geom)
            if len(shapes) != 1:
                raise ValueError(f'{link.get("name")}: a collision geometry needs exactly one shape, found {len(shapes)}')
            g = shapes[0]
            item = dict(kind=g.tag, xyz=xyz, rpy=rpy)
            if g.tag == 'mesh':
                item.update(path=os.path.join(base, g.get('filename').replace('package://', '')),
                            scale=_floats(g.get('scale'), (1.0, 1.0, 1.0)))
            elif g.tag == 'box':
                item.update(size=_floats(g.get('size'), None))
            elif g.tag == 'cylinder':
                item.update(radius=float(g.get('radius')), length=float(g.get('length')))
            elif g.tag == 'sphere':
                item.update(radius=float(g.get('radius')))
            else:
                raise ValueError(f'{link.get("name")}: unsupported collision geometry <{g.tag}>')
            items.append(item)
        if items:
            out[link.get('name')] = items
    return out


def is_plain_mesh(items):
    """One mesh, no origin offset, no scale: sampled exactly as before the primitive support (no transform is applied)."""
    return (len(items) == 1 and items[0]['kind'] == 'mesh' and all(v == 0.0 for v in items[0]['xyz'] + items[0]['rpy'])
            and items[0]['scale'] == (1.0, 1.0, 1.0))


def sample_box(size, count, rng):
    """Uniform points on the surface of a box centred at the origin."""
    a, b, c = (float(v) for v in size)
    areas = np.array([b * c, b * c, a * c, a * c, a * b, a * b])
    face = rng.choice(6, size=count, p=areas / areas.sum())
    pts = (rng.random((count, 3)) - 0.5) * np.array([a, b, c])
    axis, sign = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
    pts[np.arange(count), axis] = sign * np.array([a, b, c])[axis] / 2.0
    return pts


def sample_cylinder(radius, length, count, rng):
    """Uniform points on the surface of a cylinder along z, centred at the origin (URDF convention): side and both caps."""
    side, cap = 2.0 * np.pi * radius * length, np.pi * radius ** 2
    part = rng.choice(3, size=count, p=np.array([side, cap, cap]) / (side + 2.0 * cap))
    theta = rng.random(count) * 2.0 * np.pi
    r = np.where(part == 0, radius, radius * np.sqrt(rng.random(count)))
    z = np.where(part == 0, (rng.random(count) - 0.5) * length, np.where(part == 1, 0.5, -0.5) * length)
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)


def sample_sphere(radius, count, rng):
    v = rng.normal(size=(count, 3))
    return radius * v / np.linalg.norm(v, axis=-1, keepdims=True)


def _element_area(item, mesh):
    if item['kind'] == 'mesh':
        return float(mesh.area)
    if item['kind'] == 'box':
        a, b, c = item['size']
        return 2.0 * (a * b + b * c + a * c)
    if item['kind'] == 'cylinder':
        return 2.0 * np.pi * item['radius'] * item['length'] + 2.0 * np.pi * item['radius'] ** 2
    return 4.0 * np.pi * item['radius'] ** 2


def _split(total, areas):
    """Points per entry, proportional to area; the remainder goes to the largest entry."""
    areas = np.asarray(areas, dtype=float)
    counts = np.floor(total * areas / areas.sum()).astype(int)
    counts[np.argmax(areas)] += total - counts.sum()
    return counts


class _LinkSurface:
    """The collision surface of one link in its link frame: all of its <collision> elements, with origin and mesh scale. A link
    with one plain mesh (no offset, no scale) is sampled exactly as in the Inspire-only code of the paper's Inspire runs:
    trimesh.sample_surface(mesh, count, seed), no transform (an identity transform would turn -0.0 into +0.0)."""

    def __init__(self, items):
        import trimesh
        self.items = items
        self.plain = is_plain_mesh(items)
        self.meshes = []
        for item in items:
            mesh = None
            if item['kind'] == 'mesh':
                mesh = trimesh.load(item['path'], force='mesh')
                if item['scale'] != (1.0, 1.0, 1.0):
                    # scale the vertices directly: trimesh's apply_transform draws from the global numpy RNG for a non-rigid
                    # transform, and the env construction must leave the global RNG state alone
                    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices) * np.asarray(item['scale']),
                                           faces=np.asarray(mesh.faces), process=False)
            self.meshes.append(mesh)
        self.areas = [_element_area(item, mesh) for item, mesh in zip(items, self.meshes)]
        self.area = float(sum(self.areas))

    def sample(self, count, seed):
        import trimesh
        if self.plain:
            return trimesh.sample.sample_surface(self.meshes[0], int(count), seed=seed)[0]
        out = []
        for i, (item, mesh, n) in enumerate(zip(self.items, self.meshes, _split(count, self.areas))):
            n = int(n)
            if n == 0:
                continue
            element_seed = [int(seed), i]                  # a separate stream per collision element, global RNG untouched
            if item['kind'] == 'mesh':
                p = trimesh.sample.sample_surface(mesh, n, seed=element_seed)[0]
            else:
                rng = np.random.default_rng(element_seed)
                if item['kind'] == 'box':
                    p = sample_box(item['size'], n, rng)
                elif item['kind'] == 'cylinder':
                    p = sample_cylinder(item['radius'], item['length'], n, rng)
                else:
                    p = sample_sphere(item['radius'], n, rng)
            out.append(np.asarray(p, dtype=np.float64) @ rpy_matrix(item['rpy']).T + np.asarray(item['xyz']))
        return np.concatenate(out, 0) if out else np.zeros((0, 3))


def robot_anchor_points(urdf_path, hand_name, side, seed=0, regions=None):
    """Surface points of each region's collision links in their link frames, ordered by region and, within a region, by link.
    regions: [(region, link names)] in region order (palm first); by default the hand's finger spec (finger_spec.py) gives
    them. Every <collision> element of a link is used (mesh with origin offset and scale, box, cylinder along z, sphere); a
    link with one plain mesh (no offset, no scale) is sampled exactly as in the Inspire-only code. Returns links (names),
    link_ranges [(start, end)] per link, local points (P, 3), region slices [(start, end)], anchor_links (names per region)
    and regions."""
    if regions is None:
        from dexmachina.envs.finger_spec import load_finger_spec, load_robot_cfg
        cfg = dict(load_robot_cfg(hand_name, side))
        cfg['urdf_path'] = urdf_path
        regions = load_finger_spec(hand_name, side, robot_cfg=cfg).contact_regions()
    elements = collision_elements(urdf_path)
    links, link_ranges, pts, slices, anchor_links = [], [], [], [], []
    for region, names in regions:
        names = list(names)
        missing = [n for n in names if n not in elements]
        if missing:
            raise ValueError(f'{hand_name} {side}: {region} region links {missing} have no collision geometry in {urdf_path}')
        if region not in ANCHOR_POINTS:
            raise ValueError(f'{hand_name} {side}: unknown contact region {region!r} (expected one of {ANCHORS})')
        # points per link in proportion to its area, the remainder to the largest (as in the Inspire-only code)
        surfaces = [_LinkSurface(elements[n]) for n in names]
        counts = _split(ANCHOR_POINTS[region], [surface.area for surface in surfaces])
        start = sum(len(p) for p in pts)
        for name, surface, count in zip(names, surfaces, counts):
            sample = surface.sample(int(count), seed).astype(np.float32)
            s = sum(len(p) for p in pts)
            link_ranges.append((s, s + len(sample)))
            links.append(name)
            pts.append(sample)
        slices.append((start, sum(len(p) for p in pts)))
        anchor_links.append(names)
    return dict(links=links, link_ranges=link_ranges, points=np.concatenate(pts, 0), slices=slices, anchor_links=anchor_links,
                regions=[r for r, _ in regions])


def anchor_contacts(valid, pos, anchor_link_idx):
    """Simulator contacts per anchor: valid (N, parts, links) bool, pos (N, parts, links, 3) world; anchor_link_idx lists the
    contact-link indices of each anchor, proximal to distal. Returns touching (N, parts, A) and the contact point (N, parts, A, 3),
    taken from the most distal touching link."""
    touching, points = [], []
    for idx in anchor_link_idx:
        t = torch.zeros_like(valid[:, :, 0])
        p = torch.zeros_like(pos[:, :, 0])
        for i in idx:
            v = valid[:, :, i]
            p = torch.where(v[..., None], pos[:, :, i], p)
            t = t | v
        touching.append(t)
        points.append(p)
    return torch.stack(touching, dim=2), torch.stack(points, dim=2)


class HandObjectContact:
    def __init__(self, cfg, grid_paths, device):
        c = {**DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
        self.cap, self.tol, self.scale = float(c['cap']), float(c['tol']), float(c['scale'])
        self.touch, self.near = float(c['touch']), float(c['near'])
        if not (0.0 < self.touch < self.cap and self.tol >= 0.0 and self.scale > 0.0 and 0.0 <= self.near <= 1.0):
            raise ValueError('Hand-object contact needs 0 < touch < cap, tol >= 0, scale > 0 and 0 <= near <= 1')
        self.grids = [PartDistanceGrid(p, device) for p in grid_paths]
        if len(self.grids) != len(PARTS):
            raise ValueError(f'Need one grid per part {PARTS}')
        for path, grid in zip(grid_paths, self.grids):
            if grid.margin < self.cap:
                raise ValueError(f'{path}: grid margin {grid.margin} m is smaller than the cap {self.cap} m')

    def robot_distances(self, pts, slices, part_pos, part_quat, touching, touch_pos):
        """pts (N, P, 3) world points of one hand's anchors; part_pos (N, parts, 3), part_quat (N, parts, 4) wxyz; touching
        (N, parts, A) and touch_pos (N, parts, A, 3) from the simulator. Returns distance (N, A, parts), spot (N, A, parts, 3) in the
        part frames, and the grid distance before the contact override (N, A, parts)."""
        n, a = pts.shape[0], len(slices)
        dist = pts.new_zeros(n, a, len(self.grids))
        spot = pts.new_zeros(n, a, len(self.grids), 3)
        for p, grid in enumerate(self.grids):
            rot = matrix_from_quat(part_quat[:, p])                                         # part -> world
            local = torch.einsum('nij,npi->npj', rot, pts - part_pos[:, p, None])            # world -> part frame
            d, near = grid.query(local)
            for k, (s, e) in enumerate(slices):
                dmin, idx = d[:, s:e].min(dim=1)
                dist[:, k, p] = dmin
                spot[:, k, p] = near[:, s:e].gather(1, idx[:, None, None].expand(-1, 1, 3)).squeeze(1)
        grid_dist = dist.clone()
        for p in range(len(self.grids)):
            rot = matrix_from_quat(part_quat[:, p])
            contact_local = torch.einsum('nij,nai->naj', rot, touch_pos[:, p] - part_pos[:, p, None])
            t = touching[:, p]
            dist[:, :, p] = torch.where(t, torch.zeros_like(dist[:, :, p]), dist[:, :, p])
            spot[:, :, p] = torch.where(t[..., None], contact_local, spot[:, :, p])
        return dist, spot, grid_dist

    def score(self, d_robot, s_robot, d_human, s_human):
        """(N, A, parts), (N, A, parts, 3) robot and human -> hand score (N,) and per-entry error, score, weight."""
        dr = d_robot.clamp(min=0.0, max=self.cap)
        dh = d_human.clamp(min=0.0, max=self.cap)
        human_near = d_human < self.cap
        err = (dr - dh).abs() + torch.norm(s_robot - s_human, dim=-1) * human_near
        score = torch.exp(-torch.clamp(err - self.tol, min=0.0) / self.scale)
        touch = (d_robot < self.touch) | (d_human < self.touch)
        near = (d_robot < self.cap) | human_near
        one = torch.ones_like(err)
        weight = torch.where(touch, one, torch.where(near, one * self.near, torch.zeros_like(err)))
        wsum = weight.sum(dim=(1, 2))
        hand = torch.where(wsum > 0, (weight * score).sum(dim=(1, 2)) / wsum.clamp(min=1e-6), torch.ones_like(wsum))
        return hand, dict(err=err, score=score, weight=weight)
