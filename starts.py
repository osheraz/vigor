"""Native VIGOR motion-phase sampling, without training-repository imports."""
import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / 'assets'


def rotate(q, v):
    uv = np.cross(q[..., :3], v)
    return v + 2*(q[..., 3:4]*uv + np.cross(q[..., :3], uv))


def multiply(a, b):
    return np.concatenate((a[..., 3:4]*b[..., :3]+b[..., 3:4]*a[..., :3]
                           + np.cross(a[..., :3], b[..., :3]),
                           a[..., 3:4]*b[..., 3:4]-(a[..., :3]*b[..., :3]).sum(-1, keepdims=True)), -1)


def rotate_velocity(q, v):
    # Native my_quat_rotate: keep its arithmetic for non-unit source quaternions.
    return (2*q[3]**2-1)*v+2*q[3]*np.cross(q[:3], v)+2*q[:3]*np.dot(q[:3], v)


def interpolate_rotation(a, b, f):
    # Preserve the native source-loader arithmetic, including its sin/angle
    # interpolation. Replacing it with another SLERP would change reset poses.
    dot = (a*b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), -1, 1))
    out = (np.sin((1-f)*angle)*a+np.sin(f*angle)*b)/np.maximum(angle, 1e-8)
    if np.isclose(f, 0): out = a.copy()
    if np.isclose(f, 1): out = b.copy()
    return np.where((np.abs(np.abs(dot)-1) < 1e-8) | (angle < 1e-8), a, out)


class MotionSampler:
    def __init__(self):
        with np.load(ASSETS / 'motions.npz') as data:
            self.metadata = json.loads(str(data['metadata']))
            self.arrays = {k: data[k].copy() for k in data.files if k != 'metadata'}
        with np.load(ASSETS / 'original_terrain.npz') as data:
            self.tile_names = data['tile_names'].ravel()
            self.tile_centers = data['tile_centers'].reshape(-1, 2)
            self.tile_size = float(data['tile_size'])
        self.terrains = ('mixed', 'flat', *sorted(set(map(str, self.tile_names))))
        self.rest_times = []
        for mid, bounds in enumerate(self.metadata['time_bounds']):
            q = self.arrays[f'{mid}_root_rot']
            q = q/np.linalg.norm(q, axis=1, keepdims=True)
            times = np.arange(len(q))*self.metadata['lengths'][mid]/len(q)
            lying = np.abs(1-2*(q[:, 0]**2+q[:, 1]**2)) < .4
            self.rest_times.append(times[lying & (times >= bounds[0]) & (times <= bounds[1])])

    def state(self, motion, seconds):
        c = self.metadata
        n = c['num_frames'][motion]
        phase_frame = np.float32(seconds/c['lengths'][motion]*n)
        low, high = np.clip(int(np.floor(phase_frame)), 0, n-2), np.clip(int(np.ceil(phase_frame)), 1, n-1)
        f = np.clip(phase_frame-low, 0, 1).astype(np.float32)
        state = {}
        for key in ('root_pos', 'root_rot', 'root_vel', 'root_ang_vel', 'joints', 'joint_velocity', 'body_pos', 'body_rot'):
            a, b = self.arrays[f'{motion}_{key}'][[low, high]]
            state[key] = interpolate_rotation(a, b, f) if key.endswith('_rot') else (1-f)*a+f*b
        # Same preprocessing as G1Loader/G1FallRecoveryDataset, before online placement.
        q, p = state['root_rot'], state['root_pos']
        body = rotate(q, state['body_pos'])+p
        shift = .05-body[:, 2].min()
        p[2] += shift
        body[:, 2] += shift
        body_q = multiply(q, state['body_rot'])
        parents = c['extension_parent_indices']
        extended = body[parents]+rotate(body_q[parents], np.asarray(c['extension_offsets'], np.float32))
        state['clearance_points'] = np.concatenate((body, extended))-p
        return state

    def sample(self, seed, terrain, settled=False):
        rng = np.random.default_rng(seed)
        c = self.metadata
        weights = np.asarray(c['weights']).copy()
        if settled:
            weights *= np.array([len(t) > 0 for t in self.rest_times])
            weights /= weights.sum()
        mid = int(rng.choice(len(c['names']), p=weights))
        seconds = float(rng.choice(self.rest_times[mid]) if settled else rng.uniform(*c['time_bounds'][mid]))
        state = self.state(mid, seconds)
        available = np.arange(len(self.tile_names)) if terrain in ('mixed', 'flat') else np.flatnonzero(self.tile_names == terrain)
        tile = int(rng.choice(available))
        offset = rng.uniform(-c['xy_fraction']*self.tile_size, c['xy_fraction']*self.tile_size, size=2)
        xy = self.tile_centers[tile]+offset+state['root_pos'][:2]
        kind = 'flat' if terrain == 'flat' else str(self.tile_names[tile])
        # The trained run keeps the source heading (sample_random_yaw=false).
        q = state['root_rot']
        return dict(label=f"{kind} · {c['names'][mid]} · phase {seconds/c['lengths'][mid]:.0%}",
            seed=int(seed), terrain=kind, motion=c['names'][mid], motion_time_s=seconds,
            heightfield=None if terrain == 'flat' else 'original_terrain.npz',
            source_xy=xy.tolist(), root_wxyz_xyz=[*q[[3, 0, 1, 2]].tolist(), 0., 0., float(state['root_pos'][2])],
            joints=state['joints'].tolist(), joint_velocity=(np.zeros_like(state['joint_velocity']) if settled else state['joint_velocity']).tolist(),
            root_velocity_world=(np.zeros(6) if settled else np.r_[rotate_velocity(q, state['root_vel']), rotate_velocity(q, state['root_ang_vel'])]).tolist(),
            clearance_points=state['clearance_points'], height_buffer=c['height_buffer'],
            source_tile=tile, xy_offset=offset.tolist())

    def reposition(self, scene, attempt):
        """Resample placement in the same terrain family, preserving the motion."""
        rng = np.random.default_rng([scene['seed'], attempt, 2])
        width = self.metadata['xy_fraction']*self.tile_size
        available = np.flatnonzero(self.tile_names == scene['terrain'])
        tile = int(rng.choice(available)) if len(available) else scene['source_tile']
        offset = rng.uniform(-width, width, size=2)
        mid = self.metadata['names'].index(scene['motion'])
        root_xy = self.state(mid, scene['motion_time_s'])['root_pos'][:2]
        xy = self.tile_centers[tile]+offset+root_xy
        return dict(scene, source_tile=tile, source_xy=xy.tolist(), xy_offset=offset.tolist())
