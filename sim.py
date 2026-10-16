"""Run VIGOR in MuJoCo with a live Viser view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
import torch
from starts import MotionSampler

ROOT = Path(__file__).resolve().parent


class Simulation:
    def __init__(
        self, checkpoint_dir, terrain='mixed', seed=0, actor=None, sampler=None, start_mode='random'
    ):
        self.config = config = json.loads((checkpoint_dir / 'policy.json').read_text())
        if config.get('input_contract') != 'isaac_current_depth_lagged_history':
            raise ValueError('Unsupported policy input format')
        self.policy_name = config['candidate']
        self.actor = (
            actor
            if actor is not None
            else torch.jit.load(str(checkpoint_dir / 'student.pt'), map_location='cpu').eval()
        )
        self.sampler = sampler if sampler is not None else MotionSampler()
        self.terrain, self.seed = terrain, seed
        self.start_mode = start_mode
        self.scene = self.sampler.sample(seed, terrain, settled=start_mode == 'settled')
        self.heightfield = None
        if self.scene['heightfield']:
            with np.load(ROOT / 'assets' / self.scene['heightfield']) as terrain:
                spacing = float(terrain['spacing'])
                self.heightfield = terrain['heights']
                self.height_origin = terrain['origin']
                self.height_spacing = spacing
            extent = (np.array(self.heightfield.shape) - 1) * spacing
            self.height_center = self.height_origin + extent / 2
            self.height_min = float(self.heightfield.min())
            self.height_max = float(self.heightfield.max())
            self.height_range = max(float(np.ptp(self.heightfield)), 0.01)
        if config['joint_names'] != self.sampler.metadata['joint_names']:
            raise ValueError('Source-motion joints do not match the policy order')
        assets = ROOT / 'assets/g1'
        tree = ET.parse(assets / 'robot.xml')
        xml = tree.getroot()
        xml.find('compiler').set('meshdir', str(assets))
        option = xml.find('option')
        if option is None:
            option = ET.SubElement(xml, 'option')
        option.attrib.update(timestep=str(config['dt']), integrator='implicitfast')
        world = xml.find('worldbody')
        self.floor_z = self.height_min - 0.1 if self.heightfield is not None else 0.0
        ET.SubElement(
            world,
            'geom',
            name='floor',
            type='plane',
            pos=f'0 0 {self.floor_z}',
            size='10 10 .1',
            rgba='.7 .7 .7 1',
            friction='.8 .005 .0001',
            condim='3',
        )
        if self.heightfield is not None:
            asset = xml.find('asset')
            if asset is None:
                asset = ET.SubElement(xml, 'asset')
            # MuJoCo rows run along Y; the cached VIGOR height samples index X,Y.
            ET.SubElement(
                asset,
                'hfield',
                name='terrain',
                nrow=str(self.heightfield.shape[1]),
                ncol=str(self.heightfield.shape[0]),
                size=f'{extent[0]/2} {extent[1]/2} {self.height_range} .1',
            )
            ET.SubElement(
                world,
                'geom',
                name='height_terrain',
                type='hfield',
                hfield='terrain',
                pos=f'{self.height_center[0]} {self.height_center[1]} {self.height_min}',
                rgba='.55 .55 .55 1',
                friction='.8 .005 .0001',
                group='2',
            )
            # Full map for physics; an exact local copy for the short-range camera.
            self.patch_n = 81
            radius = (self.patch_n - 1) * spacing / 2
            ET.SubElement(
                asset,
                'hfield',
                name='depth_terrain',
                nrow=str(self.patch_n),
                ncol=str(self.patch_n),
                size=f'{radius} {radius} {self.height_range} .1',
            )
            ET.SubElement(
                world,
                'geom',
                name='depth_terrain',
                type='hfield',
                hfield='depth_terrain',
                contype='0',
                conaffinity='0',
                rgba='.55 .55 .55 1',
            )
        depth = config['depth']
        body = world.find(f".//body[@name='{depth['mount_body']}']")
        if body is None:
            raise ValueError('Camera mount is missing from robot asset')
        fx, fy, cx, cy = depth['intrinsics']
        ET.SubElement(
            body,
            'camera',
            name='depth',
            pos=' '.join(map(str, depth['position'])),
            quat=' '.join(map(str, depth['quaternion_wxyz'])),
            resolution=f"{depth['width']} {depth['height']}",
            sensorsize=f"{depth['width']} {depth['height']}",
            focalpixel=f'{fx} {fy}',
            principalpixel=f"{(depth['width']-1)/2-cx} {(depth['height']-1)/2-cy}",
        )
        self.model = model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
        if self.heightfield is not None:
            model.hfield_data[: self.heightfield.size] = (
                (self.heightfield.T - self.height_min) / self.height_range
            ).ravel()
            # Its position changes at render time.
            model.geom_sameframe[model.geom('depth_terrain').id] = 0
            self.patch_start = None
        self.joint_ids = np.array([model.joint(name).id for name in config['joint_names']])
        self.q_adr = model.jnt_qposadr[self.joint_ids]
        self.v_adr = model.jnt_dofadr[self.joint_ids]
        model.dof_armature[self.v_adr] = config['armature']
        self.data = mujoco.MjData(model)
        # Metric depth must not change when the surrounding map gets larger.
        model.vis.map.znear = 0.001 / model.stat.extent
        model.vis.map.zfar = (depth['far'] + 1) / model.stat.extent
        model.vis.quality.offsamples = 0  # One ray per pixel, as in training.
        self.renderer = mujoco.Renderer(model, height=depth['height'], width=depth['width'])
        self.renderer.enable_depth_rendering()
        self.render_options = mujoco.MjvOption()
        # Render the visual robot geometry, not a second copy of its collision hulls.
        robot_collision = (model.geom_bodyid != 0) & (
            (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        )
        model.geom_group[robot_collision] = 2
        self.render_options.geomgroup[2] = 0
        self.default = np.array(config['default_q'])
        self.kp, self.kd, self.effort = (np.array(config[k]) for k in ('kp', 'kd', 'effort'))
        self.history = np.zeros((1, config['history_steps'], 75), np.float32)
        self.depth_history = np.zeros(
            (1, config['depth_steps'], depth['height'] * depth['width']), np.float32
        )
        self.action = np.zeros(23, np.float32)
        self.control_dt = config['dt'] * config['decimation']

    def floor_height(self, x, y):
        if self.heightfield is not None:
            point = np.array([x, y, self.height_max + 1])
            distance = mujoco.mj_rayHfield(
                self.model,
                self.data,
                self.model.geom('height_terrain').id,
                point,
                np.array([0.0, 0.0, -1.0]),
            )
            return point[2] - distance if distance >= 0 else self.floor_z
        return 0.0

    def sample_start(self, terrain, seed):
        self.terrain, self.seed = terrain, seed
        self.scene = self.sampler.sample(seed, terrain, settled=self.start_mode == 'settled')

    def _place_start(self):
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)
        root = self.scene['root_wxyz_xyz']
        data.qpos[:3], data.qpos[3:7] = root[4:7], root[:4]
        if self.heightfield is not None:
            data.qpos[:2] = self.scene['source_xy']
        data.qpos[self.q_adr] = self.scene['joints']
        mujoco.mj_normalizeQuat(model, data.qpos)
        mujoco.mj_forward(model, data)
        # Online VIGOR placement: raise the sampled body above local terrain.
        points = self.scene['clearance_points']
        if self.heightfield is None:
            data.qpos[2] += self.scene['height_buffer']
        else:
            data.qpos[2] = (
                max(self.floor_height(*(point[:2] + data.qpos[:2])) - point[2] for point in points)
                + self.scene['height_buffer']
            )
        velocity = np.asarray(self.scene['root_velocity_world'])
        data.qvel[:3] = velocity[:3]
        data.qvel[3:6] = data.xmat[model.body('pelvis').id].reshape(3, 3).T @ velocity[3:]
        data.qvel[self.v_adr] = self.scene['joint_velocity']
        self.history.fill(0)
        self.depth_history.fill(0)
        # Match reset_all(): one control tick holding the sampled pose, then
        # first inference with empty histories and the issued hold action.
        self.action[:] = np.clip(
            (data.qpos[self.q_adr] - self.default) / self.config['action_scale'],
            -self.config['action_clip'],
            self.config['action_clip'],
        )
        target = self.default + self.config['action_scale'] * self.action
        for _ in range(self.config['decimation']):
            self.apply_target(target)
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

    def reset(self):
        self.placement_attempts = 1
        self.start_support_span = None
        if self.start_mode == 'settled':
            original = self.sampler.sample(self.seed, self.terrain, settled=True)
            for attempt in range(40):
                if attempt and attempt % 8 == 0:
                    pose_seed = int(
                        np.random.SeedSequence([self.seed, attempt, 3]).generate_state(1)[0]
                    )
                    original = self.sampler.sample(pose_seed, self.terrain, settled=True)
                self.scene = (
                    original if attempt % 8 == 0 else self.sampler.reposition(original, attempt)
                )
                self._place_start()
                if self.settle():
                    self.placement_attempts = attempt + 1
                    break
            else:
                raise RuntimeError(
                    f'No settled lying placement for seed {self.seed}; try another seed'
                )
        elif self.start_mode == 'gentle':
            original = self.sampler.sample(self.seed, self.terrain)
            for attempt in range(80):
                self.scene = (
                    original if attempt == 0 else self.sampler.reposition(original, attempt)
                )
                self._place_start()
                points = self.scene['clearance_points'] + self.data.qpos[:3]
                support_height_span = float(
                    np.ptp([self.floor_height(*point[:2]) for point in points])
                )
                if support_height_span <= 0.1:
                    self.placement_attempts = attempt + 1
                    self.start_support_span = support_height_span
                    break
            else:
                raise RuntimeError(f'No gentle placement for seed {self.seed}; try another seed')
        else:
            self._place_start()
        self.initial_qpos = self.data.qpos.copy()
        self.initial_qvel = self.data.qvel.copy()
        self.next_depth = 0.0
        self.last_depth = np.zeros(self.depth_history.shape[-1], np.float32)
        self.steps = 0

    def settle(self):
        """Let the sampled lying pose find support before starting the policy."""
        model, data = self.model, self.data
        target = data.qpos[self.q_adr].copy()
        quiet_steps = 0
        for step in range(round(3 / self.config['dt'])):
            torque = (
                0.15 * self.kp * (target - data.qpos[self.q_adr]) - self.kd * data.qvel[self.v_adr]
            )
            data.qfrc_applied[self.v_adr] = np.clip(torque, -self.effort, self.effort)
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or any(warning.number for warning in data.warning):
                return False
            resting = (
                np.linalg.norm(data.qvel[:3]) < 0.04
                and np.linalg.norm(data.qvel[3:6]) < 0.15
                and np.sqrt(np.mean(data.qvel[self.v_adr] ** 2)) < 0.15
            )
            quiet_steps = quiet_steps + 1 if resting else 0
            if step > 100 and quiet_steps >= round(0.25 / self.config['dt']):
                break
        mujoco.mj_forward(model, data)
        pelvis_up_z = data.xmat[model.body('pelvis').id].reshape(3, 3)[2, 2]
        supported = any(
            (model.geom_bodyid[contact.geom1] == 0) != (model.geom_bodyid[contact.geom2] == 0)
            for contact in data.contact
            if contact.dist <= 0.001
        )
        if quiet_steps < round(0.25 / self.config['dt']) or abs(pelvis_up_z) > 0.5 or not supported:
            return False
        self.settle_residual_velocity = data.qvel.copy()
        self.settle_seconds = float(data.time)
        data.qvel[:] = 0
        data.qacc_warmstart[:] = 0
        data.qfrc_applied[:] = 0
        data.time = 0.0
        self.action[:] = np.clip(
            (data.qpos[self.q_adr] - self.default) / self.config['action_scale'],
            -self.config['action_clip'],
            self.config['action_clip'],
        )
        mujoco.mj_forward(model, data)
        return True

    def apply_target(self, target):
        data = self.data
        torque = self.kp * (target - data.qpos[self.q_adr]) - self.kd * data.qvel[self.v_adr]
        torque = np.clip(torque, -self.effort, self.effort)
        data.qfrc_applied[self.v_adr] = torque

    def update_depth_terrain(self):
        if self.heightfield is None:
            return
        model, data, patch_size = self.model, self.data, self.patch_n
        start = np.clip(
            np.floor((data.cam_xpos[0, :2] - self.height_origin) / self.height_spacing).astype(int)
            - patch_size // 2,
            0,
            np.array(self.heightfield.shape) - patch_size,
        )
        if self.patch_start is not None and np.array_equal(start, self.patch_start):
            return
        x, y = start
        patch = self.heightfield[x : x + patch_size, y : y + patch_size]
        heightfield_id = model.hfield('depth_terrain').id
        heightfield_address = model.hfield_adr[heightfield_id]
        model.hfield_data[heightfield_address : heightfield_address + patch_size * patch_size] = (
            (patch.T - self.height_min) / self.height_range
        ).ravel()
        terrain_geom_id = model.geom('depth_terrain').id
        model.geom_pos[terrain_geom_id] = [
            *(self.height_origin + (start + (patch_size - 1) / 2) * self.height_spacing),
            self.height_min,
        ]
        # Static world geometry, no dynamics change.
        data.geom_xpos[terrain_geom_id] = model.geom_pos[terrain_geom_id]
        self.renderer._gl_context.make_current()
        mujoco.mjr_uploadHField(model, self.renderer._mjr_context, heightfield_id)
        self.patch_start = start

    def _build_observation(self):
        data, config = self.data, self.config
        pelvis_rotation = data.xmat[self.model.body('pelvis').id].reshape(3, 3)

        # Angular velocity and gravity are expressed in the pelvis frame.
        angular_velocity = data.qvel[3:6] * config['angular_velocity_scale']
        joint_offsets = data.qpos[self.q_adr] - self.default
        joint_velocity = data.qvel[self.v_adr] * config['joint_velocity_scale']
        projected_gravity = pelvis_rotation.T @ [0, 0, -1]

        # The checkpoint expects this order, with the previous issued action first.
        observation = np.concatenate(
            (
                self.action,
                angular_velocity,
                joint_offsets,
                joint_velocity,
                projected_gravity,
            )
        ).astype(np.float32)
        return np.clip(observation, -100.0, 100.0)

    def _update_depth(self):
        """Capture at the camera rate; reuse the last image between captures."""
        if self.data.time + 1e-8 < self.next_depth:
            return

        depth_config = self.config['depth']
        near, far = depth_config['near'], depth_config['far']
        self.update_depth_terrain()
        self.renderer.update_scene(self.data, camera='depth', scene_option=self.render_options)
        metric_depth = self.renderer.render()
        metric_depth = np.nan_to_num(metric_depth, nan=far, posinf=far)
        metric_depth = np.clip(metric_depth, near, far)
        normalized_depth = (metric_depth - near) / (far - near) - 0.5
        self.last_depth = normalized_depth.astype(np.float32).reshape(-1)
        self.next_depth += 1 / depth_config['hz']

    def _predict_action(self, observation):
        # Current inputs are separate from histories, which end at the previous step.
        action = self.actor(
            torch.from_numpy(observation[None]),
            torch.from_numpy(self.history),
            torch.from_numpy(self.last_depth[None]),
            torch.from_numpy(self.depth_history),
        ).numpy()[0]
        if not np.isfinite(action).all():
            raise RuntimeError('Policy returned nonfinite actions')
        action_limit = self.config['action_clip']
        return np.clip(action, -action_limit, action_limit)

    def _update_history(self, observation):
        """Store current inputs after inference, newest at index zero."""
        if self.steps == 0 and self.config['history_reset_from_current']:
            self.history[:] = observation
            self.depth_history[:] = self.last_depth
        else:
            self.history[:, 1:] = self.history[:, :-1].copy()
            self.history[:, 0] = observation
            self.depth_history[:, 1:] = self.depth_history[:, :-1].copy()
            self.depth_history[:, 0] = self.last_depth

    @torch.inference_mode()
    def step(self):
        """Run one policy step and the corresponding PD-controlled physics steps."""
        mujoco.mj_forward(self.model, self.data)
        observation = self._build_observation()
        self._update_depth()
        self.action[:] = self._predict_action(observation)
        self._update_history(observation)

        joint_targets = self.default + self.config['action_scale'] * self.action
        for _ in range(self.config['decimation']):
            self.apply_target(joint_targets)
            mujoco.mj_step(self.model, self.data)

        if not np.isfinite(self.data.qpos).all() or any(
            warning.number for warning in self.data.warning
        ):
            raise RuntimeError('MuJoCo instability detected')
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1

    def close(self):
        self.renderer.close()


class Viewer:
    def __init__(self, sim, args):
        import viser
        from viser.extras import ViserUrdf

        self.server = server = viser.ViserServer(port=args.port)
        server.scene.set_up_direction('+z')
        server.scene.world_axes.visible = False
        self.grid = server.scene.add_grid(
            '/ground',
            width=12,
            height=12,
            cell_size=0.5,
            plane='xy',
            cell_color=(180, 180, 180),
            section_color=(145, 145, 145),
        )
        self.terrain_handles = []
        self.base = server.scene.add_frame('/robot', show_axes=False)
        self.robot = ViserUrdf(server, ROOT / 'assets/g1/robot.urdf', root_node_name='/robot')
        self.joint_names = self.robot.get_actuated_joint_names()
        self.indices = [sim.config['joint_names'].index(name) for name in self.joint_names]
        self.play = server.gui.add_checkbox('Play', True)
        self.loop = server.gui.add_checkbox('Loop', True)
        self.speed = server.gui.add_dropdown(
            'Speed', ('0.25×', '0.5×', '1×', '2×'), initial_value='1×'
        )
        self.terrain = server.gui.add_dropdown(
            'Terrain', sim.sampler.terrains, initial_value=args.terrain
        )
        self.start_mode = server.gui.add_dropdown(
            'Start mode', ('random', 'settled', 'gentle'), initial_value=args.start_mode
        )
        self.new_start_requested = threading.Event()
        self.reset_requested = threading.Event()
        self.terrain.on_update(lambda _: self.new_start_requested.set())
        self.start_mode.on_update(lambda _: self.new_start_requested.set())
        server.gui.add_button('New random start').on_click(lambda _: self.new_start_requested.set())
        server.gui.add_button('Replay start').on_click(lambda _: self.reset_requested.set())
        server.gui.add_markdown('**VIGOR · MuJoCo**\n\nRandom motion, phase and terrain placement.')
        self.depth = server.gui.add_image(np.zeros((64, 64, 3), np.uint8), label='Policy depth')
        self.status = server.gui.add_markdown('')

        @server.on_client_connect
        def connect(client):
            self.set_camera(client, self.sim)

        self.show_case(sim)

    @staticmethod
    def set_camera(client, sim):
        x, y, z = sim.initial_qpos[:3] + [0, 0, 0.25]
        with client.atomic():
            client.camera.up_direction = (0.0, 0.0, 1.0)
            client.camera.position = (x - 2.0, y - 2.3, z + 1.5)
            client.camera.look_at = (x, y, z)

    def show_case(self, sim):
        self.sim = sim
        for handle in self.terrain_handles:
            handle.remove()
        self.terrain_handles = []
        self.grid.visible = sim.heightfield is None
        if sim.heightfield is not None:
            # Cached coplanar merging; separate tiles let the browser cull offscreen geometry.
            with np.load(ROOT / 'assets/original_terrain.npz') as terrain:
                vertices, faces = terrain['view_vertices'], terrain['view_faces']
                vertex_offsets, face_offsets = (
                    terrain['view_vertex_offsets'],
                    terrain['view_face_offsets'],
                )
                for tile_index in range(len(vertex_offsets) - 1):
                    self.terrain_handles.append(
                        self.server.scene.add_mesh_simple(
                            f'/terrain/{tile_index}',
                            vertices=vertices[
                                vertex_offsets[tile_index] : vertex_offsets[tile_index + 1]
                            ],
                            faces=faces[face_offsets[tile_index] : face_offsets[tile_index + 1]],
                            color=(145, 145, 145),
                            flat_shading=True,
                        )
                    )
        for client in self.server.get_clients().values():
            self.set_camera(client, sim)

    def update(self, sim):
        data = sim.data
        with self.server.atomic():
            self.base.position = data.qpos[:3]
            self.base.wxyz = data.qpos[3:7]
            self.robot.update_cfg(data.qpos[sim.q_adr][self.indices])
            gray = ((sim.last_depth.reshape(64, 64) + 0.5) * 255).clip(0, 255).astype(np.uint8)
            self.depth.image = np.repeat(gray[..., None], 3, axis=-1)
            self.status.content = (
                f'**{sim.scene["label"]}** · {data.time:.1f} s\n\n'
                f'{sim.start_mode} · Seed **{sim.seed}**'
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sampler = MotionSampler()
    parser.add_argument('--terrain', choices=sampler.terrains, default='mixed')
    parser.add_argument(
        '--seed', type=int, help='Repeat a sampled start; otherwise choose a fresh seed'
    )
    parser.add_argument(
        '--start-mode',
        choices=('random', 'settled', 'gentle'),
        default='random',
        help='Random phase, settled lying pose, or more even initial support',
    )
    parser.add_argument('--checkpoint-dir', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--seconds', type=float, default=24.0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if args.seconds < 0.02:
        parser.error('seconds must cover at least one control step (0.02 s)')
    torch.set_num_threads(1)
    rng = np.random.default_rng()
    seed = args.seed if args.seed is not None else int(rng.integers(2**31))
    sim = Simulation(
        args.checkpoint_dir, args.terrain, seed, sampler=sampler, start_mode=args.start_mode
    )
    viewer = None
    try:
        sim.reset()
        print(f'{sim.scene["label"]}; seed {sim.seed}; {sim.policy_name}, live MuJoCo.', flush=True)
        if args.headless:
            start = time.monotonic()
            for _ in range(round(args.seconds / sim.control_dt)):
                sim.step()
            print(f'Completed {sim.steps} steps in {time.monotonic()-start:.2f} s.', flush=True)
        else:
            viewer = Viewer(sim, args)
            print(
                f'Open http://localhost:{args.port}; forward this port if running remotely.',
                flush=True,
            )
            while True:
                step_start = time.monotonic()
                if viewer.new_start_requested.is_set():
                    seed = int(rng.integers(2**31))
                    if (viewer.terrain.value == 'flat') != (sim.terrain == 'flat'):
                        actor = sim.actor
                        sim.close()
                        sim = Simulation(
                            args.checkpoint_dir,
                            viewer.terrain.value,
                            seed,
                            actor=actor,
                            sampler=sampler,
                            start_mode=viewer.start_mode.value,
                        )
                        sim.reset()
                        viewer.show_case(sim)
                    else:
                        sim.start_mode = viewer.start_mode.value
                        sim.sample_start(viewer.terrain.value, seed)
                        sim.reset()
                        for client in viewer.server.get_clients().values():
                            viewer.set_camera(client, sim)
                    print(f'{sim.scene["label"]}; seed {sim.seed}', flush=True)
                    viewer.new_start_requested.clear()
                if viewer.reset_requested.is_set():
                    sim.reset()
                    for client in viewer.server.get_clients().values():
                        viewer.set_camera(client, sim)
                    viewer.reset_requested.clear()
                # Keep the initial scene visible until a browser is connected.
                if viewer.play.value and viewer.server.get_clients():
                    if sim.data.time >= args.seconds:
                        if viewer.loop.value:
                            sim.reset()
                        else:
                            viewer.play.value = False
                    else:
                        sim.step()
                viewer.update(sim)
                rate = float(viewer.speed.value.removesuffix('×'))
                time.sleep(max(0, sim.control_dt / rate - (time.monotonic() - step_start)))
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
        if viewer:
            viewer.server.stop()


if __name__ == '__main__':
    main()
