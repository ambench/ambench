# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR  # noqa: F401

from ambench.assets.object_specs import (
    FRAME_DEFAULT_SIZE,
    PEG_DEFAULT_LENGTH,
)
from ambench.scenes import WallSceneCfg, spawn_wall
from ambench.tasks.base.base_env_cfg import BaseEnvCfg
from ambench.tasks.base.robot_profiles import (
    EE_ABS_L1,
    EE_ABS_PID,
    FA_HEXA_ABS_L1,
    FA_HEXA_ABS_MPC,
    FA_HEXA_ABS_PID,
    OMNI_HEXA_ABS_PID,
    UA_HEXA_ABS_PID,
    UA_QUAD_ABS_PID,
)
from ambench.tasks.frame_assembly import frame_assembly_events
from ambench.utils.assets import LOCAL_ASSET_DIR


def get_peg_local_offsets(peg_separation: float, peg_offset_x: float) -> list[tuple[float, float, float]]:
    """Return peg offsets in wall-local coordinates."""
    peg_offset_x = -abs(peg_offset_x)
    return [
        (peg_offset_x, -peg_separation / 2, peg_separation / 2),
        (peg_offset_x, peg_separation / 2, peg_separation / 2),
        (peg_offset_x, -peg_separation / 2, -peg_separation / 2),
        (peg_offset_x, peg_separation / 2, -peg_separation / 2),
    ]


def spawn_peg(
    peg_index: int, position: tuple[float, float, float], rotation: tuple[float, float, float, float]
) -> RigidObjectCfg:
    """Create a peg rigid-object config."""

    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Peg_{peg_index}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/peg.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position,
            rot=rotation,
        ),
    )


@configclass
class EventCfg:
    """Configuration for domain randomization."""

    # Randomize wall pose and update pegs to follow wall pose.
    randomize_wall_pose = EventTerm(
        func=frame_assembly_events.randomize_wall_and_pegs,
        mode="reset",
        params={
            "wall_cfg": SceneEntityCfg("wall"),
            "peg_1_cfg": SceneEntityCfg("peg_1"),
            "peg_2_cfg": SceneEntityCfg("peg_2"),
            "peg_3_cfg": SceneEntityCfg("peg_3"),
            "peg_4_cfg": SceneEntityCfg("peg_4"),
            "wall_pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.2, 0.2),
            },
        },
    )

    # Randomize frame position on the ground.
    randomize_frame_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.08, 0.08),
                "y": (-0.12, 0.12),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("frame_object"),
        },
    )


@configclass
class FrameAssemblyEnvDefaultCfg(BaseEnvCfg):
    episode_length_s = 30

    # Wall configuration
    wall: WallSceneCfg = WallSceneCfg()
    wall_position = (2.0, 0.0, 1.0)
    wall_rotation = (1, 0, 0, 0)
    wall_thickness = wall.size[0]
    wall_cfg: RigidObjectCfg = spawn_wall(
        prim_path="/World/envs/env_.*/Wall",
        cfg=wall,
        translation=wall_position,
        orientation=wall_rotation,
    )

    # Peg configuration
    peg_separation = 0.4
    peg_offset_x = -0.5 * (wall_thickness + PEG_DEFAULT_LENGTH)
    num_pegs = 4

    # Frame configuration
    frame_width = 0.5
    frame_height = 0.5
    frame_thickness = 0.025
    frame_mass = 0.01

    # Frame object configuration
    frame_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Frame",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/frame.usd",
            scale=(
                frame_thickness / FRAME_DEFAULT_SIZE[2],
                frame_width / FRAME_DEFAULT_SIZE[0],
                frame_height / FRAME_DEFAULT_SIZE[1],
            ),  # scale = (0.5, 0.5, 0.5)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=frame_mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.7, 0.0, frame_height / 2),
        ),
    )

    # domain randomization
    events = EventCfg()

    # success criteria configuration
    frame_placement_tolerance_x = 0.1  # 10 cm tolerance in x direction
    frame_placement_tolerance_yz = 0.05  # 5 cm tolerance in y and z directions

    def __post_init__(self):
        super().__post_init__()

        self.peg_local_offsets = get_peg_local_offsets(self.peg_separation, self.peg_offset_x)
        peg_positions = [
            (
                self.wall_position[0] + local_offset[0],
                self.wall_position[1] + local_offset[1],
                self.wall_position[2] + local_offset[2],
            )
            for local_offset in self.peg_local_offsets
        ]
        self.peg_cfgs = []
        for i in range(self.num_pegs):
            peg_cfg = spawn_peg(i + 1, peg_positions[i], self.wall_rotation)
            self.peg_cfgs.append(peg_cfg)


@configclass
class FrameAssemblyEnvEEAbsPIDCfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class FrameAssemblyEnvEEAbsL1Cfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class FrameAssemblyEnvOmniHexaAbsPIDCfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class FrameAssemblyEnvFAHexaAbsPIDCfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class FrameAssemblyEnvFAHexaAbsL1Cfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class FrameAssemblyEnvUAHexaAbsPIDCfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class FrameAssemblyEnvUAQuadAbsPIDCfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID


@configclass
class FrameAssemblyEnvFAHexaAbsMPCCfg(FrameAssemblyEnvDefaultCfg):
    """Frame assembly environment configuration for FA-Hexa robot, absolute action with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC
