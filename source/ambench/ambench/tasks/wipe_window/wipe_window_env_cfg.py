# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

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
from ambench.tasks.wipe_window import wipe_window_events
from ambench.utils.assets import LOCAL_ASSET_DIR


def create_dot_cfg(dot_idx: int) -> RigidObjectCfg:
    """Create a dot configuration for a specific index.

    Args:
        dot_idx: Index of the dot (0-99).

    Returns:
        RigidObjectCfg: Configuration for the dot.
    """

    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Dot_{dot_idx}",
        spawn=sim_utils.CylinderCfg(
            radius=0.02,
            height=0.0005,  # 0.5mm thickness (thicker for collision detection)
            axis="X",  # Align cylinder along X-axis
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # Kinematic for stable positioning
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2),  # Red color
                roughness=0.1,
            ),
        ),
    )


@configclass
class EventCfg:
    """Configuration for wipe-window domain randomization."""

    randomize_window_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("window_object"),
        },
    )

    set_dot_positions = EventTerm(
        func=wipe_window_events.set_dots_position_on_window,
        mode="reset",
        params={
            "window_cfg": SceneEntityCfg("window_object"),
        },
    )

    randomize_window_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("window_object"),
            "static_friction_range": (0.2, 0.4),
            "dynamic_friction_range": (0.1, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )


@configclass
class WipeWindowEnvDefaultCfg(BaseEnvCfg):
    """Default configuration for the wipe-window task."""

    episode_length_s = 20

    # Environment-specific parameters

    # Window
    window_position = (2.0, 0.0, 1.0)
    window_rotation = (1, 0, 0, 0)
    window_width: float = 1.0
    window_height: float = 1.0
    window_thickness: float = 0.01
    window_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Window",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{LOCAL_ASSET_DIR}/objects/window.usd",
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=window_position,
            rot=window_rotation,
        ),
    )

    # Contact sensor
    contact_sensor_cfg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Sponge",
        filter_prim_paths_expr=["/World/envs/env_.*/Window"],
        track_pose=False,
        track_contact_points=False,
        debug_vis=False,
        update_period=0.0,
    )

    # Dot layout
    num_dots: int = 4

    # Sponge
    sponge_size = (0.12, 0.04, 0.08)
    sponge_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Sponge",
        spawn=sim_utils.CuboidCfg(
            size=sponge_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.75, 0.3),
                roughness=0.8,
            ),
        ),
    )

    # domain randomization
    events = EventCfg()

    # success criteria
    stain_removal_force_threshold: float = 1.25
    stain_removal_contact_count: int = 1
    stain_removal_contact_radius: float = 0.08

    def __post_init__(self):
        super().__post_init__()

        self.observation_space.spaces["stain_pos"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_dots, 3), dtype=np.float32
        )
        self.observation_space.spaces["stain_visible"] = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_dots,), dtype=np.float32
        )

        # Create dot object configs.
        for i in range(self.num_dots):
            setattr(self, f"dot_{i}_object_cfg", create_dot_cfg(i))


@configclass
class WipeWindowEnvEEAbsPIDCfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for End-Effector robot, absolute action with PID controller."""

    robot_profile = EE_ABS_PID


@configclass
class WipeWindowEnvOmniHexaAbsPIDCfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for Omni-Hexa, absolute action with PID controller."""

    robot_profile = OMNI_HEXA_ABS_PID


@configclass
class WipeWindowEnvFAHexaAbsPIDCfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for FA-Hexa robot, absolute action with PID controller."""

    robot_profile = FA_HEXA_ABS_PID


@configclass
class WipeWindowEnvEEAbsL1Cfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for End-Effector robot, absolute action with L1 Adaptive controller."""

    robot_profile = EE_ABS_L1


@configclass
class WipeWindowEnvFAHexaAbsL1Cfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for FA-Hexa robot, absolute action with L1 Adaptive controller."""

    robot_profile = FA_HEXA_ABS_L1


@configclass
class WipeWindowEnvUAHexaAbsPIDCfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for UA-Hexa robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_HEXA_ABS_PID


@configclass
class WipeWindowEnvUAQuadAbsPIDCfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for UA-Quad robot, absolute action with 4DOF PID controller."""

    robot_profile = UA_QUAD_ABS_PID


@configclass
class WipeWindowEnvFAHexaAbsMPCCfg(WipeWindowEnvDefaultCfg):
    """Wipe window environment configuration for FA-Hexa robot with Whole-Body MPC controller."""

    robot_profile = FA_HEXA_ABS_MPC
