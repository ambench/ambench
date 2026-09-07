# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.app import AppLauncher

# Keep the app alive for the pytest process. Closing it during this module's
# teardown terminates Kit before tests from the other package can run.
simulation_app = AppLauncher(headless=True).app

"""Integration coverage for the rotor actuator across the control pipeline."""

from types import SimpleNamespace

import pytest
import torch

from ambench.controllers.controller_cfg import BaseController, ControllerOutput
from ambench.evaluation.tracking.collect import TrackingCollector
from ambench.robots.fa_hexa import rotor_layout as fa_hexa_rotor_layout
from ambench.robots.omni_hexa import rotor_layout as omni_hexa_rotor_layout
from ambench.robots.robot_cfg import RotorActuatorCfg, RotorLayout
from ambench.robots.robot_io import RobotIO
from ambench.robots.rotor_actuator import RotorActuator
from ambench.robots.ua_hexa import rotor_layout as ua_hexa_rotor_layout
from ambench.robots.ua_quad import rotor_layout as ua_quad_rotor_layout


class _FakeRobot:
    def find_bodies(self, name: str) -> tuple[list[int], list[str]]:
        return [0], [name]

    def find_joints(self, name: str) -> tuple[list[int], list[str]]:
        return [], []


class _TestController(BaseController):
    def __init__(self, desired_wrench: torch.Tensor, **kwargs) -> None:
        super().__init__(**kwargs)
        self._test_desired_wrench = desired_wrench

    def compute_desired_wrench(self, obs: torch.Tensor) -> torch.Tensor:
        return self._test_desired_wrench.expand(obs.shape[0], -1)

    def reset(self, *args, **kwargs) -> None:
        return None


@pytest.mark.parametrize(
    ("provider", "num_rotors", "variable_tilt"),
    [
        (fa_hexa_rotor_layout, 6, False),
        (ua_hexa_rotor_layout, 6, False),
        (omni_hexa_rotor_layout, 6, True),
        (ua_quad_rotor_layout, 4, False),
    ],
)
def test_robot_layout_providers_return_valid_named_layouts(provider, num_rotors: int, variable_tilt: bool) -> None:
    layout = provider(device="cpu")

    assert layout.num_rotors == num_rotors
    assert layout.is_variable_tilt is variable_tilt
    assert layout.positions_b.device.type == "cpu"


def _layout_provider(device: str | torch.device) -> RotorLayout:
    return RotorLayout(
        positions_b=torch.tensor(
            [[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]],
            device=device,
        ),
        thrust_axes_b=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            device=device,
        ),
        spin_directions=torch.tensor([1.0, -1.0], device=device),
    )


def _robot_spec(actuator: RotorActuatorCfg) -> SimpleNamespace:
    multirotor = SimpleNamespace(
        actuator=actuator,
        layout_provider=_layout_provider,
        motor_arm_joint_names=(),
        propeller_viz=None,
        fully_actuated=True,
        aerodynamics=None,
    )
    return SimpleNamespace(
        asset=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"),
        control_body_name="base_link",
        base_body_name="base_link",
        ee_body_name=None,
        arm_joint_names=(),
        gripper=None,
        end_effector=None,
        multirotor=multirotor,
    )


def test_robot_io_keeps_instantaneous_path_without_transient_effects() -> None:
    robot_io = RobotIO(
        robot=_FakeRobot(),
        spec=_robot_spec(RotorActuatorCfg(thrust_limits=(0.0, 20.0))),
        num_envs=2,
        device="cpu",
        dt=0.01,
    )

    assert robot_io.rotor_actuator is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"thrust_limits": (0.0, float("nan"))},
        {"thrust_limits": (0.0, 20.0), "reaction_torque_ratio": float("inf")},
        {"thrust_limits": (0.0, 20.0), "response_time_constant_s": float("nan")},
        {"thrust_limits": (0.0, 20.0), "normalized_acceleration_limit_per_s": float("inf")},
    ],
)
def test_rotor_actuator_cfg_rejects_nonfinite_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="finite"):
        RotorActuatorCfg(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"response_time_constant_s": 0.0},
        {"normalized_acceleration_limit_per_s": -1.0},
    ],
)
def test_rotor_actuator_cfg_rejects_nonpositive_dynamics(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="positive"):
        RotorActuatorCfg(thrust_limits=(0.0, 20.0), **kwargs)


def test_rotor_layout_rejects_nonfinite_geometry() -> None:
    with pytest.raises(ValueError, match="finite"):
        RotorLayout(
            positions_b=torch.tensor([[float("nan"), 0.0, 0.0]]),
            thrust_axes_b=torch.tensor([[0.0, 0.0, 1.0]]),
            spin_directions=torch.tensor([1.0]),
        )


def test_rotor_layout_rejects_invalid_direction_bases() -> None:
    with pytest.raises(ValueError, match="unit vectors"):
        RotorLayout(
            positions_b=torch.zeros((1, 3)),
            thrust_axes_b=torch.tensor([[0.0, 0.0, 2.0]]),
            spin_directions=torch.tensor([1.0]),
        )
    with pytest.raises(ValueError, match="orthogonal"):
        RotorLayout(
            positions_b=torch.zeros((1, 3)),
            thrust_axes_b=torch.tensor([[0.0, 0.0, 1.0]]),
            spin_directions=torch.tensor([1.0]),
            tilt_cos_axes_b=torch.tensor([[0.0, 0.0, 1.0]]),
            tilt_sin_axes_b=torch.tensor([[0.0, 0.0, 1.0]]),
        )


def test_robot_io_owns_and_selectively_resets_reduced_actuator() -> None:
    actuator_cfg = RotorActuatorCfg(
        thrust_limits=(0.0, 20.0),
        reaction_torque_ratio=0.02,
        response_time_constant_s=0.0475,
    )
    robot_io = RobotIO(
        robot=_FakeRobot(),
        spec=_robot_spec(actuator_cfg),
        num_envs=2,
        device="cpu",
        dt=0.01,
    )
    assert robot_io.rotor_actuator is not None
    assert robot_io.rotor_actuator.normalized_speed.shape == (2, 2)

    robot_io.rotor_actuator.step(torch.full((2, 2), 5.0))
    assert torch.all(robot_io.rotor_actuator.normalized_speed > 0.0)

    robot_io.reset(torch.tensor([0]), torch.empty((1, 0)))

    assert torch.count_nonzero(robot_io.rotor_actuator.normalized_speed[0]) == 0
    assert torch.all(robot_io.rotor_actuator.normalized_speed[1] > 0.0)


def _controller(
    *,
    actuator_cfg: RotorActuatorCfg,
    rotor_layout: RotorLayout,
    desired_wrench: torch.Tensor,
    rotor_actuator: RotorActuator | None,
    dt: float,
    enable_saturation: bool = True,
) -> _TestController:
    spec = SimpleNamespace(
        base_body_name="base_link",
        multirotor=SimpleNamespace(
            actuator=actuator_cfg,
            fully_actuated=True,
            aerodynamics=None,
        ),
    )
    return _TestController(
        desired_wrench=desired_wrench,
        num_envs=1,
        device="cpu",
        dt=dt,
        robot_spec=spec,
        rotor_layout=rotor_layout,
        rotor_actuator=rotor_actuator,
        robot=_FakeRobot(),
        scene=None,
        enable_saturation=enable_saturation,
        enable_aerodynamic_effects=False,
        enable_wind_effect=False,
    )


def _single_fixed_rotor_layout() -> RotorLayout:
    return RotorLayout(
        positions_b=torch.zeros((1, 3)),
        thrust_axes_b=torch.tensor([[0.0, 0.0, 1.0]]),
        spin_directions=torch.tensor([1.0]),
    )


def test_controller_preserves_instantaneous_allocation_without_actuator() -> None:
    actuator_cfg = RotorActuatorCfg(
        thrust_limits=(0.0, 100.0),
        reaction_torque_ratio=0.1,
    )
    desired_wrench = torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0, 0.5]])
    controller = _controller(
        actuator_cfg=actuator_cfg,
        rotor_layout=_single_fixed_rotor_layout(),
        desired_wrench=desired_wrench,
        rotor_actuator=None,
        dt=0.1,
    )

    output = controller.compute(torch.zeros((1, 1)))

    torch.testing.assert_close(output.final_wrench_b, desired_wrench)
    torch.testing.assert_close(output.commanded_motor_thrusts, torch.tensor([[5.0]]))
    torch.testing.assert_close(output.motor_thrusts, torch.tensor([[5.0]]))
    assert output.rotor_actuator_output is None


def test_controller_applies_reduced_rotor_dynamics() -> None:
    output, _ = _fixed_dynamic_output()

    assert output.rotor_actuator_output is not None
    torch.testing.assert_close(output.commanded_motor_thrusts, torch.tensor([[100.0]]))
    torch.testing.assert_close(output.motor_thrusts, torch.tensor([[0.25]]))
    torch.testing.assert_close(output.final_wrench_b, torch.tensor([[0.0, 0.0, 0.25, 0.0, 0.0, 0.025]]))


def test_controller_requires_saturation_for_reduced_rotor_dynamics() -> None:
    actuator_cfg = RotorActuatorCfg(
        thrust_limits=(0.0, 100.0),
        response_time_constant_s=0.1,
    )
    rotor_layout = _single_fixed_rotor_layout()
    rotor_actuator = RotorActuator(
        actuator_cfg,
        num_envs=1,
        num_rotors=1,
        dt=0.1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="require enable_saturation=True"):
        _controller(
            actuator_cfg=actuator_cfg,
            rotor_layout=rotor_layout,
            desired_wrench=torch.zeros((1, 6)),
            rotor_actuator=rotor_actuator,
            dt=0.1,
            enable_saturation=False,
        )


def test_transient_actuator_owns_saturation_and_reports_raw_request() -> None:
    actuator_cfg = RotorActuatorCfg(
        thrust_limits=(0.0, 100.0),
        reaction_torque_ratio=0.1,
        response_time_constant_s=0.1,
    )
    rotor_layout = _single_fixed_rotor_layout()
    rotor_actuator = RotorActuator(
        actuator_cfg,
        num_envs=1,
        num_rotors=1,
        dt=0.1,
        device="cpu",
    )
    controller = _controller(
        actuator_cfg=actuator_cfg,
        rotor_layout=rotor_layout,
        desired_wrench=torch.tensor([[0.0, 0.0, 200.0, 0.0, 0.0, 20.0]]),
        rotor_actuator=rotor_actuator,
        dt=0.1,
    )

    output = controller.compute(torch.zeros((1, 1)))

    assert output.rotor_actuator_output is not None
    assert output.commanded_motor_thrusts.item() > actuator_cfg.thrust_limits[1]
    assert output.rotor_actuator_output.commanded_thrust.item() == pytest.approx(100.0)
    assert output.rotor_actuator_output.thrust_saturated.item()
    assert output.motor_thrusts.item() == pytest.approx(100.0)
    robot_spec = SimpleNamespace(
        max_arm_reach=1.0,
        multirotor=SimpleNamespace(actuator=actuator_cfg),
    )
    env_unwrapped = SimpleNamespace(
        cfg=SimpleNamespace(robot_profile=SimpleNamespace(robot=robot_spec)),
        control_output=output,
    )
    _, _, motor_saturated = TrackingCollector(SimpleNamespace(unwrapped=env_unwrapped))._motor()
    assert motor_saturated


def _fixed_dynamic_output() -> tuple[ControllerOutput, RotorActuatorCfg]:
    actuator_cfg = RotorActuatorCfg(
        thrust_limits=(0.0, 100.0),
        reaction_torque_ratio=0.1,
        normalized_acceleration_limit_per_s=0.5,
    )
    rotor_layout = _single_fixed_rotor_layout()
    rotor_actuator = RotorActuator(
        actuator_cfg,
        num_envs=1,
        num_rotors=1,
        dt=0.1,
        device="cpu",
    )
    rotor_actuator.step(torch.zeros((1, 1)))
    controller = _controller(
        actuator_cfg=actuator_cfg,
        rotor_layout=rotor_layout,
        desired_wrench=torch.tensor([[0.0, 0.0, 100.0, 0.0, 0.0, 10.0]]),
        rotor_actuator=rotor_actuator,
        dt=0.1,
    )

    return controller.compute(torch.zeros((1, 1))), actuator_cfg


def test_tracking_collector_exposes_commanded_and_realized_actuator_telemetry() -> None:
    output, actuator_cfg = _fixed_dynamic_output()
    robot_spec = SimpleNamespace(
        max_arm_reach=1.0,
        multirotor=SimpleNamespace(actuator=actuator_cfg),
    )
    env_unwrapped = SimpleNamespace(
        cfg=SimpleNamespace(robot_profile=SimpleNamespace(robot=robot_spec)),
        scene=SimpleNamespace(env_origins=torch.zeros((1, 3))),
        control_output=output,
        dt=0.1,
    )
    collector = TrackingCollector(SimpleNamespace(unwrapped=env_unwrapped))
    raw_obs = {
        "policy": [{
            "ee_pos": torch.zeros(3),
            "ee_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "base_pos": torch.zeros(3),
            "base_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "arm_joint_pos": torch.empty(0),
            "gripper_width": torch.zeros(1),
        }]
    }

    control = collector.collect(raw_obs, timestep=0)["control"]

    assert control["commanded_motor_thrusts"] == pytest.approx([100.0])
    assert control["motor_thrusts"] == pytest.approx([0.25])
    assert control["rotor_normalized_speed_commands"] == pytest.approx([1.0])
    assert control["rotor_normalized_speeds"] == pytest.approx([0.05])
    assert control["rotor_normalized_accelerations_per_s"] == pytest.approx([0.5])
    assert control["actuator_acceleration_limited"] == [True]


def test_controller_reconstructs_variable_tilt_axis_for_realized_wrench() -> None:
    actuator_cfg = RotorActuatorCfg(
        thrust_limits=(0.0, 100.0),
        reaction_torque_ratio=0.1,
        response_time_constant_s=0.1,
    )
    cos_axis = torch.tensor([[0.0, 0.0, 1.0]])
    sin_axis = torch.tensor([[1.0, 0.0, 0.0]])
    rotor_layout = RotorLayout(
        positions_b=torch.zeros((1, 3)),
        thrust_axes_b=cos_axis,
        spin_directions=torch.tensor([1.0]),
        tilt_cos_axes_b=cos_axis,
        tilt_sin_axes_b=sin_axis,
    )
    rotor_actuator = RotorActuator(
        actuator_cfg,
        num_envs=1,
        num_rotors=1,
        dt=0.1,
        device="cpu",
    )
    desired_wrench = torch.tensor([[4.0, 0.0, 3.0, 0.4, 0.0, 0.3]])
    controller = _controller(
        actuator_cfg=actuator_cfg,
        rotor_layout=rotor_layout,
        desired_wrench=desired_wrench,
        rotor_actuator=rotor_actuator,
        dt=0.1,
    )

    output = controller.compute(torch.zeros((1, 1)))

    assert output.rotor_actuator_output is not None
    torch.testing.assert_close(output.commanded_motor_thrusts, torch.tensor([[5.0]]))
    torch.testing.assert_close(output.motor_thrusts, torch.tensor([[5.0]]))
    torch.testing.assert_close(
        output.motor_arm_angles, torch.tensor([[torch.atan2(torch.tensor(4.0), torch.tensor(3.0))]])
    )
    torch.testing.assert_close(output.final_wrench_b, desired_wrench)
