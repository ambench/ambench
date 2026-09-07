# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from torch import Tensor

from ambench_learn.data.action_semantics import (
    BASE_JOINT_RELATIVE_DIM,
    EE_LOCAL_RELATIVE_DIM,
    localize_base_joint_observation_state,
    localize_ee_observation_state,
    to_base_joint_absolute_trajectory,
    to_base_joint_relative_trajectory,
    to_ee_local_absolute_trajectory,
    to_ee_local_relative_trajectory,
)


class _DecodeAnchorMixin:
    _last_state: Tensor | None
    _decode_anchor_state: Tensor | None

    def cache_decode_anchor(self, state: Tensor | None = None) -> None:
        source = self._last_state if state is None else state
        self._decode_anchor_state = None if source is None else source.detach().clone()

    def clear_decode_anchor(self) -> None:
        self._decode_anchor_state = None

    def resolve_decode_anchor(self) -> Tensor | None:
        if self._decode_anchor_state is not None:
            return self._decode_anchor_state
        return self._last_state


@ProcessorStepRegistry.register("ambench_ee_local_relative_trajectory_processor")
@dataclass
class EELocalRelativeTrajectoryProcessorStep(_DecodeAnchorMixin, ProcessorStep):
    enabled: bool = False
    _last_state: Tensor | None = field(default=None, init=False, repr=False)
    _decode_anchor_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state[..., :EE_LOCAL_RELATIVE_DIM]

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        if state is not None:
            new_observation = dict(observation)
            new_observation[OBS_STATE] = localize_ee_observation_state(state)
            new_transition[TransitionKey.OBSERVATION] = new_observation

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and state is not None:
            new_transition[TransitionKey.ACTION] = to_ee_local_relative_trajectory(action, state)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ambench_ee_local_absolute_trajectory_processor")
@dataclass
class EELocalAbsoluteTrajectoryProcessorStep(ProcessorStep):
    enabled: bool = False
    relative_step: EELocalRelativeTrajectoryProcessorStep | None = field(default=None, repr=False)
    _last_relative_action: Tensor | None = field(default=None, init=False, repr=False)
    _last_absolute_action: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError(
                "EELocalAbsoluteTrajectoryProcessorStep requires a paired EELocalRelativeTrajectoryProcessorStep."
            )
        anchor_state = self.relative_step.resolve_decode_anchor()
        if anchor_state is None:
            raise RuntimeError("EELocalAbsoluteTrajectoryProcessorStep requires a cached local decode anchor state.")

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        self._last_relative_action = action.detach().clone()
        absolute = to_ee_local_absolute_trajectory(action, anchor_state)
        self._last_absolute_action = absolute.detach().clone()
        new_transition[TransitionKey.ACTION] = absolute
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ambench_base_joint_relative_trajectory_processor")
@dataclass
class BaseJointRelativeTrajectoryProcessorStep(_DecodeAnchorMixin, ProcessorStep):
    enabled: bool = False
    _last_state: Tensor | None = field(default=None, init=False, repr=False)
    _decode_anchor_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state[..., :BASE_JOINT_RELATIVE_DIM]

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        if state is not None:
            new_observation = dict(observation)
            new_observation[OBS_STATE] = localize_base_joint_observation_state(state)
            new_transition[TransitionKey.OBSERVATION] = new_observation

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition
        new_transition[TransitionKey.ACTION] = to_base_joint_relative_trajectory(action, state)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ambench_base_joint_absolute_trajectory_processor")
@dataclass
class BaseJointAbsoluteTrajectoryProcessorStep(ProcessorStep):
    enabled: bool = False
    relative_step: BaseJointRelativeTrajectoryProcessorStep | None = field(default=None, repr=False)
    _last_relative_action: Tensor | None = field(default=None, init=False, repr=False)
    _last_absolute_action: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError(
                "BaseJointAbsoluteTrajectoryProcessorStep requires a paired BaseJointRelativeTrajectoryProcessorStep."
            )
        anchor_state = self.relative_step.resolve_decode_anchor()
        if anchor_state is None:
            raise RuntimeError(
                "BaseJointAbsoluteTrajectoryProcessorStep requires a cached relative decode anchor state."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        self._last_relative_action = action.detach().clone()
        absolute = to_base_joint_absolute_trajectory(action, anchor_state)
        self._last_absolute_action = absolute.detach().clone()
        new_transition[TransitionKey.ACTION] = absolute
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    action_representation = getattr(config, "action_representation", None)
    if action_representation == "absolute":
        raise ValueError(
            "ACT action_representation='absolute' is no longer supported. "
            "Use 'ee_local_relative' for absolute EE datasets."
        )
    if action_representation == "ee_relative":
        raise ValueError(
            "ACT action_representation='ee_relative' is no longer supported. "
            "Use 'ee_local_relative' for absolute EE datasets."
        )
    use_ee_local_relative = action_representation == "ee_local_relative"
    use_base_joint_relative = action_representation == "base_joint_relative"
    if use_ee_local_relative:
        state_feature = config.input_features.get("observation.state")
        action_feature = config.output_features.get("action")
        if state_feature is None or state_feature.shape is None or state_feature.shape[0] < 8:
            raise ValueError(
                f"ACT {action_representation} requires observation.state to begin with "
                "[ee_pos(3), ee_quat(4), gripper_width(1)]."
            )
        if action_feature is None or action_feature.shape is None or action_feature.shape[0] != 8:
            raise ValueError(f"ACT {action_representation} requires 8D absolute EE actions.")
    if use_base_joint_relative:
        state_feature = config.input_features.get("observation.state")
        action_feature = config.output_features.get("action")
        if state_feature is None or state_feature.shape is None or state_feature.shape[0] < 12:
            raise ValueError(
                "ACT base_joint_relative requires observation.state to begin with "
                "[base_pos(3), base_quat(4), arm_joint_pos(4), gripper_width(1)]."
            )
        if action_feature is None or action_feature.shape is None or action_feature.shape[0] != 12:
            raise ValueError("ACT base_joint_relative requires 12D absolute Base+joints actions.")

    ee_local_relative_step = EELocalRelativeTrajectoryProcessorStep(enabled=use_ee_local_relative)
    base_joint_relative_step = BaseJointRelativeTrajectoryProcessorStep(enabled=use_base_joint_relative)
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        ee_local_relative_step,
        base_joint_relative_step,
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        EELocalAbsoluteTrajectoryProcessorStep(
            enabled=use_ee_local_relative,
            relative_step=ee_local_relative_step,
        ),
        BaseJointAbsoluteTrajectoryProcessorStep(
            enabled=use_base_joint_relative,
            relative_step=base_joint_relative_step,
        ),
        DeviceProcessorStep(device="cpu"),
    ]
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )


def reconnect_se_relative_absolute_steps(preprocessor, postprocessor) -> None:
    ee_local_relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, EELocalRelativeTrajectoryProcessorStep)), None
    )
    base_joint_relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, BaseJointRelativeTrajectoryProcessorStep)), None
    )
    for step in postprocessor.steps:
        if (
            isinstance(step, EELocalAbsoluteTrajectoryProcessorStep)
            and step.relative_step is None
            and ee_local_relative_step is not None
        ):
            step.relative_step = ee_local_relative_step
        if (
            isinstance(step, BaseJointAbsoluteTrajectoryProcessorStep)
            and step.relative_step is None
            and base_joint_relative_step is not None
        ):
            step.relative_step = base_joint_relative_step
