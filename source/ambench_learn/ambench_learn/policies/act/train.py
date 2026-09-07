#!/usr/bin/env python
# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Train LeRobot ACT with optional benchmark-aligned on-the-fly dataset resampling."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from pprint import pformat

import lerobot.policies.act.processor_act as lerobot_act_processor
import torch
from lerobot.datasets.factory import IMAGENET_STATS, ImageTransforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_train as train_module

from ambench_learn.data.action_resampling import (
    infer_action_window_length,
    resolve_dataset_action_semantics,
)
from ambench_learn.data.action_semantics import (
    POLICY_ACTION_REPRESENTATION_CHOICES,
    resolve_policy_action_representation_for_dataset,
)
from ambench_learn.data.resampled_lerobot_dataset import ResampledLeRobotDataset
from ambench_learn.policies.act.se_relative_processor import (
    make_act_pre_post_processors as make_am_act_pre_post_processors,
)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--policy_target_hz",
        type=int,
        default=None,
        help="Logical policy FPS to derive on the fly from the canonical raw LeRobot dataset.",
    )
    parser.add_argument(
        "--policy_action_representation",
        type=str,
        default=None,
        choices=POLICY_ACTION_REPRESENTATION_CHOICES,
        help=(
            "Optional policy-side ACT action representation override. "
            "Canonical absolute datasets are trained as relative trajectories."
        ),
    )
    benchmark_args, upstream_argv = parser.parse_known_args(sys.argv[1:])
    original_make_dataset = train_module.make_dataset
    lerobot_act_processor.make_act_pre_post_processors = make_am_act_pre_post_processors

    def custom_make_dataset(cfg):
        requested_representation = benchmark_args.policy_action_representation
        if cfg.dataset.streaming:
            raise NotImplementedError("Streaming datasets are not supported by the benchmark resampler yet.")
        if not isinstance(cfg.dataset.repo_id, str):
            raise NotImplementedError("The benchmark resampler currently expects a single LeRobot dataset.")

        image_transforms = (
            ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
        )
        raw_dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=cfg.dataset.episodes,
            image_transforms=image_transforms,
            revision=cfg.dataset.revision,
            video_backend=cfg.dataset.video_backend,
        )
        dataset_action_semantics = resolve_dataset_action_semantics(raw_dataset)
        resolved_representation = resolve_policy_action_representation_for_dataset(
            dataset_action_semantics,
            requested_representation,
            policy_name="ACT",
        )
        cfg.policy.action_representation = resolved_representation
        should_wrap_dataset = benchmark_args.policy_target_hz is not None or resolved_representation.endswith(
            "_relative"
        )
        if not should_wrap_dataset:
            return original_make_dataset(cfg)
        target_fps = raw_dataset.fps if benchmark_args.policy_target_hz is None else benchmark_args.policy_target_hz
        dataset = ResampledLeRobotDataset(
            raw_dataset,
            target_fps=target_fps,
            action_window_length=infer_action_window_length(cfg.policy),
            policy_action_representation=resolved_representation,
        )

        if cfg.dataset.use_imagenet_stats:
            for key in dataset.meta.camera_keys:
                for stats_type, stats in IMAGENET_STATS.items():
                    dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

        dataset_report = dataset.build_report()
        logging.info("Benchmark resampling enabled:\n%s", pformat(dataset_report))
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "benchmark_dataset_report.json"
        report_payload = {
            **dataset_report,
            "dataset_repo_id": str(cfg.dataset.repo_id),
            "dataset_root": str(cfg.dataset.root),
        }
        report_path.write_text(json.dumps(report_payload, indent=2))
        return dataset

    train_module.make_dataset = custom_make_dataset
    sys.argv = [sys.argv[0], *upstream_argv]
    train_module.train()


if __name__ == "__main__":
    main()
