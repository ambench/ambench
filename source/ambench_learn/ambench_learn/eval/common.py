# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared runtime helpers for policy evaluation entrypoints."""

from __future__ import annotations

import argparse
import contextlib
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_NUM_ROLLOUTS = 10
DEFAULT_PROGRESS_EVERY = 100
DEFAULT_VIDEO_CAMERA_NAMES = ("ee_camera", "scene_camera")


@dataclass(frozen=True)
class EvalBatch:
    """Indices and optional seed for one vectorized rollout batch."""

    start: int
    stop: int
    seed: int | None

    @property
    def size(self) -> int:
        return self.stop - self.start

    def label(self) -> str:
        """Return the one-based rollout range shown in progress output."""

        return str(self.start + 1) if self.size == 1 else f"{self.start + 1}-{self.stop}"


def add_common_eval_args(
    parser: argparse.ArgumentParser,
) -> None:
    """Add the CLI shared by ACT, DP, and OpenPI evaluation."""

    parser.add_argument("--task", type=str, default=None, help="Name of the Isaac Lab task (gym registry id).")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel evaluation environments.")
    parser.add_argument("--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS)
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save rollout videos.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Override the environment episode length in seconds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for environment randomization.",
    )
    parser.add_argument(
        "--video-camera-names",
        nargs="+",
        default=list(DEFAULT_VIDEO_CAMERA_NAMES),
        help="Camera sensors included in saved video.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact evaluation output directory. Defaults to outputs/eval/<policy>/<task>/<timestamp>.",
    )
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument(
        "--disturbance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable saturation, aerodynamic effects, and wind during evaluation.",
    )


def validate_common_eval_args(args: argparse.Namespace) -> None:
    """Reject invalid common arguments before Isaac Sim starts."""

    if not args.task:
        raise ValueError("--task is required.")
    if args.num_envs < 1:
        raise ValueError(f"Expected --num-envs >= 1, got {args.num_envs}.")
    if args.num_rollouts < 1:
        raise ValueError(f"Expected --num-rollouts >= 1, got {args.num_rollouts}.")
    if args.episode_length_s is not None and args.episode_length_s <= 0:
        raise ValueError(f"Expected --episode-length-s > 0, got {args.episode_length_s}.")
    if args.progress_every < 1:
        raise ValueError(f"Expected --progress-every >= 1, got {args.progress_every}.")
    if args.save_video and not args.video_camera_names:
        raise ValueError("--video-camera-names must include at least one camera when video saving is enabled.")


def resolve_eval_output_dir(*, policy: str, task: str, output_dir: str | Path | None) -> Path:
    """Resolve the canonical output directory for one evaluation run."""

    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    task_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", task).strip("_") or "task"
    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S.%f")
    return (Path.cwd() / "outputs" / "eval" / policy.lower() / task_slug / timestamp).resolve()


def build_common_eval_metadata(
    args: argparse.Namespace,
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build the runtime metadata recorded by every learned-policy evaluator."""

    return {
        "Device": args.device,
        "Seed": args.seed,
        "Episode length override (s)": args.episode_length_s,
        "Disturbance": bool(args.disturbance),
        "Save video": bool(args.save_video),
        "Video cameras": list(args.video_camera_names) if args.save_video else [],
        "Output directory": str(Path(output_dir).resolve()),
        "Progress interval (steps)": int(args.progress_every),
    }


def validate_video_camera_names(env: Any, camera_names: Sequence[str]) -> list[str]:
    """Require every requested video camera to exist in the environment sensor registry."""

    requested = list(dict.fromkeys(camera_names))
    if not requested:
        raise ValueError("At least one video camera must be requested when video saving is enabled.")
    try:
        available = sorted(env.unwrapped.scene.sensors.keys())
    except AttributeError as error:
        raise ValueError("The environment does not expose a scene sensor registry for video recording.") from error
    missing = [camera_name for camera_name in requested if camera_name not in available]
    if missing:
        raise ValueError(
            f"Requested video camera(s) are unavailable: {missing}. "
            f"Available environment sensors: {available or '<none>'}."
        )
    return requested


def configure_env_cfg(
    *,
    parse_env_cfg: Callable[..., Any],
    task: str,
    device: str,
    num_envs: int,
    seed: int | None = None,
    episode_length_s: float | None = None,
    disturbance: bool = False,
    scene_camera_cfg: Any | None = None,
) -> Any:
    """Build an eval environment config with common runtime overrides."""

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if seed is not None:
        env_cfg.seed = seed
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = {}
    if disturbance:
        env_cfg.enable_saturation = True
        env_cfg.enable_aerodynamic_effects = True
        env_cfg.enable_wind_effect = True
    if scene_camera_cfg is not None:
        env_cfg.scene_camera_cfg = scene_camera_cfg
    return env_cfg


def make_scene_camera_cfg(
    *,
    position: tuple[float, float, float] = (-2.0, -2.5, 1.6),
    look_at: tuple[float, float, float] = (2.0, 0.0, 1.0),
) -> Any:
    """Build the shared side-view scene camera after Isaac Sim starts."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera.camera_cfg import CameraCfg

    from ambench.utils.camera_utils import compute_camera_quat_from_lookat

    return CameraCfg(
        prim_path="/World/envs/env_.*/scene_camera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=compute_camera_quat_from_lookat(position, look_at),
            convention="ros",
        ),
    )


def wrap_video(
    env: Any,
    *,
    enabled: bool,
    output_dir: str | Path,
    camera_names: Sequence[str],
    name_prefix: str,
    record_video_cls: type[Any],
) -> tuple[Any, Path | None]:
    """Wrap an environment with the canonical video settings when requested."""

    if not enabled:
        return env, None
    video_dir = Path(output_dir) / "videos"
    env = record_video_cls(
        env,
        video_folder=str(video_dir),
        camera_names=list(camera_names),
        name_prefix=name_prefix,
        fps=30,
        frame_skip=4,
        output_format="mp4",
    )
    return env, video_dir


def run_evaluation_batches(
    *,
    eval_run: Any,
    env: Any,
    simulation_app: Any,
    requested_rollouts: int,
    num_envs: int,
    run_batch: Callable[[EvalBatch], bool | None],
    base_seed: int | None = None,
    video_folder: str | Path | None = None,
) -> dict[str, Any]:
    """Run rollout batches and guarantee artifact finalization and environment cleanup.

    ``run_batch`` owns policy inference and must append completed records to
    ``eval_run``. Returning true requests a clean early stop.
    """

    started_at = time.perf_counter()
    status = "completed"
    error_message = None
    try:
        while len(eval_run.rollout_records) < requested_rollouts:
            if not simulation_app.is_running() or simulation_app.is_exiting():
                status = "stopped"
                break
            batch_start = len(eval_run.rollout_records)
            batch = EvalBatch(
                start=batch_start,
                stop=min(batch_start + num_envs, requested_rollouts),
                seed=None if base_seed is None else base_seed + batch_start,
            )
            stop_requested = bool(run_batch(batch))
            if len(eval_run.rollout_records) == batch_start and not stop_requested:
                raise RuntimeError("Evaluation batch completed without finalizing any rollout records.")

            # Release cached policy memory before starting the next batch.
            with contextlib.suppress(ImportError):
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if stop_requested:
                status = "stopped"
                break
    except KeyboardInterrupt:
        status = "interrupted"
        error_message = "KeyboardInterrupt"
    except Exception as error:
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            payload = eval_run.finish(
                eval_wall_time_s=time.perf_counter() - started_at,
                video_folder=video_folder,
                status=status,
                error=error_message,
            )
        finally:
            env.close()
    return payload
