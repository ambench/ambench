# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared local asset paths."""

from pathlib import Path

LOCAL_ASSET_PATH = Path(__file__).resolve().parent.parent / "assets"
LOCAL_ASSET_DIR = str(LOCAL_ASSET_PATH)

__all__ = ["LOCAL_ASSET_DIR", "LOCAL_ASSET_PATH"]
