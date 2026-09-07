# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, The AM-Bench Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the ``ambench`` Python package."""

import tomllib
from pathlib import Path

from setuptools import find_packages, setup

EXTENSION_PATH = Path(__file__).resolve().parent
PACKAGE_PATH = EXTENSION_PATH / "ambench"
with (EXTENSION_PATH / "config" / "extension.toml").open("rb") as extension_toml_file:
    EXTENSION_TOML_DATA = tomllib.load(extension_toml_file)


def collect_package_data(root: Path, suffixes: tuple[str, ...]) -> list[str]:
    """Collect non-Python package assets that must ship with the wheel."""

    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)


INSTALL_REQUIRES = [
    "casadi>=3.7.2",
    "imageio>=2.37.0",
    "matplotlib>=3.10.3",
    "moviepy>=2.2.1",
    "psutil>=7.0.0",
    "pyyaml>=6.0.2",
    "scipy>=1.15.3",
    "yourdfpy>=0.0.58",
]


setup(
    name="ambench",
    packages=find_packages(exclude=["config", "config.*"]),
    package_data={
        # Keep this list in sync with the asset formats under ``ambench/assets``:
        # USD variants (.usd/.usda/.usdz), MDL materials, meshes, textures, and the
        # JSON asset maps are all resolved at runtime and must ship with the wheel.
        "ambench": collect_package_data(
            PACKAGE_PATH,
            (".json", ".mdl", ".png", ".stl", ".urdf", ".usd", ".usda", ".usdz"),
        ),
    },
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.11",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
