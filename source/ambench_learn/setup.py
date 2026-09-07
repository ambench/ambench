# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, The AM-Bench Contributors.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the ``ambench_learn`` Python package."""

import tomllib
from pathlib import Path

from setuptools import find_namespace_packages, setup

EXTENSION_PATH = Path(__file__).resolve().parent
PACKAGE_PATH = EXTENSION_PATH / "ambench_learn"
with (EXTENSION_PATH / "config" / "extension.toml").open("rb") as extension_toml_file:
    EXTENSION_TOML_DATA = tomllib.load(extension_toml_file)


def collect_package_data(root: Path, suffixes: tuple[str, ...]) -> list[str]:
    """Collect non-Python config files that must ship with the wheel."""

    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)


# The documented install runs `uv pip install --no-deps "lerobot==0.4.4"`, so
# LeRobot's own runtime requirements are not resolved for it. Anything LeRobot
# needs at import time has to be declared here even when nothing in this
# package imports it directly: `draccus` backs the `lerobot_train` CLI that
# `policies/act/train.py` calls, and `packaging` and `ipython` are pulled in
# along the same path. Do not drop them because a usage search comes back empty.
INSTALL_REQUIRES = [
    "ambench",
    "draccus>=0.8.0",
    "einops>=0.8.1",
    "gymnasium>=1.2.0",
    "numpy>=1.26.0",
    "torch>=2.7.0",
]

EXTRAS_REQUIRE = {
    "act": [
        "ipython>=9.2.0",
        "matplotlib>=3.10.3",
        "packaging>=24.0",
        "pyyaml>=6.0.2",
        "scipy>=1.15.3",
        "torchvision>=0.14.1",
        "tqdm>=4.67.1",
        "wandb>=0.23.1",
    ],
    "dp": [
        "accelerate>=1.12.0",
        "av>=14.4.0",
        "click>=8.1.7",
        "dill>=0.4.0",
        "diffusers==0.36.0",
        "filelock>=3.0.0",
        "hydra-core>=1.3.2",
        "imagecodecs>=2026.1.1",
        "matplotlib>=3.10.3",
        "numba>=0.56.0",
        "numcodecs==0.11.0",
        "omegaconf>=2.3.0",
        "opencv-python-headless>=4.11.0.86",
        "pandas>=2.3.3",
        "psutil>=5.9.0",
        "pyyaml>=6.0.2",
        "scipy>=1.15.3",
        "threadpoolctl>=3.6.0",
        "timm>=1.0.24",
        "torchvision>=0.14.1",
        "tqdm>=4.67.1",
        "wandb>=0.23.1",
        "zarr==2.16.0",
    ],
}
EXTRAS_REQUIRE["all"] = sorted({dep for deps in EXTRAS_REQUIRE.values() for dep in deps})


setup(
    name="ambench_learn",
    # The vendored Universal Manipulation Interface tree relies on implicit
    # namespace packages, so find_packages() stopped at policies/dp and left 53
    # of 70 modules out of the built distributions. The include filter keeps
    # discovery inside this package.
    packages=find_namespace_packages(include=["ambench_learn", "ambench_learn.*"]),
    package_data={
        "ambench_learn": collect_package_data(PACKAGE_PATH, (".yaml",)),
    },
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
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
