from setuptools import setup, find_packages

setup(
    name="universal_manipulation_interface",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "hydra-core",
        "zarr",
        "numba",
        "pygame",
        "pymunk",
        "shapely",
        "scikit-image",
        "scikit-video",
        "av",
        "imagecodecs",
        "diffusers",
        "timm",
        "clip",
        "einops",
        "threadpoolctl",
    ],
)
