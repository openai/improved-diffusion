""" @xvdp fixed setup for simple import as:
>>> import improved_diffusion
"""
from setuptools import setup, find_packages

setup(
    name="improved-diffusion",
    packages = find_packages(),
    install_requires=["blobfile>=1.0.5", "torch", "tqdm", "mpi4py"],
    license=('LICENSE'),
    version='0.0.1'
)
