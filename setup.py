from setuptools import setup

setup(
    name="improved-diffusion",
    py_modules=["improved_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
