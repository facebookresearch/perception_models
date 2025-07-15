from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="perception_models",
    version="1.0.0",
    author="Meta AI Research, FAIR",
    description="Occhi package.",
    url="https://github.com/facebookresearch/perception_models",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
    ],
    license="FAIR Noncommercial Research License",
    python_requires=">=3.10",
)
