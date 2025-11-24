from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="perception_models",
    version="1.0.0",
    author="Meta AI Research, FAIR",
    description="Models of the Perception family.",
    url="https://github.com/facebookresearch/perception_models",
    packages=find_packages(),
    package_data={
        "core.vision_encoder": ["bpe_simple_vocab_16e6.txt.gz"]
    },
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
    ],
    license="FAIR Noncommercial Research License",
    python_requires=">=3.9",
    include_package_data=True,
)
