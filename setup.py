from setuptools import setup, find_packages

setup(
    name="rf-diffusion",
    version="1.0.0",
    description="RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion",
    author="RF-Diffusion Evaluation Project",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
        "psutil>=5.9.0",
        "pytorch-fid>=0.3.0",
        "pytorch-msssim>=1.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "rf-diffusion=src.cli:main",
        ],
    },
)
