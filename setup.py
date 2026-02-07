from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hqpf",
    version="0.1.0",
    author="Tommaso R. Marena",
    author_email="your.email@example.com",
    description="Hybrid Quantum-AI Energy Fusion for Protein Structure Prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChessEngineUS/hybrid-quantum-protein-folding",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hqpf-train=scripts.train_hamiltonian:main",
            "hqpf-benchmark=scripts.run_benchmark:main",
            "hqpf-analyze=scripts.analyze_results:main",
        ],
    },
)