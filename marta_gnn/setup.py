from setuptools import setup, find_packages

setup(
    name="marta_gnn",
    version="0.1.0",
    description="GNN-based delay risk prediction for MARTA transit stops",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torch-geometric>=2.4.0",
        "scikit-learn>=1.3.0",
        "gtfs-realtime-bindings>=1.0.0",
        "protobuf>=4.21.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "networkx>=3.1",
    ],
)
