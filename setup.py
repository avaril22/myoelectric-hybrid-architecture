from setuptools import setup, find_packages

setup(
    name="myoelectric-tcn",
    version="0.1.0",
    author="Armelle Varillas",
    description="Hybrid TCN for pediatric myoelectric prosthetic control",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
)