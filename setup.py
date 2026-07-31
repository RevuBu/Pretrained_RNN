from setuptools import setup, find_packages

setup(
    name="motor_cortex_pretraining",
    version="1.0.0",
    description="Developmental Pretraining of Embodied RNNs for Motor Cortex Alignment",
    author="BXD",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.60.0",
        "motornet",
        "gymnasium>=0.28.0",
        "dsa",
        "jpca",
    ],
)
