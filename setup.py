from setuptools import find_packages, setup

setup(
    name="rl-agents",
    version="0.1.0",
    description="DQN and PPO from scratch — CartPole and LunarLander",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium[box2d]>=1.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "imageio>=2.30",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
        "notebooks": ["jupyter>=1.0.0", "ipykernel>=6.0.0"],
    },
)
