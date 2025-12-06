# Dyna-CPPO: Model-Based Constrained Policy Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Dyna-CPPO** is the official implementation of a Safe Reinforcement Learning algorithm that combines **Dyna-style model-based planning** with **Constrained Policy Optimization (CPPO)**.

The algorithm leverages a learned `DynamicsModel` to generate "imaginary rollouts" (predicted trajectories of state, reward, and cost). By combining these rollouts with a **Dual-Critic** architecture and an **Expectation-Maximization (EM)** update rule, Dyna-CPPO achieves high sample efficiency while strictly adhering to safety constraints.

## 📂 Project Structure

The repository structure is organized as follows:

```text
dyna-cppo/
├── agents/                 # Neural Network Modules
│   ├── policy.py           # Actor: Gaussian Policy Network
│   ├── value.py            # Critics: ValueNet (Reward) & CostNet (Cost)
│   └── dynamics.py         # Model: Dynamics Model (s,a -> s',r,c)
├── algo/                   # CPPO Core Algorithm
│   ├── advantages.py       # Dual GAE: Reward Advantage (A_r) & Cost Advantage (A_c)
│   ├── estep.py            # E-Step: Analytical solution for optimal policy ratio v*
│   └── mstep.py            # M-Step: Supervised policy update via MSE Loss
├── envs/                   # Environment Interfaces
│   ├── safety_gym_wrappers.py # Factory for Safety-Gymnasium environments
│   └── circle_wrappers.py     # Custom Circle task wrappers
├── utils/                  # Utilities
│   ├── log.py              # Logger: Training monitoring and file saving
│   ├── seed.py             # Seed: Global random seed setting for reproducibility
│   ├── schedule.py         # Schedule: Linear annealing for Learning Rate and Clip Range
│   └── replay_buffer.py    # Buffer: Experience Replay Buffer
├── logs/                   # Experiment Outputs
│   ├── *.txt               # Detailed training logs (e.g., cppo_goal1.txt)
│   ├── *.csv               # Plotting data (e.g., cppo_goals_reward.csv)
│   └── *.pt                # Model checkpoints (e.g., policy_final_goal0.pt)
├── main.py                 # Main Entry Point
├── environment.yml         # Dependency Manifest
└── README.md               # Documentation

