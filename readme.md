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
````

## ⚙️ Dependencies

This project is built on **Python 3.10**. The necessary dependencies are listed in `environment.yml`:

  * **PyTorch**: 2.5.1 (Built with CUDA 12.1)
  * **Safety Gymnasium**: 1.0.0 (For Safety Point/Car tasks)
  * **Numpy**: 1.23.5
  * **Matplotlib**: 3.7.5

## 🚀 Setup & Run

### 1\. Installation

It is recommended to use Conda to manage the environment and avoid dependency conflicts.

```bash
# Clone the repository
git clone [https://github.com/520yrn/dyna-cppo.git](https://github.com/520yrn/dyna-cppo.git)
cd dyna-cppo

# Create environment from config file
conda env create -f environment.yml

# Activate the environment
conda activate cppo
```

### 2\. Running the Implementation

The training logic is encapsulated in `main.py`. The default configuration runs the **Dyna-CPPO** algorithm.

```bash
python main.py
```

### 3\. Configuration

You can modify hyperparameters in the `if __name__ == "__main__":` block at the bottom of `main.py`:

```python
train_dyna_once(
    env_id="SafetyPointGoal1-v0",
    total_iters=10000,
    horizon=1024,           # Real environment steps per epoch
    model_horizon=64,       # Imaginary rollout horizon
    d_constraint=25.0,      # Safety Cost Limit (d)
    model_start_iter=10,    # Epochs before using model rollouts
    log_dir="./logs"
)
```

## 🧠 Implementation Details

### 1\. Dynamics Model

  * **File:** `agents/dynamics.py`
  * **Logic:** A Feed-Forward Network that predicts the next state difference, reward, and cost given a state-action pair. It uses `one_step_loss` for supervised training and provides a `ModelEnv` interface to simulate environment interactions for planning.

### 2\. Policy Optimization (EM Approach)

  * **E-Step (`algo/estep.py`):** Solves for the optimal non-parametric policy ratio $v^*$ in the dual advantage space ($A_r, A_c$), satisfying both the KL divergence limit $\delta$ and cost constraint $d$.
  * **M-Step (`algo/mstep.py`):** Updates the parametric policy $\pi_\theta$ to minimize the MSE distance to $v^*$, applying a lower-bound clipping mechanism to ensure stability.

### 3\. Scheduling

  * **File:** `utils/schedule.py`
  * **Logic:** Implements a `LinearSchedule` that linearly decays the **Learning Rate** and **Clip Range** over total iterations to improve convergence in later training stages.

## 🔬 Reproducibility Details

To ensure exact reproducibility of results, this implementation strictly controls randomness via `utils/seed.py`.

When the program starts, the following sources of randomness are seeded:

1.  **Python Random**: `random.seed`
2.  **Numpy**: `np.random.seed`
3.  **PyTorch (CPU & GPU)**: `torch.manual_seed`
4.  **CuDNN**: Configured with `deterministic=True` and `benchmark=False`.

## 📈 Outputs & Artifacts

All experiment results are automatically saved to the `logs/` directory:

1.  **Process Logs (`.txt`)**:

      * Example: `cppo_goal1.txt` (Baseline), `dyna_cppo_goal1.txt` (Dyna).
      * Contains detailed metrics per epoch: Average Reward, Average Cost, Value/Cost Loss, Dynamics Loss, etc.

2.  **Plotting Data (`.csv`)**:

      * Example: `cppo_goals_reward.csv`, `dyna_goals_cost.csv`.
      * Formatted CSV files containing step-wise performance data, ready for generating comparison curves (Reward vs. Safety).

3.  **Model Checkpoints (`.pt`)**:

      * Example: `policy_final_goal0.pt`.
      * The final trained policy network weights.

## 📄 License

This project is open-sourced under the MIT License.