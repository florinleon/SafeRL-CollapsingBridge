# Collapsing Bridge Environment for Safe Reinforcement Learning

This repository contains the implementation of a customizable grid-based environment and associated agents for safe reinforcement learning (SafeRL) research. The environment introduces **action-dependent dynamics**, **irreversible transitions**, and **partial observability**, which makes it suitable for evaluating algorithms under safety-critical and non-stationary conditions.

## Features

- **Custom Gymnasium-compatible environment** (`BridgeEnv.py`) with collapsible bridge tiles and variable object locations based on time-of-day.
- **Partial observability** through a 3×3 perceptual grid centered on the agent.
- **Hard and soft action masking agents**:
  - `BridgeAgentQL.py`: Q-learning with hard masking (removes actions after unsafe outcomes).
  - `BridgeAgentSafe.py`: Q-learning with soft masking using empirical risk estimation and an annealing mechanism.
- **Training, evaluation, and logging** of unsafe terminations and total returns.
- **Visualization tools** (`MakePlot.py`) for plotting training progress and analyzing safety violations.
- **Interactive testing** (`TestAgent.py`) for rendering agent behavior in morning/evening scenarios.

## Citation

If you use this environment or the agents in your research, please cite:

> Florin Leon, *Action Masking Methods for Safe Reinforcement Learning in a Non-Stationary Configurable Environment*, Proceedings of the 29th International Conference on System Theory, Control and Computing (ICSTCC 2025), Cluj-Napoca, Romania, 2025.

## License

This project is provided under the MIT License.
