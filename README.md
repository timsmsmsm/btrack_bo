# Automated Parameter Tuning for btrack

## Description
This project implements automated parameter tuning for btrack cell tracking software using Optuna. Additionally, we have developed a tree visualization tool to visualize your ground truth and predicted lineage trees with your optimized parameters.

## Installation
To install the necessary dependencies, run the following commands in your Python environment:

```bash
pip install btrack traccuracy numpy pandas joblib optuna numba
pip install git+https://github.com/lowe-lab-ucl/ctc-tools
pip install git+https://github.com/lowe-lab-ucl/arboretum