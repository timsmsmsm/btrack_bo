# Automated Parameter Tuning for btrack

## Description
This project implements automated parameter tuning for btrack cell tracking software using Optuna. Additionally, we have developed a tree visualization tool to visualize your ground truth and predicted lineage trees with your optimized parameters.

## Installation
To install the necessary dependencies, run the following commands in your Python environment:

```bash
pip install btrack traccuracy numpy pandas joblib optuna numba
pip install git+https://github.com/lowe-lab-ucl/ctc-tools
pip install git+https://github.com/lowe-lab-ucl/arboretum
```

## Usage

To understand how to use this project, you can refer to the `bo_example.py` script. This script demonstrates the following tasks:

1. Downloads a dataset from the Cell Tracking Challenge website.
2. Loads the ground truth data and the dataset.
3. Chooses a sampler for the optimization process.
4. Runs the optimization process with a specified number of trials and a timeout.
5. Writes the best parameters to a JSON file.
6. Plots the ground truth lineages and saves the output image.
7. Plots the predicted lineages and saves the output image.

You can modify the parameters in the script to suit your needs, such as the dataset name, the number of trials, the timeout, and the sampler.
