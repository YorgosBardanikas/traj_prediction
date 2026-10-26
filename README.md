# Hand Trajectory Prediction with Deep Learning

A PyTorch project exploring how partial hand-movement trajectories can be used to predict movement endpoints, identify intended targets, and forecast future trajectories.

The project implements and evaluates recurrent neural networks and temporal convolutional networks on hand kinematic data. It investigates how prediction performance changes as more of a movement becomes observable, with comparisons against baseline and shuffled-label models.

## Overview

Predicting the outcome of a movement before it is completed provides a useful framework for studying how motor behavior unfolds over time. This project addresses three related prediction tasks using partial hand trajectories:

1. **Endpoint prediction** — estimate the final hand position from a partially observed trajectory.
2. **Target classification** — classify the intended target among six possible targets using only the observed movement.
3. **Future trajectory prediction** — forecast the remaining movement trajectory from its observed prefix.

The experiments use PyTorch models, held-out test trials, and task-specific evaluation metrics. The accompanying Jupyter notebooks provide visualizations and analysis of the results.

## Experiments

### 1. Endpoint prediction

**Objective:** Predict the final hand position from a partial movement trajectory.

- **Model:** Gated Recurrent Unit (GRU).
- **Input:** A partially observed 2D hand trajectory.
- **Output:** The predicted endpoint \((x, y)\).
- **Evaluation:** Euclidean endpoint error on held-out test trials.
- **Comparisons:** Shuffled-label null models and different trajectory reveal percentages.

The observed trajectory is progressively revealed to investigate how endpoint prediction improves as more movement information becomes available. Experiments are evaluated separately across movement conditions.

**Notebook:** `notebooks/plot_endpoint_error.ipynb`


### 2. Target classification

**Objective:** Predict the intended movement target from a partial hand trajectory.

- **Model:** GRU classifier.
- **Input:** A partially observed 2D hand trajectory.
- **Output:** One of six target classes.
- **Evaluation:** Held-out classification accuracy and confusion matrix.
- **Comparisons:** Appropriate null baseline and chance-level performance.

The experiment examines how early in a movement the destination can be inferred from hand kinematics. Performance is evaluated at different reveal percentages and across movement conditions.

**Notebook:** `notebooks/plot_target_accuracy.ipynb`


### 3. Future trajectory prediction

**Objective:** Predict the unobserved portion of a hand trajectory from its observed prefix.

The experiment compares different forecasting approaches:

- **Temporal CNN:** Direct multi-step prediction of the future trajectory.
- **Autoregressive encoder–decoder GRU:** Sequential prediction of future positions.
- **Linear extrapolation:** A simple kinematic baseline.

The temporal CNN predicts future displacements, which are accumulated from the final observed position to obtain the predicted trajectory. The autoregressive GRU provides an alternative sequence-generation approach.

The experiments also investigate the effect of defining the observed prefix by spatial path length rather than by elapsed time. Spatial alignment can make comparisons across movements more consistent when trajectories differ in their temporal profiles.

**Evaluation:** Point-by-point trajectory error and qualitative comparisons between predicted and observed trajectories.

**Notebook:** `notebooks/plot_trajectory_error.ipynb`

**Main result:** The temporal CNN outperformed the autoregressive GRU and the linear extrapolation. 


## Repository structure

```text
hand-trajectory-prediction/
├── scripts/
│   ├── core/
│      ├── endpoint_prediction.py
│      ├── target_classification.py
│      ├── trajectory_models.py
│      └── trajectory_prediction.py
│   ├── notebooks/
│      ├── plot_endpoint_error.py
│      ├── plot_target_accuracy.py
│      └── plot_trajectory_error.py
│   ├── __init__.py
│   ├── utils.py
│   └── helper_funcs.py
├── figures/
├── setup.py
├── requirements.txt
└── README.md
```

The notebooks focus on result exploration and visualization, while the Python scripts contain the experiment implementations. 

## Installation

Clone the repository:

```bash
git clone https://github.com/YorgosBardanikas/traj_prediction.git
cd traj_prediction
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```
