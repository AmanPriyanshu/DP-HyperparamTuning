# DP-HyperparamTuning

The offical repository for all algorithms and code.

A streamlined and basic implementation for all modules presented is available at:

* [GitHub Repo](https://github.com/AmanPriyanshu/DP-HyperparamTuning)
* [GitHub Notebook](https://github.com/AmanPriyanshu/DP-HyperparamTuning/blob/main/RL_DP_Demo.ipynb)

`Note:` [Colab Demo for the same](https://colab.research.google.com/github/AmanPriyanshu/DP-HyperparamTuning/blob/main/RL_DP_Demo.ipynb)

# Implementation:

## Imports:

```python
from DP_HyperparamTuning.experiment.train_single_model import Experiment
from DP_HyperparamTuning.algorithms.bayesian_optimization import Bayesian
from DP_HyperparamTuning.algorithms.grid_search_algorithm import GridSearch
from DP_HyperparamTuning.algorithms.evolutionary_optimization import EvolutionaryOptimization
from DP_HyperparamTuning.algorithms.reinforcement_learning_optimization import RLOptimization
```

## Running Given Modules:

```python
e = Experiment(get_model, criterion, train_dataset, test_dataset)
b = Bayesian(e.run_experiment, calculate_reward, num_limit, search_space_nm=search_space_nm, search_space_lr=search_space_nm)
```

Where, `get_model`, `calculate_reward` are functions, and `criterion` and `train_dataset, test_dataset` which are `<class 'torch.nn.modules.loss.BCELoss'>` and `torch.utils.data.Dataset` respectively.

# Contributing

When contributing to this repository, please first discuss the change you wish to make via issue,
email, or any other method with the owners of this repository before making a change.
