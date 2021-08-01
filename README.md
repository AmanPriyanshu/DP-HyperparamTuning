# DP-HyperparamTuning

The offical repository for all algorithms and code.

A streamlined and basic implementation for all modules presented is available at:

* [GitHub Repo](https://github.com/AmanPriyanshu/DP-HyperparamTuning)
* [GitHub Notebook](https://github.com/AmanPriyanshu/DP-HyperparamTuning/blob/main/RL_DP_Demo.ipynb)
`Note:` [Colab Demo for the same](https://colab.research.google.com/github/AmanPriyanshu/DP-HyperparamTuning/blob/main/RL_DP_Demo.ipynb)

# Implementation:

## Imports:

```python
from RL_DP_Project.experiment.train_single_model import Experiment
from RL_DP_Project.algorithms.bayesian_optimization import Bayesian
from RL_DP_Project.algorithms.grid_search_algorithm import GridSearch
from RL_DP_Project.algorithms.evolutionary_optimization import EvolutionaryOptimization
from RL_DP_Project.algorithms.reinforcement_learning_optimization import RLOptimization
```

## Running Given Modules:

```python
e = Experiment(get_model, criterion, train_dataset, test_dataset)
b = Bayesian(e.run_experiment, calculate_reward, num_limit, search_space_nm=search_space_nm, search_space_lr=search_space_nm)
```

Where, `get_model`, `calculate_reward` are functions and `criterion` is a `<class 'torch.nn.modules.loss.BCELoss'>`.
