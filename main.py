from experiment.train_single_model import Experiment
from sample.sample_model import get_model
from sample.sample_dataset import load_dataset
import torch
from algorithms.bayesian_optimization import Bayesian
from algorithms.grid_search_algorithm import GridSearch
from algorithms.evolutionary_optimization import EvolutionaryOptimization
from algorithms.reinforcement_learning_optimization import RLOptimization
import numpy as np

def calculate_reward(eps, train_loss, val_loss, alpha=0.33):
	return alpha*np.exp(-(0.5 * eps)) + (1-alpha)*np.exp(-(0.5 * val_loss))

def run_sample():
	criterion = torch.nn.BCELoss()
	train_dataset, test_dataset = load_dataset()
	e = Experiment(get_model, criterion, train_dataset, test_dataset)
	results = e.run_experiment(0.15, 0.001)
	print()
	print("RESULTS:")
	_ = [print(key+":", round(item, 4)) for key, item in results.items()]

def run_bayesian():
	criterion = torch.nn.BCELoss()
	train_dataset, test_dataset = load_dataset()
	e = Experiment(get_model, criterion, train_dataset, test_dataset)
	b = Bayesian(e.run_experiment, calculate_reward, 100, search_space_nm=[0.5, 2.5], search_space_lr=[0.001, 0.05])
	progress = b.run()
	return progress

def run_grid_search():
	criterion = torch.nn.BCELoss()
	train_dataset, test_dataset = load_dataset()
	e = Experiment(get_model, criterion, train_dataset, test_dataset)
	gs = GridSearch(e.run_experiment, calculate_reward, 10, search_space_nm=[0.5, 2.5], search_space_lr=[0.001, 0.05])
	progress = gs.run()
	return progress

def run_evolutionary_optimization():
	criterion = torch.nn.BCELoss()
	train_dataset, test_dataset = load_dataset()
	e = Experiment(get_model, criterion, train_dataset, test_dataset)
	eo = EvolutionaryOptimization(e.run_experiment, calculate_reward, 10, search_space_nm=[0.5, 2.5], search_space_lr=[0.001, 0.05])
	progress = eo.run()
	return progress

def run_reinforcement_learning_optimization():
	criterion = torch.nn.BCELoss()
	train_dataset, test_dataset = load_dataset()
	e = Experiment(get_model, criterion, train_dataset, test_dataset)
	rl = RLOptimization(e.run_experiment, calculate_reward, 10, search_space_nm=[0.5, 2.5], search_space_lr=[0.001, 0.05])
	progress = rl.run()
	return progress

if __name__ == '__main__':
	print("----------RUN SAMPLE-----------")
	#run_sample()
	print("----------RUN BAYESIAN---------")
	run_bayesian()
	print("----------GRID SEARCH----------")
	run_grid_search()
	print("---EVOLUTIONARY OPTIMIZATION---")
	run_evolutionary_optimization()
	print("-----REINFORCEMENT LEARNING----")
	run_reinforcement_learning_optimization()