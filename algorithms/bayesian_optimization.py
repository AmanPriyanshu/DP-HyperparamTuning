from hyperopt import hp, fmin, tpe, space_eval, Trials
import numpy as np

class Bayesian:
	def __init__(self, experiment, reward_function, max_evals, search_space_nm, search_space_lr):
		self.progress = []
		self.search_space_lr = search_space_lr
		self.search_space_nm = search_space_nm
		self.max_evals = max_evals
		self.reward_function = reward_function
		self.experiment = experiment

	def experiment_reward(self, params):
		lr, nm = params['lr'], params['nm']
		eps, train_loss, val_loss, train_acc, val_acc = self.experiment(noise_multiplier=nm, learning_rate=lr, disable=True, return_dict=False)
		reward = self.reward_function(eps, train_loss, val_loss)
		self.progress.append([nm, lr, eps, train_loss, val_loss, val_acc, train_acc, reward])
		return 1-reward

	def run(self):
		space = {
		'lr': hp.uniform('lr', self.search_space_lr[0], self.search_space_lr[1]),
		'nm': hp.uniform('nm', self.search_space_nm[0], self.search_space_nm[1])
		}
		trials = Trials()
		best = fmin(self.experiment_reward, space, algo=tpe.suggest, max_evals=self.max_evals, trials=trials)
		print(best)
		return np.array(self.progress)