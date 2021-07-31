import numpy as np
from tqdm import tqdm

class GridSearch:
	def __init__(self, experiment, reward_function, segments, search_space_nm, search_space_lr):
		self.experiment = experiment
		self.reward_function = reward_function
		self.nms = [(i/(segments-1))*(search_space_nm[-1] - search_space_nm[0])+search_space_nm[0] for i in range(segments)]
		self.lrs = [(i/(segments-1))*(search_space_lr[-1] - search_space_lr[0])+search_space_lr[0] for i in range(segments)]
		self.progress = []

	def run(self):
		space = np.array([[[nm, lr] for nm in self.nms] for lr in self.lrs])
		space = space.reshape((-1, 2))
		for (nm, lr) in tqdm(space):
			eps, train_loss, val_loss, train_acc, val_acc = self.experiment(noise_multiplier=nm, learning_rate=lr, disable=True, return_dict=False)
			reward = self.reward_function(eps, train_loss, val_loss)
			self.progress.append([nm, lr, eps, train_loss, val_loss, val_acc, train_acc, reward])
		return np.array(self.progress)