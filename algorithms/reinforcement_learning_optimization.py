import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

class RLOptimization:
	def __init__(self, experiment, reward_function, segments, search_space_nm, search_space_lr, noise_space=3, experiments_per_episode=10, number_of_episodes=10, initial_eploration=0.1, final_exploration=0.7, mutation_std=0.01):
		self.experiment = experiment
		self.noise_space = noise_space
		self.experiments_per_episode = experiments_per_episode
		self.segments = segments
		self.reward_function = reward_function
		self.search_space_nm = search_space_nm
		self.search_space_lr = search_space_lr
		self.number_of_episodes = number_of_episodes
		self.initial_eploration = initial_eploration
		self.final_exploration = final_exploration
		self.exploration_coeff = [((i*(self.final_exploration - self.initial_eploration))/(self.number_of_episodes-1))+self.initial_eploration for i in range(self.number_of_episodes)]
		self.space = self.get_space()
		self.model = self.create_model()
		self.episode_records = []
		
	def get_space(self):
		arr = [[[(i*(self.search_space_nm[1] - self.search_space_nm[0])/(self.segments - 1))+self.search_space_nm[0], (j*(self.search_space_lr[1] - self.search_space_lr[0])/(self.segments - 1))+self.search_space_lr[0]] for j in range(self.segments - 1)] for i in range(self.segments -1)]
		arr = np.array(arr)
		return arr

	def create_model(self):
		regr = RandomForestRegressor(n_estimators=500, max_depth=3, random_state=0)
		arr = self.space
		arr = np.reshape(arr, (-1, 2))
		indexes = np.random.choice(np.arange(arr.shape[0]), 100)
		arr = arr[indexes]
		regr.fit(arr, np.random.random(arr.shape[0])/100)
		return regr

	def get_random_circle(self, point):
		lims = []
		if point[0]-self.noise_space<0:
			lims.append(0)
		else:
			lims.append(point[0]-self.noise_space)
		if point[0]+self.noise_space>self.space.shape[0]-1:
			lims.append(self.space.shape[0]-1)
		else:
			lims.append(point[0]+self.noise_space)

		if point[1]-self.noise_space<0:
			lims.append(0)
		else:
			lims.append(point[1]-self.noise_space)
		if point[1]+self.noise_space>self.space.shape[0]-1:
			lims.append(self.space.shape[0]-1)
		else:
			lims.append(point[1]+self.noise_space)
		arr = self.space[lims[0]:lims[1], lims[2]:lims[3], :]
		arr = np.reshape(arr, (-1, 2))
		return arr[np.random.choice(np.arange(len(arr)), 1)[0]]

	def get_preds_from_network(self, n):
		arr = np.reshape(self.space, (-1, 2))
		arr = arr.astype(np.float32)
		expected_rewards = self.model.predict(arr)
		expected_rewards = expected_rewards.flatten()
		max_indexes = np.argsort(expected_rewards)[-n:]
		max_indexes = np.array([[i//self.space.shape[0], i%self.space.shape[0]] for i in max_indexes])
		return np.array([self.get_random_circle(i) for i in max_indexes])

	def get_random_vals(self, n):
		arr = np.reshape(self.space, (-1, 2))
		indexes = np.arange(len(arr))
		indexes = np.random.choice(indexes, n)
		return arr[indexes]

	def extract_data(self, index):
		arr = None
		for df in self.episode_records:
			df = df[:, [0, 1, -1]]
			if arr is None:
				arr = df
			else:
				arr = np.concatenate((arr, df), 0)
		return arr

	def train_model(self, index):
		data = self.extract_data(index)
		x = data.T[:-1].T
		x = x.astype(np.float32)
		y = data.T[-1].astype(np.float32)
		self.model.fit(x, y)

	def run(self):
		n = self.experiments_per_episode
		for index in range(self.number_of_episodes):
			np.random.seed(index)
			p = self.exploration_coeff[index]
			n1 = int(n*p)
			n2 = n-n1
			arr1, arr2 = self.get_preds_from_network(n1), self.get_random_vals(n2)
			points = np.concatenate((arr1, arr2), 0)
			nms, lrs = points.T[0], points.T[1]
			bar = tqdm(zip(nms, lrs), total=len(nms))
			eps_arr, train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr = [], [], [], [], []
			for nm, lr in bar:
				eps, train_loss, val_loss, train_acc, val_acc = self.experiment(noise_multiplier=nm, learning_rate=lr, disable=True, return_dict=False)
				eps_arr.append(eps)
				train_loss_arr.append(train_loss)
				val_loss_arr.append(val_loss)
				train_acc_arr.append(train_acc) 
				val_acc_arr.append(val_acc)
			eps_arr, train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr = np.array(eps_arr), np.array(train_loss_arr), np.array(val_loss_arr), np.array(train_acc_arr), np.array(val_acc_arr)
			eps, train_loss, val_loss, train_acc, val_acc = eps_arr, train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr
			rewards = self.reward_function(eps, train_loss, val_loss)
			self.episode_records.append(np.stack([nms, lrs, eps, train_loss, val_loss, train_acc, val_acc, rewards]).T)
			self.train_model(index)
		return self.episode_records
