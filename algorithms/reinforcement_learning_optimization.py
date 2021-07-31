class RLOptimization:
	def __init__(self, experiment, reward_function, segments, search_space_nm, search_space_lr, noise_space=3, experiments_per_episode=10, number_of_episodes=10, initial_eploration=0.1, final_exploration=0.7, mutation_std=0.01):
		self.experiment = experiment
		self.noise_space = noise_space
		self.segments = segments
		self.reward_function = reward_function
		self.search_space_nm = search_space_nm
		self.search_space_lr = search_space_lr
		self.number_of_episodes = number_of_episodes
		self.initial_eploration = initial_eploration
		self.final_exploration = final_exploration
		self.exploration_coeff = [((i*(self.final_exploration - self.initial_eploration))/(self.number_of_episodes-1))+self.initial_eploration for i in range(self.number_of_episodes)]
		self.space = self.get_space()
		
	def get_space(self):
		arr = [[[(i*(self.search_space_nm[1] - self.search_space_nm[0])/(self.segments - 1))+self.search_space_nm[0], (j*(self.search_space_lr[1] - self.search_space_lr[0])/(self.segments - 1))+self.search_space_lr[0]] for j in range(self.segments - 1)] for i in range(self.segments -1)]
		arr = np.array(arr)
		return arr