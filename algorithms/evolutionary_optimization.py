import numpy as np
import torch
from tqdm import tqdm

class EvolutionaryOptimization:
	def __init__(self, experiment, reward_function, segments, search_space_nm, search_space_lr, mutation_rate=0.25, population_strength=10, num_of_generations=10):
		self.experiment = experiment
		self.population_strength = population_strength
		self.reward_function = reward_function
		self.search_space_nm = search_space_nm
		self.search_space_lr = search_space_lr
		self.mutation_rate = mutation_rate
		self.segments = segments
		self.nms = [(i/(segments-1))*(search_space_nm[-1] - search_space_nm[0])+search_space_nm[0] for i in range(segments)]
		self.lrs = [(i/(segments-1))*(search_space_lr[-1] - search_space_lr[0])+search_space_lr[0] for i in range(segments)]
		self.generation_progress = []
		self.num_of_generations = num_of_generations

	def get_random_chromosomes(self, n, return_numpy=False):
		lrs = np.random.choice(((self.search_space_lr[1]-self.search_space_lr[0])*np.arange(self.segments))/self.segments+self.search_space_lr[0], n)
		nms = np.random.choice(((self.search_space_nm[1]-self.search_space_nm[0])*np.arange(self.segments))/self.segments+self.search_space_nm[0], n)
		chromosomes = [[nms[i], lrs[i]] for i in range(n)]
		if return_numpy:
			chromosomes = np.array(chromosomes)
		return chromosomes

	def run_generation(self, chromosomes):
		performances = []
		bar = tqdm(chromosomes, total=len(chromosomes))
		for genes in bar:
			eps, train_loss, val_loss, train_acc, val_acc = self.experiment(genes[0], genes[1], disable=True, return_dict=False)
			reward = self.reward_function(eps, train_loss, val_loss)
			performances.append([genes[0], genes[1], eps, train_loss, val_loss, train_acc, val_acc, reward])
			bar.set_description(str({'gen_num': self.generation_num, 'lr':round(genes[1], 4), 'nm':round(genes[0], 4), 'eps': round(eps, 4), 'val_loss': round(val_loss, 4), 'reward': round(reward, 4)}))
			bar.refresh()
		bar.close()
		self.generation_progress.append(performances)
		return performances

	def analyze_generation_performance(self, chromosomes):
		chromosomes = np.array(chromosomes)
		performances = self.run_generation(chromosomes)
		performances = np.array(performances)
		index_sort = np.argsort(performances.T[-1])
		prime_sample = chromosomes[index_sort[-1]]
		best_half = chromosomes[index_sort[int(0.5*len(chromosomes)):-1]]
		return prime_sample, best_half, chromosomes, performances

	def mutate(self, arr):
		chs = self.get_random_chromosomes(len(arr))
		for index in range(len(arr)):
			if np.random.random()<self.mutation_rate:
				arr[index] += chs[index]
				arr[index] = arr[index]/2
		return arr

	def create_new_generation(self, prime_sample, best_half, chromosomes, performances):
		np.random.shuffle(best_half)
		quarter1 = best_half[:len(best_half)//2]
		quarter2 = best_half[len(best_half)//2: 2*(len(best_half)//2)]
		top_quarter = quarter1 + quarter2
		top_quarter = prime_sample + top_quarter
		top_quarter = top_quarter/3
		mutated_best_half = self.mutate(best_half)
		new_chromosomes = np.concatenate((np.array([prime_sample]).T, top_quarter.T, mutated_best_half.T, self.get_random_chromosomes(self.population_strength - len(mutated_best_half) - len(top_quarter) - len(prime_sample), return_numpy=True).T), axis=1)
		return new_chromosomes.T

	def run(self):
		prime_sample, best_half, chromosomes, performances = None, None, None, None
		for generation_num in range(self.num_of_generations):
			np.random.seed(generation_num)
			self.generation_num = generation_num
			if generation_num==0:
				chromosomes = self.get_random_chromosomes(self.population_strength)
			else:
				chromosomes = self.create_new_generation(prime_sample, best_half, chromosomes, performances)
			prime_sample, best_half, chromosomes, performances = self.analyze_generation_performance(chromosomes)
			desc = {'gen_num':generation_num, 'reward_mean': np.mean(performances.T[-1]), 'reward_max': np.max(performances.T[-1])}
			print(desc)
		return self.generation_progress