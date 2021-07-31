import numpy as np
from dp_mnist import experiment
from tqdm import tqdm
import pandas as pd

class Environment:
	def __init__(self, search_space_lr, search_space_nm, segments=1000, mutation_rate=0.25, population_strength=13, num_of_generations=10, alpha=0.33, max_epochs_per_experiment=10):
		self.alpha = alpha
		self.segments = segments
		self.num_of_generations = num_of_generations
		self.search_space_nm = search_space_nm
		self.search_space_lr = search_space_lr
		self.population_strength = population_strength
		self.max_epochs_per_experiment = max_epochs_per_experiment
		self.generation_num = None
		self.mutation_rate = mutation_rate

	def get_random_chromosomes(self, n, return_numpy=False):
		lrs = np.random.choice(((self.search_space_lr[1]-self.search_space_lr[0])*np.arange(self.segments))/self.segments+self.search_space_lr[0], n)
		nms = np.random.choice(((self.search_space_nm[1]-self.search_space_nm[0])*np.arange(self.segments))/self.segments+self.search_space_nm[0], n)
		chromosomes = [[nms[i], lrs[i]] for i in range(n)]
		if return_numpy:
			chromosomes = np.array(chromosomes)
		return chromosomes

	def calculate_reward(self, eps, train_loss, val_loss):
		return self.alpha*np.exp(-(0.5 * eps)) + (1-self.alpha)*np.exp(-(0.5 * val_loss))

	def run_generation(self, chromosomes):
		performances = []
		bar = tqdm(chromosomes, total=len(chromosomes))
		for genes in bar:
			eps, train_loss, val_loss = experiment(genes[0], genes[1], max_epochs=self.max_epochs_per_experiment, max_epsilon=1, disable=True)
			reward = self.calculate_reward(eps, train_loss, val_loss)
			performances.append([genes[0], genes[1], eps, train_loss, val_loss, reward])
			bar.set_description(str({'gen_num': self.generation_num, 'lr':round(genes[1], 4), 'nm':round(genes[0], 4), 'eps': round(eps, 4), 'val_loss': round(val_loss, 4), 'reward': round(reward, 4)}))
			bar.refresh()
		bar.close()
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
		new_chromosomes = np.concatenate((np.array([prime_sample]).T, top_quarter.T, mutated_best_half.T, self.get_random_chromosomes(len(top_quarter), return_numpy=True).T), axis=1)
		return new_chromosomes.T

	def run_generations(self):
		prime_sample, best_half, chromosomes, performances = None, None, None, None
		for generation_num in range(self.num_of_generations):
			self.generation_num = generation_num
			if generation_num==0:
				chromosomes = self.get_random_chromosomes(self.population_strength)
			else:
				chromosomes = self.create_new_generation(prime_sample, best_half, chromosomes, performances)
			prime_sample, best_half, chromosomes, performances = self.analyze_generation_performance(chromosomes)
			df = pd.DataFrame({'1': ['generation_num'], 'generation_num': [generation_num+1]})
			df.to_csv('performances.csv', index=False, header=False, mode='a')
			desc = {'gen_num':generation_num, 'reward_mean': np.mean(performances.T[-1]), 'reward_max': np.max(performances.T[-1])}
			print(desc)
			print()
			performances = np.concatenate((np.array([np.arange(len(performances))]), performances.T)).T
			performances = pd.DataFrame(performances)
			performances.columns = ['individual_name', 'nm', 'lr', 'eps', 'train_loss', 'val_loss', 'reward']
			performances.to_csv('performances.csv', index=False, header=True, mode='a')

if __name__ == '__main__':
	env = Environment(search_space_nm=[0.5, 3], search_space_lr=[0.001, 0.1], max_epochs_per_experiment=10)
	env.run_generations()
