import torchvision
from opacus import PrivacyEngine
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import random
import os
import warnings
warnings.filterwarnings("ignore")

class Experiment:
	def __init__(self, generate_model, criterion, train_dataset, test_dataset, batch_size=8, disable=False):
		self.set_seed()
		self.criterion = criterion
		self.train_dataset, self.test_dataset = train_dataset, test_dataset
		self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)
		self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size)
		self.batch_size = batch_size
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.disable = disable
		self.delta = 1e-5
		self.generate_model = generate_model

	def set_seed(self):
		seed = 42
		random.seed(seed)
		os.environ["PYTHONHASHSEED"] = str(seed)
		np.random.seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.manual_seed(seed)

	def train(self, model, criterion, optimizer, epoch, delta):
		model.train()
		losses, accuracies = [], []
		bar = tqdm(self.train_loader, disable=self.disable)
		for _batch_idx, (data, target) in enumerate(bar):
			data, target = data.to(self.device), target.to(self.device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			losses.append(loss.item())
			preds = torch.round(output)
			acc = (preds==target).float()
			acc = torch.sum(acc)/acc.shape[0]
			accuracies.append(acc.item())
			bar.set_description(str({'type':'training', 'epoch':epoch,'loss': round(sum(losses)/(_batch_idx+1), 4), 'acc': round(sum(accuracies)/(_batch_idx+1), 4)}))
			bar.refresh()
		bar.close()
		epsilon, _ = optimizer.privacy_engine.get_privacy_spent(delta)
		return epsilon, sum(losses)/(_batch_idx+1), sum(accuracies)/(_batch_idx+1)

	def test(self, model, criterion, epoch):
		model.eval()
		losses, accuracies = [], []
		bar = tqdm(self.test_loader, disable=self.disable)
		for _batch_idx, (data, target) in enumerate(bar):
			data, target = data.to(self.device), target.to(self.device)
			output = model(data)
			loss = criterion(output, target)
			losses.append(loss.item())
			preds = torch.round(output)
			acc = (preds==target).float()
			acc = torch.sum(acc)/acc.shape[0]
			accuracies.append(acc.item())
			bar.set_description(str({'type':'testing', 'epoch':epoch,'loss': round(sum(losses)/(_batch_idx+1), 4), 'acc': round(sum(accuracies)/(_batch_idx+1), 4)}))
			bar.refresh()
		bar.close()
		return sum(losses)/(_batch_idx+1), sum(accuracies)/(_batch_idx+1)


	def run_experiment(self, noise_multiplier, learning_rate, epochs=15, disable=False, return_dict=True):
		self.disable = disable
		self.set_seed()
		model = self.generate_model()
		model = model.to(self.device)
		criterion = self.criterion
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
		privacy_engine = PrivacyEngine(model, 
							batch_size=self.batch_size, 
							sample_size=len(self.train_dataset),
							noise_multiplier=noise_multiplier, 
							max_grad_norm=1.0,
						)
		privacy_engine.attach(optimizer)

		for epoch in range(1, epochs+1):
			eps, train_loss, train_acc = self.train(model, criterion, optimizer, epoch, delta=self.delta)
			val_loss, val_acc = self.test(model, criterion, epoch)
		results = [eps, train_loss, val_loss, train_acc, val_acc]
		if return_dict:
			labels = 'eps, train_loss, val_loss, train_acc, val_acc'.split(', ')
			results = {label:val for label, val in zip(labels, results)}
		return results