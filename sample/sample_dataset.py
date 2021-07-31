import torch
import pandas as pd
import numpy as np

class ClassificationDataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return torch.from_numpy(self.x[idx].astype(np.float32)), torch.from_numpy(np.array([self.y[idx]]).astype(np.float32))

def read_csv(path='diabetes.csv', val_split=0.2):
	print("Please Create a Function as well as define its dataset class above.")
	exit()
	return train_x, train_y, test_x, test_y

def load_dataset():
	train_x, train_y, test_x, test_y = read_csv()
	train_dataset = DiabetesDataset(train_x, train_y)
	test_dataset = DiabetesDataset(test_x, test_y)
	return train_dataset, test_dataset