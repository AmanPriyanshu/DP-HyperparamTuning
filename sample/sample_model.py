import torch

def get_model():
	model = torch.nn.Sequential(
			torch.nn.Linear(8, 4),
			torch.nn.ReLU(),
			torch.nn.Linear(4, 1),
			torch.nn.Sigmoid(),
		)
	return model