import torch
import torch.nn as nn

class RLAgent(nn.Module):
	def __init__(self, drone):
		self.action_space = ['left', 'right', 'up', 'down', 'stop']
		self.drone = drone

	def forward(self, state, action):
		pass

	def act(self, state):
		pass

	def learn(self, state, action, reward, next_state, done):
		pass

	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.state_dict(torch.load(path))

	def reset(self):
		pass

	def get_action_space(self):
		return self.action_space

	def get_state_space(self):
		pass

	def get_action_size(self):
		return len(self.action_space)

	