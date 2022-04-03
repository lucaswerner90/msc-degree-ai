import torch
import numpy as np
from torch import nn
from itertools import count
from torch.autograd import Variable
from torch.distributions import Categorical
import pytorch_lightning as pl

MAX_REWARD = 1

class PolicyGradientLightning(pl.LightningModule):
	def __init__(self,actions):
		super().__init__()
		self.actions = actions
		self.num_actions = len(actions)

		self.model = nn.Sequential(
            nn.Linear(4096+1, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, self.num_actions),
            nn.Softmax(dim=-1)
        )

	def calculate_reward(self, image_width, real_x_coordinate, predicted_x_coordinate):
		"""
		Divides the images into 32 sections of fixed length and returns
		1 if the predicted_x_coordinate is in the same section as
		the real_x_coordinate.
		Otherwise it returns a result directly proportional to the distance
		between the two sections, the closer to 1, the closer the 2 sections are.
		"""
		num_sections = 32
		section_width = image_width // num_sections
		num_sections_away = abs(predicted_x_coordinate - real_x_coordinate) // section_width
		return MAX_REWARD if num_sections_away == 0 else 1 - (num_sections_away / num_sections)

	def calculate_next_movement(self, image_width:int, point_of_view:int, action:str) -> int:
		"""
		Given the current position of our view, the total image width in pixels
		and the next action, we return the next point of view
		"""
		distance = 20  # pixels distance
		if action == "LEFT":
			return max(1, point_of_view - distance)  # 200 => 200 - distancia
		if action == "RIGHT":
			return min(image_width - 1, point_of_view + distance)
		return point_of_view

	def select_action(self, probs: torch.Tensor):
		"""
		Given the output of the model, we select the action
		based on a probability distribution and we return the
		position of the action in the actions array
		"""
		m = Categorical(probs)
		return m.sample().item()

	def forward(self, x):
		x = self.model(x)
		return x

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
		return [optimizer], [lr_scheduler]

	def training_step(self, train_batch, batch_idx):
		original_image, image, real_x = train_batch["original_image"], train_batch["image"], train_batch["real_x"]
		batch_size, _, img_width, _ = original_image.shape
		reward_pool = []
		action_pool = []
		state_pool = []
		steps = 0
		for i in range(batch_size):
			# We get the image and the point of view
			image_i = image[i]
			real_i = real_x[i]
			point_of_view = torch.Tensor([0.5]) 
			initial_state = torch.concat((image_i, point_of_view))
			state = Variable(initial_state)

			for t in count():
				# We get the output of the model
				probs = self.forward(state.unsqueeze(0))
				
				# We select the action based on the output
				action = self.select_action(probs.squeeze())
				
				# Calculate reward based on the current point of view and the action selected
				point_of_view = round(point_of_view.item() * img_width)
				next_point_of_view = self.calculate_next_movement(
					img_width,
					point_of_view, 
					self.actions[action]
				)
				reward = self.calculate_reward(
					img_width,
					real_i,
					next_point_of_view
				)
				reward_pool.append(reward)
				action_pool.append(action)
				state_pool.append(state.squeeze())

				point_of_view = torch.Tensor([next_point_of_view / img_width])
				state = torch.concat((image_i, point_of_view))
				state = Variable(state)

				steps+=1

				if reward == MAX_REWARD or self.actions[action] == "NONE" or t > 20:
					break

		
		running_add = 0
		gamma = 0.99

		for i in reversed(range(steps)):
			running_add = running_add * gamma + reward_pool[i]
			reward_pool[i] = running_add

		# Normalize reward
		reward_mean = np.mean(reward_pool)
		reward_std = np.std(reward_pool)

		for i in range(steps):
			reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

		loss = 0
		for i in range(steps):
			state = state_pool[i]
			action = Variable(torch.Tensor([action_pool[i]]))
			reward = reward_pool[i]
			probs = self.forward(torch.unsqueeze(state, 0))
			probs = probs.squeeze()
			m = Categorical(probs)
			current_loss = -m.log_prob(action) * reward  # Negative score function x reward
			loss += current_loss

		loss = loss / steps
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		original_image, image, real_x = val_batch["original_image"], val_batch["image"], val_batch["real_x"]
		batch_size, _, img_width, _ = original_image.shape
		reward_pool = []
		action_pool = []
		state_pool = []
		steps = 0
		for i in range(batch_size):
			# We get the image and the point of view
			image_i = image[i]
			real_i = real_x[i]
			point_of_view = torch.Tensor([0.5]) 
			initial_state = torch.concat((image_i, point_of_view))
			state = Variable(initial_state)

			for _ in count():
				# We get the output of the model
				probs = self.forward(state.unsqueeze(0))
				
				# We select the action based on the output
				action = self.select_action(probs.squeeze())
				
				# Calculate reward based on the current point of view and the action selected
				point_of_view = round(point_of_view.item() * img_width)
				next_point_of_view = self.calculate_next_movement(
					img_width,
					point_of_view, 
					self.actions[action]
				)
				reward = self.calculate_reward(
					img_width,
					real_i,
					next_point_of_view
				)
				reward_pool.append(reward)
				action_pool.append(action)
				state_pool.append(state.squeeze())

				point_of_view = torch.Tensor([next_point_of_view / img_width])
				state = torch.concat((image_i, point_of_view))
				state = Variable(state)

				steps+=1

				if self.actions[action] == 'NONE':
					break

		running_add = 0
		gamma = 0.99

		for i in reversed(range(steps)):
			running_add = running_add * gamma + reward_pool[i]
			reward_pool[i] = running_add

		# Normalize reward
		reward_mean = np.mean(reward_pool)
		reward_std = np.std(reward_pool)

		for i in range(steps):
			reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

		loss = 0
		for i in range(steps):
			state = state_pool[i]
			action = Variable(torch.Tensor([action_pool[i]]))
			reward = reward_pool[i]
			probs = self.forward(torch.unsqueeze(state, 0))
			probs = probs.squeeze()
			m = Categorical(probs)
			current_loss = -m.log_prob(action) * reward  # Negative score function x reward
			loss += current_loss

		loss = loss / steps
		self.log('val_loss', loss)
		return loss

    
