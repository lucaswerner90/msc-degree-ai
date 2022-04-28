"""
OpenAI-inspired Environment created around the drone images.
"""

import gym
import torch
import numpy as np
import cv2
from gym import spaces

(WIDTH, HEIGHT, CHANNELS) = (640,360,3)

class DroneEnvironment(gym.Env):
	MAX_REWARD = 1
	metadata = {'render.modes': ['human']}

	def __init__(self, dataset):
		self.current_image_index = 0
		self.current_point = None
		self.action_space = spaces.Discrete(3)
		self.dataset = dataset
		self.observation_space = spaces.Box(low=-1, high=1, shape=(224, 224, CHANNELS))

	def calculate_reward(self, predicted):
		return DroneEnvironment.MAX_REWARD if abs(predicted.item() - self.real_point.item()) <= 40 else 0

	def reset(self, eval:bool = False):
		# Get the next image from the dataset
		sample = self.dataset[self.current_image_index%(len(self.dataset)-1)]
		self.current_image_index+=1

		self.real_point = sample['real_x']
		self.real_point_y = sample['real_y']
		self.image = sample['image']
		self.original_image = sample['original_image']

		self.current_point = torch.Tensor([np.random.rand()]) if not eval else torch.Tensor([0.5])

		return [self.image, self.current_point]


	def step(self, action):
		point = self.current_point + action
		point = torch.tensor([int(point.item()*WIDTH)])

		point = torch.clamp(point, 1, WIDTH-1)

		self.current_point = torch.Tensor([point / WIDTH])
		reward = self.calculate_reward(point)
		done = reward == DroneEnvironment.MAX_REWARD
		state = [self.image, self.current_point]
		return state, reward, done, {}

	def get_image(self):
		current_point = int(self.current_point.item()*WIDTH)
		original_image = np.copy(self.original_image)
		image = cv2.circle(original_image, (current_point, int(HEIGHT/2)), radius=4, color=(0, 255, 255), thickness=10)
		image = cv2.circle(image, (self.real_point, self.real_point_y), radius=4, color=(0, 0, 255), thickness=10)
		
		image = cv2.putText(
			image,
			f"Predicted: {current_point}",
			(20,HEIGHT-50),
			cv2.FONT_HERSHEY_PLAIN,
			1,
			(0, 255, 255),
			1,
			cv2.LINE_AA
		)
		image = cv2.putText(
			image,
			f"Real: {self.real_point}",
			(20,HEIGHT-20),
			cv2.FONT_HERSHEY_PLAIN,
			1,
			(0, 0, 255),
			1,
			cv2.LINE_AA
		)
		return image

	def render(self):
		cv2.imshow('image', self.get_image())
		cv2.waitKey(1)