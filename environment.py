import gym
import torch
import numpy as np
import cv2
from gym import spaces

(WIDTH, HEIGHT, CHANNELS) = (640,360,3)

class DroneEnvironment(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self, dataset):
		self.current_image_index = 0
		self.current_point = None
		self.action_space = spaces.Discrete(3)
		self.dataset = dataset
		self.observation_space = spaces.Box(low=-1, high=1, shape=(224, 224, CHANNELS))

	def calculate_reward(self, predicted):
		"""
		Divides the images into 32 sections of fixed length and returns
		1 if the predicted is in the same section as
		the real_x_coordinate.
		Otherwise it returns a result directly proportional to the distance
		between the two sections, the closer to 1, the closer the 2 sections are.
		"""
		num_sections = 32
		section_width = WIDTH // num_sections
		num_sections_away = int(abs(predicted.item() - self.real_point.item()) // section_width)
		return 1 if num_sections_away == 0 else 1 - (num_sections_away / num_sections)

	def reset(self):
		# Get the next image from the dataset
		idx = self.current_image_index % len(self.dataset)
		sample = self.dataset.__getitem__(idx)
		self.current_image_index+=1

		self.real_point = sample['real_x']
		self.image = sample['image'].squeeze()
		self.original_image = sample['original_image']
		self.current_point = torch.Tensor([np.random.rand()])
		state = torch.concat((self.image, self.current_point))
		return state

	def step(self, action):
		distance = 20  # distance in pixels
		point = torch.Tensor([int(self.current_point.item() * WIDTH)])

		if action == "LEFT":
			point -= distance
		if action == "RIGHT":
			point += distance

		point = torch.clamp(point, 1, WIDTH-1)

		self.current_point = torch.Tensor([point / WIDTH])
		reward = self.calculate_reward(point)
		done = reward == 1
		state = torch.concat((self.image, self.current_point))
		return state, reward, done, {}

	def render(self, mode='human'):
		cv2.imshow('image', self.original_image)