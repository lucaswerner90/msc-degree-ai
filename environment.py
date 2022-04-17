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
		self.actions_taken = 0
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
		
		sample = self.dataset[self.current_image_index%(len(self.dataset)-1)]
		self.current_image_index+=1

		self.real_point = sample['real_x']
		self.real_point_y = sample['real_y']
		self.image = sample['image'].squeeze()
		self.original_image = sample['original_image']

		self.current_point = torch.Tensor([np.random.rand()])
		self.actions_taken = 0

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

	def get_image(self):
		current_point = int(self.current_point.item()*WIDTH)
		image = cv2.circle(self.original_image, (current_point, int(HEIGHT/2)), radius=4, color=(0, 255, 255), thickness=10)
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