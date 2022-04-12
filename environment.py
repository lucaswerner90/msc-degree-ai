import gym
import numpy as np
import pandas as pd
from gym import spaces

N_DISCRETE_ACTIONS = 3
(HEIGHT, WIDTH, N_CHANNELS) = (360, 640, 3)

class DroneImagesEnvironment(gym.Env):
	"""Custom Environment that follows gym interface"""
	metadata = {'render.modes': ['human']}

	def __init__(self, csv_path, transform):
		super(DroneImagesEnvironment, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.df = pd.read_csv(csv_path)
		self.transform = transform
		self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
		# Example for using image as input:
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(HEIGHT, WIDTH, N_CHANNELS),
			dtype=np.uint8
		)

	def step(self, action):
		# Take a step in the environment
		# Return observation (state), reward, done, info
		distance = 20 
		next_point_of_view = None
		if action == "LEFT":
			next_point_of_view = max(1, self.point_of_view - distance)  # 200 => 200 - distancia
		if action == "RIGHT":
			next_point_of_view = min(WIDTH - 1, self.point_of_view + distance)
		
		reward = self.calculate_reward(next_point_of_view)
		return next_point_of_view, reward, reward == 1, {}

	def calculate_reward(self, next_point_of_view):
		num_sections = 32
		section_width = WIDTH // num_sections
		num_sections_away = abs(next_point_of_view - self.real_x) // section_width
		return 1 if num_sections_away == 0 else 1 - (num_sections_away / num_sections)

	def reset(self):
		# Reset the state of the environment to an initial state
		self.image = None
		self.real_x = None
		self.point_of_view = None
	def render(self, mode='human', close=False):
    	# Render the environment to the screen
		pass
