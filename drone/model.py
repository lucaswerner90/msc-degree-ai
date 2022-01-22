"""
This module controls the drone.

The most important methods here are:

1) execute_action(action)
2) streamon()

"""

import cv2
import time
from djitellopy import Tello

class DroneModel:

	LOW_BATTERY_LEVEL = 15
	AVAILABLE_ACTIONS = {
		'TAKEOFF': 'takeoff',
		'LAND': 'land',
		'ROTATE_RIGHT': 'rotate_right',
		'ROTATE_LEFT': 'rotate_left'
	}

	def __init__(self, device = Tello(), num_actions_per_second=1) -> None:
		self._device = device
		self._is_connected = False
		self._num_actions_per_second = num_actions_per_second
		self._video_frame = None
		self._is_streaming = False

	def execute_action(self, action:str, amount:int = 45):
		if not self._is_connected:
			raise Exception('Drone is not connected')

		if action not in DroneModel.AVAILABLE_ACTIONS:
			raise Exception(f'Action {action} is not available')

		# For now the only actions available would be to
		# rotate the drone to the left or to the right
		if action == DroneModel.AVAILABLE_ACTIONS['TAKEOFF']:
			self.takeoff()
		elif action == DroneModel.AVAILABLE_ACTIONS['LAND']:
			self.land()
		elif action == DroneModel.AVAILABLE_ACTIONS['ROTATE_RIGHT']:
			# Amount is between 1-360
			self._device.rotate_clockwise(amount)
		elif action == DroneModel.AVAILABLE_ACTIONS['ROTATE_LEFT']:
			# Amount is between 1-360
			self._device.rotate_counter_clockwise(amount)
		
		# Wait for the action to be executed
		time.sleep(1 / self._num_actions_per_second)

	def is_streaming(self):
		return self._is_streaming

	def get_last_video_frame(self):
		if self._is_connected:
			return self._video_frame
		raise Exception('Drone is not connected')

	def streamoff(self):
		self._device.streamoff()
		self._is_streaming = False

	def streamon(self, display_video = False):
		if not self._is_connected:
			raise Exception('Drone is not connected')

		self._device.streamon()
		self._is_streaming = True
		cap = self._device.get_video_capture()

		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				print("Can't receive frame. Exiting ...")
				break
			
			self._video_frame = frame

			if self.get_battery() < DroneModel.LOW_BATTERY_LEVEL:
				self._device.land()

			if display_video:
				cv2.imshow('Drone camera', frame)

			if cv2.waitKey(1) == ord('q'):
				self.streamoff()
				break

	def connect(self, video_bitrate=Tello.BITRATE_1MBPS, video_resolution=Tello.VIDEO_RESOLUTION_480P):
		self._device.connect()
		self._device.set_video_bitrate(video_bitrate)
		
		#Tello.RESOLUTION_480P
		#Tello.RESOLUTION_720P
		self._device.set_video_resolution(video_resolution)
		
		self._is_connected = True

	def land(self):
		self._device.land()

	def takeoff(self):
		self._device.takeoff()

	def get_battery(self):
		if self._is_connected:
			return self._device.get_battery()
		raise Exception('Drone is not connected')
	
