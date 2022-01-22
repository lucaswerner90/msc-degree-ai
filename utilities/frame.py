"""
The module contains utility functions for video processing
that is being used in the project during prediction and training.
"""

import cv2

def add_fps_to_frame(frame, fps):
	# Calculate the time it took to process the frame
	# and print the FPS within the frame
	return cv2.putText(frame, f'{fps} / FPS', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


def add_rectangle_to_frame(frame, prediction):
	# Starting point
	# represent the top left corner of the rectangle
	start_point = (int(prediction.x_min), int(prediction.y_min))
	# Ending coordinate
	# represents the bottom right corner of rectangle
	end_point = (int(prediction.x_max), int(prediction.y_max))

	# Blue color in BGR
	color = (0, 0, 255)

	# Line thickness of 2 px
	thickness = 2

	# Using cv2.rectangle() method
	# Draw a rectangle with blue line borders of thickness of 2 px
	return cv2.rectangle(frame, start_point, end_point, color, thickness)