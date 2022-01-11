#! .venv/bin/python3

"""
YOLOv5 model

This script contains the YOLOv5 model definition.
We load the model using the torch.hub module and we configure the model
to comply with our requirements.

The model.classes variable contains the list of classes that we want to detect,
in this case we want to detect only persons.

"""

from collections import namedtuple
import cv2
import torch
from torch.nn import Module


CustomDetection = namedtuple('CustomDetection', ['x_min', 'y_min', 'x_max', 'y_max'])

class YoloV5CustomModel(Module):
	def __init__(self, 
				pretrained_model='yolov5s',
				confidence=0.6,
				iou_threshold=0.45,
				classes=[0],
				multi_label=False,
				number_detections_per_image=1):

		super(YoloV5CustomModel, self).__init__()
		model = torch.hub.load('ultralytics/yolov5', pretrained_model, pretrained=True, verbose=False)
		model.conf = confidence
		model.iou = iou_threshold
		model.classes = classes
		model.multi_label = multi_label
		model.max_det = number_detections_per_image

		self.model = model
	
	def forward(self, x):
		return self.model(x)
	
	def predict_image(self, image:str):
		"""
		Returns the predictions of the model for an image.
		"""
		results = self.model(image)
		if len(results) and len(results.xyxy[0]) and len(results.xyxy[0][0]):
			# Results is of type models.common.Detections
			# https://github.com/ultralytics/yolov5/blob/a2f4a1799ba6dabea4cd74a3b1e292c102918670/models/common.py#L548
			# So in order to get the correct values,
			# we need to filter only the fields that we're gonna use
			# later on our algorithm.
			# That is, 'xmin', 'ymin', 'xmax', 'ymax' since we are only detecting
			# persons, and in this case, we only detect one person per image
			return CustomDetection(*results.xyxy[0][0][:4].tolist())
		return None

	def predict_video(self, video_path:str):
		"""
		Returns the predictions of the model for a video, frame by frame.
		"""
		predictions = []
		video = cv2.VideoCapture(video_path)
		while video.isOpened():
			ret, frame = video.read()
			if not ret:
				break
			prediction = self.predict_image(frame)
			if prediction:
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
				frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
			
			predictions.append([frame, prediction])

			cv2.imshow('Predict video frame',frame)
			cv2.waitKey(1)
		video.release()
		cv2.destroyAllWindows()
		return predictions