#! .venv/bin/python3

"""
YOLOv5 model

This script contains the YOLOv5 model definition.
We load the model using the torch.hub module and we configure the model
to comply with our requirements.

The model.classes variable contains the list of classes that we want to detect,
in this case we want to detect only persons.

"""

import torch
from torch.nn import Module
from collections import namedtuple

YOLOFrameDetection = namedtuple('YOLOFrameDetection', ['x_min', 'y_min', 'x_max', 'y_max'])

class YoloV5CustomModel(Module):
	def __init__(self, 
				pretrained_model='yolov5n',
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
		model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model.eval()

		self.model = model
	
	def forward(self, image) -> YOLOFrameDetection:
		results = self.model(image)
		if len(results) and len(results.xyxy[0]) and len(results.xyxy[0][0]):
			# Results is of type models.common.Detections
			# https://github.com/ultralytics/yolov5/blob/a2f4a1799ba6dabea4cd74a3b1e292c102918670/models/common.py#L548
			# So in order to get the correct values,
			# we need to filter only the fields that we're gonna use
			# later on our algorithm.
			# That is, 'xmin', 'ymin', 'xmax', 'ymax' since we are only detecting
			# persons, and in this case, we only detect ONE PERSON PER IMAGE
			return YOLOFrameDetection(*map(lambda x: round(x), results.xyxy[0][0][:4].tolist()))
		return YOLOFrameDetection(None, None, None, None)