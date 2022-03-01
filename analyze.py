"""
The module accepts a video path and predicts frame by frame
where the bounding box of the person is detected, adding the results
to an external Pandas dataframe that will be used later during the training
of the RL agent.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import argparse
import cv2
from tqdm import tqdm
from model.yolov5 import YoloV5CustomModel

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360

# The YOLO model is giving us a warning because of the missing CUDA GPU
# so we can simply ignore the warning (we don't need it)
warnings.filterwarnings("ignore", category=UserWarning)
model = YoloV5CustomModel()

def analyze_video(video_path, output_path) -> None:
	"""
	It opens a video, runs YoloV5 model on it
	and saves the results on the output path.

	Only frames that contain predictions will be 
	added to the output dataframe.
	"""
	output_dir = os.path.join(output_path,'processed_frames')
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	video = cv2.VideoCapture(video_path)
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

	logging.info(f'Analyzing video {video_path}, total frames: {total_frames}')
	# Add a progress bar to track the progress of the process,
	# using the total number of frames in the video
	progress_bar = tqdm(total=total_frames, desc='Processing video frames')

	filenames = []
	predictions = []
	frame_number = 0
	while video.isOpened():
		ret, frame = video.read()
		if not ret:
			break
		frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
		prediction = model(frame)
		if prediction:
			filename = os.path.join(output_dir, f'{total_frames}_{frame_number}_{prediction[0]}_{prediction[1]}.jpg')
			# [center_x, center_y]
			predictions.append(prediction)
			cv2.imwrite(filename, frame)
			filenames.append(filename)
			frame_number += 1
		progress_bar.update(1)
	
	video.release()
	progress_bar.close()

	df = pd.DataFrame({
		'filename': filenames,
		'prediction_x': list(map(lambda x: x[0], predictions)),
		'prediction_y': list(map(lambda x: x[1], predictions))
	})
	df.to_csv(os.path.join(output_path,'dataframe.csv'), index=False)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video', type=str)
	parser.add_argument('--output', type=str, default='./data')
	opt = parser.parse_args()

	if not opt.video:
		raise ValueError('You must specify a video path')

	analyze_video(opt.video, opt.output)
