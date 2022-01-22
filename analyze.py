"""
The module accepts a video path and predicts frame by frame
where the bounding box of the person is detected, adding the results
to an external Pandas dataframe that will be used later during the training
of the RL agent.
"""

import os
import logging
import warnings
import pandas as pd
import argparse
import cv2
from tqdm import tqdm
from model.yolov5 import YoloV5CustomModel

IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360
DATAFRAME_COLUMNS = ['frame', 'x_min', 'y_min', 'x_max', 'y_max']

# The YOLO model is giving us a warning because of the missing CUDA GPU
# so we can simply ignore the warning (we don't need it)
warnings.filterwarnings("ignore", category=UserWarning)

def analyze_video(video_path, output_path):
	"""
	It opens a video, runs YoloV5 model on it
	and saves the results on the output path

	Args:
		video_path ([str]): Contains the path of the video
		output_path ([str]): Path of the resulting Pandas dataframe
	"""

	video = cv2.VideoCapture(video_path)
	model = YoloV5CustomModel()
	total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

	logging.info(f'Analyzing video {video_path}, total frames: {total_frames}')
	# Add a progress bar to track the progress of the process,
	# using the total number of frames in the video
	progress_bar = tqdm(total=total_frames, desc='Processing video frames')

	if os.path.exists(output_path):
		dataframe = pd.read_csv(output_path)
	else:
		dataframe = pd.DataFrame(columns=DATAFRAME_COLUMNS)

	while video.isOpened():
		ret, frame = video.read()
		if not ret:
			break
		frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
		prediction = model(frame)
		df_prediction = pd.DataFrame([[
			frame,
			prediction.x_min,
			prediction.y_min,
			prediction.x_max,
			prediction.y_max
		]], columns=DATAFRAME_COLUMNS)
		dataframe = pd.concat([dataframe, df_prediction], ignore_index=True)
		progress_bar.update(1)
	
	video.release()
	dataframe.to_csv(output_path, index=False)
	logging.info(f'Results saved on {output_path}')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--video', type=str)
	parser.add_argument('--output', type=str, default='./data/output.csv')
	opt = parser.parse_args()

	if not opt.video:
		raise ValueError('You must specify a video path')

	analyze_video(opt.video, opt.output)
