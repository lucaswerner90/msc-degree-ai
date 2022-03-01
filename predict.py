import argparse
import pandas as pd
import numpy as np
import cv2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str)
	parser.add_argument('--output', type=str, default='./data/dataframe.csv')
	opt = parser.parse_args()

	if not opt.data:
		raise ValueError('You must specify a video path')

	df = pd.read_csv(opt.data)
	for _, row in df.iterrows():
		image, point_x, point_y = row['frame'], row['prediction_x'], row['prediction_y']
		image = cv2.circle(image, (point_x,point_y), radius=1, color=(0, 0, 255), thickness=1)
		cv2.imshow('image', image)