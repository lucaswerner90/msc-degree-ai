import argparse
import pandas as pd
import numpy as np
import cv2

def get_random_point(xmax, ymax):
	return np.random.randint(1, xmax), np.random.randint(1, ymax)

def euclidean_distance_score(x1,y1,x2,y2):
	"""
	Use the euclidean distance to calculate the score between
	the predicted point and the real one
	"""
	distance = round(np.sqrt((x2-x1)**2 + (y2-y1)**2))
	return distance

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', type=str)
	parser.add_argument('--output', type=str, default='./data/dataframe.csv')
	opt = parser.parse_args()

	if not opt.data:
		raise ValueError('You must specify a video path')

	df = pd.read_csv(opt.data)
	while True:
		row = df.iloc[np.random.randint(0,len(df))]
		filename, point_x, point_y = row['filename'], row['prediction_x'], row['prediction_y']
		image = cv2.imread(filename)
		height, width, _ = image.shape
		predicted_x, predicted_y = get_random_point(width, height)
		score = euclidean_distance_score(predicted_x, predicted_y, point_x, point_y)
		image = cv2.circle(image, (predicted_x,predicted_y), radius=4, color=(0, 255, 255), thickness=10)
		image = cv2.circle(image, (point_x,point_y), radius=4, color=(0, 0, 255), thickness=10)
		image = cv2.putText(image,f'Score: {score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
		cv2.imshow('image', image)
		cv2.waitKey()
		# cv2.waitKey(1)
	