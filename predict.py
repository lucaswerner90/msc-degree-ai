import argparse
from model.yolov5 import YoloV5CustomModel

model = YoloV5CustomModel()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str)
	parser.add_argument('--video', type=str)
	opt = parser.parse_args()
	if opt.img:
		results = model.predict_image(opt.img)
		print(results)
	elif opt.video:
		results = model.predict_video(opt.video)
		print(results)