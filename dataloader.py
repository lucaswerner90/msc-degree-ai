import pandas as pd
import cv2
from torch.utils.data import Dataset

class DroneImages(Dataset):

	def __init__(self, csv_path, transform):
		self.df = pd.read_csv(csv_path)
		self.transform = transform

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		original_image, real_x, real_y = (
			cv2.imread(row["filename"]),
			row["prediction_x"],
			row["prediction_y"],
		)
		image = original_image
		if self.transform:
			image = self.transform(image)
		
		sample = {
			"original_image": original_image,
			"image":image,
			"real_x": real_x,
			"real_y": real_y
		}

		return sample