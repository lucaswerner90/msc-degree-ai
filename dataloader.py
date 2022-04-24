import torch
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_TRANSFORMS = transforms.Compose(
	[
		transforms.ToTensor(),
		transforms.Resize([224, 224]),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		),
	]
)

pretrained_model = models.vgg19(pretrained=True)

pretrained_model.classifier = torch.nn.Sequential(
	*list(pretrained_model.classifier.children())[:-2]
)
for params in pretrained_model.parameters():
	params.requires_grad = False

class DroneImagesDataset(Dataset):

	def __init__(self, dataframe, transform=IMAGE_TRANSFORMS):
		self.df = dataframe
		self.transform = transform
		self.pretrained_model = pretrained_model

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		original_image, real_x, real_y = (
			cv2.imread(row["filename"],cv2.COLOR_BGR2RGB),
			row["prediction_x"],
			row["prediction_y"],
		)
		image = self.transform(original_image)
		image = self.pretrained_model(image.unsqueeze(0))
		image = image.squeeze()
		sample = {
			"original_image": original_image,
			"image":image,
			"real_x": real_x,
			"real_y": real_y
		}

		return sample

df = pd.read_csv('./data/dataframe.csv')

df_right = np.where(df['prediction_x'] >= 360)[0]
df_left = np.where(df['prediction_x'] <= 280)[0]
df_center = np.where((df['prediction_x'] > 280) & (df['prediction_x'] < 360))[0]

a = np.random.choice(df_right, size=340)
b = np.random.choice(df_left, size=340)
c = np.random.choice(df_center, size=340)

df= pd.concat([df.iloc[a], df.iloc[b], df.iloc[c]], axis = 0).reset_index()
df = df.sample(frac=1, random_state=42).reset_index()

df_train, df_validation = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

df_train, df_test = train_test_split(
    df_train,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

train_dataset = DroneImagesDataset(df_train)
validation_dataset = DroneImagesDataset(df_validation)
test_dataset = DroneImagesDataset(df_test)