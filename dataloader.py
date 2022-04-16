import torch
import pandas as pd
import numpy as np
import cv2
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
np.random.seed(42)
torch.manual_seed(42)

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

class DroneImages(Dataset):

	def __init__(self, csv_path='./data/dataframe.csv', transform=IMAGE_TRANSFORMS):
		self.df = pd.read_csv(csv_path)
		self.transform = transform
		self.pretrained_model = pretrained_model

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		original_image, real_x, real_y = (
			cv2.imread(row["filename"]),
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

dataset = DroneImages()
# Split the dataset into training and validation
train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

train_dataset = Subset(train_dataset, np.arange(0, int(0.8*train_size)))
test_dataset = Subset(train_dataset, np.arange(int(0.8*train_size), train_size))

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)