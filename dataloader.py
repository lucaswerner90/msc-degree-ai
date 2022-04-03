import pandas as pd
import cv2
import torchvision.models as models
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class DroneImages(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
		self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize([224, 224]),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        )
		self.pretrained_model = models.vgg19(pretrained=True)
		self.pretrained_model.classifier = nn.Sequential(
            *list(self.pretrained_model.classifier.children())[:-2]
        )
		for params in self.pretrained_model.parameters():
			params.requires_grad = False
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

		image = self.pretrained_model(image.unsqueeze(0))

		sample = {
			"original_image": original_image,
			"image":image.squeeze(),
			"real_x": real_x,
			"real_y": real_y
		}

		return sample