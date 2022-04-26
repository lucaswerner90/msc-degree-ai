"""
Define the Vision Transformer that is used to encode the image.
"""

import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel

VISUAL_TRANSFORMER_MODEL = "google/vit-base-patch16-224-in21k"

class EncoderViT(nn.Module):
    def __init__(self):
        super(EncoderViT, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(VISUAL_TRANSFORMER_MODEL)
        self.model = ViTModel.from_pretrained(VISUAL_TRANSFORMER_MODEL)
        self.model.eval()

    def forward(self, x):
        inputs = self.feature_extractor(x, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state