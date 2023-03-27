import torch 
import timm
from torch import nn
import clip
from data.data_utils import CFG


device = CFG.device
model_clip, preprocess = clip.load('ViT-B/32', device)

class Image_Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=512),
    nn.ReLU()
    )
  def forward(self, x):
    with torch.no_grad():
      image_embedding_clip = model_clip.encode_image(x)
    return self.model(image_embedding_clip.float())
