import requests
import zipfile
from pathlib import Path
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pathlib import Path
import time
import os
from torchvision import transforms
import clip
from pkg_resources import packaging
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torchvision
import timm


import os
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from timm import create_model
from tqdm.auto import tqdm


def find_classes(directory: str):
  classes = sorted([entry.name for entry in list(os.scandir(directory))])
  if not classes:
    raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
  class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
  return classes, class_to_idx

class CustomDataset(Dataset):
  def __init__(self, target_dir, transform = None):
    self.paths = list(target_dir.glob('*/*.jpg'))
    self.transform = transform
    self.classes, self.class_to_idx = find_classes(target_dir)
  def load_image(self, index):
    image_path = self.paths[index]
    return image_path
  def __len__(self):
    return len(self.paths)
  def __getitem__(self, index):
    time.sleep(0.1);
    img = self.load_image(index)
    class_name = self.paths[index].parent.name
    class_idx = self.class_to_idx[class_name]
    if self.transform:
      return self.transform(img).to(device), class_idx
    else:
      return img, class_idx
  
# Let's  make the custom transform for our dataset which will inherit from the clip preprocess

class Clip_transforms(object):
    def __call__(self,img):
        return  preprocess(Image.open(img)).to(device)
    
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(512, 16)
        self.conv2 = GCNConv(16, 512)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
class Image_Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=768 * 49, out_features=512),
    nn.ReLU())
    self.layer1 = nn.Sequential(
    vit32,
    fc)
  def forward(self, x):
    return self.layer1(x)


def cross_entropy(preds, targets, reduction='none'):
  log_softmax = nn.LogSoftmax(dim=-1)
  loss = (-targets * log_softmax(preds)).sum(1)
  if reduction == "none":
      return loss
  elif reduction == "mean":
      return loss.mean()

def Contrastive_loss(image_encoding, text_encoding):
  # Getting Image and Text Features

  # Calculating the Loss
  logits = (text_encoding @ image_encoding.T) 
  images_similarity = image_encoding @ image_encoding.T
  texts_similarity = text_encoding @ text_encoding.T
  targets = F.softmax(
      (images_similarity + texts_similarity) / 2, dim=-1
  )
  texts_loss = cross_entropy(logits, targets, reduction='none')
  images_loss = cross_entropy(logits.T, targets.T, reduction='none')
  loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
  return loss.mean()

def cyclic_contrastive_loss(image_encoding, text_encoding):
  con_logits = (text_encoding @ image_encoding.T)
  cy_con_logits = (image_encoding @ text_encoding.T)
  image_sim = image_encoding @ image_encoding.T
  text_sim = text_encoding @ text_encoding.T
  targets = F.softmax(
      (image_sim + text_sim) / 2
  )
  con_text_loss = cross_entropy(con_logits, targets, reduction='none')
  con_image_loss = cross_entropy(con_logits.T, targets.T, reduction='none')
  con_loss = (con_image_loss + con_text_loss) / 2.0

  # cyclic loss
  cy_text_loss = cross_entropy(cy_con_logits.T, targets.T, reduction='none')
  cy_image_loss = cross_entropy(cy_con_logits, targets, reduction='none')
  cy_loss = (cy_image_loss + cy_text_loss) / 2.0

  loss = con_loss + cy_loss
  return loss.mean()

# Setting up the combined model
class CLIP_GCN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = Image_Encoder()
        self.text_encoder = GCN()
        
    def forward(self, batch):
        # Getting Image and Text Features
        image, label = batch
        image_encoding = self.image_encoder(image.to(device))
        GCN_encodings = self.text_encoder(data.to(device))
        text_encoding = GCN_encodings[label]

        # Calculating the Loss
        return cyclic_contrastive_loss(image_encoding, text_encoding)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FOLDER_PATH = 'CLIP+GCN/101_Object_CategoriesNew'
    # Set the path for the data folder 
    data_path = Path('CLIP+GCN/101_Object_CategoriesNew')
    train_dir = data_path / 'Train'
    print(f"The target directory is {train_dir}")
    class_names_found = sorted([entry.name for entry in list(os.scandir(train_dir))])
    # print(class_names_found)

    class_names, class_to_idx = find_classes(train_dir)
    #print((class_names))
  
    model, preprocess = clip.load('ViT-B/32', device)



    transforms_layer = transforms.Compose([Clip_transforms()])        

    data_custom = CustomDataset(target_dir=train_dir, transform = transforms_layer)
    train_dataloader = DataLoader(dataset=data_custom, # use custom created train Dataset
                                     batch_size=32, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

    model_GCN = GCN().to(device)

    main_arr = []
    for index, cls in enumerate(class_names):
      arr = []
      for i1,cl in enumerate(class_names):
        if index != i1:
          main_arr.append([index, i1])
          main_arr.append([i1, index])
    # npArr = np.array(arr)

    # main_arr

    text_encod_arr = []
    Dictionary1 = {}; 
    for x in class_names:
      text_inputs = torch.cat([clip.tokenize(x)]).to(device)
      Dictionary1[x]=model.encode_text(text_inputs).squeeze()
      text_encod_arr.append(list(Dictionary1[x].detach().cpu().numpy()))

    x = torch.tensor(text_encod_arr, dtype=torch.float)

    edge_index = torch.tensor(main_arr, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index.t().contiguous()).to(device)


    model_name = "vit_base_patch32_224"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device = ", device)
    # create a ViT model : https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    vit32 = create_model(model_name, pretrained=True).to(device)

    vit32 = nn.Sequential(*list(vit32.children())[:-1])
    for param in vit32.parameters():
        param.requiresGrad = False

    fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=768 * 49, out_features=512),
        nn.ReLU()
    )
    model_IG = nn.Sequential(
        vit32,
        fc
    ).to(device)

    model_0 = CLIP_GCN_Model().to(device)
    optimizer = torch.optim.Adam(model_0.parameters(), lr=0.01)


    epochs = 50;
    for epoch in tqdm(range(epochs)):
      train_loss = 0
      print("The current epoch is : ", epoch + 1)
      for batch, (batch_model) in enumerate(train_dataloader):
        img, label = batch_model
        # print(model_0(batch_model))
        loss = model_0(batch_model)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
          print(f"........We have looked at : {batch * len(img)}/ {len(train_dataloader.dataset)} and the avg loss now is {train_loss/batch +1}")
      print(f"The epoch is {epoch + 1} and the avg train loss is {train_loss/len(train_dataloader)}")

# Saving the model parameters 
from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "CLIP+GCN_Model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)