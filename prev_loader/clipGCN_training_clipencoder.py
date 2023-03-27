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
from torchvision import transforms
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import random 
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from pkg_resources import packaging
import torch
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch
from torch import nn
import timm
from tqdm.auto import tqdm
import clip
import numpy as np
from sklearn.manifold import TSNE

class CFG:
    debug = False
    captions_path = "."
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr =1e-4
    text_encoder_lr =1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained = True # for both image encoder and text encoder
    trainable = False # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 224

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1
    
class Image_Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
    nn.Linear(in_features=512, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=512),
    nn.ReLU())
  def forward(self, x):
    with torch.no_grad():
      image_embedding_clip = model_clip.encode_image(x)
    return self.model(image_embedding_clip.float())


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(512, 256)
        self.conv2 = GCNConv(256, 512)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x
        
# Setting up the combined model
class CLIP_GCN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = Image_Encoder().to(device)
        self.text_encoder = GCN().to(device)
        
    def forward(self, batch):
        # Getting Image and Text Features
        image, label = batch
        image = image.to(device)
        label = label.to(device)
        image_encoding = self.image_encoder(image.to(device))
        GCN_encodings = self.text_encoder(data.to(device))
        text_encoding = GCN_encodings[label]
        # Calculating the Loss
        return contrastive_loss(image_encoding, text_encoding, label)
      
# Setting up the combined model
class CLIP_GCN_Model_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = Image_Encoder()
        self.text_encoder = GCN()
        
    def forward(self, batch):
        # Getting Image and Text Features
        image, label = batch
        image_encoding = self.image_encoder(image.to(device))
        # GCN_encodings = self.text_encoder(data.to(device))
        # text_encoding = GCN_encodings[label]

        # Calculating the Loss
        return image_encoding
      
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
      
def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        img, label = batch
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)

        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter

def train_epoch_for_batch(model,batch, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(batch, total=len(batch))
    # for batch in tqdm_object:
    img, label = batch
    loss = model(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step == "batch":
        lr_scheduler.step(loss_meter.avg)

    count = img.size(0)
    loss_meter.update(loss.item(), count)

    tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter



def cross_entropy(preds, targets, reduction='none'):
  log_softmax = nn.LogSoftmax(dim=-1)
  loss = (-targets * log_softmax(preds)).sum(1)
  if reduction == "none":
      return loss
  elif reduction == "mean":
      return loss.mean()


def contrastive_loss(image_encoding, text_encoding, labels):
  con_logits = (text_encoding @ image_encoding.T)
  diagonal_targets = torch.diag(labels)

  # Setting up the triplet loss
  triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
  anchor = image_encoding
  positive = image_encoding
  negative = image_encoding
  output = triplet_loss(anchor, positive, negative)


  con_loss = cross_entropy(con_logits, diagonal_targets, reduction='mean')

  
  return con_loss + output



def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32', device)
model_clip, preprocess = clip.load('ViT-B/32', device)
data_path = Path('cifar10')
train_dir = data_path / 'train'
transforms = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = ImageFolder(train_dir, transform=transforms)
class_names = train_dataset.classes
batch_size=128
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
model_GCN = GCN().to(device)
main_arr = []
for index, cls in enumerate(class_names):
  arr = []
  for i1,cl in enumerate(class_names):
    if index != i1:
      main_arr.append([index, i1])
      main_arr.append([i1, index])
text_encod_arr = []
Dictionary1 = {}; 
for x in class_names:
  text_inputs = torch.cat([clip.tokenize(x)]).to(device)
  Dictionary1[x]=model.encode_text(text_inputs).squeeze()
  text_encod_arr.append(list(Dictionary1[x].detach().cpu().numpy()))
x = torch.tensor(text_encod_arr, dtype=torch.float)
edge_index = torch.tensor(main_arr, dtype=torch.long)
data = Data(x=x, edge_index=edge_index.t().contiguous()).to(device)
GCN_encodings = model_GCN(data)
model = CLIP_GCN_Model().to(device)
params = [
        {"params": model.image_encoder.parameters(), "lr": 1e-3},
        {"params": model.text_encoder.parameters(), "lr": 1e-4},
    ]
optimizer = torch.optim.AdamW(params, weight_decay=0.)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )

print('Training the model')

step = "epoch"
best_loss = float('inf')
for epoch in range(50):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_dataloader, optimizer, lr_scheduler, step)

# Saving the model parameters 
from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "CLIP+GCN_Model_newclipend.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)

print('Testing the model for tsne on the train data')

MODEL_PATH = Path("models")
MODEL_NAME = "CLIP+GCN_Model_newclipend.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model = CLIP_GCN_Model_test() 

# Load in the saved state_dict()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model = loaded_model.to(device)
loaded_model.eval()
features = np.array([])
labels = np.array([])
with torch.inference_mode():
  for batch in tqdm(train_dataloader, desc='Running the model inference'):
    img, label = batch
    features = np.vstack([features, loaded_model(batch).cpu().numpy()]) if features.shape[0] > 0 else loaded_model(batch).cpu().numpy()
    labels = np.append(labels, label.cpu().numpy())
      
tsne = TSNE(n_components=2).fit_transform(features)


tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)
colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
classes = class_names
fig = plt.figure()
ax = fig.add_subplot(111)
for idx, c in enumerate(colors):
    indices = [i for i, l in enumerate(labels) if idx == l]
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
    ax.scatter(current_tx, current_ty, c=c, label=classes[idx])
    

ax.legend(loc='best')
plt.show()