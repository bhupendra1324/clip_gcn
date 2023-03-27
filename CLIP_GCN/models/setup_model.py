import torch
from torch import nn
import clip
from models.Image_encoder import Image_Encoder
from models.text_encodder import GCN
from data.getdatasets import getdataset
from data.data_utils import CFG
from losses.Triplet_contrastive_loss import *


# Setting up the device agnostic code and the clip model 
device = CFG.device
model, preprocess = clip.load('ViT-B/32', device)


# Calling the datasets to make the imports
_, _, _, _, class_names, train_classes = getdataset(CFG.dataset)
train_class_names = [class_names[i] for i in train_classes]
# print("The train classes are: ", train_class_names)

# Making the clip encodings and the edge index
main_arr = []
for index, cls in enumerate(train_class_names):
  arr = []
  for i1,cl in enumerate(train_class_names):
    if index != i1:
      main_arr.append([index, i1])
      main_arr.append([i1, index])
      
text_encod_arr = []
Dictionary1 = {}; 
for x in train_class_names:
  text_inputs = torch.cat([clip.tokenize(f'This is a photo of a {x}')]).to(device)
  Dictionary1[x]=model.encode_text(text_inputs).squeeze()
  text_encod_arr.append(list(Dictionary1[x].detach().cpu().numpy()))
  
x = torch.tensor(text_encod_arr, dtype=torch.float)

edge_index = torch.tensor(main_arr, dtype=torch.long)
n = len(train_class_names)

mappings_idx = {i:idx for idx, i in enumerate(train_classes)}

# Setting up the combined model

# Setting up the combined model
class CLIP_GCN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = Image_Encoder().to(device)
        self.text_encoder = GCN(n = n, edges=edge_index, in_channels=512, out_channels=512, hidden_layers='d256,d').to(device)
        
    def forward(self, batch, unlab = False):
        # Getting Image and Text Features
        image, label, _ = batch
        image = image.to(device)
        label = label.to(device)
        # mapped_labels = []
        # mapped_labels = [mappings_idx[i.item()] for i in label]
        # mapped_labels = torch.tensor(mapped_labels, dtype=torch.long)
        image_encoding = self.image_encoder(image.to(device))
        if unlab:
           return image_encoding
        
        text_enc = []
        for img in range(image_encoding.shape[0]):
            GCN_encodings = self.text_encoder(x.to(device))
            text_enc.append(GCN_encodings)
        text_encoding = torch.stack(text_enc).to(device)
        
        return contrastive_loss(image_encoding, text_encoding, label)
    


class CLIP_GCN_Model_Image_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = Image_Encoder().to(device)
        self.text_encoder = GCN(n = n, edges=edge_index, in_channels=512, out_channels=512, hidden_layers='d256,d').to(device)
        
    def forward(self, batch):
        # Getting Image and Text Features
        image, label, _ = batch
        image = image.to(device)
        label = label.to(device)
        image_encoding = self.image_encoder(image.to(device))
        # GCN_encodings = self.text_encoder(x.to(device))
        # text_encoding = GCN_encodings[label]
        # Calculating the Loss
        return image_encoding



class CLIP_GCN_logits(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = Image_Encoder().to(device)
        self.text_encoder = GCN(n = n, edges=edge_index, in_channels=512, out_channels=512, hidden_layers='d256,d').to(device)
        
    def forward(self, batch, unlab = False):
        # Getting Image and Text Features
        image, label, _ = batch
        image = image.to(device)
        label = label.to(device)
        
        image_encoding = self.image_encoder(image.to(device))
        if unlab:
           return image_encoding
        
        text_enc = []
        for img in range(image_encoding.shape[0]):
            GCN_encodings = self.text_encoder(x.to(device))
            text_enc.append(GCN_encodings)
        text_encoding = torch.stack(text_enc).to(device)
        
        return contrastive_logits(image_encoding, text_encoding, label)
    
