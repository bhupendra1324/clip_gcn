import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from models.setup_model import *
from data.getdatasets import getdataset
from data.data_utils import CFG
from random import randint

def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

device = CFG.device
whole_training_set, train_dataset_labelled, train_dataset_unlabelled, mask_lab, class_names, train_classes = getdataset(CFG.dataset)
train_dataloader = DataLoader(train_dataset_labelled, CFG.batch_size, shuffle=False)
train_class_names = class_names

img, label, _ = next(iter(train_dataloader))
MODEL_PATH = Path("CLIP_GCN/models/saved_models")
MODEL_NAME = "cifar10_2epoch.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model = CLIP_GCN_logits() 
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model = loaded_model.to(device)

loaded_model.eval()
features = np.array([])
labels = np.array([])
with torch.inference_mode():
  for batch in tqdm(train_dataloader, desc='Running the model inference'):
    img, label,_ = batch
    features = np.vstack([features, loaded_model(batch).cpu().numpy()]) if features.shape[0] > 0 else loaded_model(batch).cpu().numpy()
    labels = np.append(labels, label.cpu().numpy())
    
tsne = TSNE(n_components=2).fit_transform(features)
tx = tsne[:, 0]
ty = tsne[:, 1]
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

colors = []
classes = train_class_names
for i in range(len(classes)):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

if CFG.dataset == 'cifar10':
    colors =['red', 'orange', 'yellow', 'lime', 'green', 'turquoise', 'blue', 'purple', 'pink', 'magenta']

colors = colors[:len(classes)]
fig = plt.figure()
ax = fig.add_subplot(111)
for idx, c in enumerate(colors):
    indices = [i for i, l in enumerate(labels) if idx == l]
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
    ax.scatter(current_tx, current_ty, c=c, label=classes[idx])
    

ax.legend(loc='best')
plt.show()

    