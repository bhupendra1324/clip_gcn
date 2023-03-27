import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
from models.setup_model import CLIP_GCN_Model_Image_Encoder
from data.getdatasets import getdataset
from data.data_utils import CFG
from random import randint
from GCD.methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from GCD.project_utils.cluster_and_log_utils import log_accs_from_preds


device = CFG.device
whole_training_set, train_dataset_labelled, train_dataset_unlabelled, mask_lab, class_names, train_classes = getdataset(CFG.dataset)
train_dataloader = DataLoader(train_dataset_labelled, CFG.batch_size, shuffle=False)
train_class_names = [class_names[i] for i in train_classes] 

img, label, _ = next(iter(train_dataloader))
MODEL_PATH = Path("CLIP_GCN/models/saved_models")
MODEL_NAME = "cifar10_b32.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

loaded_model = CLIP_GCN_Model_Image_Encoder() 
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
loaded_model = loaded_model.to(device)


# Getting all the features from the train data
loaded_model.eval()
all_feats = np.array([])
targets = np.array([])
mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
mask_cls = np.array([])     # From all the data, which instances belong to Old classes

lab_features = np.array([])
lab_labels = np.array([])

with torch.inference_mode():
  for batch in tqdm(train_dataloader, desc='Running the model inference'):
    img, label,_ = batch
    lab_features = np.vstack([lab_features, loaded_model(batch).cpu().numpy()]) if lab_features.shape[0] > 0 else loaded_model(batch).cpu().numpy()
    lab_labels = np.append(lab_labels, label.cpu().numpy())
    mask_cls = np.append(mask_cls, [False]*len(label))
    mask_lab = np.append(mask_lab, [True]*len(label))

all_feats = lab_features
targets = np.append(targets, lab_labels)

# Getting the data from the test data

train_dataloader = DataLoader(whole_training_set, CFG.batch_size, shuffle=False)
train_class_names = class_names

unlab_features = np.array([])
unlab_labels = np.array([])

with torch.inference_mode():
  for batch in tqdm(train_dataloader, desc='Running the model inference'):
    img, label,_ = batch
    unlab_features = np.vstack([unlab_features, loaded_model(batch).cpu().numpy()]) if unlab_features.shape[0] > 0 else loaded_model(batch).cpu().numpy()
    unlab_labels = np.append(unlab_labels, label.cpu().numpy())
    mask_lab = np.append(mask_lab, [False]*len(label))
    mask_cls = np.append(mask_cls, np.isin(label.cpu().numpy(), train_classes))

all_feats = np.vstack([all_feats, unlab_features])
targets = np.append(targets, unlab_labels)    

#########
# GCD EVAL

mask_lab = mask_lab.astype(bool)
mask_cls = mask_cls.astype(bool)

# all_feats = np.concatenate(all_feats)

l_feats = all_feats[mask_lab]       # Get labelled set
u_feats = all_feats[~mask_lab]      # Get unlabelled set
l_targets = targets[mask_lab]       # Get labelled targets
u_targets = targets[~mask_lab]       # Get unlabelled target

print('Fitting Semi-Supervised K-Means...')
if CFG.dataset == 'cifar10':
  n = 10
if CFG.dataset == 'sbcars':
  n = 196
kmeans = SemiSupKMeans(k=n, tolerance=1e-4, max_iterations=10000, init='k-means++',
                        n_init=10, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for
                                            x in (l_feats, u_feats, l_targets, u_targets))

kmeans.fit_mix(u_feats, l_feats, l_targets)
all_preds = kmeans.labels_.cpu().numpy()
u_targets = u_targets.cpu().numpy()

# -----------------------
# EVALUATE
# -----------------------
# Get preds corresponding to unlabelled set
preds = all_preds[~mask_lab]

# Get portion of mask_cls which corresponds to the unlabelled set
mask = mask_cls[~mask_lab]
mask = mask.astype(bool)

# -----------------------
# EVALUATE
# -----------------------
all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets, y_pred=preds, mask=mask, eval_funcs=['v1', 'v2'],
                                                save_name='SS-K-Means Train ACC Unlabelled', print_output=True)

