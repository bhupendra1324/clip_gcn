import torch
import numpy as np
from GCD.methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from GCD.project_utils.cluster_utils import cluster_acc, np, linear_assignment
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



def split_cluster_acc(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    return old_acc

def cluster_acc_eval(loaded_model, lab_features, lab_labels, eval_batch):
    

    device = CFG.device
   
    # Getting all the features from the train data
    loaded_model.eval()
    with torch.no_grad():
        lab_features = lab_features.cpu().detach().numpy()
        lab_labels = lab_labels.cpu().detach().numpy()
   
    all_feats = np.array([])
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes

    mask_cls = np.append(mask_cls, [False]*len(lab_labels))
    mask_lab = np.append(mask_lab, [True]*len(lab_labels))
    all_feats = lab_features
    targets = np.append(targets, lab_labels)

    # Getting the data from the test data

    unlab_features = np.array([])
    unlab_labels = np.array([])

    

    with torch.inference_mode():
        img, label,_ = eval_batch
        unlab_features = np.vstack([unlab_features, loaded_model(eval_batch, unlab = True).cpu().numpy()]) if unlab_features.shape[0] > 0 else loaded_model(eval_batch, unlab = True).cpu().numpy()
        unlab_labels = np.append(unlab_labels, label.cpu().numpy())
        mask_lab = np.append(mask_lab, [False]*len(label))
        mask_cls = np.append(mask_cls, np.isin(label.cpu().numpy(), lab_labels))
    

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

    n = len(set(targets))
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
    
    acc = split_cluster_acc(y_true=u_targets, y_pred=preds, mask=mask)

    return 100 * (1- acc)

def epoch_acc(loaded_model):
    device = CFG.device
    whole_training_set, train_dataset_labelled, train_dataset_unlabelled, mask_lab, class_names, train_classes = getdataset(CFG.dataset)
    train_dataloader = DataLoader(train_dataset_labelled, CFG.batch_size, shuffle=False)
    train_class_names = [class_names[i] for i in train_classes] 

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
            lab_features = np.vstack([lab_features, loaded_model(batch, unlab = True).cpu().numpy()]) if lab_features.shape[0] > 0 else loaded_model(batch, unlab = True).cpu().numpy()
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
            unlab_features = np.vstack([unlab_features, loaded_model(batch, unlab = True).cpu().numpy()]) if unlab_features.shape[0] > 0 else loaded_model(batch, unlab = True).cpu().numpy()
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
    n = len(set(targets))
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

