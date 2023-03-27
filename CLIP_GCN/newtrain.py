import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
from models.setup_model import CLIP_GCN_Model_eval
from data.getdatasets import getdataset
from data.data_utils import CFG
from eval_methods.gcd_eval import cluster_acc_eval, epoch_acc
from data.cifar10 import *


# Setting up the cfg's 


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

def train_epoch(model, train_loader, val_data, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        model.train()
        img_emb, label, con_loss = model(batch)
        eval_loss = []
        for eval_batch in val_data:
            eval_loss.append(cluster_acc_eval(model, img_emb, label, eval_batch))

        avg_eval_loss = torch.tensor(sum(eval_loss)/len(eval_loss), requires_grad = True )
        loss = torch.tensor(avg_eval_loss + con_loss, requires_grad = True )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)

        count = img.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter



if __name__ == '__main__':
    device = CFG.device
    whole_training_set, train_dataset_labelled, train_dataset_unlabelled, mask_lab, class_names, train_classes = getdataset(CFG.dataset)

    # Getting the validation data
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    train_dataset_labelled = train_dataset_labelled_split 
    val_dataset_labelled = val_dataset_labelled_split 
    print(len(val_dataset_labelled))

    train_dataloader = DataLoader(train_dataset_labelled, CFG.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset_labelled, CFG.batch_size, shuffle=True) 
    
    
    img, _, _ = next(iter(train_dataloader))

    model = CLIP_GCN_Model_eval()
    model.to(device)
    params = [
            {"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
        )
    step = "epoch"
    best_loss = float('inf')
    epoch_acc(model)
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, step)
        epoch_acc(model)
        

    # Saving the model parameters 
    # Create models directory (if it doesn't already exist)
    MODEL_PATH = Path("CLIP_GCN/models/saved_models")
    MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                    exist_ok=True # if models directory already exists, don't error
    )

    # Create model save path
    MODEL_NAME = "cifar10_b32_cluster.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    # Save the model state dict
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
            f=MODEL_SAVE_PATH)
    
    