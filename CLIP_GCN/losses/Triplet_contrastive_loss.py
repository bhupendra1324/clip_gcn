import torch
from torch import nn
from online_triplet_loss.losses import *
from data.data_utils import CFG

def cross_entropy(preds, targets, reduction='none'):
  log_softmax = nn.LogSoftmax(dim=-1)
  loss = (-targets * log_softmax(preds)).sum(1)
  if reduction == "none":
      return loss
  elif reduction == "mean":
      return loss.mean()

ce=nn.CrossEntropyLoss()


logit_scale = 4.0652
def contrastive_loss(image_encoding, text_encoding, labels):
  # normalizing the encodings
  
  logits=[]
  for txt,im in zip(text_encoding,image_encoding):
    l_i = logit_scale * im @ txt.t()
    logits.append(l_i)
  cy_con_logits = torch.stack(logits) 
  
  self_logits = (image_encoding @ image_encoding.T)
  diagonal_targets = torch.diag(labels)

  
  # Setting up the triplet loss
  triplet_loss, fraction_pos = batch_all_triplet_loss(labels, image_encoding, squared=False, margin=CFG.margin)
  # triplet_loss2, fraction_pos2 = batch_all_triplet_loss(labels, text_encoding, squared=False, margin=2)


  self_loss = cross_entropy(self_logits, diagonal_targets, reduction='mean')
  
  cy_loss = ce(cy_con_logits, labels)


  # print(f'The triplet_loss is {triplet_loss}| con_loss {con_loss}| cyclic_con_loss {cy_loss}' )

  return triplet_loss + cy_loss+self_loss
  # return 0.3*triplet_loss + 0.7*cy_loss
  # return (0.6)*triplet_loss + (0.4)*(con_loss+cy_loss)/2.0

def contrastive_logits(image_encoding, text_encoding, labels):
  # normalizing the encodings
  
  logits=[]
  for txt,im in zip(text_encoding,image_encoding):
    l_i = logit_scale * im @ txt.t()
    logits.append(l_i)
  cy_con_logits = torch.stack(logits) 
  
  self_logits = (image_encoding @ image_encoding.T)
  diagonal_targets = torch.diag(labels)

  
  # Setting up the triplet loss
  triplet_loss, fraction_pos = batch_all_triplet_loss(labels, image_encoding, squared=False, margin=0.8)


  self_loss = cross_entropy(self_logits, diagonal_targets, reduction='mean')
  
  cy_loss = ce(cy_con_logits, labels)


  # print(f'The triplet_loss is {triplet_loss}| con_loss {con_loss}| cyclic_con_loss {cy_loss}' )

  return cy_con_logits
  # return 0.3*triplet_loss + 0.7*cy_loss
  # return (0.6)*triplet_loss + (0.4)*(con_loss+cy_loss)/2.0
