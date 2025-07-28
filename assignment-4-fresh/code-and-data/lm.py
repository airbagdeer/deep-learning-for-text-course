from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> [torch.IntTensor, torch.IntTensor]:
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    return (inputs, labels)

def compute_loss(logits, gold_labels):
    # logits size is (batch, seq_len, vocab_size)
    # gold_bales size is (batch, seq_len)
    # NOTE remember to handle padding (ignore them in loss calculation!)
    # NOTE cross-entropy expects other dimensions for logits
    # NOTE you can either use cross_entropy from PyTorch, or implement the loss on your own.
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), gold_labels.view(-1), ignore_index=-1)
    return loss

