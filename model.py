import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

import vocab


class albertModel(nn.Module):
  def __init__(self, albert, d_model=33, n_vocab=16000, dropout=0.1):
    super().__init__()
    self.albert = albert
    self.linear = nn.Linear(d_model, n_vocab)

  def forward(self, src):
    output = self.albert(src)
    output = self.linear(output)
    return F.log_softmax(output, dim=-1)
