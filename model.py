import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

import vocab


class myAlbertModel(nn.Module):
  def __init__(self, albert, d_model, vocab_size=16000, dropout=0.1):
    super().__init__()
    self.albert = albert
    self.linear = nn.Linear(d_model, vocab_size)

  def forward(self, src):
    # print(src.size()) # torch.Size([8, 128])
    output = self.albert(src)[0]
    # print(output[0].size()) # torch.Size([8, 128, 4096])
    output = self.linear(output)
    return F.log_softmax(output, dim=-1)
