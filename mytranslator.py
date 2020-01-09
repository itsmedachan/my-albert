import time
import torch
import torch.nn as nn
import numpy as np
import vocab
from fastprogress import progress_bar
import pandas as pd


class myTranslator:
  def __init__(self, model, dataloader, device, en_tokenizer, ja_tokenizer, result_file):
    self.model = model

    if torch.cuda.device_count() > 1 and device != torch.device("cpu"):
      print("[Info] Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)

    self.model = self.model.to(device)
    self.dataloader = dataloader
    self.device = device
    self.en_tokenizer = en_tokenizer
    self.ja_tokenizer = ja_tokenizer

    self.result_file = result_file

  def translate(self):
    self.model.eval()
    dataloader = self.dataloader

    data_iter = progress_bar(dataloader)

    input_english_list = []
    target_japanese_list = []
    output_japanese_list = []

    for i, batch in enumerate(data_iter):
      src = batch["input_english"].to(self.device)
      tgt = batch["target_japanese"].to(self.device)

      output = self.model(src)
      # print(output.size())
      output = torch.argmax(output, dim=-1)
      output = output.cpu().numpy()

      for n in range(len(src)):
        input_english = batch["input_english"][n].tolist()
        target_japanese = batch["target_japanese"][n].tolist()
        output_japanese = output[n].tolist()

        input_english = self.en_tokenizer.ids_to_text(input_english)
        input_english_list.append(input_english)
        target_japanese = self.ja_tokenizer.ids_to_text(target_japanese)
        target_japanese_list.append(target_japanese)
        output_japanese = self.ja_tokenizer.ids_to_text(output_japanese)
        output_japanese_list.append(output_japanese)

    df = pd.DataFrame(
        [input_english_list, output_japanese_list, target_japanese_list]).T
    df.to_csv(self.result_file)
