from torch.utils.data import Dataset
import tqdm
import torch
import random

source_path = 'english-dev-margin-removed'
target_path = 'japanese-dev'

corpus_lines = None

with open(source_path, "r", encoding="utf-8") as f:
  en_lines = [line[:-1]
              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
  corpus_lines = len(en_lines)
  print(corpus_lines)

with open(target_path, "r", encoding="utf-8") as f:
  ja_lines = [line[:-1]
              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
  corpus_lines = len(ja_lines)
  print(corpus_lines)
