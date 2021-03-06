from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np

import vocab


class JESCDataloaders:
  def __init__(self, data_paths, en_tokenizer, ja_tokenizer, option):
    self.en_train = data_paths["train"]["english"]
    self.ja_train = data_paths["train"]["japanese"]
    self.en_valid = data_paths["valid"]["english"]
    self.ja_valid = data_paths["valid"]["japanese"]
    self.en_test = data_paths["test"]["english"]
    self.ja_test = data_paths["test"]["japanese"]
    self.en_tokenizer = en_tokenizer
    self.ja_tokenizer = ja_tokenizer
    self.option = option
    self.vocab_size = option.vocab_size

    self.prepare_dataloaders()

  def prepare_dataloaders(self):
    # initialize seed
    np.random.seed(0)
    torch.manual_seed(0)
    # prepare data
    self.train = torch.utils.data.DataLoader(
        JESCDataset(
            source_path=self.en_train,
            target_path=self.ja_train,
            seq_len=self.option.seq_len,
            en_tokenizer=self.en_tokenizer,
            ja_tokenizer=self.ja_tokenizer,
            vocab_size=self.vocab_size,
        ),
        batch_size=self.option.batch_size,
        shuffle=True,
        num_workers=0,
    )

    self.valid = torch.utils.data.DataLoader(
        JESCDataset(
            source_path=self.en_valid,
            target_path=self.ja_valid,
            seq_len=self.option.seq_len,
            en_tokenizer=self.en_tokenizer,
            ja_tokenizer=self.ja_tokenizer,
            vocab_size=self.vocab_size,
        ),
        batch_size=self.option.batch_size,
        num_workers=0,
    )

    self.test = torch.utils.data.DataLoader(
        JESCDataset(
            source_path=self.en_test,
            target_path=self.ja_test,
            seq_len=self.option.seq_len,
            en_tokenizer=self.en_tokenizer,
            ja_tokenizer=self.ja_tokenizer,
            vocab_size=self.vocab_size,
        ),
        batch_size=self.option.batch_size,
        num_workers=0,
    )


class JESCDataset(Dataset):
  def __init__(self, source_path, target_path, seq_len, en_tokenizer, ja_tokenizer, vocab_size, corpus_lines=None, train=True):
    self.seq_len = seq_len
    self.en_tokenizer = en_tokenizer
    self.ja_tokenizer = ja_tokenizer
    self.corpus_lines = corpus_lines
    self.vocab_size = vocab_size
    self.train = train

    with open(source_path, "r", encoding="utf-8") as f:
      en_lines = [
          line[:-1] for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
      self.corpus_lines = len(en_lines)

    with open(target_path, "r", encoding="utf-8") as f:
      ja_lines = [
          line[:-1] for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
      # corpus_lines = len(ja_lines)

    self.lines = []
    for en, ja in zip(en_lines, ja_lines):
      self.lines.append([en, ja])

  def __len__(self):
    return self.corpus_lines  # 約2,000,000

  def __getitem__(self, item):
    t1, t2 = self.get_corpus_line(item)
    t1 = self.en_tokenizer.text_to_ids(t1)
    t2 = self.ja_tokenizer.text_to_ids(t2)
    # t1, t2, is_next_label = self.random_sent(item)
    if self.train:
      t1_random = self.random_word(t1)
      t1 = [vocab.BOS] + t1_random + [vocab.EOS]
    else:
      t1 = [vocab.BOS] + t1 + [vocab.EOS]

    # [CLS] tag = SOS tag, [SEP] tag = EOS tag
    t2 = [vocab.BOS] + t2 + [vocab.EOS]

    # t1_label = [vocab.pad_index] + t1_label + [vocab.pad_index]
    # t2_label = t2_label + [vocab.pad_index]

    # segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
    # bert_input = (t1 + t2)[:self.seq_len]
    # bert_label = (t1_label + t2_label)[:self.seq_len]

    # for t1 (english)
    padding = [vocab.PAD for _ in range(self.seq_len - len(t1))]
    t1.extend(padding)

    # for t2 (japanese)
    padding = [vocab.PAD for _ in range(self.seq_len - len(t2))]
    t2.extend(padding)

    output = {
        "input_english": t1,
        "target_japanese": t2,
    }

    return {key: torch.tensor(value) for key, value in output.items()}

  def random_word(self, tokens):
    for i, token in enumerate(tokens):
      prob = random.random()
      if prob < 0.15:
        prob /= 0.15

        # 80% randomly change token to mask token
        if prob < 0.8:
          tokens[i] = vocab.MASK

        # 10% randomly change token to random token
        elif prob < 0.9:
          tokens[i] = random.randrange(self.vocab_size)

        # 10% randomly change token to current token
        else:
          continue

    return tokens

  def random_sent(self, index):
    t1, t2 = self.get_corpus_line(index)

    # output_text, label(isNotNext:0, isNext:1)
    if random.random() > 0.5:
      return t1, t2, 1
    else:
      return t1, self.get_random_line(), 0

  def get_corpus_line(self, item):
    return self.lines[item][0], self.lines[item][1]

  # def get_random_line(self):
  #     if self.on_memory:
  #         return self.lines[random.randrange(len(self.lines))][1]

  #     line = self.file.__next__()
  #     if line is None:
  #         self.file.close()
  #         self.file = open(self.corpus_path, "r", encoding=self.encoding)
  #         for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
  #             self.random_file.__next__()
  #         line = self.random_file.__next__()
  #     return line[:-1].split("\t")[1]
