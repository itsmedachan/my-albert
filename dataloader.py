from torch.utils.data import Dataset
import tqdm
import torch
import random

import vocab


class BERTDataset(Dataset):
               def __init__(self, source_path, target_path, vocab, seq_len, encoding="utf-8", corpus_lines=None):
    self.vocab = vocab
    self.seq_len = seq_len

    self.corpus_lines = corpus_lines
    self.corpus_path = corpus_path
    self.encoding = encoding

    with open(source_path, "r", encoding="utf-8") as f:
      en_lines = [line[:-1] for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
      self.corpus_lines = len(en_lines)

    with open(target_path, "r", encoding="utf-8") as f:
      ja_lines = [line[:-1] for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
      # corpus_lines = len(ja_lines)

    self.lines = []
    for en, ja in zip(en_lines, ja_lines):
      self.lines.append( [en, ja] )

  def __len__(self):
    return self.corpus_lines #約2,000,000

  def __getitem__(self, item):
    t1, t2 = self.get_corpus_line(item)
    # t1, t2, is_next_label = self.random_sent(item)
    t1_random = self.random_word(t1)

    # [CLS] tag = SOS tag, [SEP] tag = EOS tag
    t1 = [self.vocab.SOS] + t1_random + [self.vocab.EOS]
    t2 = t2 + [self.vocab.EOS]

    # t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
    # t2_label = t2_label + [self.vocab.pad_index]

    # segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
    # bert_input = (t1 + t2)[:self.seq_len]
    # bert_label = (t1_label + t2_label)[:self.seq_len]

    # for t1 (english)
    padding = [self.vocab.PAD for _ in range(self.seq_len - len(t1))]
    t1.extend(padding)
    
    # for t2 (japanese)
    padding = [self.vocab.PAD for _ in range(self.seq_len - len(t2))]
    t2.extend(padding)


    output = {"bert_input": bert_input,
              "bert_label": bert_label,
              "segment_label": segment_label,
              "is_next": is_next_label}

    return {key: torch.tensor(value) for key, value in output.items()}

  def random_word(self, sentence):
    tokens = sentence.split()
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = self.vocab.mask_index

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.randrange(len(self.vocab))

            # 10% randomly change token to current token
            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

            output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

        else:
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            # output_label.append(0)

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
