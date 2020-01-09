import datetime
import numpy as np
import pandas as pd
import nltk

from tokenizer import SentencePieceTokenizer


def calc_bleu():
  df = pd.read_csv("result/result_2020_01_09_14_47.csv")
  en_tokenizer = SentencePieceTokenizer("my-spm/enwiki.model")
  ja_tokenizer = SentencePieceTokenizer("my-spm/jawiki.model")

  corpus_bleu = 0.
  for index, row in df.iterrows():
    if index % 500 == 0:
      print("[info] {} / {} finished".format(index, len(df)))
      source = row[1]
      prediction = row[2]
      target = row[3]

      try:
        source_tokens = en_tokenizer.text_to_tokens(source)
        prediction_tokens = ja_tokenizer.text_to_tokens(prediction)
        target_tokens = ja_tokenizer.text_to_tokens(target)

        targets = [target_tokens]
        sentence_bleu = nltk.translate.bleu_score.sentence_blue(
            targets, prediction_tokens)
        corpus_bleu += sentence_bleu
      except:
        print(source_tokens)
        print(prediction_tokens)
        print(target_tokens)

  corpus_bleu /= len(df.index)

  now = '{0:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
  FILE_NAME = 'bleu_' + now
  FILE_PATH = 'result/' + FILE_NAME

  with open(FILE_PATH, "w") as f:
    f.writelines(str(corpus_bleu))
    print("data_size:", len(df))
    print("corpus_bleu:", corpus_bleu)


if __name__ == '__main__':
  calc_bleu()
