import math
import datetime
import argparse
import torch
from transformers import AlbertModel, AlbertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from trainer import albertTrainer
from dataloader import JESCDataloaders
from tokenizer import SentencePieceTokenizer
from model import myAlbertModel


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-epoch', type=int, default=30)
  parser.add_argument('-batch_size', type=int, default=36)

  parser.add_argument('-vocab_size', type=int, default=16000)
  parser.add_argument('-hidden_size', type=int, default=1024)  # official: 4096
  parser.add_argument('-num_hidden_layers', type=int,
                      default=6)  # official: 12
  parser.add_argument('-seq_len', type=int, default=128)
  parser.add_argument('-learning_rate', type=float, default=0.0001)
  parser.add_argument('-warmup_proportion', type=float, default=0.1)
  parser.add_argument('-max_grad_norm', type=float, default=1.0)

  parser.add_argument('-save_model', type=str, default='model/trained')
  parser.add_argument('-save_mode', type=str,
                      choices=['all', 'best'], default='all')
  parser.add_argument('-log', type=str, default='log_files/log')

  parser.add_argument('-no_cuda', action='store_true')
  # parser.add_argument('-user_apex', action='store_true')

  option = parser.parse_args()
  now = '{0:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
  option.log = option.log + '_' + now
  print('[Info] log files: ', option.log)

  print('[Info] building tokenizer')
  en_tokenizer = SentencePieceTokenizer("my-spm/enwiki.model")
  ja_tokenizer = SentencePieceTokenizer("my-spm/jawiki.model")

  print('[Info] building dataloader')
  data_paths = {
      "train": {"english": "data/en_train", "japanese": "data/ja_train"},
      "valid": {"english": "data/en_valid", "japanese": "data/ja_valid"},
      "test": {"english": "data/en_test", "japanese": "data/ja_test"},
  }
  dataloaders = JESCDataloaders(data_paths, en_tokenizer, ja_tokenizer, option)

  print('[Info] building albert')
  config = AlbertConfig(
      vocab_size_or_config_json_file=option.vocab_size,
      hidden_size=option.hidden_size,
      num_hidden_layers=option.num_hidden_layers,
  )
  albert = AlbertModel(config=config)
  model = myAlbertModel(albert, d_model=option.hidden_size)

  device = torch.device('cuda:0' if not option.no_cuda else 'cpu')

  print('[Info] preparing optimizer')
  # Parameters:
  lr = option.learning_rate
  data_size = dataloaders.train.__len__()
  num_training_steps = option.epoch * math.ceil(data_size / option.batch_size)
  num_warmup_steps = int(num_training_steps * option.warmup_proportion)

  # In Transformers, optimizer and schedules are splitted and instantiated like this:
  # To reproduce BertAdam specific behavior set correct_bias=False
  optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
  schedular = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

  print('[Info] building trainer')
  trainer = albertTrainer(
      model, dataloaders, optimizer, schedular, option, device
  )

  print('[Info] training start!')
  trainer.run()


if __name__ == '__main__':
  main()
