import math
import datetime
import argparse
import torch
from transformers import AlbertModel, AlbertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import collections

from trainer import albertTrainer
from dataloader import JESCDataloaders, JESCDataset
from tokenizer import SentencePieceTokenizer
from model import myAlbertModel
from mytranslator import myTranslator


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-epoch', type=int, default=30)
  parser.add_argument('-batch_size', type=int, default=18)

  parser.add_argument('-vocab_size', type=int, default=16000)
  parser.add_argument('-hidden_size', type=int, default=1024)  # official: 4096
  parser.add_argument('-num_hidden_layers', type=int,
                      default=6)  # official: 12
  parser.add_argument('-seq_len', type=int, default=128)
#   parser.add_argument('-learning_rate', type=float, default=0.001)
#   parser.add_argument('-warmup_proportion', type=float, default=0.1)
#   parser.add_argument('-max_grad_norm', type=float, default=1.0)

#   parser.add_argument('-save_model', type=str, default='model/trained')
#   parser.add_argument('-save_mode', type=str,
  #   choices=['all', 'best'], default='all')
#   parser.add_argument('-log', type=str, default='log_files/log')

  parser.add_argument('-no_cuda', action='store_true')
  # parser.add_argument('-user_apex', action='store_true')

  option = parser.parse_args()
  now = '{0:%Y_%m_%d_%H_%M}'.format(datetime.datetime.now())
#   option.log = option.log + '_' + now
#   print('[Info] log files: ', option.log)

  print('[Info] building tokenizer')
  en_tokenizer = SentencePieceTokenizer("my-spm/enwiki.model")
  ja_tokenizer = SentencePieceTokenizer("my-spm/jawiki.model")

  print('[Info] building dataloader')
  dataloader = torch.utils.data.DataLoader(
      JESCDataset(
          source_path='./data/en_test',
          target_path='./data/ja_test',
          seq_len=option.seq_len,
          en_tokenizer=en_tokenizer,
          ja_tokenizer=ja_tokenizer,
          vocab_size=option.vocab_size,
          train=False,
      ),
      batch_size=option.batch_size,
      num_workers=0,
  )

  print('[Info] building albert')
  config = AlbertConfig(
      vocab_size_or_config_json_file=option.vocab_size,
      hidden_size=option.hidden_size,
      num_hidden_layers=option.num_hidden_layers,
  )
  albert = AlbertModel(config=config)
  model = myAlbertModel(albert, d_model=option.hidden_size)

  print('[Info] loading pretrained model')
  PATH = './model/trained_loss_4.712.chkpt'
  checkpoint = torch.load(PATH)
  state_dict = fix_model_state_dict(checkpoint['model'])
  model.load_state_dict(state_dict)

  device = torch.device('cuda:0' if not option.no_cuda else 'cpu')

  result_file = './result/result_' + now + '.csv'

  print('[Info] building translator')
  translator = myTranslator(
      model, dataloader, device, en_tokenizer, ja_tokenizer, result_file
  )
  translator.translate()


#   print('[Info] preparing optimizer')
#   # Parameters:
#   lr = option.learning_rate
#   data_size = dataloaders.train.__len__()
#   num_training_steps = option.epoch * math.ceil(data_size / option.batch_size)
#   num_warmup_steps = int(num_training_steps * option.warmup_proportion)

  # In Transformers, optimizer and schedules are splitted and instantiated like this:
  # To reproduce BertAdam specific behavior set correct_bias=False
#   optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
#   schedular = get_linear_schedule_with_warmup(
#       optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

#   print('[Info] building trainer')
#   trainer = albertTrainer(
#       model, dataloaders, optimizer, schedular, option, device
#   )

#   print('[Info] training start!')
#   trainer.run()

def fix_model_state_dict(state_dict):
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict


if __name__ == '__main__':
  main()
