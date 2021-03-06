import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import vocab
from fastprogress import progress_bar


class albertTrainer:
  def __init__(self, model, dataloaders, optimizer, schedular, option, device):
    self.model = model

    if torch.cuda.device_count() > 1 and device != torch.device("cpu"):
      print("[Info] Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)

    self.model = self.model.to(device)

    self.dataloaders = dataloaders
    self.optimizer = optimizer
    self.schedular = schedular
    self.option = option
    self.device = device

    self.criterion = nn.NLLLoss(ignore_index=vocab.PAD)

    # self.use_apex = use_apex

    self.log = {
        "train": {"loss": [], "acc": []},
        "valid": {"loss": [], "acc": []},
    }

    # ログ記述用ファイル
    self.log_train_file = None
    self.log_valid_file = None
    if option.log:
      self.log_train_file = option.log + "_train.log"
      self.log_valid_file = option.log + "_valid.log"

      print('[Info] Training performance will be written to file: {} and {}'.format(
          self.log_train_file, self.log_valid_file
      ))

      with open(self.log_train_file, 'w') as log_tf, open(self.log_valid_file, 'w') as log_vf:
        log_tf.write('epoch, loss, accuracy\n')
        log_vf.write('epoch, loss, accuracy\n')

  def run(self):
    start_time = time.time()
    epoch_num = self.option.epoch
    # step_num = self.option.step
    self.step = 0
    self.epoch = 0
    for epoch in range(epoch_num):
      # while self.step < step_num:
      self.epoch += 1
      self.train()
      self.valid()
      self.schedular.step()
      self.save()
      self.draw_graph()
      # self.output_examples()
      print("[Info] successfully finished!")
      end_time = time.time()
      processing_time = end_time - start_time
      print("[Info] processing time {}".format(processing_time))

  def train(self):
    self.iter(train=True)

  def valid(self):
    self.iter(train=False)

  def iter(self, train):
    if train:
      self.model.train()
      dataloader = self.dataloaders.train
    else:
      self.model.eval()
      dataloader = self.dataloaders.valid

    total_loss = 0.  # float
    total_correct = 0
    total_element = 0

    data_iter = progress_bar(dataloader)
    for i, batch in enumerate(data_iter):
      src = batch["input_english"].to(self.device)
      tgt = batch["target_japanese"].to(self.device)

      # forward
      output = self.model(src)

      # calc loss
      loss = self.criterion(
          output.view(-1, self.option.vocab_size),
          tgt.view(-1),
      )

      if torch.cuda.device_count() > 1:
        loss.mean()

      if train:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.option.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1

      # keep notes
      total_loss += loss.item()
      # total_correct
      # total_element

    train = "train" if train else "valid"
    print("[Info] epoch {}, step {}@{}: loss = {}".format(
        self.epoch, self.step, train, total_loss / (i+1)
    ))
    self.log[train]["loss"].append(total_loss / (i+1))

  def save(self, out_dir="output"):
    model_state_dict = self.model.state_dict()
    optimizer_state_dict = self.optimizer.state_dict()

    checkpoint = {
        'model': model_state_dict,
        'optimizer': optimizer_state_dict,
        'settings': self.option,
        'epoch': self.epoch
    }

    if self.option.save_model:
      if self.option.save_mode == 'all':
        model_name = self.option.save_model + \
            '_loss_{loss:3.3f}.chkpt'.format(
                loss=self.log["valid"]["loss"][-1]
            )
        torch.save(checkpoint, model_name)
      elif self.save_mode == 'best':
        model_name = self.option.save_model + '.chkpt'
        if len(self.log["valid"]["loss"]) == 1 or self.log["valid"]["loss"][-1] <= min(self.log["valid"]["loss"][:-1]):
          torch.save(checkpoint, model_name)
          print('  - [Info] The checkpoint file has been updated!')

    if self.log_train_file and self.log_valid_file:
      with open(self.log_train_file, 'a') as log_tf, open(self.log_valid_file, 'a') as log_vf:
        log_tf.write('{epoch}, {loss: 8.5f}\n'.format(
            epoch=self.epoch, loss=self.log["train"]["loss"][-1]
        ))
        log_vf.write('{epoch}, {loss: 8.5f}\n'.format(
            epoch=self.epoch, loss=self.log["valid"]["loss"][-1]
        ))

  def draw_graph(self):
    x = np.arange(self.epoch)
    y = np.array([self.log["train"]["loss"], self.log["valid"]["loss"]]).T
    plots = plt.plot(x, y)
    plt.legend(plots, ("train", "valid"), loc="best", framealpha=0.25, prop={
        "size": "small", "family": "monospace"
    })
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(self.option.log + "_graph.png")
