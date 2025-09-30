import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from data import GenerationDataset
from evaluation.base_evaluator import BaseEvaluator
import matplotlib.pyplot as plt

class Trainer:
    """
    Trainer for conditional generation model.
    """
    def __init__(self, configs, eval_configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_opt()
        self._init_data(dataset)
        self._init_eval(eval_configs)
        self._best_valid_loss = 1e10
        self.tf_writer = SummaryWriter(log_dir=self.output_folder)

    def _init_eval(self, eval_configs):
        dataset = GenerationDataset(eval_configs["data"])
        self.evaluator = BaseEvaluator(eval_configs["eval"], dataset, None)
        self.eval_configs = eval_configs

    def _init_cfgs(self, configs):
        self.configs = configs
        
        self.n_epochs = self.configs["epochs"]
        self.itr_per_epoch = self.configs["itr_per_epoch"]
        self.valid_epoch_interval = self.configs["val_epoch_interval"]
        self.display_epoch_interval = self.configs["display_interval"]

        self.lr = self.configs["lr"]
        self.batch_size = self.configs["batch_size"]

        self.model_path = self.configs["model_path"]
        self.output_folder = configs["output_folder"]
        os.makedirs(self.output_folder, exist_ok=True)

    def _init_model(self, model):
        self.model = model
        if self.model_path != "":
            print("Loading pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))

    def _init_opt(self):
        self.opt = Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        p1 = int(0.75 * self.n_epochs)
        p2 = int(0.9 * self.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=[p1, p2], gamma=0.1)

    def _init_data(self, dataset):
        self.dataset = dataset
        self.train_loader = dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=True, include_self=True)
        self.valid_loader = dataset.get_loader(split="valid", batch_size=self.batch_size, shuffle=False, include_self=True)

    def _reset_train(self):
        self._best_valid_loss = 1e10
        self._global_batch_no = 0

    """
    Train.
    """
    def train(self):
        self._reset_train()
        for epoch_no in range(self.n_epochs):
            self._train_epoch(epoch_no)
            if self.valid_loader is not None and (epoch_no + 1) % self.valid_epoch_interval == 0:
                self.valid(epoch_no)
                self.evaluate(epoch_no)
    
    def evaluate(self, epoch_no):
        self.model.eval()
        self.evaluator.model = self.model
        self.evaluator.n_samples = 10
        res_dict = self.evaluator.evaluate(mode="cond_gen", sampler="ddim", save_pred=False)
        for k in res_dict["tensorboard"].keys():
            self.tf_writer.add_scalar(fr"Cond_gen/{k}", res_dict["tensorboard"][k], epoch_no)
        
    def _train_epoch(self, epoch_no):
            start_time = time.time()
            avg_loss = 0
            self.model.train()

            for batch_no, train_batch in enumerate(self.train_loader):
                self._global_batch_no += 1
                self.opt.zero_grad()
                loss_dict = self.model(train_batch, is_train=True)

                loss_dict["all"].backward()
                self.opt.step()

                avg_loss += loss_dict["all"].item()
                for k in loss_dict.keys():
                    self.tf_writer.add_scalar(fr"Train/{k}", loss_dict[k].item(), self._global_batch_no)

                if batch_no >= self.itr_per_epoch:
                    break
            self.lr_scheduler.step()
            avg_loss /= len(self.train_loader)
            self.tf_writer.add_scalar("Train/epoch_loss", avg_loss, epoch_no)
            self.tf_writer.add_scalar("Train/lr", self.opt.param_groups[0]['lr'], epoch_no)
            end_time = time.time()
            
            if (epoch_no+1)%self.display_epoch_interval==0:
                print("Epoch:", epoch_no,
                      "Loss:", avg_loss,
                      "Time: {:.2f}".format(end_time-start_time))

    """
    Valid.
    """
    def valid(self, epoch_no=-1):
        self.model.eval()
        avg_loss_valid = 0
        with torch.no_grad():
            for batch_no, valid_batch in enumerate(self.valid_loader):
                loss_dict = self.model(valid_batch, is_train=False)
                avg_loss_valid += loss_dict["all"].item()

        avg_loss_valid = avg_loss_valid/len(self.valid_loader)
        self.tf_writer.add_scalar("Valid/epoch_loss", avg_loss_valid, epoch_no)

        if self._best_valid_loss > avg_loss_valid:
            self._best_valid_loss = avg_loss_valid
            print(f"\n*** Best loss is updated to {avg_loss_valid} at {epoch_no}.\n")
            self.save_model("model_best_loss")
        if (epoch_no+1) % 100 == 0:
            self.save_model(f"model_epoch_{epoch_no}")
    """
    Save.
    """
    def save_model(self, comment):
        os.makedirs(fr"{self.output_folder}/ckpts", exist_ok=True)
        path = os.path.join(fr"{self.output_folder}/ckpts", f"{comment}.pth")
        torch.save(self.model.state_dict(), path)
