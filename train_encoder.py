
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from variational_trajectory import VariationalTrajectory
from train_utils import DEVICE, train, Logger

import random
import csv
import matplotlib.pyplot as plt
import numpy as np


TRAIN_TAUS_FILE = "data/train_taus.pt"
VAL_TAUS_FILE = "data/val_taus.pt"

SAVE_FILE = "data/latent_policy.pt"

LOG_FILE = "logs/latent_policy.csv"
GRAFF_FILE = "logs/latent_policy.png"

N_EPOCHS = 512
LR = 5e-7
BATCH_SIZE = 192

MIN_SEQ_LEN = 10

KL_COEF = 1


class TrajectoryDataset:
    
    def __init__(self, file):
        self.states, self.actions, r = torch.load(file)
        
        self.states = self.states.to(DEVICE)
        self.actions = self.actions.to(DEVICE)[:,:,:4] # messed up shape
        
        assert self.states.shape[0] == self.actions.shape[0]
        assert self.states.shape[1] == self.actions.shape[1]
        
        self.size = self.states.shape[0]
        
        self.x_states = None
        self.x_actions = None
        self.shuffler = []
        
        self.reset()
    
    
    def reset(self):
        self.x_states = self.states
        self.x_actions = self.actions
        self.shuffler = list(range(len(self)))
        
    def shuffle(self):
        random.shuffle(self.shuffler)
        
        seq_len = random.randint(MIN_SEQ_LEN, self.actions.shape[1])
        seq_start = random.randint(0, self.actions.shape[1] - seq_len)
        
        self.x_states = self.states[:, seq_start:seq_start+seq_len, :]
        self.x_actions = self.actions[:, seq_start:seq_start+seq_len]
        
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, getter):

        # handle input
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # use shuffler as indexes, if batchsize overhangs then batch is truncated
        s = self.x_states[self.shuffler[index : index+batchsize]]
        a = self.x_actions[self.shuffler[index : index+batchsize]]
        
        if s.numel() == 0:
            return None
        
        return (s, a), a


class VariationalLogger(Logger):
    
    def __init__(self):
        self.epoch = 0
        
        self.train_ps = []
        self.val_ps = []
        
        self.train_log_ps = []
        self.val_log_ps = []
        
        self.train_kls = []
        self.val_kls = []
    
        with open(LOG_FILE, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch", "train_p", "val_p", "train_log_p", "val_log_p", "train_kl", "val_kl"])
    
    
    def initialize(self, model):
        self.model = model
        
        
    def log(self, train_log, val_log):
        
        # get training metrics
        train_seq = torch.cat([p[0] for p in train_log[0]])
        train_mu = torch.cat([p[1] for p in train_log[0]])
        train_sig = torch.cat([p[2] for p in train_log[0]])
        train_p, train_log_p, train_kl = ELBOLoss((train_seq, train_mu, train_sig), torch.cat(train_log[1]), split=True)

        # get validation metrics
        val_seq = torch.cat([p[0] for p in val_log[0]])
        val_mu = torch.cat([p[1] for p in val_log[0]])
        val_sig = torch.cat([p[2] for p in val_log[0]])
        val_p, val_log_p, val_kl = ELBOLoss((val_seq, val_mu, val_sig), torch.cat(val_log[1]), split=True)
        
        # save metrics
        self.train_ps.append(train_p)
        self.val_ps.append(val_p)
        
        self.train_log_ps.append(train_log_p)
        self.val_log_ps.append(val_log_p)
        
        self.train_kls.append(np.log(train_kl))
        self.val_kls.append(np.log(val_kl))
        
        # write to csv
        with open(LOG_FILE, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([self.epoch, train_p, val_p, train_log_p, val_log_p, train_kl, val_kl])
        
        fig, ax = plt.subplots(3)
        
        ax[0].plot(self.val_ps)
        ax[0].plot(self.train_ps)
        ax[0].legend(["val", "train"])
        ax[0].set_ylabel("Mean P(a|s))")
        ax[0].set_xlabel("Epoch")
        ax[0].set_title("Correct Action Probability Progress")
        
        ax[1].plot(self.val_log_ps)
        ax[1].plot(self.train_log_ps)
        ax[1].legend(["val", "train"])
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Mean Log P(a|s)")
        ax[1].set_title("Log Likelihood Progress")
        
        ax[2].plot(self.val_kls)
        ax[2].plot(self.train_kls)
        ax[2].legend(["val", "train"])
        ax[2].set_ylabel("Log KL Divergence")
        ax[2].set_xlabel("Epoch")
        ax[2].set_title("KL Divergence Progress")
        
        fig.set_figwidth(6)
        fig.set_figheight(12)
        plt.tight_layout()
        plt.savefig(GRAFF_FILE)
        plt.close(fig)
        
        if val_p >= max(self.val_ps):
            torch.save(self.model.state_dict(), SAVE_FILE)
        
        self.epoch += 1
        
        
def ELBOLoss(pred, target, split=False):
    act_probs, mus, sigmas = pred
    
    probs = torch.mean(act_probs[target.bool()])
    log_probs = torch.mean(torch.log(act_probs[target.bool()]))
    
    kl = torch.sum(sigmas**2 + mus**2 - torch.log(sigmas) - 1/2) / sigmas.numel()
    
    loss = -log_probs + KL_COEF*kl
    
    if split:
        return probs.item(), log_probs.item(), kl.item()
    
    return loss


class ELBOMetric:
    def __init__(self):
        self.title = "[p, log_p, kl]"
    
    def __call__(self, pred, target):
        p, log_p, kl = ELBOLoss(pred, target, split=True)
        return np.array([p, log_p, np.log(kl)], dtype=np.float16)


def main():

    model = VariationalTrajectory()
    model = model.to(DEVICE)
    
    train_data = TrajectoryDataset(TRAIN_TAUS_FILE)
    val_data = TrajectoryDataset(VAL_TAUS_FILE)
    
    logger = VariationalLogger()
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    schedule = get_linear_schedule_with_warmup(optimizer, N_EPOCHS//8 * (1 + len(train_data)//BATCH_SIZE), N_EPOCHS * (1 + len(train_data)//BATCH_SIZE))
    
    train(
        model,
        optimizer,
        train_data,
        ELBOLoss,
        val_data,
        num_epochs = N_EPOCHS,
        batch_size = BATCH_SIZE,
        logger = logger,
        lr_scheduler = schedule,
        metric=ELBOMetric(),
        rolling_avg=0.99,
    )
    
    print("\nDone!\n")


if __name__ == '__main__':
    main()