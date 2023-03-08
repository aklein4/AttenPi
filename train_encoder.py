
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

DEAD_RECKON = False

LOAD_FILE = "data/encoder.pt"
SAVE_FILE = "data/encoder.pt" if not DEAD_RECKON else "data/reckoner.pt"

LOG_FILE = "logs/encoder.csv" if not DEAD_RECKON else "logs/reckoner.csv"
GRAFF_FILE = "logs/encoder.png" if not DEAD_RECKON else "logs/reckoner.png"

N_EPOCHS = 50
LR = 1e-7 if DEAD_RECKON else 1e-5
BATCH_SIZE = 256 if DEAD_RECKON else 64

MIN_SEQ_LEN = 10

KL_COEF = 1


class TrajectoryDataset:
    
    def __init__(self, file):
        self.states, _, _ = torch.load(file)
        self.states = self.states.to(DEVICE)
        
        self.size = self.states.shape[0]
        
        self.x = None
        self.shuffler = []
        
        self.reset()
    
    
    def reset(self):
        self.x = self.states
        self.shuffler = list(range(len(self)))
        
    def shuffle(self):
        random.shuffle(self.shuffler)
        
        seq_len = random.randint(MIN_SEQ_LEN, self.x.shape[1])
        seq_start = random.randint(0, self.x.shape[1] - seq_len)
        self.x = self.x[:, seq_start:seq_start+seq_len, :]
        
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, getter):

        # handle input
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # use shuffler as indexes, if batchsize overhangs then batch is truncated
        seqs = self.x[self.shuffler[index : index+batchsize]]
        
        if seqs.numel() == 0:
            return None
        
        return seqs, seqs


class VariationalLogger(Logger):
    
    def __init__(self):
        self.epoch = 0
        
        self.train_losses = []
        self.val_losses = []
        
        self.train_errors = []
        self.val_errors = []
        
        self.train_kls = []
        self.val_kls = []
    
        with open(LOG_FILE, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch", "train_loss", "val_loss", "train_error", "val_error", "train_kl", "val_kl"])
    
    
    def initialize(self, model):
        self.model = model
        
        
    def log(self, train_log, val_log):
        
        # get training metrics
        train_seq = torch.cat([p[0] for p in train_log[0]])
        train_mu = torch.cat([p[1] for p in train_log[0]])
        train_sig = torch.cat([p[2] for p in train_log[0]])
        train_loss, train_error, train_kl = ELBOLoss((train_seq, train_mu, train_sig), torch.cat(train_log[1]), split=True)
        
        train_loss, train_error, train_kl = train_loss.item(), train_error.item(), train_kl.item()

        # get validation metrics
        val_seq = torch.cat([p[0] for p in val_log[0]])
        val_mu = torch.cat([p[1] for p in val_log[0]])
        val_sig = torch.cat([p[2] for p in val_log[0]])
        val_loss, val_error, val_kl = ELBOLoss((val_seq, val_mu, val_sig), torch.cat(val_log[1]), split=True)
        
        val_loss, val_error, val_kl = val_loss.item(), val_error.item(), val_kl.item()
        
        # save metrics
        self.train_losses.append(np.log(train_loss))
        self.val_losses.append(np.log(val_loss))
        
        self.train_errors.append(np.log(train_error))
        self.val_errors.append(np.log(val_error))
        
        self.train_kls.append(np.log(train_kl))
        self.val_kls.append(np.log(val_kl))
        
        # write to csv
        with open(LOG_FILE, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([self.epoch, train_loss, val_loss, train_error, val_error, train_kl, val_kl])
        
        fig, ax = plt.subplots(3)
        
        ax[0].plot(self.val_losses)
        ax[0].plot(self.train_losses)
        ax[0].legend(["val", "train"])
        ax[0].set_ylabel("Log ELBO Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_title("ELBO Loss Progress")
        
        ax[1].plot(self.val_errors)
        ax[1].plot(self.train_errors)
        ax[1].legend(["val", "train"])
        ax[1].set_xlabel("Epoch")
        if DEAD_RECKON:
            ax[1].set_ylabel("Log Mean L2 Error")
            ax[1].set_title("L2 Error Progress")
        else:
            ax[1].set_ylabel("Log Mean L1 Error")
            ax[1].set_title("L1 Error Progress")
        
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
        
        torch.save(self.model.state_dict(), SAVE_FILE)
        
        self.epoch += 1
        
        
def ELBOLoss(pred, target, split=False):
    seq, mus, sigmas = pred
    
    error = F.l1_loss(seq, target)
    # error -= LOSS_DELTA * F.l1_loss(x_seq[:,1:], target[:,:-1])
    
    if DEAD_RECKON:
        error = F.mse_loss(seq, target)
    
    kl = torch.sum(sigmas**2 + mus**2 - torch.log(sigmas) - 1/2) / sigmas.numel()
    
    loss = error + KL_COEF*kl
    
    if split:
        return loss, error, kl
    return loss


class ELBOMetric:
    def __init__(self):
        self.title = "[loss, error, kl]"
    
    def __call__(self, pred, target):
        loss, error, kl = ELBOLoss(pred, target, split=True)
        return np.array([np.log(loss.item()), error.item(), np.log(kl.item())], dtype=np.float16)


def main():

    model = VariationalTrajectory(dead_reckon=DEAD_RECKON)
    if DEAD_RECKON:
        model.load_state_dict(torch.load(LOAD_FILE))
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