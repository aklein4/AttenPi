
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from path_finder import PathFinder
from train_utils import DEVICE, train, Logger

import random
import csv
import matplotlib.pyplot as plt
import numpy as np

TRAIN_TAUS_FILE = "data/train_taus.pt"
VAL_TAUS_FILE = "data/val_taus.pt"

MODEL_FILE = "data/PathFinder.pt"
LOG_FILE = "logs/PathFinder.csv"
GRAFF_FILE = "logs/PathFinder.png"

N_EPOCHS = 100
LR = 1e-5
BATCH_SIZE = 64

MIN_SEQ_LEN = 10

NOISE = 0.003

LOSS_LAMBDA = 0.1
LOSS_DELTA = 0.1


class TrajectoryDataset:
    
    def __init__(self, file, noise=0):
        self.states, _, _ = torch.load(file)
        self.states = self.states.to(DEVICE)
        
        self.size = self.states.shape[0]
        
        self.x = None
        self.shuffler = []
        
        self.noise = noise
        
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
        
        noised_seqs = seqs.clone()
        noised_seqs[:,1:] += torch.randn_like(seqs[:,1:]) * self.noise
        
        return noised_seqs, seqs


class TrajectoryLogger(Logger):
    
    def __init__(self):
        self.epoch = 0
        
        self.train_losses = []
        self.val_losses = []
        
        self.train_lambdas = []
        self.val_lambdas = []
    
        with open(LOG_FILE, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow(["epoch", "train_loss", "val_loss", "train_lambda", "val_lambda"])
    
    
    def initialize(self, model):
        self.model = model
        
        
    def log(self, train_log, val_log):
        
        train_seq = torch.cat([p[0] for p in train_log[0]])
        train_end = torch.cat([p[1] for p in train_log[0]])
        train_loss, train_lamda = TrajectoryLoss((train_seq, train_end), torch.cat(train_log[1]), split=True)
        train_loss, train_lamda = train_loss.item(), train_lamda.item()

        val_seq = torch.cat([p[0] for p in val_log[0]])
        val_end = torch.cat([p[1] for p in val_log[0]])
        val_loss, val_lamda = TrajectoryLoss((val_seq, val_end), torch.cat(val_log[1]), split=True)
        val_loss, val_lamda = val_loss.item(), val_lamda.item()
        
        self.train_losses.append(np.log(train_loss))
        self.val_losses.append(np.log(val_loss))
        
        self.train_lambdas.append(train_lamda)
        self.val_lambdas.append(val_lamda)
        
        with open(LOG_FILE, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([self.epoch, train_loss, val_loss, train_lamda, val_lamda])
            
        fig, ax = plt.subplots(1)
        
        ax.plot(self.val_losses)
        ax.plot(self.train_losses)
        ax.legend(["val", "train"])
        ax.set_ylabel("Log of Mean L1 Loss")
        ax.set_xlabel("Epoch")
        ax.set_title("PathFinder Training Progress")
        
        # ax[1].plot(self.val_lambdas)
        # ax[1].plot(self.train_lambdas)
        # ax[1].legend(["val", "train"])
        # ax[1].set_ylabel("Lambda")
        # ax[1].set_xlabel("Epoch")
        # ax[1].set_title("PathFinder Lambda Progress")
        
        fig.set_figwidth(6)
        fig.set_figheight(4)
        plt.tight_layout()
        plt.savefig(GRAFF_FILE)
        plt.close(fig)
        
        torch.save(self.model.state_dict(), MODEL_FILE)
        
        self.epoch += 1
        
        
def TrajectoryLoss(pred, target, split=False):
    x_seq, x_ends = pred
    
    loss = F.l1_loss(x_seq, target)
    # loss -= LOSS_DELTA * F.l1_loss(x_seq[:,1:], target[:,:-1])
    
    # end_target = torch.full_like(x_ends[:,0], x_ends.shape[1]-1).long()
    loss_end = torch.tensor([0]).to(loss.device) # LOSS_LAMBDA * F.cross_entropy(x_ends, end_target)
    
    if split:
        return loss, loss_end
    return loss + loss_end


def main():

    model = PathFinder()
    # model.load_state_dict(torch.load(MODEL_FILE))
    model = model.to(DEVICE)
    
    train_data = TrajectoryDataset(TRAIN_TAUS_FILE, noise=NOISE)
    val_data = TrajectoryDataset(VAL_TAUS_FILE)
    
    logger = TrajectoryLogger()
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    schedule = get_linear_schedule_with_warmup(optimizer, N_EPOCHS//8 * (1 + len(train_data)//BATCH_SIZE), N_EPOCHS * (1 + len(train_data)//BATCH_SIZE))
    
    train(
        model,
        optimizer,
        train_data,
        TrajectoryLoss,
        val_data,
        num_epochs = N_EPOCHS,
        batch_size = BATCH_SIZE,
        logger = logger,
        lr_scheduler = schedule
    )
    
    torch.no_grad()
    model.eval()
    
    test_x = val_data[0][0]
    test_seq, test_end = model(test_x, test_x)
    test_seq = test_seq.squeeze(0)
    test_end = F.softmax(test_end, dim=-1).squeeze(0)
    
    print("\n --- TEST --- \n")
    for i in range(test_seq.shape[0]):
        print(test_x[0,i,0].item(), '->', test_seq[i,0].item(), '|', test_seq[i,0].item() - test_x[0,i,0].item())
    print('\n')


if __name__ == '__main__':
    main()