
import torch
import torch.nn as nn
import torch.nn.functional as F

from procgen import ProcgenGym3Env

from train_utils import train, Logger
from model_utils import MobileNet

from tqdm import tqdm
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


# device to use for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_VERSION = True

N_ENVS = 256
N_PER_ENV = 16

CHANGE_PROB = 0.25

# csv log output location
LOG_LOC = "logs/baseline.csv"
# graph output location
GRAFF = "logs/baseline.png"

LR = 1e-4
BATCH_SIZE = 8


class randomPolicy:
    def __init__(self, n_actions, n_envs):
        self.n_actions = n_actions
        self.n_envs = n_envs
        self.n_actions = n_actions

        self.prev = np.random.randint(0, self.n_actions-1, size=(n_envs,))

    def __call__(self, s):
        if random.random() < CHANGE_PROB:
            self.prev = np.random.randint(0, self.n_actions-1, size=(self.n_envs,))
        return self.prev


class Sampler:
    def __init__(
            self,
            num_envs,
            num_per_env
        ):
        self.env = env = ProcgenGym3Env(num=num_envs, env_name="coinrun", use_sequential_levels=True, use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True)
        self.num_envs = num_envs
        self.num_per_env = num_per_env

        self.policy = randomPolicy(15, self.num_envs)
        self.data = []


    def shuffle(self):
        """ Sample from the environment and shuffle the data buffer.
        """
        self.data = []

        prev_states = [[] for _ in range(self.num_envs)]
        
        for t in range(self.num_per_env):

            a = self.policy(None)

            self.env.act(a)
            obs = self.env.observe()[1]['rgb']
            obs = torch.tensor(obs).to(DEVICE).permute(0, 3, 1, 2).float()

            for i in range(self.num_envs):
                prev_states[i].append(obs[i])

                if len(prev_states[i]) == 6:
                    self.data.append((torch.cat(prev_states[i][:3], dim=0), torch.cat(prev_states[i][3:], dim=0)))
                    prev_states[i].pop(0)

        random.shuffle(self.data)
        return
    

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, getter):
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter
        
        return torch.stack([p[0] for p in self.data[index:index+batchsize]]), torch.stack([p[1] for p in self.data[index:index+batchsize]])


class BatchIdLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pred, target):
        target_outputs = self.model(target)

        predictions = 5 * F.normalize(pred, dim=-1) @ F.normalize(target_outputs, dim=-1).T

        return F.cross_entropy(predictions, torch.arange(pred.shape[0]).to(DEVICE).long())


def main():

    # initialize the model that we will train
    model = MobileNet(64)
    model = model.to(DEVICE)

    # initialize our training environment
    env = Sampler(
        num_envs = N_ENVS,
        num_per_env = N_PER_ENV
    )

    loser = BatchIdLoss(model)

    # initialize the model's optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # train ad infinitum
    train(
        model = model,
        optimizer = optimizer,
        train_data = env,
        loss_fn = loser,
        batch_size = BATCH_SIZE,
        num_epochs=16
    )

    torch.save(model.state_dict(), "local_data/pre-conv.pt")


if __name__ == '__main__':
    main()


