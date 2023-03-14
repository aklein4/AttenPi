
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
# from stable_baselines3 import PPO

from latent_policy import LatentPolicy
from train_utils import train, Logger
from configs import DefaultLatentPolicy

from tqdm import tqdm
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ENVS = 4
SHUFFLE_RUNS = 4
MAX_BUF_SIZE = 16

N_SKILLS = DefaultLatentPolicy.num_skills
SKILL_LEN = 4

EVAL_ITERS = 2

LOG_LOC = "logs/log.csv"
GRAFF = "logs/graff.png"

CHECKPOINT = "local_data/checkpoint.pt"

LEARNING_RATE = 1e-4
BATCH_SIZE = 1

R_NORM = 8
ACC_LAMBDA = 0.0

BASE_DIM = 32
BASE_LAYERS = 4
BASE_LR = 1e-4
BASE_BATCH = 4


class TrainingEnv:
    def __init__(
            self,
            env_name,
            num_envs,
            model,
            skill_len,
            shuffle_runs=1,
            discount=1,
            max_buf_size=100000,
            device=DEVICE
        ):
        self.env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=False)
        self.num_envs = num_envs

        self.model = model
        self.discount = discount
        self.skill_len = skill_len
        self.max_buf_size = max_buf_size
        self.device = device
        self.shuffle_runs = shuffle_runs

        # should hold (s, a, r, k, d) tuples
        # temporal length should be skill_len
        self.data = []

        self.action_size = model.config.action_size
        self.state_size = model.config.state_size
        self.num_skills = model.config.num_skills

        self.shuffler = []


    def __len__(self):
        return len(self.data)


    def shuffle(self):
        for i in tqdm(range(self.shuffle_runs), desc="Exploring", leave=False):
            self.sample()
        self.data = self.data[-self.max_buf_size:]

        self.shuffler = list(range(len(self.data)))
        random.shuffle(self.shuffler)


    def sample(self):

        obs = self.env.reset()
        curr_dones = np.zeros((self.num_envs,), dtype=bool)
        prev_rewards = []

        with torch.no_grad():
            while True:
                # TODO: skill choosing model
                skill = torch.randint(0, self.num_skills-1, (self.num_envs,)).to(self.device)
                self.model.setSkill(skill)

                rewards = []
                dones = []

                for t in range(self.skill_len):

                    a = self.model.policy(torch.tensor(obs).to(self.device))

                    dones.append(torch.tensor(curr_dones))

                    obs, r, this_done, info = self.env.step(a.squeeze().detach().cpu().numpy())
                    r /= R_NORM

                    rewards.append(torch.tensor(r).to(self.device))
                    rewards[-1][curr_dones] = 0
                    curr_dones = np.logical_or(curr_dones, this_done)

                actions = self.model.action_history.to(self.device)
                states = self.model.state_history.to(self.device)
                rewards = torch.stack(rewards)
                for i in range(rewards.shape[0]):
                    prev_rewards.append(rewards[i])
                dones = torch.stack(dones).to(self.device)

                for i in range(self.num_envs):
                    s, a, r, d = states[i], actions[i], rewards[:,i], dones[:,i]

                    if torch.any(torch.logical_not(d)):
                        self.data.append((s, a, r, skill[i], d))

                if np.all(curr_dones):
                    break

            for tau in range(2, 1+len(prev_rewards)):
                prev_rewards[-tau] += self.discount * prev_rewards[-tau+1]

    
    def evaluate(self, iterations=1):

        seedo = random.randrange(0xFFFF)
        self.env.seed(0)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        obs = self.env.reset()
        curr_dones = np.zeros((self.num_envs,), dtype=bool)
        rewards = np.zeros((self.num_envs,), dtype=float)

        tot_rewards = 0
        tot = 0

        with torch.no_grad():
            for it in tqdm(range(iterations), desc="Evaluating", leave=False):
                while True:
                    # TODO: skill choosing model
                    skill = torch.randint(0, self.num_skills-1, (self.num_envs,)).to(self.device)
                    self.model.setSkill(skill)

                    for t in range(self.skill_len):

                        a = self.model.policy(torch.tensor(obs).to(self.device), stochastic=True)

                        obs, r, this_done, info = self.env.step(a.squeeze().detach().cpu().numpy())
                        r /= R_NORM

                        rewards[np.logical_not(curr_dones)] += r[np.logical_not(curr_dones)]

                        curr_dones = np.logical_or(curr_dones, this_done)

                    if np.all(curr_dones):
                        break
                
                tot_rewards += np.sum(rewards)
                tot += self.num_envs
            
            self.env.seed(seedo)
            torch.manual_seed(seedo)
            np.random.seed(seedo)
            random.seed(seedo)
            return tot_rewards / tot


    def __getitem__(self, getter):
        # unpack index and batchsize
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices we are going to use
        indices = self.shuffler[index:index+batchsize]

        x = ([], [], [], [])
        y = []
        
        # unpack data tuples onto batch tuples
        for i in indices:
            s, a, r, k, d = self.data[i]

            x[0].append(s)
            x[1].append(a)
            x[2].append(k)
            x[3].append(d)
            
            y.append(r)

        x = [torch.stack(x[i]) for i in range(len(x))]

        return tuple(x), (x[0], x[1], torch.stack(y), x[2], x[3])


class BaseREINFORCE(nn.Module):
    def __init__(self, state_size, h_dim, n_layers, batch_size, lr, device):
        super().__init__()
        
        in_layer = [
            nn.Linear(state_size, h_dim),
            nn.Dropout(0.1),
            nn.ELU(),
        ]
        mid_layers = [nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.Dropout(0.1),
            nn.ELU(),
        ) for _ in range(n_layers)]
        out_layer = [
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, 1)
        ]
        self.baseline = nn.Sequential(
            *(in_layer + mid_layers + out_layer)
        )
        self.baseline = self.baseline.to(device)
        self.optimizer = torch.optim.Adam(self.baseline.parameters(), lr=lr)

        self.batch_size = batch_size


    def train_baseline(self, x, y):
        self.train()
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)
        for b in range(0, x.shape[0], self.batch_size):
            x_b, y_b = x[b:b+self.batch_size], y[b:b+self.batch_size]

            y_hat = self.baseline(x_b).squeeze(-1)
            loss = F.mse_loss(y_hat.float(), y_b.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

    def predict_baseline(self, x):
        self.eval()
        with torch.no_grad():
            V = self.baseline(x).squeeze(-1)
        return V


    def loss(self, pred, y):
        pi_logits, mon = pred
        s, a, r, k, d = y

        batch_size = pi_logits.shape[0]

        correct_mon = ACC_LAMBDA*(torch.argmax(mon, dim=-1) == k).float()

        base = self.predict_baseline(s)
        self.train_baseline(s, r)
        assert base.shape == r.shape
        r = (r-base + correct_mon.unsqueeze(1)).unsqueeze(-1)

        log_probs = torch.log_softmax(pi_logits, dim=-1)
        multed = log_probs * r
        masked = multed.view(-1, multed.shape[-1])[a.view(-1)]
        reinforce_loss = -torch.mean(masked)

        monitor_loss = F.cross_entropy(mon, k)

        return reinforce_loss + monitor_loss


def DualLoss(pred, y):
    pi_logits, mon = pred
    s, a, r, k, d = y

    batch_size = pi_logits.shape[0]

    correct_mon = ACC_LAMBDA*(torch.argmax(mon, dim=-1) == k).float()

    r = (r + correct_mon.unsqueeze(1)).unsqueeze(-1)

    log_probs = torch.log_softmax(pi_logits, dim=-1)
    multed = log_probs * r
    masked = multed.view(-1, multed.shape[-1])[a.view(-1)][torch.logical_not(d).view(-1)]
    reinforce_loss = -torch.mean(masked)

    monitor_loss = F.cross_entropy(mon, k)

    return reinforce_loss + monitor_loss


class EnvLogger(Logger):
    def __init__(self, env, eval_iters, log_loc, graff):

        self.env = env
        self.eval_iters = eval_iters

        # accuracies
        self.avg_rewards = []

        # save locations
        self.log_loc = log_loc
        self.graff = graff

        # create metric file and write header
        with open(self.log_loc, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "avg_reward"])


    def initialize(self, model):
        # get reference to the model so we can save it
        self.model = model
    

    def log(self, train_log, val_log):

        curr_r = self.env.evaluate(self.eval_iters)
        self.avg_rewards.append(curr_r)
        
        # append metrics to csv file
        with open(self.log_loc, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.avg_rewards)-2, curr_r])

        plt.plot(self.avg_rewards)
        plt.title(r"% Average Reward")

        plt.savefig(self.graff)
        plt.clf()

        self.save_checkpoint()
    

    def save_checkpoint(self):
        # save the model to a new folder
        torch.save(self.model.state_dict(), CHECKPOINT)


def main():

    model = LatentPolicy()
    model.to(DEVICE)

    env = TrainingEnv(
        env_name = "CartPole-v1",
        num_envs = N_ENVS,
        model = model,
        skill_len = SKILL_LEN,
        shuffle_runs = SHUFFLE_RUNS,
        discount = 1,
        max_buf_size = MAX_BUF_SIZE,
        device = DEVICE
    )

    logger = EnvLogger(
        env = env,
        eval_iters = EVAL_ITERS,
        log_loc = LOG_LOC,
        graff = GRAFF
    )
    logger.initialize(model)
    logger.log(None, None)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loser = BaseREINFORCE(model.config.state_size, BASE_DIM, BASE_LAYERS, BATCH_SIZE, BASE_LR, DEVICE)

    train(
        model = model,
        optimizer = optimizer,
        train_data = env,
        loss_fn = DualLoss,
        logger = logger,
        batch_size = BATCH_SIZE
    )


if __name__ == '__main__':
    main()
