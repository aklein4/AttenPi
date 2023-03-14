
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

N_ENVS = 32*16
SHUFFLE_RUNS = 1
MAX_BUF_SIZE = 1024

N_SKILLS = DefaultLatentPolicy.num_skills
SKILL_LEN = 32

EVAL_ITERS = 1

LOG_LOC = "logs/log.csv"
GRAFF = "logs/graff.png"

CHECKPOINT = "local_data/checkpoint.pt"

LEARNING_RATE = 1e-4
BATCH_SIZE = 64


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
                    
                    rewards.append(torch.tensor(r))
                    prev_rewards.append(rewards[-1])
                    for tau in range(1, 1+len(prev_rewards)):
                        prev_rewards[-tau][np.logical_not(curr_dones)] += (self.discount ** tau) * rewards[-1][np.logical_not(curr_dones)]

                    curr_dones = np.logical_or(curr_dones, this_done)

                actions = self.model.action_history.to(self.device)
                states = self.model.state_history.to(self.device)
                rewards = torch.stack(rewards).to(self.device)
                dones = torch.stack(dones).to(self.device)

                for i in range(self.num_envs):
                    s, a, r, d = states[i], actions[i], rewards[:,i], dones[:,i]

                    if torch.any(torch.logical_not(d)):
                        self.data.append((s, a, r, skill[i], d))

                if np.all(curr_dones):
                    break

    
    def evaluate(self, iterations=1):

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

                        a = self.model.policy(torch.tensor(obs).to(self.device))

                        obs, r, this_done, info = self.env.step(a.squeeze().detach().cpu().numpy())
                        
                        rewards[np.logical_not(curr_dones)] += r[np.logical_not(curr_dones)]

                        curr_dones = np.logical_or(curr_dones, this_done)

                    if np.all(curr_dones):
                        break
                
                tot_rewards += np.sum(rewards)
                tot += 1
            
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
    

def DualLoss(pred, y):
    pi_logits, mon = pred
    s, a, r, k, d = y

    batch_size = pi_logits.shape[0]

    correct_mon = (torch.argmax(mon, dim=-1) == k).float()

    r = (r + correct_mon.unsqueeze(1)).unsqueeze(-1)

    log_probs = torch.log_softmax(pi_logits, dim=-1)
    multed = log_probs * r
    masked = multed.view(-1, multed.shape[-1])[a.view(-1)]
    reinforce_loss = -torch.sum(masked)

    monitor_loss = F.cross_entropy(mon, k)

    return (reinforce_loss + monitor_loss)/batch_size


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
        self.model.save_state_dict(CHECKPOINT)


def main():

    model = LatentPolicy()
    model.to(DEVICE)

    env = TrainingEnv(
        env_name = "LunarLander-v2",
        num_envs = N_ENVS,
        model = model,
        skill_len = SKILL_LEN,
        shuffle_runs = SHUFFLE_RUNS,
        discount = 0.95,
        max_buf_size = MAX_BUF_SIZE,
        device = DEVICE
    )

    logger = EnvLogger(
        env = env,
        eval_iters = EVAL_ITERS,
        log_loc = LOG_LOC,
        graff = GRAFF
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
