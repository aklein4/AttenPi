
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
# from stable_baselines3 import PPO

from latent_policy import LatentPolicy

from tqdm import tqdm
import numpy as np
import random


N_SAMPLE_TAUS = 2


class TrainingEnv:
    def __init__(self, env_name, num_envs, model, discount, skill_len, max_buf_size):
        self.env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=False)
        self.num_envs = num_envs

        self.model = model
        self.discount = discount
        self.skill_len = skill_len
        self.max_buf_size = max_buf_size

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
        self.sample()
        self.data = self.data[-self.max_buf_size:]

        self.shuffler = list(range(len(self.data)))
        random.shuffle(self.shuffler)


    def sample(self):

        obs = self.env.reset()[0]
        curr_dones = np.zeros((self.num_envs,), dtype=bool)
        prev_rewards = []

        while True:
            # TODO: skill choosing model
            skill = torch.randint(0, self.num_skills-1, (self.num_envs,))
            self.model.setSkill(skill)

            rewards = []
            dones = []

            for t in range(self.skill_len):

                a = self.model.policy(torch.tensor(obs))

                dones.append(torch.tensor(curr_dones))

                obs, r, this_done, info, _ = self.env.step(a.squeeze().detach().cpu().numpy())
                
                curr_dones = np.logical_or(curr_dones, this_done)
                rewards.append(torch.tensor(r))
                prev_rewards.append(rewards[-1])
                for tau in range(1, 1+len(prev_rewards)):
                    prev_rewards[-tau][np.logical_not(curr_dones)] += (self.discount ** tau) * rewards[-1][np.logical_not(curr_dones)]

            actions = self.model.action_history
            states = self.model.state_history
            rewards = torch.stack(rewards)
            dones = torch.stack(dones)

            for i in range(self.num_envs):
                s, a, r, d = states[i], actions[i], rewards[:,i], dones[:,i]

                if torch.any(torch.logical_not(d)):
                    self.data.append((s, a, r, skill[i], d))

            if np.all(curr_dones):
                break


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


def main():

    model = LatentPolicy()

    env = TrainingEnv("LunarLander-v2", num_envs=N_SAMPLE_TAUS, model=model, discount=0.99, skill_len=8)

    env.sample()

    x, y = env[(0, 2)]
    
    pred = model.forward(x)
    loss = DualLoss(pred, y)

    print(loss)

if __name__ == '__main__':
    main()