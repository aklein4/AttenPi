
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

from encoding_models import LatentPolicy
from configs import AcrobatLatentPolicy

from tqdm import tqdm
import numpy as np
import os
import random


N_SAMPLE_TAUS = 100
EPISODE_LENGTH = 64
SAMPLE_TAUS_FILE = "data/sample_taus.pt"

N_EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 8
LATENT_FILE = "data/latent_model.pt"

CHANGE_PROB = 0.25


def sampleTau(env, policy, render=False):

    states = []
    actions = []
    reward = 0

    obs = torch.tensor(env.reset()[0]).float()
    
    while True:
    
        states.append(obs)

        action_ind = torch.tensor(policy(obs)).long()
        actions.append(F.one_hot(action_ind, 3).float().squeeze(0))

        obs, r, done, info, _ = env.step(action_ind.item())
        obs = torch.tensor(obs).float()

        reward += r

        if render:
            env.render()

        if done:
            return states, actions, reward


def collect_data(env, policy, num_episodes, episode_length=None):
    
    states = []
    actions = []
    rewards = []

    pbar = tqdm(range(num_episodes), desc="Sampling", leave=False)

    while len(rewards) < num_episodes:
        s, a, r = sampleTau(env, policy)

        if episode_length is not None and len(a) < episode_length:
            continue

        if episode_length is not None:
            states.append(torch.stack(s[-episode_length:]))
            actions.append(torch.stack(a[-episode_length:]))
        
        rewards.append(r)
        pbar.update(1)

    pbar.close()

    if episode_length is not None:
        return torch.stack(states), torch.stack(actions), torch.tensor(rewards).float()
    
    return torch.tensor(rewards).float()


class randomPolicy:
    def __init__(self):
        self.prev = torch.randint(0, 3, (1,)).item()

    def __call__(self, s):
        if random.random() < CHANGE_PROB:
            self.prev = torch.randint(0, 3, (1,)).item()
        return self.prev


def main():

    env = gym.make("Acrobot-v1")

    pi_random = randomPolicy()

    states, actions, rewards = None, None, None
    if not os.path.exists(SAMPLE_TAUS_FILE):
        states, actions, rewards = collect_data(env, pi_random, N_SAMPLE_TAUS, EPISODE_LENGTH)
        torch.save((states, actions, rewards), SAMPLE_TAUS_FILE)
    else:
        states, actions, rewards = torch.load(SAMPLE_TAUS_FILE)

    model = LatentPolicy(config=AcrobatLatentPolicy)
    
    if os.path.exists(LATENT_FILE):
        model.load_state_dict(torch.load(LATENT_FILE))
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        model.train()

        shuffler = list(range(N_EPOCHS))
        random.shuffle(shuffler)

        tot_loss = 0
        tot_acc = 0
        num_seen = 0

        for epoch in (pbar := tqdm(range(N_EPOCHS))):
            tot_loss = 0
            tot_acc = 0
            num_seen = 0

            for b in range(0, N_SAMPLE_TAUS, BATCH_SIZE):

                s, a = states[shuffler[b:b+BATCH_SIZE]], actions[shuffler[b:b+BATCH_SIZE]]
                if s.shape[0] == 0:
                    continue

                policy = model.encodePolicy(s, a)
                pred = model.decodePolicy(policy, s).reshape(-1, a.shape[-1])
                
                target = a.reshape(-1, a.shape[-1])
                loss = F.cross_entropy(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_seen += 1
                tot_loss += loss.item()
                tot_acc += torch.sum(torch.argmax(pred, dim=-1) == torch.argmax(target, dim=-1)).item() / pred.shape[0]

            pbar.set_postfix({"Loss": tot_loss/num_seen, 'acc': tot_acc/num_seen})

        torch.save(model.state_dict(), LATENT_FILE)

    torch.no_grad()
    model.eval()

    reward = []
    for p in (pbar := tqdm(range(N_SAMPLE_TAUS))):
        reward.append(sampleTau(env, pi_random)[2])
        pbar.set_postfix({"Mean": np.mean(reward), 'std': np.std(reward)})
    
    print("\nRandom:")
    print("Mean:", np.mean(reward))
    print("Std:", np.std(reward))
    print("Max:", np.max(reward))
    print("Min:", np.min(reward))

    reward = []
    for p in (pbar := tqdm(range(N_SAMPLE_TAUS))):
        model.setInferencePolicy(model.encodePolicy(states[p], actions[p]))
        reward.append(sampleTau(env, model.sampleAction)[2])

        pbar.set_postfix({"Mean": np.mean(reward), 'std': np.std(reward)})

    print("\nLatent:")
    print("Mean:", np.mean(reward))
    print("Std:", np.std(reward))
    print("Max:", np.max(reward))
    print("Min:", np.min(reward))


if __name__ == '__main__':
    main()