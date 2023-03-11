
import torch

import gym
# from stable_baselines3 import PPO

from tqdm import tqdm
import numpy as np
import random


TRAIN_TAUS_FILE = "data/train_taus.pt"
VAL_TAUS_FILE = "data/val_taus.pt"

N_SAMPLE_TAUS = 4096*8
VAL_SIZE = 512*8

CHANGE_PROB = 0.25


def sampleTau(env, policy):

    states = []
    actions = []
    rewards = np.zeros((env.num_envs,))

    obs = env.reset()
    
    with tqdm(None, leave=True) as pbar:
        while True:
        
            action_ind = policy(obs)
        
            states.append(obs)
            
            # action = np.zeros((env.num_envs, 4))
            # for i in range(env.num_envs):
            #     action[i, action_ind[i]] = 1
            actions.append(action_ind)

            obs, r, done, info = env.step(action_ind)
            
            rewards += r
            
            pbar.update(1)
            if np.any(done):
                break

    return torch.tensor(np.array(states)).permute(1, 0, 2), torch.tensor(np.array(actions)).permute(1, 0), torch.tensor(rewards)


class randomPolicy:
    def __init__(self, n_actions, n_envs):
        self.n_actions = n_actions
        self.n_envs = n_envs
        
        self.prev = np.random.randint(0, self.n_actions-1, size=(n_envs,))

    def __call__(self, s):
        if random.random() < CHANGE_PROB:
            self.prev = np.random.randint(0, self.n_actions-1, size=(self.n_envs,))
        return self.prev


def main():

    env = gym.vector.make("LunarLander-v2", num_envs=N_SAMPLE_TAUS, asynchronous=False)

    pi_random = randomPolicy(4, N_SAMPLE_TAUS)
    
    states, actions, rewards = sampleTau(env, pi_random)
    
    states, val_states = states[:-VAL_SIZE], states[-VAL_SIZE:]
    actions, val_actions = actions[:-VAL_SIZE], actions[-VAL_SIZE:]
    rewards, val_reward = rewards[:-VAL_SIZE], rewards[-VAL_SIZE:]
    
    torch.save((states, actions, rewards), TRAIN_TAUS_FILE)
    torch.save((val_states, val_actions, val_reward), VAL_TAUS_FILE)


if __name__ == '__main__':
    main()