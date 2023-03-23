
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
# from stable_baselines3 import PPO

from quantum_policy import QuantumPolicy
from train_utils import train, Logger
from model_utils import SkipNet
import configs

from tqdm import tqdm
import numpy as np
import random
import csv
import matplotlib.pyplot as plt

TEST_MODEL = "./local_data/enc_reg.pt"

# device to use for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_VERSION = True

# model config class
CONFIG = configs.CheetahPolicy
ENV_NAME = "HalfCheetah-v4"

MAX_EPISODE = 200

# length of each skill sequence
SKILL_LEN = CONFIG.skill_len


def visualize(env, model, env_handler=(lambda x: x)):
    """ Sample (s, a, r, d) tuples from the environment and store them in the data buffer,
    discarding the oldest tuples if the buffer is full.
    """
    
    obs = env.reset()
    if LOCAL_VERSION:
        obs = obs[0]

    time = 0
    done = False

    # run until all envs are done
    while True:

        if obs.dtype == np.uint8:
            obs = obs.astype(np.float32) / 255

        # get the chosen skill and set it in the model
        skill = model.setChooseSkill(torch.tensor(obs).unsqueeze(0).to(DEVICE).float()).squeeze(0)
        print(skill.to(DEVICE).detach().cpu().numpy().round(3))

        # iterate through the skill sequence
        for t in range(SKILL_LEN):
            time += 1

            if obs.dtype == np.uint8:
                obs = obs.astype(np.float32) / 255

            # sample an action using the current state
            a = model.policy(torch.tensor(obs).to(DEVICE).float().unsqueeze(0)).squeeze(0)

            # take a step in the environment, caching done to temp variable
            out = env.step(env_handler(a).detach().cpu().numpy())
            if LOCAL_VERSION:
                out = out[:-1]
            obs, r, done, info = out
            done = time >= MAX_EPISODE

            env.render()

        if done:
            break


def CheetahHandler(a):
    return (a-2).float() / 2.0


def main():

    # initialize the model that we will train
    model = QuantumPolicy(CONFIG)
    model.load_state_dict(torch.load(TEST_MODEL))
    model.to(DEVICE)

    # initialize our training environment

    torch.no_grad()
    model.eval()

    env = gym.make(ENV_NAME, render_mode='human')

    while True:
        visualize(env, model, CheetahHandler)


if __name__ == '__main__':
    main()


