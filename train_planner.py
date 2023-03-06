
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
# from stable_baselines3 import PPO

from latent_models import LatentTrajectory
from configs import LunarLatentTrajectoryStable

from tqdm import tqdm
import numpy as np
import os
import random
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SAMPLE_TAUS = 4096
EPISODE_LENGTH = 64
SAMPLE_PERIOD = 4

TRAIN_TAUS_FILE = "data/train_taus.pt"
VAL_TAUS_FILE = "data/val_taus.pt"

VAL_SIZE = 512

N_EPOCHS = 50
LR = 1e-5
BATCH_SIZE = 32
LATENT_FILE = "data/latent_model.pt"

CHANGE_PROB = 0.25


def sampleTau(env, policy, n_steps=1000):

    states = []
    dones = []
    rewards = np.zeros((env.num_envs,))

    obs = env.reset()
    done = np.zeros((env.num_envs,), dtype=bool)
    
    for _ in tqdm(range(n_steps), desc="Sampling"):
    
        states.append(obs)
        dones.append(done)

        action_ind = policy(obs)

        obs, r, done, info = env.step(action_ind)
        
        rewards += r

    return torch.tensor(np.array(states)).permute(1, 0, 2), torch.tensor(np.array(dones)).permute(1, 0), torch.tensor(rewards)


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
    
    if not os.path.exists(TRAIN_TAUS_FILE):
        states, dones, rewards = sampleTau(env, pi_random, n_steps=EPISODE_LENGTH)
        
        states = states[:,::SAMPLE_PERIOD]
        dones = dones[:,::SAMPLE_PERIOD]
        
        states, val_states = states[:-VAL_SIZE], states[-VAL_SIZE:]
        dones, val_dones = dones[:-VAL_SIZE], dones[-VAL_SIZE:]
        rewards, val_reward = rewards[:-VAL_SIZE], rewards[-VAL_SIZE:]
        
        torch.save((states, dones, rewards), TRAIN_TAUS_FILE)
        torch.save((val_states, val_dones, val_reward), VAL_TAUS_FILE)
      
    states, dones, rewards = torch.load(TRAIN_TAUS_FILE, map_location=DEVICE)
    val_states, val_dones, val_reward = torch.load(VAL_TAUS_FILE, map_location=DEVICE)

    model = LatentTrajectory(config=LunarLatentTrajectoryStable)
    model = model.to(DEVICE)
    
    if os.path.exists(LATENT_FILE):
        model.load_state_dict(torch.load(LATENT_FILE))
    
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in (pbar := tqdm(range(N_EPOCHS))):
            tot_loss = 0
            num_seen = 0
            
            model.train()
            
            shuffler = list(range(states.shape[0]))
            random.shuffle(shuffler)

            for b in tqdm(range(0, states.shape[0], BATCH_SIZE), desc="Training", leave=False):

                s, d = states[shuffler[b:b+BATCH_SIZE]], dones[shuffler[b:b+BATCH_SIZE]]
                if s.shape[0] == 0:
                    continue

                s_noised = s + torch.normal(torch.zeros_like(s), 0.08).to(DEVICE)

                policy = model.encode(s)
                pred = model.decode(policy, s_noised)
                
                d[:,0] = True
                mask = torch.logical_not(d).reshape(-1)
                
                loss = F.l1_loss(
                    pred.reshape(-1, pred.shape[-1])[mask,:],
                    s.reshape(-1, s.shape[-1])[mask,:]
                )
                
                d = d[:,1:]
                mask = torch.logical_not(d).reshape(-1)
                    
                loss -= F.l1_loss(
                    pred[:,1:].reshape(-1, pred.shape[-1])[mask,:],
                    s[:,:-1].reshape(-1, s.shape[-1])[mask,:]
                ) / 10

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_seen += 1
                tot_loss += loss.item()

            with torch.no_grad():
                val_loss = 0
                val_comp = 0
                val_seen = 0

                model.eval()

                val_shuffler = list(range(val_states.shape[0]))

                for b in tqdm(range(0, val_states.shape[0], BATCH_SIZE), desc="Validating", leave=False):

                    s, d = val_states[val_shuffler[b:b+BATCH_SIZE]], val_dones[val_shuffler[b:b+BATCH_SIZE]]
                    if s.shape[0] == 0:
                        continue

                    policy = model.encode(s)
                    pred = model.decode(policy, s)
                    
                    d[:,0] = True
                    mask = torch.logical_not(d).reshape(-1)
                    
                    loss = F.l1_loss(
                        pred.reshape(-1, pred.shape[-1])[mask,:],
                        s.reshape(-1, s.shape[-1])[mask,:]
                    )

                    val_seen += 1
                    val_loss += loss.item()
                    
                    d = d[:,1:]
                    mask = torch.logical_not(d).reshape(-1)
                    
                    val_comp += F.l1_loss(
                        s[:,1:].reshape(-1, pred.shape[-1])[mask,:],
                        s[:,:-1].reshape(-1, s.shape[-1])[mask,:]
                    ).item()

            pbar.set_postfix({"train_loss": tot_loss/num_seen, 'val_loss': val_loss/val_seen, 'val_comp': val_loss/val_comp})

        torch.save(model.state_dict(), LATENT_FILE)

    model.eval()
    torch.no_grad()
    
    example_traj = val_states[9, torch.logical_not(val_dones[9])]
    
    true_x, true_y = [], []
    for i in range(example_traj.shape[0]):
        true_x.append(example_traj[i,0].item())
        true_y.append(example_traj[i,1].item())

    encoding = model.encode(example_traj)
    pred_traj = example_traj[:1]
    
    for i in range(1, example_traj.shape[0]):
        next_state = model.decode(encoding, pred_traj)[-1].unsqueeze(0)
        pred_traj = torch.cat((pred_traj, next_state), dim=0)
    
    pred_x, pred_y = [], []
    for i in range(pred_traj.shape[0]):
        pred_x.append(pred_traj[i,0].item())
        pred_y.append(pred_traj[i,1].item())
    
    for i in range(len(pred_x)):
        print(pred_x[i], pred_y[i], "-", true_x[i], true_y[i])
    
    plt.plot(pred_x, pred_y, label="Predicted")
    plt.plot(true_x, true_y, label="True")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Decoded Trajectory vs. Original")
    plt.savefig("decoded.png")


if __name__ == '__main__':
    main()