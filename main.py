
import torch

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from encoding_models import StateActionEncoderDecoder, PositionalEncoding, TauEncoderDecoder

import numpy as np
from tqdm import tqdm

def collect_data(num_episodes, episode_length=100):
    model = PPO.load("ppo_cartpole")
    env = make_vec_env("CartPole-v1")
    
    data = []
    episodes = []

    for _ in range(num_episodes):

        epso = []
        obs = env.reset()
        while True:

            action, _states = model.predict(obs)

            s_a = np.concatenate((obs[0], action, np.zeros_like(action)), axis=-1)
            data.append(torch.tensor(s_a, dtype=torch.float32))
            epso.append(torch.tensor(s_a, dtype=torch.float32))

            obs, rewards, dones, info = env.step(action)

            if dones[0] or len(epso) == episode_length-1:
                
                s_a = np.concatenate((np.zeros_like(obs[0]), np.zeros_like(action), np.ones_like(action)), axis=-1)
                data.append(torch.tensor(s_a, dtype=torch.float32))

                while len(epso) < episode_length:
                    epso.append(torch.tensor(s_a, dtype=torch.float32))

                break
        episodes.append(torch.stack(epso))
    
    return torch.stack(data), torch.stack(episodes)


def main():

    encoder = StateActionEncoderDecoder(6, 0, 32, 8)
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)

    data, episodes = collect_data(100, 8)

    val_episode = episodes[0].unsqueeze(0)
    episodes = episodes[1:]

    weights = 1 / torch.mean(torch.mean(data), dim=0)
    weighted_data = data * weights

    for epoch in (pbar := tqdm(range(1000))):

        optimizer.zero_grad()

        pred = encoder(data)

        enc = encoder.encode(episodes)

        # sim_loss = torch.sum(enc[:,1:] * enc[:,:-1]) / enc[:,:-1].numel()
        # sim_loss -= torch.sum(enc[1:,:] * enc[:-1,:]) / enc[1:].numel()

        loss = torch.nn.functional.mse_loss(pred * weights, weighted_data)

        loss.backward()
        optimizer.step()

        data.detach_()

        pbar.set_postfix({'loss': loss.item()})


    tau_enc = TauEncoderDecoder(6, 64, 8, 16, 8, 128)
    optimizer = torch.optim.Adam(tau_enc.parameters(), lr=1e-4)

    encoder.eval()
    tau_enc.train()

    encoded_episodes = encoder.encode(episodes)

    tgt = episodes.clone()
    end_token = torch.zeros_like(tgt[:,0,:]).unsqueeze(1)
    end_token[:,:,-1] = 1
    tgt = torch.cat((tgt, end_token), dim=1)

    for epoch in (pbar := tqdm(range(50))):
            
        optimizer.zero_grad()

        pred = tau_enc(episodes)

        loss = torch.nn.functional.mse_loss(pred, tgt * weights.unsqueeze(0).unsqueeze(0))

        loss.backward()

        optimizer.step()
        encoded_episodes.detach_()
        tgt.detach_()

        pbar.set_postfix({'loss': loss.item()})

    tau_enc.eval()

    val_enc = encoder.encode(val_episode)
    val_h = tau_enc.encode(val_episode)
    val_pred = tau_enc.decode(val_h) * weights.unsqueeze(0).unsqueeze(0)

    for i in range(episodes.shape[1]):
        print(val_episode[0, i].tolist())
    print("-----------------")

    for i in range(val_pred.shape[1]):
        print(val_pred[0, i].tolist())
    print("-----------------")


if __name__ == '__main__':
    main()