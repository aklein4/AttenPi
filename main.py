
import torch

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from encoding_models import Trajectory2Policy
from configs import AcrobatTrajectory2Policy

from tqdm import tqdm
import numpy as np


def collect_data(env, num_episodes, episode_length):
    
    states = []
    actions = []
    dones = []

    for _ in tqdm(range(num_episodes)):

        s = []
        a = []
        d = []

        action = np.random.randint(0, 2)

        obs = env.reset()[0]
        while True:

            if np.random.rand() < 0.25:
                action = np.max(np.random.randint(0, 2), 0)

            s.append(torch.tensor(obs, dtype=torch.float32))
            a.append(torch.tensor([0, 0, 0], dtype=torch.float32))
            a[-1][action] = 1
            d.append(torch.tensor([0], dtype=torch.float32))

            obs, rewards, done, info, _ = env.step(action)

            if done:

                while len(s) < episode_length:
                    s.append(torch.zeros_like(s[-1]))
                    a.append(torch.zeros_like(a[-1]))
                    a[-1][1] = 1
                    d.append(torch.tensor([1], dtype=torch.float32))
                
                break

        seg_start = np.random.randint(0, len(s) - episode_length)
        s = s[seg_start:seg_start + episode_length]
        a = a[seg_start:seg_start + episode_length]
        d = d[seg_start:seg_start + episode_length]

        states.append(torch.stack(s))
        actions.append(torch.stack(a))
        dones.append(torch.stack(d))
    
    return torch.cat([torch.stack(states), torch.stack(dones)], dim=-1), torch.stack(actions)


def evalPolicy(env, model, policy, max_steps, render=False):

    obs = env.reset()[0]

    s = [torch.cat([torch.tensor(obs, dtype=torch.float32), torch.tensor([0], dtype=torch.float32)], dim=-1)]
    a = [torch.zeros(3, dtype=torch.float32)]
    probs = model.decodePolicy(policy, torch.stack(s), torch.stack(a), probs=True)[-1].numpy()
    action = np.random.choice(3, 1, p =  probs/ np.sum(probs))[0]

    s = []
    a = []

    r = 0
    seen = 0
    while True:

        s.append(torch.cat([torch.tensor(obs, dtype=torch.float32), torch.tensor([0], dtype=torch.float32)], dim=-1))
        a.append(torch.tensor([0, 0, 0], dtype=torch.float32))
        a[-1][action] = 1

        obs, reward, done, info, _ = env.step(action)
        r += reward
        seen += 1

        s = s[:model.config.seq_len]
        a = a[:model.config.seq_len]

        probs = model.decodePolicy(policy, torch.stack(s), torch.stack(a), probs=True)[-1].numpy()
        action = np.random.choice(3, 1, p =  probs/ np.sum(probs))[0]

        if render:
            print(action)
            env.render()

        if done or seen >= max_steps:
            return r


def main():

    env = gym.make("Acrobot-v1")

    states, actions = collect_data(env, 1000, 64)

    model = Trajectory2Policy(config=AcrobatTrajectory2Policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    done_mask = states.reshape((-1, 7))[:,-1] != 1

    target_actions = actions.reshape(-1, actions.shape[-1])[done_mask]

    for epoch in (pbar := tqdm(range(50))):

        policy = model.encodePolicy(states, actions)

        policy_actions = model.decodePolicy(policy, states, actions)
        policy_actions = policy_actions.reshape(-1, policy_actions.shape[-1])
        policy_actions = policy_actions[done_mask]

        policy_actions = torch.nn.functional.log_softmax(policy_actions, dim=-1)
            
        loss = -torch.sum(torch.where(target_actions > 0.5, policy_actions, 0)) / target_actions.shape[0]

        acc = torch.sum(torch.argmax(policy_actions, dim=-1) == torch.argmax(target_actions, dim=-1)).item() / policy_actions.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"Loss": loss.item(), 'acc': acc})

    with torch.no_grad():
        model.eval()

        scores = []
        best_p = None
        for i in tqdm(range(states.shape[0])):
            scores.append(evalPolicy(env, model, model.encodePolicy(states[i], actions[i]), 100))
            if best_p is None or scores[-1] > scores[best_p]:
                best_p = i

        print("Mean:", np.mean(scores), "Max:", np.max(scores))

        env = gym.make("Acrobot-v1", render_mode="human")
        evalPolicy(env, model, model.encodePolicy(states[best_p], actions[best_p]), 10000, render=True)

if __name__ == '__main__':
    main()