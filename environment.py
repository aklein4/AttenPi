
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
# from stable_baselines3 import PPO

from latent_policy import LatentPolicy
from train_utils import train, Logger
from model_utils import getFeedForward
import configs

from tqdm import tqdm
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


# device to use for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_VERSION = DEVICE == torch.device("cpu")

# number of concurrent environments
N_ENVS = 16
# number of passes through all envs to make per epoch
SHUFFLE_RUNS = 4
# maximum number of (s, a, r, k, d) tuples to store, truncated to newest
MAX_BUF_SIZE = 256

# model config class
CONFIG = configs.CartpolePolicy
ENV_NAME = "CartPole-v1"

# number of skills in model dict
N_SKILLS = CONFIG.num_skills
# length of each skill sequence
SKILL_LEN = CONFIG.skill_len

# number of evaluation iterations (over all envs)
EVAL_ITERS = 1

# csv log output location
LOG_LOC = "logs/log.csv"
# graph output location
GRAFF = "logs/graff.png"

# model checkpoint location
CHECKPOINT = "local_data/checkpoint.pt"

# model leaarning rate
LEARNING_RATE = 1e-3
# model batch size
BATCH_SIZE = 16

# MDP discount factor
DISCOUNT = 0.95
# divide rewards by this factor for normalization
R_NORM = 10
# balance between policy and skill rewards
Kl_LAMBDA = 0.1

# whether to perform evaluation stochastically
STOCH_EVAL = True

# baseline hidden layer size
BASE_DIM = 16
# baseline number of hidden layers
BASE_LAYERS = 2
# baseline learning rate
BASE_LR = 1e-2
# baseline batch size
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
            device=DEVICE,
            action_handler=(lambda x: x)
        ):
        """Handles the environment and data collection for reinforcement training.

        Args:
            env_name (str): Name of the gym environment
            num_envs (it): Number of concurrent environments
            model (LatentPolicy): Policy model to train
            skill_len (int): Length of each skill sequence
            shuffle_runs (int, optional): Number of passes through envs per epoch. Defaults to 1.
            discount (int, optional): MDP reward discount factor. Defaults to 1.
            max_buf_size (int, optional): Maximum data buffer size (truncated to newest). Defaults to 100000.
            device (torch.device, optional): Device to handle tensors on. Defaults to DEVICE.
        """

        # create environment
        self.env = gym.vector.make(env_name, num_envs=num_envs, asynchronous=False)
        self.num_envs = num_envs

        # store model reference
        self.model = model

        # basic params
        self.discount = discount
        self.skill_len = skill_len
        self.shuffle_runs = shuffle_runs
        self.max_buf_size = max_buf_size
        self.action_handler = action_handler

        # device to store everything on
        self.device = device

        # should hold (s, a, r, k, d) tuples
        # temporal length in each tuple should be skill_len
        self.data = []

        # info related to the env/model
        self.action_size = model.config.action_size
        self.state_size = model.config.state_size
        self.num_skills = model.config.num_skills

        # suffle outgoing indices during __getitem__
        self.shuffler = []


    def shuffle(self):
        """ Sample from the environment and shuffle the data buffer.
        """
        # sample shuffler_runs times
        for i in tqdm(range(self.shuffle_runs), desc="Exploring", leave=False):
            self.sample()
        self.data = self.data[-self.max_buf_size:]

        # shuffle shuffler
        self.shuffler = list(range(len(self.data)))
        random.shuffle(self.shuffler)


    def sample(self):
        """ Sample (s, a, r, k, d) tuples from the environment and store them in the data buffer,
        discarding the oldest tuples if the buffer is full.
        """

        # reset the environment and get the initial state
        obs = self.env.reset()
        if LOCAL_VERSION:
            obs = obs[0]
        # keep track of which envs are done
        curr_dones = np.zeros((self.num_envs,), dtype=bool)
        # hold previous rewards as references for return calculation
        prev_rewards = []

        # torch setup
        with torch.no_grad():
            self.model.eval()

            # run until all envs are done
            while True:

                # get the chosen skill and set it in the model
                skill = self.model.setChooseSkill(torch.tensor(obs).to(self.device))

                # get the skill's reward sequence
                rewards = []
                # get the skill's done sequence
                dones = []

                # iterate through the skill sequence
                for t in range(self.skill_len):

                    # sample an action using the current state
                    a = self.model.policy(torch.tensor(obs).to(self.device))

                    # this item in the sequence is done if the _previous state_ was done
                    dones.append(torch.tensor(curr_dones))

                    # take a step in the environment, caching done to temp variable
                    out = self.env.step(self.action_handler(a).squeeze().detach().cpu().numpy())
                    if LOCAL_VERSION:
                        out = out[:-1]
                    obs, r, this_done, info = out
                    # normalize reward
                    r /= R_NORM

                    # store the reward to the sequence
                    rewards.append(torch.tensor(r).to(self.device))
                    # if we were done before this action, the reward doesn't count
                    rewards[-1][curr_dones] = 0
                    # update the done array -> everything done after the first done is masked out
                    curr_dones = np.logical_or(curr_dones, this_done)

                # get the states and actions from the model's history
                actions = self.model.action_history.to(self.device) # (num_envs, skill_len, action_size)
                states = self.model.state_history.to(self.device) # (num_envs, skill_len)

                # stack the rewards, and store a reference (sorted temporally)
                rewards = torch.stack(rewards) # (skill_len, num_envs)
                for i in range(rewards.shape[0]):
                    prev_rewards.append(rewards[i])

                # stack dones into tensor
                dones = torch.stack(dones).to(self.device) # (skill_len, num_envs)

                # iterate over the batch
                for i in range(self.num_envs):
                    # get the tuple from each env
                    s, a, r, d = states[i], actions[i], rewards[:,i], dones[:,i]

                    # store it if it won't be completely masked out
                    if torch.any(torch.logical_not(d)):
                        self.data.append((s, a, r, skill[i], d))

                # break if all envs are done, TODO: make this more efficient
                if np.all(curr_dones):
                    break

            # propogate rewards backwards through time to replace referenced rewards with returns
            for tau in range(2, 1+len(prev_rewards)):
                # dones are accounted for by having r=0
                prev_rewards[-tau] += self.discount * prev_rewards[-tau+1]

        return
    

    def evaluate(self, iterations=1):
        """ Calculate the average reward achieved over a series of deterministic episodes.

        Args:
            iterations (int, optional): Number of times to iterate over all envs (wasted if not STOCH_EVAL). Defaults to 1.

        Returns:
            float: Average reward from trials
        """

        # store a random seed to reinitialize randomness
        seedo = random.randrange(0xFFFF)

        # set all seeds for deterministic evaluation
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

        # accumulate rewards vs number of trials
        tot_rewards = 0
        tot = 0

        # torch setup
        with torch.no_grad():
            self.model.eval()

            # do all iterations
            for it in tqdm(range(iterations), desc="Evaluating", leave=False):

                # reset the environment and get the initial state
                obs = self.env.reset(seed=0)
                if LOCAL_VERSION:
                    obs = obs[0]
                # keep track of which envs are done
                curr_dones = np.zeros((self.num_envs,), dtype=bool)
                # hold rewards (each new reward is added)
                rewards = np.zeros((self.num_envs,), dtype=float)

                # run until all envs are done
                while True:

                    # get the chosen skill and set it in the model
                    self.model.setChooseSkill(torch.tensor(obs).to(self.device))

                    # iterate through the skill sequence
                    for t in range(self.skill_len):

                        # sample an action using the current state, greedy if not STOCH_EVAL
                        a = self.model.policy(torch.tensor(obs).to(self.device), stochastic=STOCH_EVAL)

                        # take a step in the environment, caching done to temp variable
                        out = self.env.step(self.action_handler(a).squeeze().detach().cpu().numpy())
                        if LOCAL_VERSION:
                            out = out[:-1]
                        obs, r, this_done, info = out
                        # normalize reward
                        r /= R_NORM

                        # add new rewards, masked by done _before_ this action
                        rewards[np.logical_not(curr_dones)] += r[np.logical_not(curr_dones)]

                        # update the done array -> everything done after the first done is masked out
                        curr_dones = np.logical_or(curr_dones, this_done)

                    # break if all envs are done, TODO: make this more efficient
                    if np.all(curr_dones):
                        break
                
                # store episode in accumulator
                tot_rewards += np.sum(rewards)
                tot += self.num_envs
            
            # reset all randomness
            torch.manual_seed(seedo)
            np.random.seed(seedo)
            random.seed(seedo)

            # return the average reward as float
            return (tot_rewards / tot).item()


    def __len__(self):
        # get the current length of the data buffer
        return len(self.data)


    def __getitem__(self, getter):
        """ Get a batch of data from the dataset.

        Args:
            getter (int or tuple): Index to get the batch from, or (index, batchsize) tuple

        Returns:
            tuple: (s, a, k, d) x, and (s, a, r, k, d) y
        """
        # handle input
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices we are going to use
        indices = self.shuffler[index:index+batchsize]
        
        # accumulate the data tuples into batch lists
        accum = ([], [], [], [], [])
        for i in indices:
            s, a, r, k, d = self.data[i]

            # maintain order in tuples
            accum[0].append(s)
            accum[1].append(a)
            accum[2].append(r)
            accum[3].append(k)
            accum[4].append(d)

        # stack into tensors
        accum = [torch.stack(accum[i]) for i in range(len(accum))]

        # remove r from x
        x = accum.copy()
        x.pop(2)

        return x, accum


class BaseREINFORCE(nn.Module):
    def __init__(self, state_size, h_dim, n_layers, batch_size, lr, device, disable_baseline=False):
        """ Calculated loss for the REINFORCE algorithm.
        Internally trains a baseline model as loss is called.

        Args:
            state_size (int): elements in the environment state
            h_dim (int): baseline model hidden layer size
            n_layers (int): baseline model number of hidden layers
            batch_size (int): baseline model training batch size
            lr (float): baseline model learning rate
            device (torch.device): Device to store baseline model on
            disable_baseline (bool, optional): Disable baseline training. Defaults to False.
        """
        super().__init__()
        
        # create a basic feedforward model for the baseline
        self.baseline = getFeedForward(state_size, h_dim, 1, n_layers, dropout=0.1)
        self.baseline = self.baseline.to(device)

        # create an optimizer to train the baseline
        self.optimizer = torch.optim.Adam(self.baseline.parameters(), lr=lr)

        # basic parameters
        self.batch_size = batch_size
        self.disable_baseline = disable_baseline


    def train_baseline(self, x, y):
        """ Perform a training epoch on the baseline model.

        Args:
            x (tensor): current states (..., state_size)
            y (_type_): returns for each state (..., 1) - should be same size as x, excluding state_size dim
        """
        # don't train if we are not using baseline
        if self.disable_baseline:
            return

        # set to training mode
        self.train()

        # vectorize the data
        assert x.shape[:-1] == y.shape
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1)

        # iterate over batches
        for b in range(0, x.shape[0], self.batch_size):
            x_b, y_b = x[b:b+self.batch_size], y[b:b+self.batch_size]

            # calculate loss and perform backprop
            y_hat = self.baseline(x_b).squeeze(-1)
            loss = F.mse_loss(y_hat.float(), y_b.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

    def predict_baseline(self, x):
        """ Calculate the baseline prediction for given states

        Args:
            x (tensor): Tensor of current states (..., state_size)

        Returns:
            tensor: baseline predictions, same shape as x (excluding state_size dim)
        """
        # use zero if we are not using baseline
        if self.disable_baseline:
            return torch.zeros_like(x[..., 0])

        # set to eval mode
        self.eval()

        # calculate the baseline prediction without grad
        with torch.no_grad():
            V = self.baseline(x)
        return V


    def loss(self, pred, y):
        """ Calculate the combined loss for the REINFORCE algorithm:
        - policy loss
        - skill loss
        - chooser loss

        Args:
            pred (tuple): (pi_logits, monitor_logits, pred_k) x tuple
            y (tuple): (s, a, r, k, d) y tuple

        Returns:
            _type_: _description_
        """

        # (b, t, a_d, a_s), (b, k), (b, k), (b, t, a_d, a_s)
        pi_logits, mon, pred_k, pred_opt = pred
        # (b, t, s), (b, t, s_d), (b, t), (b), (b, t)
        s, a, r, k, d = y

        # get the baseline prediction, squeeze to match r
        base = self.predict_baseline(s).squeeze(-1)
        assert base.shape == r.shape

        # train the baseline now that we are done with it
        self.train_baseline(s, r)

        # combine the monitor reward with the policy reward
        policy_r = r-base
        # unsqueeze to broadcast with pred_opt
        policy_r = policy_r.unsqueeze(-1).unsqueeze(-1)

        opt_probs = torch.softmax(pred_opt, dim=-1)
        opt_multed = opt_probs * policy_r
        opt_chosen = opt_multed.view(-1, opt_multed.shape[-1])[range(a.numel()),a.view(-1)]
        opt_masked = opt_chosen[torch.logical_not(d).view(-1).repeat_interleave(a.shape[-1])]
        opt_loss = -torch.mean(opt_masked)

        # get 1 where the monitor is correct, -1 where it is wrong
        mon_rewards = 2*(torch.argmax(mon, dim=-1) == k).float() - 1
        # unsqueeze to broadcast with r
        mon_rewards.unsqueeze_(1).unsqueeze_(-1).unsqueeze_(-1)
        assert mon_rewards.dim() == pi_logits.dim()

        # get log probabilities of each action
        log_probs = torch.log_softmax(pi_logits, dim=-1)
        # multiply log probabilities by rewards
        multed = log_probs * mon_rewards
        # index into vector with only the chosen actions
        chosen = multed.view(-1, multed.shape[-1])[range(a.numel()),a.view(-1)]
        # mask out actions that were taken in a done state
        masked = chosen[torch.logical_not(d).view(-1).repeat_interleave(a.shape[-1])]
        # calculate the policy loss according to REINFORCE
        elbo_loss = -(1-Kl_LAMBDA)*torch.mean(masked) + Kl_LAMBDA*F.kl_div(torch.softmax(pi_logits, dim=-1), torch.softmax(pred_opt, dim=-1).detach(), reduction='batchmean')

        # monitor loss is simply cross entropy of pred vs actual
        monitor_loss = F.cross_entropy(mon, k)

        # chooser reward is return to first state where skill was chosen
        chooser_r = (r-base)[:,0]

        # get log probabilities of each skill
        chooser_probs = torch.log_softmax(pred_k, dim=-1)
        # index into vector with only the chosen skills
        chooser_chosen = chooser_probs[range(k.numel()),k] # k is vector
        assert chooser_chosen.shape == chooser_r.shape

        # chooser loss is REINFORCE, similar to policy
        chooser_loss = -torch.mean(chooser_chosen * chooser_r)

        # return superposition of all losses
        return opt_loss + elbo_loss + monitor_loss + chooser_loss


class EnvLogger(Logger):
    def __init__(self, env, eval_iters, log_loc, graff):
        """ A logger to track the performance of the model in the environment.

        Args:
            env (TrainingEnv): Environment to evaluate the model in
            eval_iters (int): Number of iterations per env.evaluation() call
            log_loc (str): File location for the output csv
            graff (str): File location for the output graph
        """

        # save environment reference
        self.env = env

        # store metrics over epochs
        self.avg_rewards = []
        self.accs = []

        # basic parameters
        self.log_loc = log_loc
        self.graff = graff
        self.eval_iters = eval_iters

        # create metric file and write header
        with open(self.log_loc, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, dialect='excel')
            spamwriter.writerow(["epoch", "avg_reward", "skill_acc"])


    def initialize(self, model):
        # get reference to the model so we can save it
        self.model = model
    

    def log(self, train_log, val_log):
        """ Log the current epoch's metrics.

        Args:
            train_log (tuple): tuple of (x, y) lists from the training function
            val_log (None): Unused, but required by the Logger interface
        """

        # calculate skill accuracy
        if train_log is not None:
            # extract a single monitor prediction tensor
            mon = torch.cat([
                train_log[0][t][1].view(-1, train_log[0][t][1].shape[-1])
                for t in range(len(train_log[0]))
            ], dim=0)
            # extract a single target skill tensor
            k = torch.cat([
                train_log[1][t][3].view(-1)
                for t in range(len(train_log[1]))
            ], dim=0)
            # calculate prediction accuracy
            acc = (torch.argmax(mon, dim=-1) == k).float().mean().item()

        # during init call, we just use an average
        else:
            acc = 1 / N_SKILLS

        # deterministically evaluate the model's average reward
        curr_r = self.env.evaluate(self.eval_iters)

        # save metrics
        self.accs.append(acc)
        self.avg_rewards.append(curr_r)
        
        # append metrics to csv file
        with open(self.log_loc, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
            spamwriter.writerow([len(self.avg_rewards)-2, curr_r, acc]) # -2 because of init call

        """ Plot the metrics """
        fig, ax = plt.subplots(2)

        ax[0].plot(self.avg_rewards)
        ax[0].set_title(r"Average Reward")

        ax[1].plot(self.accs)
        ax[1].set_title(r"Skill Accuracy")

        fig.set_figwidth(6)
        fig.set_figheight(8)
        plt.tight_layout()
        plt.savefig(self.graff)
        plt.close(fig)

        # save a checkpoint, TODO: add metric instead of overriding every epoch
        self.save_checkpoint()
    

    def save_checkpoint(self):
        # save the model's state dict to the checkpoint location
        torch.save(self.model.state_dict(), CHECKPOINT)


def walkerHandler(a):
    return (a-1).float()


def main():

    # initialize the model that we will train
    model = LatentPolicy(CONFIG)
    model.to(DEVICE)

    # initialize our training environment
    env = TrainingEnv(
        env_name = ENV_NAME,
        num_envs = N_ENVS,
        model = model,
        skill_len = SKILL_LEN,
        shuffle_runs = SHUFFLE_RUNS,
        discount = DISCOUNT,
        max_buf_size = MAX_BUF_SIZE,
        device = DEVICE
    )

    # env.sample()
    # s, a, r, k, d = env.data[0]
    # k = k.unsqueeze(0)
    # model.setSkill(k)
    # print(model.forward((s.unsqueeze(0), a.unsqueeze(0), k, d.unsqueeze(0)))[0])
    # for i in range(s.shape[0]):
    #     model.policy(s[i:i+1], action_override=a[i:i+1].unsqueeze(1))
    # exit()

    # initialize the logger
    logger = EnvLogger(
        env = env,
        eval_iters = EVAL_ITERS,
        log_loc = LOG_LOC,
        graff = GRAFF
    )
    logger.initialize(model)

    # make an init call to the logger to save the before-training performance
    logger.log(None, None)

    # initialize the loss function object
    loser = BaseREINFORCE(model.config.state_size, BASE_DIM, BASE_LAYERS, BATCH_SIZE, BASE_LR, DEVICE)

    # initialize the model's optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train ad infinitum
    train(
        model = model,
        optimizer = optimizer,
        train_data = env,
        loss_fn = loser.loss,
        logger = logger,
        batch_size = BATCH_SIZE
    )


if __name__ == '__main__':
    main()
