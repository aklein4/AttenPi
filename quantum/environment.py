
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

INIT_MODEL = "./local_data/cheetah_init.pt"

# device to use for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_VERSION = True

# number of concurrent environments
N_ENVS = 4
# number of passes through all envs to make per epoch
SHUFFLE_RUNS = 1

# model config class
CONFIG = configs.CheetahPolicy
ENV_NAME = "HalfCheetah-v4"

# length of each skill sequence
SKILL_LEN = CONFIG.skill_len

# number of evaluation iterations (over all envs)
EVAL_ITERS = 1
MAX_BUF_SIZE = 512

MAX_EPISODE = 1000

# csv log output location
LOG_LOC = "logs/semisup_regulator.csv"
# graph output location
GRAFF = "logs/semisup_regulator.png"

# model checkpoint location
CHECKPOINT = "local_data/semisup_regulator.pt"

# model leaarning rate
LEARNING_RATE = 3e-4
# model batch size
BATCH_SIZE = 32

# MDP discount factor
DISCOUNT = 0.98
# divide rewards by this factor for normalization
R_NORM = 100

LAMBDA_SKILL = 0.0
LAMBDA_PI = 1
LAMBDA_SEMISUP = 0.1

# whether to perform evaluation stochastically
STOCH_EVAL = True

BASELINE = True

# baseline hidden layer size
BASE_DIM = 32
# baseline number of hidden layers
BASE_LAYERS = 2
# baseline learning rate
BASE_LR = 1e-3
# baseline batch size
BASE_BATCH = BATCH_SIZE//4


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

        # should hold (s, a, r, d) tuples
        # temporal length in each tuple should be skill_len
        self.data = []

        # info related to the env/model
        self.action_size = model.config.action_size
        self.state_size = model.config.state_size

        # suffle outgoing indices during __getitem__
        self.shuffler = []


    def shuffle(self):
        """ Sample from the environment and shuffle the data buffer.
        """
        if len(self.data) > 0:
            self.data = random.choices(self.data, k=min(len(self.data), self.max_buf_size))
        # sample shuffler_runs times
        self.pbar = tqdm(range(self.shuffle_runs), desc="Exploring", leave=False)
        for _ in self.pbar:
            self.sample()
        self.pbar.close()
        self.pbar = None

        # shuffle shuffler
        self.shuffler = list(range(len(self.data)))
        random.shuffle(self.shuffler)


    def sample(self):
        """ Sample (s, a, r, d) tuples from the environment and store them in the data buffer,
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
            time = -1
        
            # run until all envs are done
            while True:

                if obs.dtype == np.uint8:
                    obs = obs.astype(np.float32) / 255

                # get the chosen skill and set it in the model
                skill = self.model.setChooseSkill(torch.tensor(obs).to(self.device).float())

                # get the skill's reward sequence
                rewards = []
                # get the skill's done sequence
                dones = []

                # iterate through the skill sequence
                for t in range(self.skill_len):
                    time += 1

                    if obs.dtype == np.uint8:
                        obs = obs.astype(np.float32) / 255

                    # sample an action using the current state
                    a = self.model.policy(torch.tensor(obs).to(self.device).float())

                    # this item in the sequence is done if the _previous state_ was done
                    dones.append(torch.tensor(curr_dones))

                    # take a step in the environment, caching done to temp variable
                    out = self.env.step(self.action_handler(a).detach().cpu().numpy())
                    if LOCAL_VERSION:
                        out = out[:-1]
                    obs, r, this_done, info = out
                    this_done |= time >= MAX_EPISODE
                    # normalize reward
                    r /= R_NORM

                    # store the reward to the sequence
                    rewards.append(torch.tensor(r).to(self.device))
                    # if we were done before this action, the reward doesn't count
                    rewards[-1][curr_dones] = 0
                    # update the done array -> everything done after the first done is masked out
                    curr_dones = np.logical_or(curr_dones, this_done)

                self.pbar.set_postfix({"t": time})

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
                        self.data.append((s, a, r, d))

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
            pbar = tqdm(range(iterations), desc="Evaluating", leave=False)
            for it in pbar:

                # reset the environment and get the initial state
                obs = self.env.reset(seed=0)
                if LOCAL_VERSION:
                    obs = obs[0]
                # keep track of which envs are done
                curr_dones = np.zeros((self.num_envs,), dtype=bool)
                # hold rewards (each new reward is added)
                rewards = np.zeros((self.num_envs,), dtype=float)

                time = -1

                # run until all envs are done
                while True:

                    if obs.dtype == np.uint8:
                        obs = obs.astype(np.float32) / 255
        
                    # get the chosen skill and set it in the model
                    self.model.setChooseSkill(torch.tensor(obs).to(self.device).float())

                    # iterate through the skill sequence
                    for t in range(self.skill_len):
                        time += 1

                        if obs.dtype == np.uint8:
                            obs = obs.astype(np.float32) / 255

                        # sample an action using the current state, greedy if not STOCH_EVAL
                        a = self.model.policy(torch.tensor(obs).to(self.device).float(), stochastic=STOCH_EVAL)

                        # take a step in the environment, caching done to temp variable
                        out = self.env.step(self.action_handler(a).detach().cpu().numpy())
                        if LOCAL_VERSION:
                            out = out[:-1]
                        obs, r, this_done, info = out
                        this_done |= time >= MAX_EPISODE
                        # normalize reward
                        r /= R_NORM

                        # add new rewards, masked by done _before_ this action
                        rewards[np.logical_not(curr_dones)] += r[np.logical_not(curr_dones)]

                        # update the done array -> everything done after the first done is masked out
                        curr_dones = np.logical_or(curr_dones, this_done)

                        

                        pbar.set_postfix({"t": time})

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
            pbar.close()
            return (tot_rewards / tot).item()


    def __len__(self):
        # get the current length of the data buffer
        return len(self.data)


    def __getitem__(self, getter):
        """ Get a batch of data from the dataset.

        Args:
            getter (int or tuple): Index to get the batch from, or (index, batchsize) tuple

        Returns:
            tuple: s as x, and (s, a, r, d) as y
        """
        # handle input
        index = getter
        batchsize = 1
        if isinstance(getter, tuple):
            index, batchsize = getter

        # get the indices we are going to use
        indices = self.shuffler[index:index+batchsize]
        
        # accumulate the data tuples into batch lists
        accum = ([], [], [], [])
        for i in indices:
            s, a, r, d = self.data[i]

            # maintain order in tuples
            accum[0].append(s)
            accum[1].append(a)
            accum[2].append(r)
            accum[3].append(d)

        # stack into tensors
        accum = [torch.stack(accum[i]) for i in range(len(accum))]

        return (accum[0], accum[3]), accum


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
        self.baseline = SkipNet(state_size, h_dim, 1, n_layers, dropout=0.1)
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
            return torch.zeros_like(x[..., :1])

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
            pred (tuple): (pi_logits, skill_logits, logits) x tuple
            y (tuple): (s, a, r, d) y tuple

        Returns:
            torch.tensor: Combined loss of the model
        """

        # (b, t, pi, a_d, a_s), (b, pi), (b, t, a_d, a_s), (b, b)
        pi_logits, skill_logits, logits, enc_outs, pi_preds = pred
        # (b, t, s), (b, t, s_d), (b, t), (b, t)
        s, a, r, d = y

        batch_size = s.shape[0]

        # get the baseline prediction, squeeze to match r
        base = self.predict_baseline(s).squeeze(-1)
        assert base.shape == r.shape

        # train the baseline now that we are done with it
        self.train_baseline(s, r)

        # get the advantage from the reward and baseline
        A = r - base.detach()
        # unsqueeze to broadcast with pred_opt
        A = A.unsqueeze(-1).unsqueeze(-1)
        assert A.dim() == logits.dim()

        multed = torch.log_softmax(logits, dim=-1) * A
        chosen = multed.view(-1, multed.shape[-1])[range(a.numel()),a.view(-1)]
        masked = chosen[torch.logical_not(d).view(-1).repeat_interleave(a.shape[-1])]
        loss = -torch.mean(masked)
        
        enc_loss = F.cross_entropy(enc_outs, torch.arange(0, enc_outs.shape[0], dtype=torch.long).to(enc_outs.device))

        pi_mask = torch.diag(torch.ones(pi_preds.shape[1], dtype=torch.bool)).to(pi_preds.device)
        pi_mask = pi_mask.unsqueeze(0).repeat(pi_preds.shape[0], 1, 1)
        pi_probs = torch.log_softmax(pi_preds, -1)[pi_mask]
        # pi_probs *= torch.softmax(-skill_logits, -1).view(-1).detach()
        pi_loss = -torch.mean(pi_probs)

        skill_target = torch.argmax(skill_logits, dim=-1)
        semisup_loss = F.cross_entropy(skill_logits, skill_target)

        return loss + (LAMBDA_SKILL * enc_loss) + (LAMBDA_PI * pi_loss) + (LAMBDA_SEMISUP * semisup_loss)


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
        self.enc_accs = []
        self.skill_maxs = []
        self.skill_kls = []
        self.pi_kls = []
        self.pi_accs = []

        # basic parameters
        self.log_loc = log_loc
        self.graff = graff
        self.eval_iters = eval_iters

        # create metric file and write header
        try:
            with open(self.log_loc, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, dialect='excel')
                spamwriter.writerow(["epoch", "avg_reward", "skill_max", "enc_acc", "skill_kl", "pi_acc", "pi_kl"])
        except:
            pass


    def initialize(self, model):
        # get reference to the model so we can save it
        self.model = model
    

    def log(self, train_log, val_log):
        """ Log the current epoch's metrics.

        Args:
            train_log (tuple): tuple of (x, y) lists from the training function
            val_log (None): Unused, but required by the Logger interface
        """

        # deterministically evaluate the model's average reward
        curr_r = self.env.evaluate(self.eval_iters)

        if train_log is not None:
            corr = 0
            tot = 0
            for i in range(len(train_log[0])):
                corr += (torch.argmax(train_log[0][i][3], dim=-1) == torch.arange(0, train_log[0][i][3].shape[0], dtype=torch.long).to(train_log[0][i][3].device)).sum()
                tot += train_log[0][i][3].shape[0]
            acc = (corr / tot).item()

            p = 0
            tot = 0
            for i in range(len(train_log[0])):
                p += torch.softmax(train_log[0][i][1], dim=-1).max(dim=-1)[0].sum()
                tot += train_log[0][i][1].shape[0]
            skill_max = (p / tot).item()

            avg_skill = torch.zeros_like(train_log[0][0][1][0])
            tot = 0
            for i in range(len(train_log[0])):
                avg_skill += torch.sum(torch.softmax(train_log[0][i][1], dim=-1), dim=0)
                tot += train_log[0][i][1].shape[0]
            avg_skill /= tot
            kl = 0
            tot = 0
            for i in range(len(train_log[0])):
                kl += F.kl_div(torch.log_softmax(train_log[0][i][1], dim=-1), avg_skill.unsqueeze(0), reduction='sum')
                tot += train_log[0][i][1].shape[0]
            skill_kl = (kl / tot).item()

            kl = 0
            tot = 0
            for i in range(len(train_log[0])):
                avg_pi = torch.mean(torch.softmax(train_log[0][i][0], dim=-1), dim=-3)
                avg_pi = torch.stack([avg_pi]*train_log[0][i][0].shape[-3], dim=-3)
                kl += F.kl_div(torch.log_softmax(train_log[0][i][0], dim=-1), avg_pi, reduction='sum')
                tot += train_log[0][i][0].shape[0]*train_log[0][i][0].shape[1]*train_log[0][i][0].shape[2]
            pi_kl = (kl / tot).item()

            corr = 0
            tot = 0
            for i in range(len(train_log[0])):
                pi_targ = torch.arange(0, train_log[0][i][4].shape[1], dtype=torch.long).to(train_log[0][i][4].device)
                pi_targ = torch.stack([pi_targ]*train_log[0][i][4].shape[0], dim=0)
                corr += (torch.argmax(train_log[0][i][4], dim=-1) == pi_targ).sum()
                tot += train_log[0][i][4].shape[0]*train_log[0][i][4].shape[1]
            pi_acc = (corr / tot).item()

        else:
            acc = 1/self.model.config.batch_keep
            skill_max = 1/self.model.config.num_pi
            skill_kl = None
            pi_kl = None
            pi_acc = 1/self.model.config.num_pi


        # save metrics
        self.avg_rewards.append(curr_r)
        self.enc_accs.append(acc)
        self.skill_maxs.append(skill_max)
        self.skill_kls.append(skill_kl)
        self.pi_kls.append(pi_kl)
        self.pi_accs.append(pi_acc)

        # append metrics to csv file
        try:
            with open(self.log_loc, 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                spamwriter.writerow([len(self.avg_rewards)-2, curr_r, skill_max, acc, skill_kl, pi_acc, pi_kl]) # -2 because of init call
        except:
            pass

        """ Plot the metrics """
        fig, ax = plt.subplots(3, 2)

        ax[0,0].plot(self.avg_rewards)
        ax[0,0].set_title(r"Average Reward")

        ax[1,0].plot(self.enc_accs)
        ax[1,0].set_title(r"Choice-State Identification Accuracy")

        ax[0,1].plot(self.skill_maxs)
        ax[0,1].set_title(r"Avg Max Choice")

        ax[2,0].plot(self.skill_kls)
        ax[2,0].set_title(r"Batch-wise Choice Information Radius")

        ax[2,1].plot(self.pi_kls)
        ax[2,1].set_title(r"Inter-Policy Information Radius")

        ax[1,1].plot(self.pi_accs)
        ax[1,1].set_title(r"Policy Identification Accuracy")

        fig.set_figwidth(12)
        fig.set_figheight(12)
        plt.tight_layout()
        try:
            plt.savefig(self.graff)
        except:
            pass
        plt.close(fig)

        # save a checkpoint, TODO: add metric instead of overriding every epoch
        self.save_checkpoint()
    

    def save_checkpoint(self):
        # save the model's state dict to the checkpoint location
        torch.save(self.model.state_dict(), CHECKPOINT)


def CheetahHandler(a):
    return (a-2).float() / 2.0

def KangarooHandler(a):
    return a.int()

def main():

    # initialize the model that we will train
    model = QuantumPolicy(CONFIG)
    model.load_state_dict(torch.load(INIT_MODEL))
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
        device = DEVICE,
        action_handler=CheetahHandler
    )

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
    loser = BaseREINFORCE(model.config.state_size, BASE_DIM, BASE_LAYERS, BATCH_SIZE, BASE_LR, DEVICE, disable_baseline=(not BASELINE))

    # initialize the model's optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train ad infinitum
    train(
        model = model,
        optimizer = optimizer,
        train_data = env,
        loss_fn = loser.loss,
        logger = logger,
        batch_size = BATCH_SIZE,
        num_epochs = 50
    )


if __name__ == '__main__':
    main()


