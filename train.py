
import torch
import torch.nn as nn
import torch.nn.functional as F

from procgen import ProcgenGym3Env

from quantum_policy import QuantumPolicy
from train_utils import train, Logger
from model_utils import SkipNet, MobileNet
import configs

from tqdm import tqdm
import numpy as np
import random
import csv
import matplotlib.pyplot as plt


# device to use for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model checkpoint location
CHECKPOINT = "local_data/all_reg.pt"

# csv log output location
LOG_LOC = "logs/all_reg.csv"
# graph output location
GRAFF = "logs/all_reg.png"

# model config class
CONFIG = configs.DefaultQuantumPolicy
ENV_NAME = "coinrun"

# number of concurrent environments
N_ENVS = 8
# number of passes through all envs to make per epoch
SHUFFLE_RUNS = 1
MAX_BUF_SIZE = 1024
MAX_EPISODE = 127

# length of each skill sequence
SKILL_LEN = CONFIG.skill_len

# MDP discount factor
DISCOUNT = 0.97
# divide rewards by this factor for normalization
R_NORM = 1

LAMBDA_SKILL = 1
LAMBDA_ENC = 3
LAMBDA_SEMISUP = 0

# model leaarning rate
LEARNING_RATE = 1e-4
# model batch size
BATCH_SIZE = 64

BASELINE = True

RECENT_DECAY = 0.8

# baseline hidden layer size
BASE_DIM = CONFIG.hidden_dim
# baseline number of hidden layers
BASE_LAYERS = CONFIG.num_layers
# baseline learning rate
BASE_LR = 1e-5
# baseline batch size
BASE_BATCH = BATCH_SIZE


class Environment:
    def __init__(
            self,
            env_name,
            num_envs,
            model,
            skill_len,
            shuffle_runs=1,
            max_buf_size=0,
            discount=1,
            device=DEVICE,
            action_handler=(lambda x: x),
            recent_decay=RECENT_DECAY
        ):
        """Handles the environment and data collection for reinforcement training.

        Args:
            env_name (str): Name of the gym environment
            num_envs (it): Number of concurrent environments
            model (LatentPolicy): Policy model to train
            discount (int, optional): MDP reward discount factor. Defaults to 1.
            device (torch.device, optional): Device to handle tensors on. Defaults to DEVICE.
        """

        # create environment
        self.env = env = ProcgenGym3Env(num=num_envs, env_name=env_name, distribution_mode='easy', use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True)
        self.num_envs = num_envs

        # store model reference
        self.model = model

        # basic params
        self.discount = discount
        self.skill_len = skill_len
        self.shuffle_runs = shuffle_runs
        self.max_buf_size = max_buf_size
        self.action_handler = action_handler
        self.recent_decay = recent_decay

        # device to store everything on
        self.device = device

        # should hold (s, a, r, d) tuples
        # temporal length in each tuple should be skill_len
        self.data = []

        # suffle outgoing indices during __getitem__
        self.shuffler = []

        self.recent_rewards = 0
        self.recent_div = 0


    def shuffle(self):
        """ Sample from the environment and shuffle the data buffer.
        """
        if len(self.data) > 0 and self.max_buf_size > 0:
            self.data = random.choices(self.data, k=min(len(self.data), self.max_buf_size))
        else:
            self.data = []

        self.recent_rewards *= self.recent_decay
        self.recent_div *= self.recent_decay
        self.recent_div += 1

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

        # keep track of which envs are done
        dones = np.zeros((self.num_envs,), dtype=bool)
        # hold previous rewards as references for return calculation
        prev_rewards = []

        prev_states = []
        temp_obs = self.env.observe()
        for i in range(3):
            prev_states.append(torch.tensor(temp_obs[1]['rgb']).permute(0, 3, 1, 2).to(DEVICE).float())

        # torch setup
        with torch.no_grad():
            self.model.eval()
            time = -1
        
            # run until all envs are done
            while True:

                seed_state = torch.cat(prev_states, dim=1)

                # get the chosen skill and set it in the model
                self.model.setChooseSkill(seed_state)

                # iterate through the skill sequence
                for t in range(self.skill_len):
                    time += 1

                    obs = torch.cat(prev_states, dim=1)

                    # sample an action using the current state
                    a = self.model.policy(obs)

                    # take a step in the environment, caching done to temp variable
                    self.env.act(a.squeeze().cpu().detach().numpy())

                    out = self.env.observe()

                    r, new_obs, curr_dones = out
                    new_obs = new_obs['rgb']
                    curr_dones |= time >= MAX_EPISODE
                    r /= R_NORM

                    r = torch.tensor(r).to(self.device)
                    r[dones] = 0
                    prev_rewards.append(r)

                    for i in range(self.num_envs):
                        if not dones[i]:
                            self.data.append(((obs[i], seed_state[i]), (a[i], r[i])))

                    prev_states.pop(0)
                    prev_states.append(torch.tensor(new_obs).permute(0, 3, 1, 2).to(DEVICE).float())

                    dones = np.logical_or(curr_dones, dones)

                    self.pbar.set_postfix({"t": time, "done": float(dones.sum())/self.num_envs})

                if dones.all():
                    break

            # propogate rewards backwards through time to replace referenced rewards with returns
            for tau in range(2, 1+len(prev_rewards)):
                # dones are accounted for by having r=0
                prev_rewards[-tau] += self.discount * prev_rewards[-tau+1]

        self.recent_rewards += prev_rewards[0].sum().item() / (self.shuffle_runs * self.num_envs)

        return


    def getRecent(self):
        return self.recent_rewards / self.recent_div


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
        x = ([], [])
        y = ([], [], [])

        for i in indices:
            s, s_ = self.data[i][0]
            a, r = self.data[i][1]

            # maintain order in tuples
            x[0].append(s)
            x[1].append(s_)

            y[0].append(s)
            y[1].append(a)
            y[2].append(r)

        return tuple(torch.stack(x[i]) for i in range(len(x))), tuple(torch.stack(y[i]) for i in range(len(y)))


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
        self.baseline = nn.Sequential(
            MobileNet(state_size),
            SkipNet(state_size, h_dim, 1, n_layers, dropout=0.1)
        )
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

        # iterate over batches
        for b in range(0, x.shape[0], self.batch_size):
            x_b, y_b = x[b:b+self.batch_size], y[b:b+self.batch_size]

            # calculate loss and perform backprop
            y_hat = self.baseline(x_b)
            loss = F.mse_loss(y_hat.view(-1).float(), y_b.view(-1).float())

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
            return torch.zeros_like((x.shape[0], 1))

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

        pi_logits, skill_logits, enc_logits, pi_outs, skill_outs = pred
        s, a, r = y

        # get the baseline prediction, squeeze to match r
        base = self.predict_baseline(s).squeeze(-1)
        assert base.shape == r.shape

        # train the baseline now that we are done with it
        self.train_baseline(s, r)

        # get the advantage from the reward and baseline
        A = r - base.detach()
        # unsqueeze to broadcast with pred_opt
        A = A.unsqueeze(-1).unsqueeze(-1)
        assert A.dim() == pi_logits.dim()

        pi_multed = torch.log_softmax(pi_logits, dim=-1) * A
        pi_chosen = pi_multed.view(-1, pi_multed.shape[-1])[range(a.numel()),a.view(-1)]
        pi_loss = -torch.mean(pi_chosen)

        skill_multed = torch.log_softmax(skill_logits, dim=-1) * A
        skill_chosen = skill_multed.view(-1, skill_multed.shape[-1])[range(a.numel()),a.view(-1)]
        skill_loss = -torch.mean(skill_chosen)
        
        enc_loss = F.cross_entropy(enc_logits, torch.arange(0, enc_logits.shape[0], dtype=torch.long).to(enc_logits.device))

        skill_target = torch.argmax(skill_outs, dim=-1)
        semisup_loss = F.cross_entropy(skill_outs, skill_target)

        return pi_loss + (LAMBDA_SKILL * skill_loss) + (LAMBDA_ENC * enc_loss) + (LAMBDA_SEMISUP * semisup_loss)


class EnvLogger(Logger):
    def __init__(self, env, log_loc, graff):
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

        # basic parameters
        self.log_loc = log_loc
        self.graff = graff

        # create metric file and write header
        try:
            with open(self.log_loc, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, dialect='excel')
                spamwriter.writerow(["epoch", "avg_reward", "skill_max", "enc_acc", "skill_kl", "pi_kl"])
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
        curr_r = self.env.getRecent()

        if train_log is not None:
            preds = train_log[0]

            # get decoder accuracy
            corr = 0
            tot = 0
            for i in range(len(preds)):
                corr += (torch.argmax(preds[i][2], dim=-1) == torch.arange(0, preds[i][2].shape[0], dtype=torch.long).to(preds[i][2].device)).sum()
                tot += preds[i][2].shape[0]
            acc = (corr / tot).item()

            # get skill max
            p = 0
            tot = 0
            for i in range(len(preds)):
                p += torch.softmax(preds[i][4], dim=-1).max(dim=-1)[0].sum()
                tot += preds[i][4].shape[0]
            skill_max = (p / tot).item()

            # get skill kl
            avg_skill = torch.zeros_like(preds[0][4][0])
            tot = 0
            for i in range(len(preds)):
                avg_skill += torch.sum(torch.softmax(preds[i][4], dim=-1), dim=0)
                tot += preds[i][4].shape[0]
            avg_skill /= tot
            kl = 0
            tot = 0
            for i in range(len(preds)):
                kl += F.kl_div(torch.log_softmax(preds[i][4], dim=-1), avg_skill.unsqueeze(0), reduction='sum')
                tot += preds[i][4].shape[0]
            skill_kl = (kl / tot).item()

            kl = 0
            tot = 0
            for i in range(len(preds)):
                avg_pi = torch.mean(torch.softmax(preds[i][3], dim=-1), dim=-3)
                avg_pi = torch.stack([avg_pi]*preds[i][3].shape[-3], dim=-3)
                kl += F.kl_div(torch.log_softmax(preds[i][3], dim=-1), avg_pi, reduction='sum')
                tot += preds[i][3].shape[0]*preds[i][3].shape[1]*preds[i][3].shape[2]
            pi_kl = (kl / tot).item()

        else:
            acc = 1/self.model.config.batch_keep
            skill_max = 1/self.model.config.num_pi
            skill_kl = None
            pi_kl = None

        # save metrics
        self.avg_rewards.append(curr_r)
        self.enc_accs.append(acc)
        self.skill_maxs.append(skill_max)
        self.skill_kls.append(skill_kl)
        self.pi_kls.append(pi_kl)

        # append metrics to csv file
        try:
            with open(self.log_loc, 'a') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                spamwriter.writerow([len(self.avg_rewards)-2, curr_r, skill_max, acc, skill_kl, pi_kl]) # -2 because of init call
        except:
            pass

        """ Plot the metrics """
        fig, ax = plt.subplots(3, 2)

        ax[0,0].plot(self.avg_rewards)
        ax[0,0].set_title(r"Average Reward")

        ax[1,1].plot(self.enc_accs)
        ax[1,1].set_title(r"Choice-State Identification Accuracy")

        ax[1,0].plot(self.skill_maxs)
        ax[1,0].set_title(r"Avg Max Choice")

        ax[2,0].plot(self.skill_kls)
        ax[2,0].set_title(r"Epoch-wise Choice Information Radius")

        ax[2,1].plot(self.pi_kls)
        ax[2,1].set_title(r"Step-Wise Inter-Policy Information Radius")

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


def main():

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # initialize the model that we will train
    model = QuantumPolicy(CONFIG)
    # model.load_state_dict(torch.load(INIT_MODEL))
    model.to(DEVICE)

    # initialize our training environment
    env = Environment(
        env_name = ENV_NAME,
        num_envs = N_ENVS,
        model = model,
        skill_len = SKILL_LEN,
        shuffle_runs = SHUFFLE_RUNS,
        discount = DISCOUNT,
        max_buf_size = MAX_BUF_SIZE,
    )

    # initialize the logger
    logger = EnvLogger(
        env = env,
        log_loc = LOG_LOC,
        graff = GRAFF
    )
    logger.initialize(model)

    # make an init call to the logger to save the before-training performance
    env.shuffle()
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


