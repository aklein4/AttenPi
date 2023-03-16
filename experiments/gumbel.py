
import torch
import numpy as np

N_SAMPLE = 30000

def sample(mu, B):
    count = torch.zeros_like(mu)
    dist = torch.distributions.gumbel.Gumbel(torch.zeros_like(B), B)
    for i in range(N_SAMPLE):
        count += mu + dist.sample()
    return count / N_SAMPLE

def main():
    pi = torch.tensor([1.0, 1.0, 1.0])
    pi /= pi.sum()

    B =  torch.tensor([1, 1.0, 1.0])

    print(sample(pi, B))


if __name__ == '__main__':
       main()