import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import pandas as pd
from agents.dqn import DQN
import math

from collections import namedtuple, deque

#input training arguments and hyperparameters from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seed for rng", type=int, default=None)
parser.add_argument("--learn", help="learning rate", type=float, default=3e-4)
parser.add_argument("--e_init", help="initial exploration rate", type=float, default=0.9)
parser.add_argument("--e_final", help="final exploration rate", type=float, default=0.01)
parser.add_argument("--e_decay", help="exploration decay factor (higher = slower decay)", type=int, default=2500)
parser.add_argument("--discount", help="discount rate", type=float, default=0.99)
parser.add_argument("--update", help="soft update rate", type=float, default=0.005)
parser.add_argument("--batch", help="batch size", type=int, default=128)
args = parser.parse_args()
seed, learn_rate, e_init, e_final, e_decay, discount, update_rate, batch_size = args.seed, args.learn, args.e_init, args.e_final, args.e_decay, args.discount, args.update, args.batch


Sample = namedtuple('Sample', 'curr_obs action reward next_obs')

class ReplayBuffer(object):
    
    def __init__(self, buffer_size, rng):
        self.buffer = deque([], maxlen=buffer_size)
        self.rng = rng

    def push(self, *args):
        self.buffer.append(Sample(*args))

    def get_batch(self, batch_size):
        return self.rng.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    


env = gym.make("CartPole-v1")

#establish universal seed
if seed is not None:
    rng = random.Random(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    torch.manual_seed(seed)
else:
    rng = random.Random()
    env.reset()

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)


if torch.cuda.is_available() or torch.backends.mps.is_available():
    n_eps = 300
else:
    n_eps = 50


#linearly decay over 90% of episodes

dqn = DQN(env, rng, e_init, learn_rate, batch_size, device)
buffer = ReplayBuffer(int(1e4), rng)
log = pd.DataFrame(columns=['total_reward', 'episode_length'], index =[i+1 for i in range(n_eps)])
log.index.name='episode_num'
epsilon = e_init
num_iter = 0
def decay(num_iter):
    return e_final + (e_init - e_final) * math.exp(-1.0 * num_iter / e_decay)

for i in range(n_eps):

    curr_obs, info = env.reset()
    curr_obs = torch.tensor(curr_obs, dtype=torch.float32, device=device).unsqueeze(0)
    terminated = False
    truncated = False

    #logging variables
    total_reward = 0
    episode_length = 0

    while not (terminated or truncated):
        action = dqn.choose_action(curr_obs, epsilon)

        next_obs, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0) if not terminated else None

        buffer.push(curr_obs, action, reward, next_obs)

        #optimize the DQN's main network
        if len(buffer) >= batch_size: 
            batch = buffer.get_batch(batch_size)
            batch = Sample(*zip(*batch)) #converts the batch of Samples into a single Sample of tuples containing the values for the entire batch
            dqn.optimize(batch, discount)


        #soft update of target network parameters from the current state of the main network parameters
        dqn.soft_update(update_rate)

        curr_obs = next_obs

        total_reward += reward.item()
        episode_length += 1
        num_iter +=1

    epsilon = decay(num_iter)

    episode_log = [total_reward, episode_length]
    log.iloc[i] = episode_log

env.close()

log.to_csv(f"logs/cartpole_dqn_{seed}.csv")


print(f"Finished training {n_eps} episodes on CartPole-v1 (seed={seed})")
print("Average reward over last 10 eps: ",sum(log['total_reward'][-10:])/10)
print(f"Results saved to logs/cartpole_dqn_{seed}.csv")

    





