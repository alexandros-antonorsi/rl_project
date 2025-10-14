import gymnasium as gym
import numpy as np
import argparse
import pandas as pd
from datetime import date

#sets up argparse to initialize the experimental variables
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--episodes", help="number of episodes to run", type=int, default=5)
parser.add_argument("-s", "--seed", help="seed to use for rng", type=int, default=25)
args = parser.parse_args()
episodes, seed = args.episodes, args.seed

#set up environment, random seeding, and log dataframe
env = gym.make('CartPole-v1')
env.reset(seed=seed)
env.action_space.seed(seed)
np.random.seed(seed)
log = pd.DataFrame(columns=['total_reward', 'episode_length'], index=[i+1 for i in range(episodes)])
log.index.names=['episode_num']


for i in range(episodes):
    observation, info = env.reset()

    episode_over = False
    total_reward=0
    step_count = 0
    while not episode_over:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
        step_count+=1

    episode_log = [total_reward, step_count]
    log.loc[i+1]=episode_log

env.close()

log.to_csv(f"results/cartpole_random_{date.today()}_seed{seed}.csv")

print(f"Finished {episodes} episodes on CartPole-v1 (seed={seed})")
print("Mean reward: ",np.sum([log.loc[i+1, 'total_reward'] for i in range(episodes)])/episodes)
print(f"Results saved to results/cartpole_random_{date.today()}_seed{seed}.csv")