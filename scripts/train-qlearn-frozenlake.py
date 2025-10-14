import gymnasium as gym
import numpy as np
import argparse
import pandas as pd
from datetime import date
from ..src.agents.qlearn_tabular import QLearnAgent

#input training arguments and hyperparameters from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", help="number of episodes to train", type=int, default=1e5)
parser.add_argument("--seed", help="seed for rng", type=int, default=25)
parser.add_argument("--learn", help="learning rate", type=float, default=0.01)
parser.add_argument("--epsilon", help="exploration rate", type=float, default=0.1)
parser.add_argument("--discount", help="discount rate", type=float, default=0.95)
args = parser.parse_args()
n_eps, seed, learn_rate, epsilon, discount = args.episodes, args.seed, args.learn, args.epsilon, args.discount

#set up env and log, make agent
env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset(seed=seed)
env.action_space.seed(seed)
np.random.seed(seed)
log = pd.DataFrame(columns=['total_reward', 'episode_length', 'final_td_error'], index =[i+1 for i in range(n_eps)])
log.index.name='episode_num'

qlearn = QLearnAgent(env, learn_rate, discount, epsilon)

for i in range(n_eps):
    #iteration variables
    obs, info = env.reset()
    terminated = False
    truncated = False

    #logging variables
    total_reward = 0
    episode_length = 0

    while not terminated or truncated:
        action = qlearn.choose_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        qlearn.update(obs, action, reward, next_obs, terminated)

        obs = next_obs

        total_reward += reward
        episode_length += 1
    
    episode_log = [total_reward, episode_length, qlearn.td_errors[-1]]
    log.iloc[i] = episode_log

env.close()

log.to_csv(f"results/frozenlake_qlearn_{date.today()}_{seed}.csv")

print(f"Finished {n_eps} episodes on FrozenLake-v1 (seed={seed})")
print("Average reward over last 100 eps: ",np.sum(log['total_rewards'][:-100])/100)
print(f"Results saved to results/frozenlake_qlearn_{date.today()}_{seed}.csv")

    