import gymnasium as gym
import numpy as np
import argparse
import pandas as pd
from datetime import date
from src.agents.qlearn_tabular import QLearnAgent

#input training arguments and hyperparameters from CLI
parser = argparse.ArgumentParser()
parser.add_argument("--episodes", help="number of episodes to train", type=int, default=int(1e5))
parser.add_argument("--seed", help="seed for rng", type=int, default=25)
parser.add_argument("--learn", help="learning rate", type=float, default=0.1)
parser.add_argument("--e_init", help="initial exploration rate", type=float, default=1)
parser.add_argument("--e_final", help="final exploration rate", type=float, default=0)
parser.add_argument("--discount", help="discount rate", type=float, default=0.99)
args = parser.parse_args()
n_eps, seed, learn_rate, e_init, e_final, discount = int(args.episodes), args.seed, args.learn, args.e_init, args.e_final, args.discount

#set up env and log, make agent
env = gym.make('FrozenLake-v1', is_slippery=True)
#env.metadata['render_fps']=240
env.reset(seed=seed)
env.action_space.seed(seed)
log = pd.DataFrame(columns=['total_reward', 'episode_length'], index =[i+1 for i in range(n_eps)])
log.index.name='episode_num'

#linearly decay over 90% of episodes
e_decay = (e_init-e_final) / (0.9 * n_eps) 

qlearn = QLearnAgent(env, seed, learn_rate, discount, e_init, e_final, e_decay)

for i in range(n_eps):
    #iteration variables
    obs, info = env.reset()
    terminated = False
    truncated = False

    #logging variables
    total_reward = 0
    episode_length = 0

    while not (terminated or truncated):
        action = qlearn.choose_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        qlearn.update(obs, action, reward, next_obs, terminated)

        obs = next_obs

        total_reward += reward
        episode_length += 1

    qlearn.decay()
    
    episode_log = [total_reward, episode_length]
    log.iloc[i] = episode_log

env.close()

log.to_csv(f"results/frozenlake_qlearn_{date.today()}_seed{seed}.csv")

print(f"Finished training {n_eps} episodes on FrozenLake-v1 (seed={seed})")
print("Average reward over last 100 eps: ",np.sum(log['total_reward'][-100:])/100)
print(f"Results saved to results/frozenlake_qlearn_{date.today()}_seed{seed}.csv")

