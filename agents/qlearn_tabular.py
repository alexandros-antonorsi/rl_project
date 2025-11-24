import numpy as np
import gymnasium as gym
from collections import defaultdict

class QLearnAgent:
    
    def __init__(
            self, 
            env: gym.Env, 
            seed: int,
            learn_rate: float,
            discount: float,
            e_init: float,
            e_final: float,
            e_decay: float):
        self.env = env
        self.learn_rate = learn_rate
        self.discount = discount
        self.epsilon = e_init
        self.epsilon_final = e_final
        self.decay_rate = e_decay
        self.rng = np.random.default_rng(seed)
        
        #instead of instantiating for all state,action pairs, use defaultdict so we can only store values for observed states
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))


    def update(self, curr_obs, action, reward, next_obs, terminated):
        q_now = self.q_table[curr_obs][action]
        #if our action results in termination, our future rewards are 0 so we should remove that term from our approximation update
        target = reward + (1 - terminated) * self.discount * np.max(self.q_table[next_obs])
        self.q_table[curr_obs][action] = q_now - self.learn_rate * (q_now - target)


    #we use an epsilon-greedy policy. in Gymnasium environments the action space is the same for all states, so we don't need to store/update policy values per state
    def choose_action(self, curr_obs):
        if self.rng.random() < self.epsilon:
            return self.env.action_space.sample()
        else: 
            #break ties randomly
            max = np.max(self.q_table[curr_obs])
            choices = np.nonzero(self.q_table[curr_obs] == max)[0]
            return self.rng.choice(choices)
        
    def decay(self):
        self.epsilon = max(self.epsilon_final, self.epsilon-self.decay_rate)

