import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import gymnasium as gym
from collections import namedtuple

class DQN:

    

    def __init__(
            self, 
            env: gym.Env, 
            rng: random.Random,
            e_init: float,
            learn_rate: float,
            batch_size: int,
            device: torch.device):
        self.env = env
        self.rng = rng
        self.epsilon = e_init
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.device = device

        num_actions = env.action_space.n 
        dim_states = env.observation_space.shape[0] 

        self.target_network = nn.Sequential(
            nn.Linear(dim_states, batch_size),
            nn.ReLU(),
            nn.Linear(batch_size, batch_size),
            nn.ReLU(),
            nn.Linear(batch_size, num_actions)
        ).to(device)

        self.main_network = nn.Sequential(
            nn.Linear(dim_states, batch_size),
            nn.ReLU(),
            nn.Linear(batch_size, batch_size),
            nn.ReLU(),
            nn.Linear(batch_size, num_actions)
        ).to(device)

        self.target_network.load_state_dict(self.main_network.state_dict())

        self.optimizer = optim.AdamW(self.main_network.parameters(), lr=self.learn_rate, amsgrad=True)


    
    def choose_action(self, curr_obs, epsilon):
        if self.rng.random() < epsilon:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.main_network(curr_obs).max(1).indices.view(1,1) #takes argmax of the output and reshapes it to a 1x1 tensor


    def optimize(self, batch, discount):
        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_obs)), dtype=torch.bool, device=self.device) #indices of states which are not terminal
        non_terminal_next_obs = torch.cat([s for s in batch.next_obs if s is not None])

        curr_obs_batch = torch.cat(batch.curr_obs)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values = self.main_network(curr_obs_batch).gather(1, action_batch) #the Q(s,a) pairs for each state, action pair in the batch
       
        #expected reward from future states is 0 if state is terminal, otherwise we compute via target network
        next_obs_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_obs_values[non_terminal_mask] = self.target_network(non_terminal_next_obs).max(1).values 
        
        target_values = reward_batch + discount * next_obs_values

        huber = nn.SmoothL1Loss() #Huber loss function
        loss = huber(q_values, target_values.unsqueeze(1))
        

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.main_network.parameters(), 1)
        self.optimizer.step()

        


    def soft_update(self, update_rate):
        target_dict = self.target_network.state_dict()
        main_dict = self.main_network.state_dict()
        for key in main_dict:
            target_dict[key] = update_rate * main_dict[key] + (1 - update_rate) * target_dict[key]
        self.target_network.load_state_dict(target_dict)

