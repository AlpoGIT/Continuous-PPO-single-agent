import numpy as np
import torch, torch.nn as nn
from collections import deque, namedtuple
import random
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class actor_critic_continuous(nn.Module):
    def __init__(self, state_dim, action_dim, fc1=64, fc2=64):
        super(actor_critic_continuous, self).__init__()
        
        self.actor =  nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, action_dim),
                                    nn.Tanh()
                                    )
        self.critic = nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, 1)
                                    )
        '''
        def init_weight(m):
            if type(m) == nn.Linear:
                #nn.init.xavier_normal_(m.weight.data)
                nn.init.orthogonal_(m.weight.data)
                #if m.bias is not None:
                    #nn.init.normal_(m.bias.data)

        self.actor.apply(init_weight)
        self.critic.apply(init_weight)
        '''

        self.action_dim = action_dim
        self.sigma = nn.Parameter(torch.zeros(action_dim))

    def act(self, state):
        mean = self.actor(state)
        d = torch.distributions.Normal(mean, nn.functional.softplus(self.sigma))
        action = d.sample()
        log_prob = d.log_prob(action)

        return action.cpu().numpy() , log_prob.sum(-1).unsqueeze(-1), d.entropy().sum(-1).unsqueeze(-1)
    '''
    def linearly_anneal(self, episode, time_limit):
        T = time_limit/2
        initial_value = -0.7
        final_value = -1.6 
        log_sigma = np.maximum((final_value- initial_value)*(episode/T)+initial_value, final_value)

        # std deviations not learned
        self.sigma = torch.full((1, self.action_dim), np.exp(log_sigma)).detach().to(device)
     '''   
    def value(self, state):

        return self.critic(state)

    def give_log_prob(self, state, action):
        mean = self.actor(state)
        d = torch.distributions.Normal(mean, nn.functional.softplus( self.sigma))
        log_prob = d.log_prob(action)

        return log_prob.sum(-1).unsqueeze(-1), d.entropy().sum(-1).unsqueeze(-1)

class storage: 
    """Memory based on previous replay buffer, but need some cleaning"""
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "done", "proba",  "value"])
        
    def add(self, state, action, reward, done, proba, value):
        e = self.experience(state, action, reward, done, proba, value)
        self.memory.append(e)
    
    def sample(self, advantages, returns):
        batch_size = self.batch_size
        all_batch = []
        indices = np.random.permutation(np.arange(self.__len__())) 
        batch_number = int(self.__len__() / self.batch_size)

        for k in np.arange(batch_number-1):
            experiences = [self.memory[i] for i in indices[k*batch_size : (k + 1)*batch_size]]
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
            rewards = None  # torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
            probas = torch.from_numpy(np.vstack([e.proba for e in experiences if e is not None])).float().to(device)
            values = torch.from_numpy(np.vstack([e.value for e in experiences if e is not None])).float().to(device)
            batch_returns = torch.from_numpy(np.vstack( [returns[i] for i in indices[k*batch_size : (k+1)*batch_size]])).float().to(device)
            batch_advantages = torch.from_numpy(np.vstack( [advantages[i] for i in indices[k*batch_size : (k+1)*batch_size]])).float().to(device)

            batch = (states, actions, rewards, dones, probas, values, batch_returns, batch_advantages)
            all_batch.append(batch)
        
       #last batch is longer
        experiences = [self.memory[i] for i in indices[(batch_number-1)*batch_size:]]
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions =  torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = None  #torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        probas = torch.from_numpy(np.vstack([e.proba for e in experiences if e is not None])).float().to(device)
        values = torch.from_numpy(np.vstack([e.value for e in experiences if e is not None])).float().to(device)
        batch_returns = torch.from_numpy(np.vstack( [returns[i] for i in indices[(batch_number-1)*batch_size:]])).float().to(device)
        batch_advantages = torch.from_numpy(np.vstack( [advantages[i] for i in indices[(batch_number-1)*batch_size:]])).float().to(device)

        batch = (states, actions, rewards, dones, probas, values, batch_returns, batch_advantages)
        all_batch.append(batch)

        return all_batch

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()