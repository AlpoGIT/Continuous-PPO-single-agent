## POLICY GRADIENT -- PPO ##
import itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import gym
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from network import  actor_critic_continuous, storage 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

"""Define environment"""
# need to be in p2_continuous-control
from unityagents import UnityEnvironment
env = UnityEnvironment(file_name="C:\\Users\AL\Documents\GitHub\deep-reinforcement-learning\p2_continuous-control\Reacher_Windows_x86_64\Reacher.exe", no_graphics=True)

target_score = 30

brain_name = env.brain_names[0]         # get default brain
brain = env.brains[brain_name]

state_dim = 33
action_dim = 4

time_limit = 2000           
keep_print_every = 50
print_every = 1

# Current best params
# c1=0.005, eps=0.3, K=8, alpha=3e-4, default init

"""Define hyperparameters"""
c_1 = 0.5                 # critic loss weigth
optimization_epoch = 8      # gradient descent epoch K
eps = 0.2                   # PPO clip epsilon
alpha = 3e-4                # Adam learning rate
beta = 0.01
grad_clip = 0.5
rollout_length = 2048       # horizon
mini_batch_size = 64

GAE_tau = 0.95
discount = 0.99

''' Adam decay, not used
ln_alpha_i =  -8.11 #for 3e-4, -6.9 for 1e-3
ln_alpha_f = -9.21
alpha_time_limit = time_limit
alpha_scheduler = np.array([(ln_alpha_f-ln_alpha_i)*x/alpha_time_limit+ln_alpha_i for x in np.arange(time_limit)])
alpha_scheduler = np.exp(alpha_scheduler)
'''

"""Define utils"""
buffer_size = rollout_length
memory = storage(buffer_size, mini_batch_size)

network = actor_critic_continuous(state_dim, action_dim).to(device)
#network.linearly_anneal(0, time_limit)          # initialize standard deviation

optim = torch.optim.Adam(network.parameters(), lr=alpha, betas=(0.9,0.999), eps=1e-8)

scores_deque = deque(maxlen=100)
max_score = - 99999
av_score = 0
scores = []
score = 0

rewards = deque(maxlen=rollout_length)
dones = deque(maxlen=rollout_length)
returns = deque(maxlen=rollout_length)
deque_advantages = deque(maxlen=rollout_length)
values = deque(maxlen=rollout_length+1)         # there's a +1 to stock value[i+1]

########################################################
#               PPO algo                               #
########################################################
env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
state = env_info.vector_observations[0]             # get the current state

t = -1
episode = 0

while episode < time_limit:
    t += 1

    with torch.no_grad():
        torch_state = torch.tensor(state, dtype=torch.float).to(device)
        action, proba, _ = network.act(torch_state)

        if t % rollout_length == 0 and t > 1: # already calculated
            value = values[-1]
        else:
            value = network.value(torch_state).cpu().numpy()
    
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0] 
    done = env_info.local_done[0] 

    score += reward

    memory.add(
                state, 
                action, 
                reward, 
                done, 
                proba.detach().cpu().numpy(), 
                value
                )
    rewards.append(reward)
    values.append(value)
    dones.append(done)  

    if done:
        episode += 1

        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        next_state = env_info.vector_observations[0]            # get the current state

        scores_deque.append(score)
        scores.append(score)

        if score > max_score:
            max_score = score

        score = 0

        if episode % print_every == 0:
            av_score = np.mean(scores_deque)
            print('\r{}/{}\t average score: {:.2f}\tmax deque: {:.2f}'\
                .format(episode, time_limit, av_score, np.max(scores_deque) ), end ='')
        if episode % keep_print_every == 0:
            av_score = np.mean(scores_deque)
            print('\r{}/{}\t average score: {:.2f}\tmax score: {:.2f}\tsigma: {}'\
                .format(episode, time_limit, av_score, max_score, F.softplus(network.sigma)))
            np.savetxt('scores.txt', scores, fmt='%f')
        if np.mean(scores_deque) > target_score :
            print("\nsolved in {} episodes.".format(episode), end ='')
            torch.save(network.state_dict(), 'network_parameters.pth')
            np.savetxt('scores.txt', scores, fmt='%f')
            break
    #
    state = next_state

    if (t+1) % rollout_length == 0 :    # t begins at 0

        ''' Not used here. Attempt to rescale returns
        #--------------------------------------
        # rescale rewards
        gammas = [discount**(rollout_length-i) for i in np.arange(1,rollout_length+1)]
        #rewards = np.array([1,2,3,4])  # test
        reshaped = [ np.sum(np.asarray(rewards)[0:i]*gammas[rollout_length-i:]) for i in np.arange(1,rollout_length+1) ]
        std_reshaped = np.array([np.std(reshaped[0:i]) for i in np.arange(1, rollout_length+1)])

        reshaped = np.asarray(rewards)/(std_reshaped+1e-5)
        reshaped = np.clip(reshaped, -10, 10)
        #--------------------------------------
        '''

        with torch.no_grad():
            torch_state = torch.tensor(state, dtype=torch.float).to(device).detach()
            values.append(network.value(torch_state).cpu().numpy())
        #ret = values[-1]
        advantage = 0

        for i in reversed(range(rollout_length)):
            #ret = rewards[i] + discount * (1-dones[i]) * ret
            TD_error = rewards[i] + discount * (1-dones[i])*values[i+1] - values[i]
            advantage = advantage * GAE_tau * discount * (1-dones[i]) + TD_error
            deque_advantages.appendleft(advantage.item())
            #returns.appendleft(ret)    # a possible estimate for V_target
            returns.appendleft(advantage+values[i]) # better V_target

        #
        # Usual normalization of advantages
        advantages = (deque_advantages - np.mean(deque_advantages)) /  (np.std(deque_advantages) + 1e-8)
   
        ''' Not used: Adam decay
        optim = torch.optim.Adam(
                        network.parameters(),
                        lr=alpha_scheduler[episode],
                        betas=(0.9,0.999),
                        eps=1e-8
                        )
        '''

        for k in range(optimization_epoch):
            batches = memory.sample(advantages, returns)
        
            for batch in batches: 
                batch_states, batch_actions, _ , _ , old_probas, batch_values , batch_returns, batch_advantages= batch 
                new_probas, entropies = network.give_log_prob(batch_states, batch_actions)

                # PPO loss
                ratio = (new_probas - old_probas).exp()
                ppo_clipped = ratio.clamp(1 - eps, 1 + eps) * batch_advantages
                ppo_not_clipped = ratio * batch_advantages
                policy_loss = - torch.min(ppo_not_clipped, ppo_clipped).mean()

                entropy_penalty = - entropies.mean() 

                # PPO-like value loss: keep V_\theta not too far from previous value
                v_theta = network.value(batch_states)
                v1 =  (batch_returns - v_theta).pow(2)
                v2 = (batch_values + (v_theta-batch_values).clamp(-eps, eps) - batch_returns).pow(2)
                value_loss = torch.max(v1,v2).mean()
              
                # vanilla value loss
                #value_loss = v1.mean()

                loss = policy_loss + c_1 * value_loss + beta * entropy_penalty

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
                optim.step()
            #
        #
        # clear storage and linearly anneal std deviation
        #network.linearly_anneal(episode, time_limit)
        memory.clear()
        
    #
#

"""Print results"""
average = []
scores_deque = deque(maxlen=100)

for x in scores:
    scores_deque.append(x)
    average.append(np.mean(scores_deque))
goal = [target_score  for x in range(len(average))]

plt.plot(scores, 'b-', label = 'score')
plt.plot(average, 'r-', label = 'average score')
plt.plot(goal, 'k--', label = 'goal')
plt.xlabel('trajectory #')
plt.ylabel('score')
plt.legend()
plt.show()