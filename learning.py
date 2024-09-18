from turtle import end_fill
import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import time
import math
import scipy.spatial.distance as ssd

total_episodes = 2000         # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 6000              # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.001            # Exponential decay rate for exploration prob

env_MQL = gym.make("myFirstEnv-v0")

action_size = env_MQL.action_space.nvec
state_size = env_MQL.observation_space.n

qtable_MQL = {}
for s in range(state_size):
    row = {}
    qtable_MQL[s] = row
    for ai in range(action_size[0]):
        for aj in range(action_size[1]):
            row[(ai,aj)]= 0



# List of rewards
rewards_MQL = []

for episode in range(total_episodes):
    # Reset the environment
    state = env_MQL.reset()
    step = 0
    total_rewards_MQL = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = max(qtable_MQL[state], key=qtable_MQL[state].get)


        # Else doing a random choice --> exploration
        else:
            action = env_MQL.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env_MQL.step(action)

        qtable_MQL[state][(action[0],action[1])] = (qtable_MQL[state][(action[0],action[1])] + learning_rate * (reward + gamma * max(qtable_MQL[new_state].values()) - qtable_MQL[state][(action[0],action[1])]))

        if reward > total_rewards_MQL:
            total_rewards_MQL = reward

        # Our new state is state
        state = new_state
  

        
    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards_MQL.append(total_rewards_MQL)




env_QL = gym.make("myEnvNActions-v0")

action_size = env_QL.action_space.nvec
state_size = env_QL.observation_space.n

qtable_QL = {}
for s in range(state_size):
    row = {}
    qtable_QL[s] = row
    for ai in range(action_size[0]):
        for aj in range(action_size[1]):
            row[(ai,aj)]= 0


# List of rewards
rewards_QL = []

for episode in range(total_episodes):
    # Reset the environment
    state = env_QL.reset()
    step = 0
    total_rewards = 0
    
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = max(qtable_QL[state], key=qtable_QL[state].get)


        # Else doing a random choice --> exploration
        else:
            action = env_QL.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env_QL.step(action)

        qtable_QL[state][(action[0],action[1])] = (qtable_QL[state][(action[0],action[1])] + learning_rate * (reward + gamma * max(qtable_QL[new_state].values()) - qtable_QL[state][(action[0],action[1])]))
        
        if reward > total_rewards:
            total_rewards = reward

        # Our new state is state
        state = new_state
  

        
    episode += 1
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    rewards_QL.append(total_rewards)





users = []
users.append(np.array([[1, 1, 0]]))
users.append(np.array([[3, 3, 0]]))
users.append(np.array([[2, 2, 0]]))
users.append(np.array([[1, 2, 0]]))

x_n = 4 
y_n = 4 
location_size = (x_n, y_n)  
rotation_size = 181 

Rtable = {}
for x in range(location_size[0]):
    for y in range(location_size[1]):
        row = {}
        Rtable[(x,y)] = row
        for angle in range(rotation_size):
            row[angle]= 0

def calculate_reward(x_coordinate, y_coordinate, orientation):
    distance = []
    cos_phi = []
    phi = []
    cos_sigma_ori = []
    sigma_ori = []
    sigma_final = []
    cos_sigma = []
    h = []
    dr =[]
    for i in range(4):
        distance.append(ssd.euclidean([x_coordinate, y_coordinate, 10],[users[i]]))
        cos_phi.append((10)/distance[i])
        phi.append(math.acos(cos_phi[i]))
        cos_sigma_ori.append(cos_phi[i])
        sigma_ori.append(math.acos(cos_sigma_ori[i]))
        sigma_final.append(abs(sigma_ori[i] - orientation*(math.pi/180)))
        cos_sigma.append(math.cos(sigma_final[i]))
        h.append((((1+1)*1)/(2*math.pi*(distance[i]*distance[i])))*(((1.5*1.5)/((math.sin(math.pi/3))*(math.sin(math.pi/3)))))*(cos_sigma[i])*cos_phi[i])
        dr.append((20*(10**6))*math.log2(1+ (math.exp(1)/(2*math.pi))*((h[i])**2)))
    reward = sum(dr)
    return reward


for x in range(location_size[0]):
    for y in range(location_size[1]):
        for angle in range(rotation_size):
            Rtable[(x,y)][angle] = calculate_reward(x, y, angle)



# print('xxxxx', Rtable)
res = {key: max(val.values()) for key, val in Rtable.items()}

# print("The modified dictionary : " + str(res)) 

new_val = res.values()
maximum_val = max(new_val)