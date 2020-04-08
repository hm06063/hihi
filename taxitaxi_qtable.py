#!/usr/bin/env python
# coding: utf-8

# step 0 : import the dependencies
# agent 만들기 위해 필요한 libraries import.
# Numpy -> Qtable
# OpenAI Gym -> Taxi 환경
# Random -> Random 수 생성

# In[2]:


import numpy as np
import gym
import random


# step 1: create the environment
# OpenAI gym is a library composed of many environments that we can use to train our agents

# In[8]:


env = gym.make("Taxi-v3")
env.render()


# step 2 : create the Q-Table and intialize it
# rows(state) & columns(actions) 얼마나 필요한지 알기 위해서 action_size랑 state_size를 계산해야함
# OpenAI Gym -> env.action_space.n & env.observation_space.n

# In[9]:


action_size = env.action_space.n
print("Action size", action_size)

state_size = env.observation_space.n
print("State size", state_size)


# In[10]:


qtable = np.zeros((state_size,action_size)) # 위에서 정한 크기로 0으로 초기화
print(qtable)


# step 3 : Create the hyperparameters
# 이 변수들은 우리가 tune the training of our 알고리즘 하기 위해 필요

# In[19]:


total_episodes = 50000
total_test_episodes = 100
max_steps = 99 #Max steps per episode

learning_rate = 0.7
gamma = 0.618 #Discounting rate

#Exploration parameters
epsilon = 1.0 #Exploration rate 시간이 지날 수록 exploration<exploitation
max_epsilon = 1.0 #Exploration probability at start
min_epsilon = 0.01 #Minimum exploration probability
decay_rate = 0.01 #Exponential decay rate for exploration prob


# step 4 : The Q learning algorithm
# 1. 임의로 Initalize Q-values (Q(s,a)) for all state-action pairs
# 2. For life or until learning is stopped
# 3. Choose an action(a) in the current world state(s) based on current Q-values estimates (Q(s,-))
# 4. Take the action(a) and observe the outcome state (s') and reward (r)
# 5. Update Q(s,a) := Q(s,a) + a[r+rmaxQ(s',a')-Q(s,a)]
# 
# 우리가 exploration 상태라면 우리는 action을 random하게 받을 것이고, exploitation 상태라면 우리는 action을 그 sate에서 highest Q-value를 가진 애를 고를 것이다.
# 

# In[20]:


#2 For life or until learning is stopped
for episode in range(total_episodes):
    #Reset the environment
    state = env.reset()
    step = 0
    done = False
    
    #above max xtep -> episode must end (we take too much steps)
    for step in range(max_steps):
        #3. Choose an action in the current world state
        ##First we randomize a number
        exp_exp_tradeoff = random.uniform(0,1) #exploration_exploitation_tradeoff
        
        ## If this number > greater than epsilon => exploitation (taking the biggest Q-Value for this)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
            
        #Else doing a random choice => exploration
        else:
            action = env.action_space.sample()
            
        #Take the action (a) and observe the outcome state(s') and reward(r)
        new_state, reward, done, info = env.step(action)
        
        #Update Q(s,a):= Q(s,a) + lr{R(s,a) + gamma * maxQ(s',a') - Q(s,a)}
        qtable[state, action] = qtable[state,action] + learning_rate * (reward + gamma * 
                                        np.max(qtable[new_state, : ]) - qtable[state,action])
        
        #Our new state is state
        state = new_state
        
        #If done : finish episode
        if done == True:
            break
            
    episode += 1
    
    # Reduce epsilon ( cuz we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)


# step 5 : Use our Q-table to play taxi~!~
# 

# In[21]:


env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    print("*******************************")
    print("EPISODE ",episode)
    
    for step in range(max_steps):
        env.render()
        #Take the action(index) that have the max expected future reward given that state
        action = np.argmax(qtable[state,:])
        
        new_state, reward, done, info = env.step(action)
        
        total_rewards += reward
        
        if done :
            rewards.append(total_rewards)
            print("Score", total_rewards)
            break
        state = new_state
        
env.close()
print("Score over time: "+ str(sum(rewards)/total_test_episodes))

        


# In[ ]:




