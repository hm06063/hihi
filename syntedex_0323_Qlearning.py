#!/usr/bin/env python
# coding: utf-8

# In[12]:


import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset() #꼭 해줘야함

LEARNING_RATE = 0.1 # 학습률. (상수) 0~1사이값으로 지정하면 됨. 일반적으로는 학습률을 저하시켜
DISCOUNT = 0.95 #weight(가중치) -> measure of how important do we find future actions over current actions basically or future reward vs current reward
#cuz the way the agent work always going to go off the max Q value 항상 max 큐 값으로 가버림.
#your max Q vqlue도 always looking ahead to future max Q values -> that's gonna back propagate all the way down for a long~ chain
#어쨌든 저 DISCOUNT 값 = how much we value future reward over current reward. 0~1사이 값.
EPISODE = 2000

SHOW_EVERY = 500 )

#measure of how much random this you want to do or how much exploration you want to do
epsilon = 0.5 #chance. 0~1. 높을 수록 perform a random action&exploratory it - 시간지날수록 감소시켜야됨 exploitation비율높이게
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODE//2 #// - int로 나누기

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)# amount that we want to decay by 각 에피소드 

#discrete하게 만들어줌. 최고점에서 최저점까지 20바이 20으로 나눠주기
#DISCRETE_OS_SIZE = [20,20] - 감당 가능한 크기로 바꿔줌. 항상 같은 크기일 필요는 ㅇ없음.
DISCRETE_OS_SIZE = [20]*len(env.observation_space.high) #functional하고 모든 환경에서 가능하게 하고싶기 때문에.

discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

#print(discrete_os_win_size) 

#finally create the Q table!!
# 우리는 액션이 3개가 있어(0~2). 이제 20x20의 combination이 잇어
# what are the Q values for these actions' combination
# what's highest values ? 
# over time, agents gonna explore and pick random value. using the Q func 
#-> slowly back propagate to change Q value to lovely reward
#  그전에 초기화를 해줘야대

#DISCRETE~SIZE는 20바이20이엇t어 -> contains every combinations of position, velocity.
# every single action을 보기 위해 +[env.action_space.n]
q_table = np.random.uniform(low=-2,high=0,size = (DISCRETE_OS_SIZE)+[env.action_space.n]) #these are two variables . 
# reward는 항상 음수. 
#도착하면 reward 0될거서 테이블을 음수로 만드는 거
#print(q_table.shape)

ep_rewards = [] # 각 에피소드 reward저장
aggr_ep_rewards = {'ep':[],'avg':[], 'min':[],'max':[]}
 #'ep':gonna track the episode number basically XY역할
 #'avg':will trailing average. average for any given window so every 500 episodes for ex. average over time so as our model imporves average should go up
 #'min : track for every what was worst model we had
 #'max' : best one
 # avg might actually be going up but the min or worst performing model is still in the dump
 # and so you might have the cases where you actually prefer that the 최악의 모델이 still somewhat decent then to have highest avg or something like that


#q table 이산적으로  만들어주는 함수
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODE):#iterate
    episode_reward = 0
    
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
        
    discrete_state = get_discrete_state(env.reset()) #env.step() return 4 values, env.reset() returns just initial state
    print(discrete_state)
    print(np.argmax(q_table[discrete_state])) #Q values. starting values (random and meanlingless) 


    done = False 

    while not done: #환경통해 step하기 위해선 action 필요

        if np.random.random()>epsilon:#random값이 epsilon값 보다 크면 q값중에 젤 큰거 갖고와!
            action = np.argmax(q_table[discrete_state]) 
        else: 
            action = np.random.randint(0,env.action_space.n)
    
        new_state,reward,done,_ = env.step(action) #step할 때마다 환경으로부터 new state를 받음.
  
        episode_reward+=reward


        new_discrete_state = get_discrete_state(new_state)
  # print(reward,new_state) #1
        if render:
            env.render() 
 
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) #왜 argmax말고 max썼냐? gonna use max future Q in our new Q 식이므로 Arg max가 뭔지 알기보다는 그 max값을 알고싶은 거얌
      #Q value gets back propagated down that down the table so  여기가 중요한 곳~!~!
            current_q = q_table[discrete_state+(action, )]#grab the current Q value
      #discrete_state만 하면 값 3개나오고 action 하면 the Q value 나옴

      #모든 Q값 계산하는 식. DISCOUNT * max_future_q가 back propagate하는 방향에 근거함.
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT  *max_future_q) 
      #새 q값으로 q 테이블 업뎃
            q_table[discrete_state + (action, )] = new_q 
        elif new_state[0] >= env.goal_position:#new_state[0] - have position&velocity
            q_table[discrete_state + (action, )] = 0 # -> reward for completing things(no punishment)

        discrete_state = new_discrete_state
  
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)

#우리 이제 dictionary 쓸 거니까 calculate the avg reard

    #if not episode%10:
      #  np.save(f"qtables/{episode}-qtable.py",q_table)
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:]) #-SHOW EVERY : just means like the last 500
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Epsiode: {episode} avg:{average_reward} min:{min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])}")



env.close()
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label="avg")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label="min")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label="max")
plt.legend(loc=4) # location  4: lower right
plt.show()



# In[ ]:




