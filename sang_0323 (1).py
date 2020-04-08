#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
import numpy as np
import tensorflow as tf
import random
import dqn
from collections import deque

import gym
from gym import wrappers

env = gym.make('CartPole-v0')

#신경망 정의하는 상수들
input_size = env.observation_space.shape[0] #네개의 정보
output_size = env.action_space.n #두개의 action

dis = 0.9
REPLAY_MEMORY = 50000
'''


# In[2]:


'''
class DQN:

  #Session 받아오기 중요!
  def __init__(self, session, input_size, output_size, name = "main"):
    self.session = session
    self.input_size = input_size
    self.output_size = output_size
    self.net_name = name

    self._build_network()

  #Network Build
  def _build_network(self, h_size=10, l_rate = 1e-1):
    with tf.variable_scope(self.net_name):
        self._X = tf.placeholder(
          tf.float32, [None,self.input_size], name = "input_x") 
      #입력사이즈 만큼 받기. None - 입력 한개~여러개 동시에 받기 위해서
      
      #First layer of Weights
        W1 = tf.get_variable("W1", shape=[self.input_size, h_size], 
                           initializer=tf.contrib.layers.xavier_initializer())
        layer1 = tf.nn.tanh(tf.matmul(self._X,W1))
      #tanh : activation function

      #Second layer of Weights
        W2 = tf.get_variable("W2", shape=[h_size, self.output_size], #원하는 outputsize
                           initializer=tf.contrib.layers.xavier_initializer())
      
      #마지막 layer는 activation function해주지 않아도 됨. Q가 linear regression이라서

      #Q prediction
        self._Qpred = tf.matmul(layer1, W2)

    #We need to define the parts of the network needed for learning a policy
    self._Y = tf.placeholder(
        shape=[None,self.output_size],dtype=tf.float32)
    #Y는 정답. Y를 hold할 수 있는 공간 생성!
  
    
    #Loss function
    self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
    #Learning 학습시킬 때마다 train호출!
    self._train = tf.train.AdamOptimizer(
        learning_rate = l_rate).minimize(self._loss)
    
  #상태를 받아서 실행시킨 다음에 결과를 받아주는 함수
    def predict(self,state):
        x = np.reshape(state,[1,self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

  #학습시키는 함수.
    def update(self,x_stack, y_stack):
        return self.session.run([self._loss, self._train],feed_dict={
            self._X: x_stack, self._Y: y_stack})
'''


# In[3]:


'''
#Train from Replay Buffer ## update는 메인, 읽어오는 것은 target!!
def replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty(0).reshape(0,input_size)
    y_stack = np.empty(0).reshape(0,output_size)

  #Get stored info from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

    #Terminal?
        if done: 
            Q[0,action] = reward
        else:
      #Q-value를 가져와(by feeding the new_state through our network)
            #Q[0,action] = reward + dis * np.max(targetDQN.predict(next_state))
            Q[0,action] = reward + dis * np.max(targetDQN.predict(next_state))

        y_stack = np.vstack([y_stack,Q])
        x_stack = np.vstack([x_stack,state])

    #Train our network using target and predicted Q values on each epsiode
    return mainDQN.update(x_stack, y_stack)
    
'''


# In[4]:


'''
#Code 5 : network(variable) copy
def get_copy_var_ops(*, dest_scope_name = "target", src_scope_name = "main"):
    #Copy variables src_scope to dest_scope
    op_holder =[]

    src_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope = src_scope_name)
    dest_vars = tf.get_collection(
      tf.GraphKeys.TRAINABLE_VARIABLES, scope = dest_scope_name)
  
    for src_var, dest_var in zip(src_vars,dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder
  '''


# In[5]:


'''
#bot play - 학습된 network 받아와서 환경 초기화후 계산 결과 보여줌
def bot_play(mainDQN):
  #See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:
        env.render()
        #k = mainDQN.predict(s)
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        reward_sum += reward
        if done:
            print("Total score :{}".format(reward_sum))
            break
'''


# In[2]:


'''
#main
def main():
    max_episodes = 5000

  #state the previous observations in replay memory
    replay_buffer = deque()

  #네트워크 생성
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, input_size, output_size, name = "main")
        targetDQN = dqn.DQN(sess, input_size, output_size, name = "target")
        tf.global_variables_initializer().run()

         #initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name = "target", 
                                src_scope_name = "main")
        sess.run(copy_ops)

        for episodes in range(max_episodes):
            e = 1. / ((episodes/10)+1)
            done = False
            step_count = 0
            state = env.reset()

            while not done:
                if np.random.rand(1) <e:
                    action = env.action_space.sample()
                else:
                      #Choose an action by greedily from the Q Network
                    action = np.argmax(mainDQN.predict(state))

                    #Get new state and reward from environment
                    next_state, reward, done, _ = env.step(action)
                    if done: #big penalty
                        reward = -100
        
        #save the 경험 to our buffer
                    replay_buffer.append((state, action, reward, next_state, done))
                    if len(replay_buffer)>REPLAY_MEMORY:
                      replay_buffer.popleft()

                    state = next_state
                    step_count += 1
                    if step_count > 10000: #충분
                        break
                    print("Episodes: {}, steps:{}".format(episodes, step_count))
                    if step_count>10000:
                        pass #break
        
                    if episodes % 10 == 1: #train every 10 episodes
                    #Get a random batch of experience
                        for _ in range(50):
                        #Minibatch works better
                            minibatch = random.sample(replay_buffer,10)
                            loss, _ = simple_replay_train(mainDQN, minibatch)
                        print("Loss : ", loss)

                        #copy q_net -> target_net
                        sess.run(copy_ops)
                
                #See our trained bot in action
               # env2 = wrappers.Monitor(env,'gym-results',force=True)

                for i in range(200):
                    bot_play(mainDQN)
                    #bot_play(mainDQN,env=env2)
                #env2.close()
                #gym.upload('gym-results',api_key='sk_VT2wPcss0ylnlPORltmQ')
                
if __name__ == "__main__":
    main()

'''


# In[3]:


"""
DQN (NIPS 2013)
Playing Atari with Deep Reinforcement Learning
https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""
import numpy as np
import tensorflow as tf
import random
import dqn
import gym
from collections import deque
 
env = gym.make('CartPole-v0')
#env = gym.wrappers.Monitor(env, 'gym-results/', force=True)
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n
 
DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
MAX_EPISODE = 5000
BATCH_SIZE = 64
 
# minimum epsilon for epsilon greedy
MIN_E = 0.0
# epsilon will be 

EPSILON_DECAYING_EPISODE = MAX_EPISODE * 0.01
 
def bot_play(mainDQN: dqn.DQN) -> None:
    """Runs a single episode with rendering and prints a reward
    Args:
        mainDQN (dqn.DQN): DQN Agent
    """
    state = env.reset()
    total_reward = 0
 
    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            print("Total score: {}".format(total_reward))
            break
 
 
def train_minibatch(DQN: dqn.DQN, train_batch: list) -> float:
    """Prepare X_batch, y_batch and train them
    Recall our loss function is
        target = reward + discount * max Q(s',a)
                 or reward if done early
        Loss function: [target - Q(s, a)]^2
    Hence,
        X_batch is a state list
        y_batch is reward + discount * max Q
                   or reward if terminated early
    Args:
        DQN (dqn.DQN): DQN Agent to train & run
        train_batch (list): Minibatch of Replay memory
            Eeach element is a tuple of (s, a, r, s', done)
    Returns:
        loss: Returns a loss
    """
    state_array = np.vstack([x[0] for x in train_batch])
    action_array = np.array([x[1] for x in train_batch])
    reward_array = np.array([x[2] for x in train_batch])
    next_state_array = np.vstack([x[3] for x in train_batch])
    done_array = np.array([x[4] for x in train_batch])
 
    X_batch = state_array
    y_batch = DQN.predict(state_array)
 
    Q_target = reward_array + DISCOUNT_RATE * np.max(DQN.predict(next_state_array), axis=1) * ~done_array
    y_batch[np.arange(len(X_batch)), action_array] = Q_target
 
    # Train our network using target and predicted Q values on each episode
    loss, _ = DQN.update(X_batch, y_batch)
 
    return loss
 
def annealing_epsilon(episode: int, min_e: float, max_e: float, target_episode: int) -> float:
    """Return an linearly annealed epsilon
    Epsilon will decrease over time until it reaches 

         (epsilon)
             |
    max_e ---|\
             | \
             |  \
             |   \
    min_e ---|____\_______________(episode)
                  |
                 target_episode
     slope = (min_e - max_e) / (target_episode)
     intercept = max_e
     e = slope * episode + intercept
    Args:
        episode (int): Current episode
        min_e (float): Minimum epsilon
        max_e (float): Maximum epsilon
        target_episode (int): epsilon becomes the  

    Returns:
        float: epsilon between  

    """
 
    slope = (min_e - max_e) / (target_episode)
    intercept = max_e
 
    return max(min_e, slope * episode + intercept)
 
def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    last_100_game_reward = deque(maxlen=100)
 
    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
        init = tf.global_variables_initializer()
        sess.run(init)
 
        for episode in range(MAX_EPISODE):
            e = annealing_epsilon(episode, MIN_E, 1.0, EPSILON_DECAYING_EPISODE)
            done = False
            state = env.reset()
 
            step_count = 0
            while not done:
 
                if np.random.rand() < e:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(mainDQN.predict(state))
 
                next_state, reward, done, _ = env.step(action)
 
                if done:
                    reward = -1
 
                replay_buffer.append((state, action, reward, next_state, done))
 
                state = next_state
                step_count += 1
 
                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    train_minibatch(mainDQN, minibatch)
 
            print("[Episode {:>5}]  steps: {:>5} e: {:>5.2f}".format(episode, step_count, e))
 
            # CartPole-v0 Game Clear Logic
            last_100_game_reward.append(step_count)
            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)
                if avg_reward > 199.0:
                    print("Game Cleared within {} episodes with avg reward {}".format(episode, avg_reward))
                    break
 
 
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




