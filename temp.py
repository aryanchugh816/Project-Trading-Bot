from keras.models import Sequential
from keras.layers import *
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import gym
import gym_trading_bot

class Agent:
    
    def __init__(self, state_size=3, action_size=3):
        
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        
        self.model = self._create_model_action_predict()
        
    def _create_model_action_predict(self):
        
        model = Sequential()
        model.add(LSTM(4,input_shape=(self.state_size,1), return_sequences=True))
        #model.add(Dropout(0.3))
        model.add(LSTM(4))
        model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation="softmax"))
        model.compile(loss="mean_squared_error", optimizer='adam', metrics=['acc'])
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        
        if np.random.rand() <= self.epsilon:
            
            return random.randrange(self.action_size)
        
        return np.argmax(self.model.predict(state)[0])
    
    def train(self, batch_size=32):
        
        minibatch = random.sample(self.memory, batch_size)
        
        for experience in minibatch:
            
            state, action, reward, next_state, done = experience
            
            if not done:
                
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
                
            else:
                
                target = reward
                
            target_f = self.model.predict(state)
            target_f[0][action] = target
            
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        self.model.load_weights(name)
        
    def save(self, name):
        self.model.save_weights(name)

n_episodes = 500
output_dir = 'trading_bot/'

done = False
state_size = 3
action_size = 3
batch_size = 32
agent = Agent(state_size, action_size)


env = gym.make('TradingBot-v0')

for e in range(n_episodes): # this game is won if we complete 200 episodes
    state = env.reset()
    state = np.reshape(state, [1,state_size,1]) 
    # We reshape it because we have to pass it to our neural network as an example
    
    for time in range(5000): # we choose a value higher than 200 as this game completes we if survive for 
                            # 200 time-steps
        env.render()
        action = agent.act(state)
        next_state,reward,done,other_info = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state,[1,state_size,1])
        agent.remember(state,action,reward,next_state,done) # Gaining Experience
        state = next_state
        
        if done:
            print("Game Episode :{}/{} High Score :{} Exploration Rate:{:.2}\n--------------------------------------".format(e+1,n_episodes,time,agent.epsilon))
            print("Profit: {}\nLoss: {}".format((max(0,(env.balance-env.initial_balance_b))), (min(0, (env.balance-env.initial_balance_b)))))
            break
            
    
    if len(agent.memory)>batch_size:
        agent.train(batch_size)
        
    # We will save our updated model weights every 50 game episodes
    if e%50 == 0:
        agent.save(output_dir+"weights_"+'{:04d}'.format(e)+".hdf5")
        
print("Deep Q-Learner Model Trained")
env.close()