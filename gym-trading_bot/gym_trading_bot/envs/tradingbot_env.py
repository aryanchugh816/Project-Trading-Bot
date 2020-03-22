import gym
from gym import spaces
import numpy as np
import pandas as pd
from Bot_Functions import *

class TradingBotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        # At the initialization as for starting balance and path for data file

        data_file="original"
        self.data = create_dataset(data_file)
        self.last_index = len(self.data[0])
        self.balance = 500000
        self.initial_balance = self.balance
        self.initial_balance_b = self.balance
        self.stock_volume_aquired = 0
        self.profit = 0
        self.loss = 0
        print("Environment Created\nInitail Balance: {}\nInitial Stock Volume held: {}\nInitail Profit: {}".format(self.balance,0,0))

        self.index = 0
        self.done = False
        self.other_info = {}
        self.state = self.data[0][self.index]
        self.action_space = spaces.Discrete(3)  
        # 0: Buy
        # 1: Hold
        # 2: Sell

    def step(self, action):
        
        # price of per stock:
        price = self.data[0][self.index][-1]
        loss_previous = self.loss
        profit_previous = self.profit

        self.initial_balance = self.balance

        if self.index == self.last_index-2:
            action=2

        if action == 0:

            # Deciding how much of the stocks we can purchase
            volume_purchasable = np.int(np.floor(self.balance/price))

            # for the case when we have more money and volume of stock are lesser 
            volumes_bought =  min(volume_purchasable, self.data[-1][self.index])

            # Updating balance for our purchase
            self.balance -= price*volumes_bought
            self.stock_volume_aquired += volumes_bought

        elif action == 2:

            # Selling whole volume of our stock at the current price
            self.balance += price*self.stock_volume_aquired
            self.stock_volume_aquired = 0

        # Changing and returning observation(next_state), reward, done, other_info
        self.index += 1
        self.state = self.data[0][self.index]

        diff = self.balance - self.initial_balance

        if self.balance < self.initial_balance:
            self.loss += abs(diff)
            self.profit -= abs(diff)
            if self.profit < 0:
                self.profit = 0

        else:
            self.profit += (diff)
            self.loss -= (diff)
            if self.loss < 0:
                self.loss = 0
        """
        if max(0, (self.balance - self.initial_balance)) == 0:
            self.loss += abs(min(0, (self.balance - self.initial_balance)))
            self.profit += max(0, (self.balance - self.initial_balance))
            
        self.loss += abs(min(0, (self.balance - self.initial_balance)))
        """
        reward = calculate_reward(self.loss,self.profit)

        if self.index == self.last_index-1:
            self.done = True

        return (self.state, reward, self.done, self.other_info)


    def reset(self):
        self.index = 0
        self.balance = 500000
        self.initial_balance = self.balance
        self.stock_volume_aquired = 0
        self.profit = 0
        self.loss = 0
        self.done = False
        self.other_info = {}
        self.state = self.data[0][self.index]

        # Returning the current state of our environment ==> in our case is the row of the data that we are at/ we havn't passed to our agent yet
        return (self.state)

    def render(self, mode='human', close=False):
        #print("Step: {}\nBalance: {}\nProfit: {}\nLoss: {}\nStocks Volume: {}".format(self.index+1, self.balance, self.profit, self.loss, self.stock_volume_aquired))
        print("Step: {}\nBalance: {}\nStocks Volume: {}".format(self.index+1, self.balance, self.stock_volume_aquired))

    

    

