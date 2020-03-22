from alpha_vantage.timeseries import TimeSeries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from gym_trading_bot.Data.Data_Functions import *

def create_dataset(data_file, look_back=3):
    
    if data_file == "original":
        data_file = return_address()
        data_file += "/AAPL_1min_full.csv"

    dataset = pd.read_csv(data_file)
    volumes = dataset["5. volume"].values
    volumes = volumes[look_back-1:]
    data = dataset["1. open"].values
    
    X = []
    Y = []
    
    for i in range(len(data)-look_back):
        temp = data[i:(i+look_back)]
        X.append(temp)
        Y.append(data[i+look_back])
    
    return np.array(X),np.array(Y),np.array(volumes)

def stock_data_to_csv(stock, interval="1min", output="compact", API_KEY="OMORWG3DQWN5BWUV"):
    
    """Function to gather data from alpha_vantage API and save in a csv file"""
    
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    
    data, meta_data = ts.get_intraday(symbol=stock, interval=interval, outputsize=output)
    
    data = data.iloc[::-1]
    
    name = stock+"_"+interval+"_"+output+".csv"
    data.to_csv(name)
    
    return

def plot_line_graph(file="Data/Timeseries_alpha_vantage/AAPL_1min_compact.csv", cols=[2], separate=False):
    
    """cols: {1 : open, 2 : high, 3 : low, 4 : close, 5 : volume}"""
    
    data = pd.read_csv(file)
    
    columns = data.columns
    
    data = data.get(columns[1:])
    
    plt.style.use("seaborn")
    
    if not separate:
        plt.figure(figsize=(16,12))
        for i in cols:
            plt.plot(data[columns[i]], label=columns[i].split()[-1])

        plt.legend()
        plt.show()
        
    else:
        for i in cols:
            plt.figure(figsize=(16,12))
            plt.plot(data[columns[i]], label=columns[i].split()[-1])
            plt.legend()
            plt.show()
    
    return

def calculate_reward(loss, profit):

    # Profit:
    if profit > loss:
        return 10.0

    # Loss:
    elif profit < loss:
        return 2.0

    # Neither Loss nor Profit:
    else:
        return 5.0


