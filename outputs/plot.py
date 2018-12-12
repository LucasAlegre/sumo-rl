import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


if __name__ == '__main__':

    fig, ax = plt.subplots()
    #plt.ylim([20,220])
    #plt.xlim([250,19800])
    plt.axvline(x=10000, color='k', linestyle='--')

    df = pd.read_csv("outputs/resultc1.csv")
    ax.plot(df['step'], moving_average(df['total_stopped'], window_size=5))

    plt.title("")
    plt.xlabel("Time Step")
    plt.ylabel("Total Nº Stopped Vehicles")
    plt.grid()
    plt.show()
