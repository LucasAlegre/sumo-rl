import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    df = pd.read_csv("outputs/result2.csv")
    plt.plot(df['step'], df['total_stopped'])

    plt.grid()
    plt.show()