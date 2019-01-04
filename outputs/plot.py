import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-f", dest="file", required=True, help="The csv file to plot.\n")
    prs.add_argument("-w", dest="window", required=False, default=5, type=int, help="The moving average window.\n")
    prs.add_argument("-average", action='store_true', default=False)
    args = prs.parse_args()

    if args.average:
        main_df = pd.DataFrame()
        for file in glob.glob(args.file+'*'):
            df = pd.read_csv(file)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        steps = main_df.groupby('step_time').total_stopped.mean().keys()
        mean_stopped = moving_average(main_df.groupby('step_time').mean()['total_stopped'], window_size=args.window)
        sem = moving_average(main_df.groupby('step_time').sem()['total_stopped'], window_size=args.window)
        plt.figure(1, figsize=(12, 9))
        ax = plt.subplot()
        #plt.xlim([0, 20000])
        plt.ylim([0, 20])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.grid()
        plt.plot(steps, mean_stopped)
        plt.fill_between(steps, mean_stopped + sem*1.96, mean_stopped - sem*1.96, alpha=0.5)

    else:
        #plt.xlim([0, 20000])
        #plt.ylim([0, 20])
        #plt.axvline(x=20000, color='k', linestyle='--')
        #plt.axvline(x=40000, color='k', linestyle='--')
        #plt.axvline(x=60000, color='k', linestyle='--')

        df = pd.read_csv(args.file)

        df['cum_total_stopped'] = df.total_stopped.cumsum()
        df['cum_wait_time'] = df.total_wait_time.cumsum()

        plt.figure(1)
        plt.plot(df['step_time'], moving_average(df['total_stopped'], window_size=args.window))
        plt.title("")
        plt.xlabel("Time Step")
        plt.ylabel("Total Number of Stopped Vehicles")
        plt.grid()

        plt.figure(2)
        plt.plot(df['step_time'], moving_average(df['cum_total_stopped'], window_size=args.window))
        plt.title("")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Number of Stopped Vehicles")
        plt.grid()

        plt.figure(3)
        plt.plot(df['step_time'], moving_average(df['total_wait_time'], window_size=args.window))
        plt.title("")
        plt.xlabel("Time Step")
        plt.ylabel("Total Waiting Time of Vehicles")
        plt.grid()

        plt.figure(4)
        plt.plot(df['step_time'], moving_average(df['cum_wait_time'], window_size=args.window))
        plt.title("")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Total Waiting Time of Vehicles")
        plt.grid()

        #plt.figure(3)
        #plt.plot(df['step_time'], moving_average(df['buffer_size'], window_size=args.window))
        #plt.title("")
        #plt.xlabel("Time Step")
        #plt.ylabel("Buffer Size")
        #plt.grid()

    plt.show()

