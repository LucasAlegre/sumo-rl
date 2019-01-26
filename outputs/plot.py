import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob


def fig():
    fig = 1
    while True:
        yield fig
        fig += 1
fig_gen = fig()


def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def plot_figure(figsize=(12, 9), x_label='', y_label='', title=''):
    plt.figure(next(fig_gen), figsize=figsize)
    plt.rcParams.update({'font.size': 16})
    ax = plt.subplot()

    # manually change this:
    #plt.xlim([300, 79900])
    #plt.yticks([x for x in range(0, 7001, 500)])
    #plt.ylim([0, 7000])
    #plt.axvline(x=20000, color='k', linestyle='--')
    #plt.axvline(x=40000, color='k', linestyle='--')
    #plt.axvline(x=60000, color='k', linestyle='--')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-f", dest="file", nargs='+', required=True, help="The csv file to plot.\n")
    prs.add_argument("-w", dest="window", required=False, default=5, type=int, help="The moving average window.\n")
    prs.add_argument("-average", action='store_true', default=False)
    args = prs.parse_args()

    if args.average:

        plot_figure(x_label='Time Step (s)', y_label='Total Waiting Time of Vehicles (s)')

        for filename in args.file:
            main_df = pd.DataFrame()
            for file in glob.glob(filename+'*'):
                df = pd.read_csv(file)
                if main_df.empty:
                    main_df = df
                else:
                    main_df = pd.concat((main_df, df))

            '''
            steps = main_df.groupby('step_time').total_stopped.mean().keys()
            mean = moving_average(main_df.groupby('step_time').mean()['total_stopped'], window_size=args.window)
            std = moving_average(main_df.groupby('step_time').std()['total_stopped'], window_size=args.window)
    
            plot_figure(steps, mean, x_label='Time Step (s)', y_label='Total Waiting Time of Vehicles (s)')
            #plt.fill_between(steps, mean + sem*1.96, mean - sem*1.96, alpha=0.5)
            plt.fill_between(steps, mean + std, mean - std, alpha=0.5)
            '''

            steps = main_df.groupby('step_time').total_stopped.mean().keys()
            mean = moving_average(main_df.groupby('step_time').mean()['total_wait_time'], window_size=args.window)
            sem = moving_average(main_df.groupby('step_time').sem()['total_wait_time'], window_size=args.window)
            std = moving_average(main_df.groupby('step_time').std()['total_wait_time'], window_size=args.window)

            #plt.fill_between(steps, mean + sem*1.96, mean - sem*1.96, alpha=0.5)
            plt.plot(steps, mean)
            plt.fill_between(steps, mean + std, mean - std, alpha=0.3, color='gray')

    else:
        df = pd.read_csv(args.file)

        df['cum_total_stopped'] = df.total_stopped.cumsum()
        df['cum_wait_time'] = df.total_wait_time.cumsum()

        plot_figure(x=df['step_time'],
                    y=moving_average(df['total_stopped'], window_size=args.window),
                    x_label="Time Step (s)",
                    y_label="Total Number of Stopped Vehicles")

        plot_figure(x=df['step_time'],
                    y=moving_average(df['total_wait_time'], window_size=args.window),
                    x_label="Time Step (s)",
                    y_label="Total Waiting Time of Vehicles (s)")

    plt.savefig("saved.png", bbox_inches="tight")
    plt.show()
