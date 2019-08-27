import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import glob

sns.set(rc={'figure.figsize':(12,9)}, font_scale=2, style='darkgrid')
colors = sns.color_palette('colorblind', 4)
sns.set_palette(colors)

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
    ax = plt.subplot()

    # manually change this:
    #plt.xlim([380, 399900])
    #plt.yticks([0]+[x for x in range(1500, 3001, 250)])
    #plt.ylim([1500, 3001])
    #for i in range(0,400000,100000):
    #    plt.axvline(x=i, color='k', linestyle='--')
    #plt.axvline(x=25000, color='k', linestyle='--')
    #plt.axvline(x=50000, color='k', linestyle='--')
    #plt.axvline(x=75000, color='k', linestyle='--')
    plt.grid(axis='y')
    #plt.text(8000,2850,'Context 1')
    #plt.text(28000,2850,'Context 2')
    #plt.text(44500,5000,'Context 1')
    #plt.text(64500,5000,'Context 2')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-f", dest="file", nargs='+', required=True, help="The csv file to plot.\n")
    prs.add_argument("-label", dest="label", nargs='+', required=False, help="Figure labels.\n")
    prs.add_argument("-out", dest="out", required=False, default='', help="The .pdf filename in which the figure will be saved.\n")
    prs.add_argument("-w", dest="window", required=False, default=5, type=int, help="The moving average window.\n")
    args = prs.parse_args()
    if args.label:
        labels = args.label
    else:
        labels = ['' for _ in range(len(args.file))]

    plot_figure(x_label='Time Step (s)', y_label='Total Waiting Time of Vehicles (s)')

    for filename in args.file:
        main_df = pd.DataFrame()
        for file in glob.glob(filename+'*'):
            df = pd.read_csv(file)
            if main_df.empty:
                main_df = df
            else:
                main_df = pd.concat((main_df, df))

        steps = main_df.groupby('step_time').total_stopped.mean().keys().values
        mean = moving_average(main_df.groupby('step_time').mean()['total_wait_time'], window_size=args.window)
        #sem = moving_average(main_df.groupby('step_time').sem()['total_wait_time'], window_size=args.window)
        std = moving_average(main_df.groupby('step_time').std()['total_wait_time'], window_size=args.window)

        #plt.fill_between(steps, mean + sem*1.96, mean - sem*1.96, alpha=0.5)
        plt.plot(steps, mean)
        labels.pop(0)
        plt.fill_between(steps, mean + std, mean - std, alpha=0.3)

    if args.label is not None:
        plt.legend()

    if args.out != '':
        plt.savefig(args.out+'.pdf', bbox_inches="tight")
    plt.show()
