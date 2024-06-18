import sys

import matplotlib.pyplot as plt
import numpy as np
from sumolib.output import parse_fast


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


if __name__ == "__main__":
    fig, ax = plt.subplots()
    a = [0, 0.1, 0.5, 0.8]
    plt.ylim([10, 40])
    plt.xlim([200, 19900])
    plt.axvline(x=10000, color="k", linestyle="--")

    for i in range(1, len(sys.argv)):
        with open(sys.argv[i]) as f:
            l = f.readlines()
        x = []
        y = []
        # for line in l[2:]:
        #    s = [float(x) for x in line.split()]
        #    y.append(s[-2])
        # x = [i for i in range(1,len(y)+1)]

        file = sys.argv[i]
        # y = list(map(float, [e.halting for e in parse_fast(file, 'step',['halting'])]))
        y = list(map(lambda x: float(x) * 3.6, [e.meanSpeed for e in parse_fast(file, "step", ["meanSpeed"])]))
        x = [i for i in range(1, len(y) + 1)]
        # plt.plot(x, y)
        y_av = movingaverage(y, 360)
        ax.plot(x, y_av, "navy", label=f"alpha = {a[i]}")
    # ax.legend()

    # plt.xticks(np.arange(0, 20001, step=1000))
    plt.xlabel("Time Step")
    plt.ylabel("Mean Speed Km/h")
    # plt.ylabel("Total Nº Vehicles")
    # plt.ylabel("Average Nº Stopped Cars per TL")
    # plt.title("100% Exploração - Troca de Contexto no Timestep 5.000")
    # plt.title("QL alpha=0.5 gama=0.8 decay=0.90 epsilon=1 - Troca de contexto no Tempo 6.000")
    plt.grid()
    plt.show()
