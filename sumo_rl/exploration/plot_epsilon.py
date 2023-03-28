"""Plot epsilon decay."""
import argparse

import matplotlib.pyplot as plt


if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    prs.add_argument("-e", dest="epsilon", type=float, required=True, help="Epsilon\n")
    prs.add_argument("-d", dest="decay", type=float, required=True, help="Epsilon\n")
    args = prs.parse_args()

    plt.plot([i for i in range(0, 20000, 5)], [args.epsilon * args.decay**i for i in range(0, 4000)])

    plt.grid()
    plt.show()
