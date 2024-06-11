import pandas as pd
import numpy as np


def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    if "TimeLimit.truncated" in df.columns:
        row_list = df[df['TimeLimit.truncated'].astype('bool')].index.to_list()
        df = df.drop(row_list)
        df.to_csv("dqn_conn0_ep101.csv", index=False)


read_csv("ppo_conn0_ep3.csv")
