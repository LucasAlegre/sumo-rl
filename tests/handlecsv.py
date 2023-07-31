import os
import sys
import numpy as np

import pandas as pd


def delete_bad_line(filename):
    name, ext = os.path.splitext(filename)
    new_name = name + '-1' + ext
    with open(filename, 'r') as f:
        head = f.readline()
        len = head.count(',')
        with open(new_name, 'w') as f1:
            f1.write(head)
            for line in f:
                cnt = line.count(',')
                if cnt <= len:
                    f1.write(line)


def check_csv(filename):
    df = pd.read_csv(filename, sep=",", engine='python')
    df_a = pd.to_numeric(df["system_mean_waiting_time"], errors="coerce")
    df_a.fillna(0, inplace=True)
    # print(df_a.info())
    # print(df_a.head())
    # print(df_a.tail())
    # print(df_a[15988])
    # print(df_a[1765:1775])
    df_group = df.groupby("step")
    df_group.fillna(0, inplace=True)
    print(df_group.head())
    # df_group.to_csv("group.csv")

    # df_group_mean = df.groupby("step").mean().to_csv("group_mean.csv")
    # df_group_mean().to_csv("group_mean.csv")
    # print(df_group_mean.head())

    mean = df.groupby("step")["system_mean_waiting_time"].mean()
    print(mean.head())


def nobleprize():
    df = pd.read_csv("nobleprize.csv", sep=",", engine='python')
    # print(df.head())
    print(df.info())

    grouped = df.groupby('category')

    for name, entries in grouped:
        print(f'\n\n================Entries for the "{name}" category================')
        print(entries[['awardYear', 'prizeAmount', 'prizeAmountAdjusted', 'name']].head(10))

    print("\n\n==============aggregation============")
    agg = grouped.agg({'prizeAmount': [np.sum, np.mean], 'prizeAmountAdjusted': [np.sum, np.mean], 'name': np.size})
    # print(agg.info())
    print(agg.head(10))

    print("\n\n==============transform============")
    trans = grouped[['prizeAmount', 'prizeAmountAdjusted']].transform(lambda x: (x - x.mean()) / x.std())
    # print(trans.info())
    print(trans.head(10))

    print("\n\n==============sort & group============")
    # sorted = df.sort_values(by=['awardYear'], ascending=True)
    groupByAwardYear = df.groupby('awardYear', sort=True)
    for name, entries in groupByAwardYear:
        print(f'\n\n================Entries for the "{name}" year================')
        print(entries[['category', 'prizeAmount', 'prizeAmountAdjusted']].head(10))



def group_byCategory():
    df = pd.read_csv("nobleprize.csv", sep=",", engine='python')

    groupByCategory = df.groupby('category', sort=True)
    return groupByCategory


def agg_byMean():
    df = pd.read_csv("nobleprize.csv", sep=",", engine='python')

    groupByAwardYear = df.groupby('awardYear', sort=True)
    agg = groupByAwardYear.agg({'prizeAmount': 'median', 'prizeAmountAdjusted': 'mean'})
    return agg


def trans_byMean():
    df = pd.read_csv("nobleprize.csv", sep=",", engine='python')

    groupByCategory = df.groupby('category', sort=True)
    df["med_prizeAmount"] = groupByCategory["prizeAmount"].transform('median')
    df["avg_prizeAmountAdjusted"] = groupByCategory["prizeAmountAdjusted"].transform('mean')
    return df


if __name__ == '__main__':
    # filename = sys.argv[1]
    # delete_bad_line(filename)

    # check_csv("./ppo_conn0_ep2-1.csv")

    # byCategory = group_byCategory()
    # print(byCategory[["awardYear", "category", "prizeAmount", "prizeAmountAdjusted"]].head())

    # aggByMean = agg_byMean()
    # print(aggByMean.head())

    transByMean = trans_byMean()
    print(transByMean.head())










"""
Data columns (total 52 columns):
 #   Column                      Non-Null Count  Dtype 
---  ------                      --------------  ----- 
 0   awardYear                   950 non-null    int64 
 1   category                    950 non-null    object
 2   categoryFullName            950 non-null    object
 3   sortOrder                   950 non-null    int64 
 4   portion                     950 non-null    object
 5   prizeAmount                 950 non-null    int64 
 6   prizeAmountAdjusted         950 non-null    int64 
 7   dateAwarded                 533 non-null    object
 8   prizeStatus                 950 non-null    object
 9   motivation                  950 non-null    object
 10  categoryTopMotivation       20 non-null     object
 11  award_link                  950 non-null    object
 12  id                          950 non-null    int64 
 13  name                        950 non-null    object
 14  knownName                   923 non-null    object
 15  givenName                   923 non-null    object
 16  familyName                  921 non-null    object
 17  fullName                    923 non-null    object
 18  penName                     11 non-null     object
 19  gender                      923 non-null    object
 20  laureate_link               950 non-null    object
 21  birth_date                  923 non-null    object
 22  birth_city                  922 non-null    object
 23  birth_cityNow               922 non-null    object
 24  birth_continent             923 non-null    object
 25  birth_country               923 non-null    object
 26  birth_countryNow            923 non-null    object
 27  birth_locationString        923 non-null    object
 28  death_date                  630 non-null    object
 29  death_city                  611 non-null    object
 30  death_cityNow               611 non-null    object
 31  death_continent             617 non-null    object
 32  death_country               617 non-null    object
 33  death_countryNow            617 non-null    object
 34  death_locationString        618 non-null    object
 35  orgName                     27 non-null     object
 36  nativeName                  27 non-null     object
 37  acronym                     11 non-null     object
 38  org_founded_date            26 non-null     object
 39  org_founded_city            22 non-null     object
 40  org_founded_cityNow         22 non-null     object
 41  org_founded_continent       24 non-null     object
 42  org_founded_country         24 non-null     object
 43  org_founded_countryNow      24 non-null     object
 44  org_founded_locationString  24 non-null     object
 45  ind_or_org                  950 non-null    object
 46  residence_1                 219 non-null    object
 47  residence_2                 2 non-null      object
 48  affiliation_1               697 non-null    object
 49  affiliation_2               67 non-null     object
 50  affiliation_3               2 non-null      object
 51  affiliation_4               1 non-null      object
"""
