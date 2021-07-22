import os

import pandas as pd

path = '/home/juncislab/PycharmProjects/0722skels/data/skel_dir_2'
file_list = os.listdir(path)


for i, name in enumerate(file_list):
    skel_df = pd.read_csv(path + '/' + name, index_col=False)
    skel_df = skel_df.fillna(method='backfill')
    skel_df.to_csv(path + '/' + name, index=False)
    print(skel_df)