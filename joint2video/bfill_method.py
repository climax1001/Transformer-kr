import os
import pandas as pd

path = '/home/juncislab/dataset/test'
file_list = os.listdir(path)

for i, name in enumerate(file_list):
    skel_df = pd.read_csv(path + '/' + name, index_col=False)
    skel_df = skel_df.fillna(method='ffill')
    skel_df.to_csv(path + '/' + name, index=False)
    print(path + name)
    print(skel_df)