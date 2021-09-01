

import os
import pandas as pd
path = '/home/juncislab/PycharmProjects/0722skels/data/skel_dir_2'
read_path = os.listdir('/home/juncislab/PycharmProjects/0722skels/data/skel_dir_2')
del_list = []
for i in range(140):
    if (i+1) % 3 == 0:
        del_list.append(i)

for i, name in enumerate(read_path):

    cs = pd.read_csv(path + '/' + name)
    cs.drop(['Unnamed: 0'], axis='columns' ,inplace=True)
    cs.to_csv(path + '/' + name, index=False)
    print(cs)
    print(i)