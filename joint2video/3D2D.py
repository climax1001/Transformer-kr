

import os
import pandas as pd
origin = '/home/juncislab/dataset/train'

path = '/home/juncislab/PycharmProjects/0722skels/data/train_skel_dir'
read_path = os.listdir(origin)
del_list = []

for i in range(140):
    if (i+1) % 3 == 0:
        del_list.append(i)

del_list = list(map(str, del_list))
for i, name in enumerate(read_path):
    cs = pd.read_csv(origin + '/' + name)
    cs.drop(del_list, axis='columns' ,inplace=True)
    cs.to_csv(path + '/' + name, index=False, na_rep=0)
    print(i)