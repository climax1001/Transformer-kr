import operator
import os

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def skels(list_):
    point_list = np.reshape(list_,(46,2))
    return point_list

def draw_point(one_list, height, width):
    skel_dict = {}
    for num, [x,y] in enumerate(one_list):
        x = float(x) * width
        y = float(y) * height
        skel_dict[num] = (int(x),int(y))

    return skel_dict

def skel_connection(img, dict_, color = (255,0,0)):
    left_connection = [[0,1], [1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[0,17]]

    right_connection = [[x+21,y+21] for x,y, in left_connection]

    arm_connection = [[0,42], [21,43],[42,44],[43,45],[44,45]]

    for [x,y] in left_connection:
        cv2.line(img, dict_[x],dict_[y], color, 1)

    for [x,y] in right_connection:
        cv2.line(img, dict_[x],dict_[y], color, 1)

    for [x,y] in arm_connection:
        cv2.line(img, dict_[x], dict_[y], color, 1)

def _2D_frame(filename):
    tmp = [i for i in range(138)]
    del_list = []
    for seq, i in enumerate(tmp):
        if (i + 1) % 3 == 0:
           del_list.append(str(i))
    print(del_list)
    skel_data = pd.read_csv(filename)
    skel_data = skel_data.drop(del_list, axis=1)
    return skel_data

def monitor_video(filepath):
    skel = pd.read_csv(filepath)

    for i in range(len(skel)):
        img = np.zeros((210, 260, 3), np.uint8)
        get_one_line = np.array(skel.iloc[i].tolist())
        my_list = skels(get_one_line)

        mydict = draw_point(my_list, 210, 260)
        skel_connection(img, mydict)
        cv2.imshow('image', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # folder_path = '/home/juncislab/PycharmProjects/0722skels/data/skel_dir_2'
    # dir_list = os.listdir(folder_path)
    # dic = {}
    # new_skel_path = open('/home/juncislab/PycharmProjects/0722skels/data/tmp/new_train.skels', 'a')
    #
    # for i in range(len(dir_list)):
    #     index = int(dir_list[i].split('.csv')[0].split('-')[1])
    #     dic[dir_list[i]] = index
    #
    # sdic = sorted(dic.items(), key=operator.itemgetter(1))
    # for i in range(len(sdic)):
    #     print(sdic[i][0])
    #     data = pd.read_csv(folder_path + '/' + sdic[i][0])
    #     for row in range(len(data)):
    #         csv_one_line = list(map(float,data.iloc[row]))
    #         csv_one_line = list(np.round(csv_one_line, 6))
    #         writable = ' '.join(map(str,csv_one_line))
    #
    #         new_skel_path.write(writable)
    #     new_skel_path.write('\n')
    # new_skel_path.close()
    monitor_video('/home/juncislab/PycharmProjects/0722skels/data/skel_dir_2/11August_2010_Wednesday_tagesschau-1.csv')