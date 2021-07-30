import os

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def skels(list_):
    point_list = []
    tmp = []
    for i, point in enumerate(list_):
        if (i+1) % 3 != 0:
            tmp.append(point)

        elif (i+1) % 3 == 0:
            point_list.append(tmp)
            tmp = []

    return point_list

def draw_point(one_list, height, width):
    skel_dict = {}
    for num, [x,y] in enumerate(one_list):
        x = float(x) * width
        y = float(y) * height
        skel_dict[num] = (int(x),int(y))

    return skel_dict

def skel_connection(img, dict_, color = (255,0,0)):
    left_connection = [[0,1], [1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[9,10],[10,11],[11,12],[13,14],[14,15],[15,16],
    [17,18],[18,19],[19,20],[5,9],[9,13],[13,17],[0,17]]

    right_connection = [[x+21,y+21] for x,y, in left_connection]

    arm_connection = [[0,42], [21,43],[42,44],[43,45],[44,45]]

    print(dict_)
    for [x,y] in left_connection:
        cv2.line(img, dict_[x],dict_[y], color, 1)

    for [x,y] in right_connection:
        cv2.line(img, dict_[x],dict_[y], color, 1)

    for [x,y] in arm_connection:
        cv2.line(img, dict_[x], dict_[y], color, 1)

def _2D_frame(src_path, to_path = None):
    filename = os.listdir(src_path)
    del_list = [i for i in range(138)]
    for seq, i in enumerate(del_list):
        print(i)
        if (i+1) % 3 == 0:
            del_list.remove(i)
    print(del_list)
    for _,name in enumerate(filename):
        skel = pd.read_csv(src_path + '/' + name)
        skel = skel.drop(del_list, axis=1)
        print(skel)

    # print(len(skel))
if __name__ == "__main__":
    _2D_frame('../data/skel_dir_2')
    skel = pd.read_csv('../data/01April_2010_Thursday_heute-6694.csv')
    get_one_line = np.array(skel.iloc[28].tolist())
    my_list = skels(get_one_line)
    for i in range(len(skel)):
        img = np.zeros((210, 260, 3), np.uint8)

        get_one_line = np.array(skel.iloc[i].tolist())
        my_list = skels(get_one_line)
        mydict = draw_point(my_list, 210, 260)
        skel_connection(img, mydict)

        cv2.imshow('image',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    # l = []
    # for [i,j] in my_list:
    #     l.append(i)
    #     l.append(j)
    #
    # # print(l)
    # torch_list = np.array(my_list)
    # torch_list = torch.FloatTensor(torch_list)
    # # print("shape: ", print(torch_list))
    # emb = nn.Embedding(len(torch_list), 512)
    # # print(emb(torch_list))