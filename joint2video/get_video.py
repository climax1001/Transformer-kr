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

    skel_data = pd.read_csv(filename)
    skel_data = skel_data.drop(del_list, axis=1)

    return skel_data

def show_video(skel,
               file_path,
               video_name,
               skip_frames = 0.4):

    # 비디오 파일 저장
    print(skel.shape)
    FPS = (25 // skip_frames)
    video_file = file_path + "/{}.mp4".format(video_name.split(".")[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video = cv2.VideoWriter(video_file, fourcc, FPS, (260, 210))
    # print(len(skel))
    for i in range(len(skel)):
        img = np.zeros((210, 260, 3), np.uint8)
        get_one_line = skel[i]
        my_list = skels(get_one_line)

        mydict = draw_point(my_list, 210, 260)
        skel_connection(img, mydict)

        video.write(img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
if __name__ == "__main__":
    # 파일 경로를 넣으면 골격을 뽑아준다.
    show_video('../data/01April_2010_Thursday_heute-6694.csv')

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