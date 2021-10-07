import os

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def skeleton_array():
    array = (
        (0, 1),

        (1, 2),
        (2, 3),
        (3, 4),

        (1, 5),
        (5, 6),
        (6, 7),

        (0, 15),
        (0, 16),

        (1, 8),

        (25, 26),
        (26, 27),
        (27, 28),
        (28, 29),

        (25, 30),
        (30, 31),
        (31, 32),
        (32, 33),

        (25, 34),
        (34, 35),
        (35, 36),
        (36, 37),

        (25, 38),
        (38, 39),
        (39, 40),
        (40, 41),

        (25, 42),
        (42, 43),
        (43, 44),

        (25 + 21, 26 + 21),
        (26 + 21, 27 + 21),
        (27 + 21, 28 + 21),
        (28 + 21, 29 + 21),

        (25 + 21, 30 + 21),
        (30 + 21, 31 + 21),
        (31 + 21, 32 + 21),
        (32 + 21, 33 + 21),

        (25 + 21, 34 + 21),
        (34 + 21, 35 + 21),
        (35 + 21, 36 + 21),
        (36 + 21, 37 + 21),

        (25 + 21, 38 + 21),
        (38 + 21, 39 + 21),
        (39 + 21, 40 + 21),
        (40 + 21, 41 + 21),

        (25 + 21, 42 + 21),
        (42 + 21, 43 + 21),
        (43 + 21, 44 + 21),

    )

    return array

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
               references,
               skip_frames = 1):

    # 리스트 분할 함수
    def list_chunk(lst, n):
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    # 비디오 파일 저장
    # timing_hyp_seq_1 : (85, 134)
    FPS = (25 // skip_frames)
    video_file = file_path + "{}.mp4".format(video_name.split(".")[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, FPS, (2048, 2048))
    width, height = 1920, 1080
    joints = skeleton_array()
    print(video_file)
    print(references.shape)
    for i, cordi in enumerate(skel):
        one_frame_skel = skel[i]
        one_frame_ref = references[i]

        img_pred = np.zeros((2048, 2048, 3), np.uint8)
        img_ref = np.zeros((2048, 2048, 3), np.uint8)

        chunked_skel = list_chunk(one_frame_skel, 2)
        chunked_ref = list_chunk(one_frame_ref, 2)
        # print('chu_skel : ', chunked_skel.shape)
        # print('chu_ref_skel : ', chunked_ref.shape)
        for (x, y) in joints:
            # print("X : ",chunked_skel[x].tolist(),"Y : ",type(chunked_skel[y].tolist()))
            # print("ch_x : ", int(ch_x[0] * width),int(ch_x[1] * height))
            ch_x, ch_y = chunked_skel[x].tolist(), chunked_skel[y].tolist()
            ch_x = [int(ch_x[0] * width), int(ch_x[1] * height)]
            ch_y = [int(ch_y[0] * width), int(ch_y[1] * height)]
            img_pred = cv2.line(img_pred, ch_x, ch_y, (0, 125, 125), 5)

            # ch_ref_x, ch_ref_y = chunked_ref[x].tolist(), chunked_ref[y].tolist()
            # ch_ref_x = [int(ch_ref_x[0] * width), int(ch_ref_x[1] * height)]
            # ch_ref_y = [int(ch_ref_y[0] * width), int(ch_ref_y[1] * height)]
            # img_ref_line = cv2.line(img_ref, ch_ref_x, ch_ref_y , (125, 0, 125), 5)

        # img = cv2.hconcat([img_ref_line, img_pred_line])
        video.write(img_pred)
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