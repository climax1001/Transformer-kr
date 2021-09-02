import os

import cv2
import numpy as np
from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset
import torch
from joint2video.get_video import skels, draw_point, skel_connection

f = open('/home/juncislab/PycharmProjects/0722skels/data/tmp/dev.skels','r')
wanna_line = 1
i = 0
pose = []
while True:
    line = f.readline()
    i = i + 1
    if line == '':
        break
    if i == wanna_line:
        pose = line.replace('\n','').split(' ')
# # 마지막 공백 뺌
pose.pop()
f.close()

# 46, 2포인트로 변환?
point = np.reshape(pose, (92, -1))



if __name__ == "__main__":
    i_poses = []
    FPS = 25
    for line in range(len(pose)):
        i_pose = float(pose[line])
        i_poses.append(i_pose)

    point = np.reshape(pose,(-1,92))
    print(point.shape)
    video_file = '/home/juncislab/gt_video/MORGEN_MAL_SONNE.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_file, fourcc, FPS, (260,210))
    for i in range(len(point)):
        img = np.zeros((210,260,3), np.uint8)
        get_one_line = point[i]
        my_list = skels(get_one_line)
        print(my_list)
        mydict = draw_point(my_list, 210, 260)
        skel_connection(img, mydict)

        video.write(img)
        # cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# paths = os.listdir(train_path)
# for path in paths:
#     file_path.write(path + '\n')
# def tokenize_features(features):
#     features = torch.as_tensor(features)
#     ft_list = torch.split(features, 1, dim=0)
#     return [ft.squeeze() for ft in ft_list]
#
# def stack_features(features, something):
#     return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
#
# files_field = data.RawField()
#
# reg_trg_field = data.Field(sequential=True,
#                                use_vocab=False,
#                                dtype=torch.float32,
#                                batch_first=True,
#                                include_lengths=False,
#                                pad_token=torch.ones((150,))*TARGET_PAD,
#                                preprocessing=tokenize_features,
#                                postprocessing=stack_features,)
#
# src_field = data.Field(init_token=None, eos_token=EOS_TOKEN,
#                            pad_token=PAD_TOKEN, tokenize=tok_fun,
#                            batch_first=True, lower=False,
#                            unk_token=UNK_TOKEN,
#                            include_lengths=True)
# SKEL = data.Field(sequential=True,
#                                use_vocab=False,
#                                dtype=torch.float32,
#                                batch_first=True,
#                                include_lengths=False,
#                                pad_token=torch.ones((150,))*TARGET_PAD,
#                                preprocessing=tokenize_features,
#                                postprocessing=stack_features,)
#
# FILE = data.Field(sequential=False,use_vocab=False)
# TEXT = data.Field(sequential=True, use_vocab=True)
# print(files_field)
# # train_data , test_data = TabularDataset.split(path='.', train = )