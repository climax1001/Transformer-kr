import os

from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset
import torch

from constants import TARGET_PAD, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN

train_path = '/home/juncislab/Downloads/bbinix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev'

tok_fun = lambda s: list(s) if "word" == "char" else s.split()


f = open('/home/juncislab/PycharmProjects/0722skels/data/tmp/train.skels','r')
for i in range(1, 10):
    line = f.readline()
    print("=========================")
    print(line)
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