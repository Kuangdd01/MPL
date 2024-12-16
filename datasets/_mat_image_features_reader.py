import copy
import json
import os
import pickle
from functools import lru_cache

import h5py
import numpy as np
import torch
from torch.nn import functional as F


class MatImageFeaturesH5Reader:
    def __init__(self, config, h5file, idxfile, boxfile):
        # using config h5file
        # using config objects_vocab
        # using config attributes_vocab

        # config
        self.config = config
        self.feature_size = 2048
        # h5py
        self.f = h5py.File(h5file, 'r')
        # self.features = self.f['features']
        self.features = np.asarray(self.f['features'])
        self.pos_boxes = np.asarray(self.f['pos_bboxes'])
        # imgid2idx
        self.img2idx = pickle.load(open(idxfile, 'rb'))
        # boxes and class
        self.objs_words = json.load(open(boxfile, 'r'))

    def __len__(self):
        return len(self.f)

    def get_num_boxes(self, img):
        idx = self.img2idx[int(img)]
        st, ed = self.pos_boxes[idx]
        return ed - st

    # @lru_cache(maxsize=125)
    def __getitem__(self, img):
        idx = self.img2idx[int(img)]
        st, ed = self.pos_boxes[idx]
        features = self.features[st:ed]
        boxes = self.objs_words[str(img)]['bboxes']
        objs_words = self.objs_words[str(img)]['classes']

        num_boxes = features.shape[0]

        return torch.tensor(features), torch.tensor(num_boxes), None, \
               boxes, objs_words, None
