# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import lmdb  # install lmdb by "pip install lmdb"
import base64
import pickle
from typing import List
from functools import lru_cache

import numpy as np
import torch


class RefImageFeaturesH5Reader(object):

    def __init__(self, config, features_path: str, boxfile, in_memory: bool = False):
        self.config = config
        self.features_path = features_path
        self._in_memory = in_memory
        self.feature_size = 512 if config.backbone == "clip" else 2048
        self.num_locs = 4

        with open(boxfile) as f:
            self.lid2words = [str(line).strip() for line in f]

        # If not loaded in memory, then list of None.
        self.env = lmdb.open(
            self.features_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get("keys".encode()))

    def keys(self) -> List[int]:
        return self._image_ids

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        image_id = str(image_id).encode()
        if self._in_memory:
            return self._get_item_in_memory(image_id)
        else:
            return self._load_item_from_disk(image_id)

    @lru_cache(maxsize=25000)
    def _get_item_in_memory(self, image_id):
        return self._load_item_from_disk(image_id)

    def _load_item_from_disk(self, image_id):
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(image_id))
            try:
                features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(-1, self.feature_size)
                boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(-1, 4)
            except:
                features = item["features"].reshape(-1, self.feature_size)
                boxes = item['boxes'].reshape(-1, 4)

            assert features.shape[0] == boxes.shape[0]
            num_boxes = features.shape[0]
            obj_labels = np.frombuffer(base64.b64decode(item["objects_id"]), dtype=np.int64)
            obj_words = [self.lid2words[i] for i in obj_labels]
            return torch.tensor(features), torch.tensor(num_boxes), None, boxes, obj_words, None

    @lru_cache(maxsize=None)
    def get_num_boxes(self, image_id):
        image_id = str(image_id).encode()
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(image_id))
            try:
                boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(-1, 4)
            except:
                boxes = item['boxes'].reshape(-1, 4)
            return boxes.shape[0]
