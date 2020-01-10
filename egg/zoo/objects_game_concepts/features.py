# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
#from torch.utils import data
import numpy as np

from functools import reduce
from egg.zoo.objects_game.util import compute_binomial

import itertools
import os
import pathlib
import json
from . import data

class VectorsLoader:
    def __init__(self,
            n_distractors=1,
            batch_size=32,
            shuffle_train_data=False,
            dump_data_folder = "./egg/zoo/objects_game_concepts/messages",
            load_data_path = "./egg/zoo/objects_game_concepts/data/concepts",
            seed=None):
        self.n_distractors = n_distractors

        self.batch_size = batch_size
        self.shuffle_train_data = shuffle_train_data

        self.load_data_path = load_data_path

        self.dump_data_folder = pathlib.Path(dump_data_folder) if dump_data_folder is not None else None

        seed = seed if seed else np.random.randint(0, 2 ** 31)
        self.random_state = np.random.RandomState(seed)

        with open(f"{self.load_data_path}/summary.json", "r") as jf:
            dataset = json.load(jf)
        self.features = dataset["features"]
        self.task = data.Task(
            num_features=dataset["num_features"],
            num_context=n_distractors+1,
        )

    def get_iterators(self):
        train = data.load_dataloader(
            self.load_data_path,
            "train_1",
            self.task,
            self.features,
            self.batch_size)
        test = data.load_dataloader(
            self.load_data_path,
            "test_1",
            self.task,
            self.features,
            self.batch_size)
        dev = data.load_dataloader(
            self.load_data_path,
            "dev_1",
            self.task,
            self.features,
            self.batch_size)
        return train, test, dev
