# Copyright 2020 Jérémie Donà, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import numpy as np
import torch

from PIL import Image


class Chairs(object):
    max_length = 62

    def __init__(self, train, data_root, nt_cond, seq_len=15, image_size=64):
        self.train = train
        self.nt_cond = nt_cond
        assert seq_len <= self.max_length
        self.seq_len = seq_len
        assert image_size == 64
        self.image_size = image_size
        self.data_root = os.path.join(data_root, 'rendered_chairs')
        self.sequences = sorted(os.listdir(self.data_root))
        self.sequences.remove('all_chair_names.mat')
        rng = np.random.RandomState(42)
        rng.shuffle(self.sequences)
        if self.train:
            self.start_idx = 0
            self.stop_idx = int(len(self.sequences) * 0.85)
        else:
            self.start_idx = int(len(self.sequences) * 0.85)
            self.stop_idx = len(self.sequences)

    def get_sequence(self, index, chosen_idx=None, chosen_id_st=None):
        index, idx = divmod(index, self.stop_idx - self.start_idx)
        if chosen_idx is not None:
            idx = chosen_idx
        obj_dir = self.sequences[self.start_idx + idx]
        dname = os.path.join(self.data_root, obj_dir)
        index, id_st = divmod(index, self.max_length)
        if chosen_id_st is not None:
            id_st = chosen_id_st
        assert index == 0
        sequence = []
        for i in range(id_st, id_st + self.seq_len):
            fname = os.path.join(dname, 'renders', f'{i % self.max_length}.png')
            sequence.append(np.array(Image.open(fname)))
        sequence = np.array(sequence)
        return sequence

    def __getitem__(self, index):
        sequence = torch.tensor(self.get_sequence(index) / 255).permute(0, 3, 1, 2).float()
        return sequence[:self.nt_cond], sequence[self.nt_cond:]

    def __len__(self):
        return (self.max_length) * (self.stop_idx - self.start_idx)
