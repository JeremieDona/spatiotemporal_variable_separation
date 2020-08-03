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
import re
import torch

import numpy as np

from torch.utils.data import Dataset


def extract_id(string):
    return int(re.findall(r'\d+', string)[0])


class WaveEq(Dataset):
    def __init__(self, data_dir, nt_cond, seq_len, train, downsample):
        super(WaveEq, self).__init__()

        self.nt_cond = nt_cond
        self.seq_len = seq_len

        base_path = os.path.join(data_dir, 'data')
        files = os.listdir(base_path)
        files = [os.path.join(base_path, f) for f in files]

        self.train = train
        max_seq = int(0.8 * len(files))

        if train:
            files = [file for file in files if extract_id(file) < max_seq]
        else:
            files = [file for file in files if extract_id(file) >= max_seq]

        self.size = len(files)
        self.all_data = []

        self.downsample = downsample

        for file in files:
            data_dict = torch.load(file)
            data = data_dict.get('simul')
            max_, min_ = data.max(), data.min()
            data = (data - min_) / (max_ - min_)
            data = data[::self.downsample]
            self.nt = len(data)
            self.all_data.append(data)

        self.full_seq_len = data[0].size(0)

    def __len__(self):
        return self.size * (self.full_seq_len - self.seq_len + 1)

    def __getitem__(self, idx):
        # extract data of lenght seq_len
        idx_seq = idx // (self.nt + 1 - self.seq_len)  # seq is of size nt+1
        idx_in_seq = idx % (self.nt + 1 - self.seq_len)
        full_state = self.all_data[idx_seq][idx_in_seq: idx_in_seq + self.seq_len].unsqueeze(1)
        return full_state[:self.nt_cond], full_state[self.nt_cond: self.seq_len]


class WaveEqPartial(WaveEq):

    def __init__(self, data_dir, nt_cond, seq_len, train, downsample, n_pixels):
        super(WaveEqPartial, self).__init__(data_dir, nt_cond, seq_len, train, downsample)

        data_dir = os.path.join(data_dir, 'pixels')
        pixels = np.load(os.path.join(data_dir, 'pixels.npz'), allow_pickle=True)
        self.rand_w = pixels['rand_w']
        self.rand_h = pixels['rand_h']
        self.n_wave_points = n_pixels

    def __getitem__(self, idx):
        cond, target = super().__getitem__(idx)
        cond = cond[:, :, self.rand_w[:self.n_wave_points], self.rand_h[:self.n_wave_points]]
        target = target[:, :, self.rand_w[:self.n_wave_points], self.rand_h[:self.n_wave_points]]
        return cond, target
