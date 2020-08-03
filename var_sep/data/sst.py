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
import torch

from torch.utils.data import Dataset

from netCDF4 import Dataset as netCDFDataset


def extract_data(fp, variables):
    loaded_file = netCDFDataset(fp, 'r')
    data_dict = {}
    for var in variables:
        data_dict[var] = loaded_file.variables[var][:].data
    return data_dict


class SST(Dataset):
    var_names = ['thetao', 'daily_mean', 'daily_std']

    def __init__(self, data_dir, nt_cond, nt_pred, train, zones=range(1, 30), eval=False):
        super(SST, self).__init__()

        self.data_dir = data_dir
        self.pred_h = nt_pred
        self.zones = list(zones)
        self.lb = nt_cond
        self.zone_size = 64

        self.data = {}
        self.cst = {}
        self.climato = {}

        self.train = train
        self.eval = eval

        self._normalize()

        self.first = 0 if self.train else int(0.8 * self.len_)

        # Retrieve length
        if self.train:
            self.len_ = int(0.8 * self.len_)
        else:
            self.len_ = self.len_ - int(0.8 * self.len_)

        self.len_ = self.len_ - self.pred_h - self.lb - 1
        self._total_len = len(self.zones) * self.len_

    def _normalize(self):
        for zone in self.zones:
            zdata = extract_data(os.path.join(self.data_dir, f'data_{zone}.nc'), variables=self.var_names)
            self.len_ = len(zdata["thetao"])

            climate_mean, climiate_std = zdata['daily_mean'].reshape(-1, 1, 1), zdata['daily_std'].reshape(-1, 1, 1)
            zdata["thetao"] = (zdata["thetao"] - climate_mean) / climiate_std
            self.climato[zone] = (climate_mean, climiate_std)

            mean = zdata["thetao"].mean(axis=(1, 2)).reshape(-1, 1, 1)
            std = zdata["thetao"].std(axis=(1, 2)).reshape(-1, 1, 1)
            zdata["thetao"] = (zdata["thetao"] - mean) / std
            self.cst[zone] = (mean, std)

            self.data[zone] = zdata["thetao"]

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        file_id = self.zones[idx // self.len_]
        idx_id = (idx % self.len_) + self.lb + 1 + self.first
        inputs = self.data[file_id][idx_id - self.lb + 1: idx_id + 1].reshape(self.lb, 1, self.zone_size,
                                                                              self.zone_size)
        target = self.data[file_id][idx_id + 1: idx_id + self.pred_h + 1].reshape(self.pred_h, 1, self.zone_size,
                                                                                  self.zone_size)

        if self.eval:
            inputs, target = torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)
            mu_clim, std_clim = (self.climato[file_id][0][idx_id + 1: idx_id + self.pred_h + 1],
                                 self.climato[file_id][1][idx_id + 1: idx_id + self.pred_h + 1])
            mu_norm, std_norm = (self.cst[file_id][0][idx_id + 1: idx_id + self.pred_h + 1],
                                 self.cst[file_id][1][idx_id + 1: idx_id + self.pred_h + 1])
            return inputs, target, mu_clim, std_clim, mu_norm, std_norm
        else:
            return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)
