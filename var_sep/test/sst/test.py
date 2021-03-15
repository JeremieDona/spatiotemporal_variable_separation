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


import argparse
import os
import torch

import numpy as np

from tqdm import tqdm

from var_sep.data.sst import SST
from var_sep.utils.helper import DotDict, load_json
from var_sep.test.utils import load_model, _ssim_wrapper


def get_min(test_loader):
    mins, maxs = {}, {}
    for zone in test_loader.zones:
        mins[zone] = test_loader.data[zone].min()
        maxs[zone] = test_loader.data[zone].max()
    return mins, maxs


def load_dataset(args, train=False, zones=range(17, 21)):
    return SST(args.data_dir, args.nt_cond, args.nt_pred, train, zones=zones, eval=True)


def compute_mse_ssim(args, test_set, sep_net):
    mins, maxs = get_min(test_set)
    all_mse = []
    all_ssim = []
    torch.set_grad_enabled(False)
    for cond, target, mu_clim, std_clim, mu_norm, std_norm, file_id in tqdm(test_set):
        cond, target = cond.unsqueeze(0).to(args.device), target.unsqueeze(0).to(args.device)
        if args.offset:
            forecasts = sep_net.get_forecast(cond, target.size(1) + args.nt_cond)[0]
            forecasts = forecasts[:, args.nt_cond:]
        else:
            forecasts = sep_net.get_forecast(cond, target.size(1))[0]

        mu_norm, std_norm = (torch.tensor(mu_norm, dtype=torch.float).to(args.device),
                             torch.tensor(std_norm, dtype=torch.float).to(args.device))

        forecasts = (forecasts * std_norm) + mu_norm
        target = (target * std_norm) + mu_norm

        # Original space for MSE
        mu_clim, std_clim = (torch.tensor(mu_clim, dtype=torch.float).to(args.device),
                             torch.tensor(std_clim, dtype=torch.float).to(args.device))
        forecasts = (forecasts * std_clim) + mu_clim
        target = (target * std_clim) + mu_clim
        mse = (forecasts - target).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)

        # Normalize by min and max per zone for SSIM
        min_, max_ = mins[file_id], maxs[file_id]
        forecasts = (forecasts - min_)/ (max_ - min_)
        target = (target - min_) / (max_ - min_)
        ssim = _ssim_wrapper(forecasts, target)

        all_mse.append(mse.cpu().numpy())
        all_ssim.append(ssim.cpu().numpy())

    return all_mse, all_ssim


def main(args):
    if args.device is None:
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    # Load XP config
    xp_config = load_json(os.path.join(args.xp_dir, 'params.json'))
    xp_config.device = device
    xp_config.data_dir = args.data_dir
    xp_config.xp_dir = args.xp_dir
    xp_config.nt_pred = 10
    args.nt_pred = 10

    test_set = load_dataset(xp_config, train=False)
    sep_net = load_model(xp_config, args.epoch)

    all_mse, all_ssim = compute_mse_ssim(xp_config, test_set, sep_net)
    mse_array = np.concatenate(all_mse, axis=0)
    ssim_array = np.concatenate(all_ssim, axis=0)
    print(f'MSE at t+10: {np.mean(mse_array.mean(axis=0)[:10])}')
    print(f'MSE at t+6: {np.mean(mse_array.mean(axis=0)[:6])}')
    print(f'SSIM at t+10: {np.mean(ssim_array.mean(axis=0)[:10])}')
    print(f'SSIM at t+6: {np.mean(ssim_array.mean(axis=0)[:6])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="PDE-Driven Spatiotemporal Disentanglement (Moving MNIST testing)")
    p.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the dataset is saved.')
    p.add_argument('--xp_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the model configuration file and checkpoints are saved.')
    p.add_argument('--epoch', type=int, metavar='EPOCH', default=None,
                   help='If specified, loads the checkpoint of the corresponding epoch number.')
    p.add_argument('--device', type=int, metavar='DEVICE', default=None,
                   help='GPU where the model should be placed when testing (if None, on the CPU)')
    args = DotDict(vars(p.parse_args()))
    main(args)
