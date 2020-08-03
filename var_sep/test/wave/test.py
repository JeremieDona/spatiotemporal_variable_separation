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

from torch.utils.data import DataLoader
from tqdm import tqdm

from var_sep.data.wave_eq import WaveEq, WaveEqPartial
from var_sep.utils.helper import DotDict, load_json
from var_sep.networks.conv import DCGAN64Encoder, VGG64Encoder, DCGAN64Decoder, VGG64Decoder
from var_sep.networks.mlp_encdec import MLPEncoder, MLPDecoder
from var_sep.networks.model import SeparableNetwork


def load_dataset(args, train=False):
    if args.data == 'wave':
        return WaveEq(args.data_dir, args.nt_cond, args.nt_cond + args.nt_pred, train, args.downsample)
    else:
        return WaveEqPartial(args.data_dir, args.nt_cond, args.nt_cond + args.nt_pred, True, args.downsample,
                             args.n_wave_points)


def build_model(args):
    Es = torch.load(os.path.join(args.xp_dir, 'ov_Es.pt'), map_location=args.device).to(args.device)
    Et = torch.load(os.path.join(args.xp_dir, 'ov_Et.pt'), map_location=args.device).to(args.device)
    t_resnet = torch.load(os.path.join(args.xp_dir, 't_resnet.pt'), map_location=args.device).to(args.device)
    decoder = torch.load(os.path.join(args.xp_dir, 'decoder.pt'), map_location=args.device).to(args.device)
    sep_net = SeparableNetwork(Es, Et, t_resnet, decoder, args.nt_cond, args.skipco)
    sep_net.eval()
    return sep_net


def compute_mse(args, batch_size, test_set, sep_net):
    all_mse = []
    loader = DataLoader(test_set, batch_size=batch_size, pin_memory=False, shuffle=False, num_workers=3)
    torch.set_grad_enabled(False)
    for cond, target in tqdm(loader):
        cond, target = cond.to(args.device), target.to(args.device)
        if args.offset:
            forecasts, t_codes, s_codes, t_residuals = sep_net.get_forecast(cond, target.size(1) + args.nt_cond)
            forecasts = forecasts[:, args.nt_cond:]
        else:
            forecasts, t_codes, s_codes, t_residuals = sep_net.get_forecast(cond, target.size(1))

        forecasts = forecasts.view(target.shape)

        if args.data == 'wave':
            mse = (forecasts - target).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        else:
            mse = (forecasts - target).pow(2).mean(dim=-1)

        all_mse.append(mse.data.cpu().numpy())

    return all_mse


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
    xp_config.nt_pred = 40
    args.nt_pred = 40

    test_set = load_dataset(xp_config, train=False)
    sep_net = build_model(xp_config)

    all_mse = compute_mse(xp_config, args.batch_size, test_set, sep_net)
    mse_array = np.concatenate(all_mse, axis=0)
    print(f'MSE at t+40: {np.mean(mse_array.mean(axis=0)[:40])}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="PDE-Driven Spatiotemporal Disentanglement (Moving MNIST testing)")
    p.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the dataset is saved.')
    p.add_argument('--xp_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the model configuration file and checkpoints are saved.')
    p.add_argument('--batch_size', type=int, metavar='BATCH', default=256,
                   help='Batch size used to compute metrics.')
    p.add_argument('--device', type=int, metavar='DEVICE', default=None,
                   help='GPU where the model should be placed when testing (if None, put it on the CPU)')
    args = DotDict(vars(p.parse_args()))
    main(args)
