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


# Code heavily modified from SRVP https://github.com/edouardelasalles/srvp; see license notice and copyrights below.

# Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

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
import random
import torch
import math
import itertools

import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm

from var_sep.data.moving_mnist import MovingMNIST
from var_sep.utils.helper import DotDict, load_json
from var_sep.test.utils import load_model, _ssim_wrapper


class SwapDataset(Dataset):

    def __init__(self, data_dir, seq_len, nt_cond, n_object):
        self.seq_len = seq_len
        self.n_object = n_object
        self.nt_cond = nt_cond
        self.frame_size = 64
        self.object_size = 28
        self.digits_permutation = np.random.permutation(10000)
        self.trajectories = np.load(os.path.join(data_dir, f'mmnist_test_{n_object}digits_{self.frame_size}.npz'),
                                    allow_pickle=True)['latents']
        self.images = datasets.MNIST(data_dir, train=False, download=True)

    def __len__(self):
        return 10000 // self.n_object

    def __getitem__(self, index):
        # get trajectory
        x_trajectory_reverse = np.zeros((self.seq_len, 1, self.frame_size, self.frame_size), dtype=np.float32)
        x_swap = np.zeros((math.factorial(self.n_object), self.seq_len, 1, self.frame_size, self.frame_size),
                          dtype=np.float32)
        img = [self.images[self.digits_permutation[index + i * (10000 // self.n_object)]][0]
               for i in range(self.n_object)]
        trajectory = self.trajectories[:, index]
        trajectory_reverse = self.trajectories[:, len(self) - index - 1]
        for t in range(self.seq_len):
            for i in range(self.n_object):
                sx, sy, _, _ = trajectory_reverse[t, i]
                x_trajectory_reverse[t, 0, sx:sx + self.object_size, sy:sy + self.object_size] += img[i]
            for j, reordering in enumerate(itertools.permutations(range(self.n_object))):
                for i in range(self.n_object):
                    sx, sy, _, _ = trajectory[t, i]
                    x_swap[j, t, 0, sx:sx + self.object_size, sy:sy + self.object_size] += img[reordering[i]]
        x_trajectory_reverse[x_trajectory_reverse > 255] = 255
        x_swap[x_swap > 255] = 255
        return (torch.tensor(x_trajectory_reverse[:self.nt_cond]) / 255,
                torch.tensor(x_trajectory_reverse[self.nt_cond:]) / 255,
                torch.tensor(x_swap[:, :self.nt_cond]) / 255, torch.tensor(x_swap[:, self.nt_cond:]) / 255)


def load_dataset(args, train=False):
    return MovingMNIST.make_dataset(args.data_dir, 64, args.nt_cond, args.nt_cond + args.nt_pred, 4, True,
                                    args.n_object, train)


def main(args):
    ##################################################################################################################
    # Setup
    ##################################################################################################################
    # -- Device handling (CPU, GPU)
    if args.device is None:
        device = torch.device('cpu')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    # Seed
    random.seed(args.test_seed)
    np.random.seed(args.test_seed)
    torch.manual_seed(args.test_seed)
    # Load XP config
    xp_config = load_json(os.path.join(args.xp_dir, 'params.json'))
    xp_config.device = device
    xp_config.data_dir = args.data_dir
    xp_config.xp_dir = args.xp_dir
    xp_config.nt_pred = args.nt_pred

    ##################################################################################################################
    # Load test data
    ##################################################################################################################
    print('Loading data...')
    test_dataset = load_dataset(xp_config, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,  pin_memory=True)
    swap_dataset = SwapDataset(args.data_dir, xp_config.nt_cond + args.nt_pred, xp_config.nt_cond, xp_config.n_object)
    swap_loader = DataLoader(swap_dataset, batch_size=args.batch_size, pin_memory=True)
    nc = 1
    size = 64

    ##################################################################################################################
    # Load model
    ##################################################################################################################
    print('Loading model...')
    sep_net = load_model(xp_config, args.epoch)

    ##################################################################################################################
    # Eval
    ##################################################################################################################
    print('Generating samples...')
    torch.set_grad_enabled(False)
    swap_iterator = iter(swap_loader)
    nt_test = xp_config.nt_cond + args.nt_pred
    gt_swap = []
    content_swap = []
    cond_swap = []
    target_swap = []
    results = defaultdict(list)
    # Evaluation is done by batch
    for batch in tqdm(test_loader, ncols=80, desc='evaluation'):
        # Data
        x_cond, x_target, _, x_gt_swap = next(swap_iterator)
        x_gt_swap = x_gt_swap.to(device)
        x_cond = x_cond.to(device)

        # Extraction of S
        _, _, s_code, _ = sep_net.get_forecast(x_cond, nt_test)

        # Content swap
        x_swap_cond, x_swap_target = batch
        x_swap_cond = x_swap_cond.to(device)
        x_swap_target = x_swap_target.to(device)
        x_swap_cond_byte = x_cond.cpu().mul(255).byte()
        x_swap_target_byte = x_swap_target.cpu().mul(255).byte()
        cond_swap.append(x_swap_cond_byte.permute(0, 1, 3, 4, 2))
        target_swap.append(x_swap_target_byte.permute(0, 1, 3, 4, 2))
        x_swap_pred = sep_net.get_forecast(x_swap_cond, nt_test, init_s_code=s_code)[0]
        x_swap_pred = x_swap_pred[:, xp_config.nt_cond:]
        content_swap.append(x_swap_pred.cpu().mul(255).byte().permute(0, 1, 3, 4, 2))
        gt_swap.append(x_gt_swap[:, 0].cpu().mul(255).byte().permute(0, 1, 3, 4, 2))

        # Pixelwise quantitative eval
        x_gt_swap = x_gt_swap.view(-1, xp_config.n_object, args.nt_pred, nc, size, size).to(device)
        metrics_batch = {'mse': [], 'psnr': [], 'ssim': []}
        for j, reordering in enumerate(itertools.permutations(range(xp_config.n_object))):
            mse = torch.mean(F.mse_loss(x_swap_pred, x_gt_swap[:, j], reduction='none'), dim=[3, 4])
            metrics_batch['mse'].append(mse.mean(2).mean(1).cpu())
            metrics_batch['psnr'].append(10 * torch.log10(1 / mse).mean(2).mean(1).cpu())
            metrics_batch['ssim'].append(_ssim_wrapper(x_swap_pred, x_gt_swap[:, j]).mean(2).mean(1).cpu())

        # Compute metrics for best samples and register
        results['mse'].append(torch.min(torch.stack(metrics_batch['mse']), 0)[0])
        results['psnr'].append(torch.max(torch.stack(metrics_batch['psnr']), 0)[0])
        results['ssim'].append(torch.max(torch.stack(metrics_batch['ssim']), 0)[0])

    ##################################################################################################################
    # Print results
    ##################################################################################################################
    print('\n')
    print('Results:')
    for name in results.keys():
        res = torch.cat(results[name]).numpy()
        results[name] = res
        print(name, res.mean())

    ##################################################################################################################
    # Save samples
    ##################################################################################################################
    np.savez_compressed(os.path.join(args.xp_dir, 'results_swap.npz'), **results)
    np.savez_compressed(os.path.join(args.xp_dir, 'content_swap_gt.npz'), gt_swap=torch.cat(gt_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'content_swap_test.npz'), content_swap=torch.cat(content_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'cond_swap_test.npz'), cond_swap=torch.cat(cond_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'target_swap_test.npz'), target_swap=torch.cat(target_swap).numpy())


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="PDE-Driven Spatiotemporal Disentanglement (Moving MNIST content swap testing)")
    p.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the dataset is saved.')
    p.add_argument('--xp_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the model configuration file and checkpoints are saved.')
    p.add_argument('--epoch', type=int, metavar='EPOCH', default=None,
                   help='If specified, loads the checkpoint of the corresponding epoch number.')
    p.add_argument('--batch_size', type=int, metavar='BATCH', default=16,
                   help='Batch size used to compute metrics.')
    p.add_argument('--nt_pred', type=int, metavar='PRED', required=True,
                   help='Total of frames to predict.')
    p.add_argument('--device', type=int, metavar='DEVICE', default=None,
                   help='GPU where the model should be placed when testing (if None, on the CPU)')
    p.add_argument('--test_seed', type=int, metavar='SEED', default=1,
                   help='Manual seed.')
    args = DotDict(vars(p.parse_args()))
    main(args)
