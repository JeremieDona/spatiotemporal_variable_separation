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


# Code adapted from SRVP https://github.com/edouardelasalles/srvp; see license notice and copyrights below.

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

import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from var_sep.data.moving_mnist import MovingMNIST
from var_sep.utils.helper import load_json
from var_sep.test.utils import load_model, _ssim_wrapper


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
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)
    train_dataset = load_dataset(xp_config, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
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
    train_iterator = iter(train_loader)
    nt_test = xp_config.nt_cond + args.nt_pred
    predictions = []
    content_swap = []
    cond_swap = []
    target_swap = []
    cond = []
    gt = []
    results = defaultdict(list)
    # Evaluation is done by batch
    for batch in tqdm(test_loader, ncols=80, desc='evaluation'):
        # Data
        x_cond, x_target = batch
        bsz = len(x_cond)
        x_cond = x_cond.to(device)
        x_target = x_target.to(device)
        cond.append(x_cond.cpu().mul(255).byte().permute(0, 1, 3, 4, 2))
        gt.append(x_target.cpu().mul(255).byte().permute(0, 1, 3, 4, 2))

        # Prediction
        x_pred, _, s_code, _ = sep_net.get_forecast(x_cond, nt_test)
        x_pred = x_pred[:, xp_config.nt_cond:]

        # Content swap
        x_swap_cond, x_swap_target = next(train_iterator)
        x_swap_cond = x_swap_cond[:bsz].to(device)
        x_swap_target = x_swap_target[:bsz].to(device)
        x_swap_cond_byte = x_swap_cond.cpu().mul(255).byte()
        x_swap_target_byte = x_swap_target.cpu().mul(255).byte()
        cond_swap.append(x_swap_cond_byte.permute(0, 1, 3, 4, 2))
        target_swap.append(x_swap_target_byte.permute(0, 1, 3, 4, 2))
        x_swap_pred = sep_net.get_forecast(x_swap_cond, nt_test, init_s_code=s_code)[0]
        x_swap_pred = x_swap_pred[:, xp_config.dt:]
        content_swap.append(x_swap_pred.cpu().mul(255).byte().permute(0, 1, 3, 4, 2))

        # Pixelwise quantitative eval
        x_target = x_target.view(-1, args.nt_pred, nc, size, size)
        mse = torch.mean(F.mse_loss(x_pred, x_target, reduction='none'), dim=[3, 4])
        metrics_batch = {
            'mse': mse.mean(2).mean(1).cpu(),
            'psnr': 10 * torch.log10(1 / mse).mean(2).mean(1).cpu(),
            'ssim': _ssim_wrapper(x_pred, x_target).mean(2).mean(1).cpu()
        }
        predictions.append(x_pred.cpu().mul(255).byte().permute(0, 1, 3, 4, 2))

        # Compute metrics for best samples and register
        for name in metrics_batch.keys():
            results[name].append(metrics_batch[name])

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
    np.savez_compressed(os.path.join(args.xp_dir, 'results.npz'), **results)
    np.savez_compressed(os.path.join(args.xp_dir, 'predictions.npz'), predictions=torch.cat(predictions).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'gt.npz'), gt=torch.cat(gt).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'cond.npz'), cond=torch.cat(cond).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'content_swap.npz'), content_swap=torch.cat(content_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'cond_swap.npz'), target_swap=torch.cat(cond_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'target_swap.npz'), target_swap=torch.cat(target_swap).numpy())


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="PDE-Driven Spatiotemporal Disentanglement (Moving MNIST testing)")
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
    args = p.parse_args()
    main(args)
