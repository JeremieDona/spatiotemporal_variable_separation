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
import itertools

import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

from var_sep.data.chairs import Chairs
from var_sep.utils.helper import DotDict, load_json
from var_sep.utils.ssim import ssim_loss
from var_sep.networks.conv import DCGAN64Encoder, VGG64Encoder, DCGAN64Decoder, VGG64Decoder, ResNet18
from var_sep.networks.mlp_encdec import MLPEncoder, MLPDecoder
from var_sep.networks.model import SeparableNetwork


def _ssim_wrapper(pred, gt):
    bsz, nt_pred = pred.shape[0], pred.shape[1]
    img_shape = pred.shape[2:]
    ssim = ssim_loss(pred.reshape(bsz * nt_pred, *img_shape), gt.reshape(bsz * nt_pred, *img_shape), max_val=1., reduction='none')
    return ssim.mean(dim=[2, 3]).view(bsz, nt_pred, img_shape[0])


class SwapDataset(Chairs):

    def __init__(self, train, data_root, nt_cond, seq_len=20, image_size=64):
        super(SwapDataset, self).__init__(train, data_root, nt_cond, seq_len=seq_len, image_size=image_size)

    def __getitem__(self, index):
        idx_content = np.random.randint(self.stop_idx - self.start_idx)
        id_st_content = np.random.randint(self.max_length - self.seq_len)
        sequence = torch.tensor(self.get_sequence(index, chosen_idx=idx_content,
                                                  chosen_id_st=id_st_content) / 255).permute(0, 3, 1, 2).float()
        sequence_swap = torch.tensor(self.get_sequence(index,
                                                       chosen_idx=idx_content) / 255).permute(0, 3, 1, 2).float()
        return (sequence[:self.nt_cond], sequence[self.nt_cond:],
                sequence_swap[:self.nt_cond].unsqueeze(0), sequence_swap[self.nt_cond:].unsqueeze(0))


def load_dataset(args, train=False):
    return Chairs(train, args.data_dir, args.nt_cond, seq_len=args.nt_cond + args.nt_pred)


def build_model(args):
    Es = torch.load(os.path.join(args.xp_dir, 'ov_Es.pt'), map_location=args.device).to(args.device)
    Et = torch.load(os.path.join(args.xp_dir, 'ov_Et.pt'), map_location=args.device).to(args.device)
    t_resnet = torch.load(os.path.join(args.xp_dir, 't_resnet.pt'), map_location=args.device).to(args.device)
    decoder = torch.load(os.path.join(args.xp_dir, 'decoder.pt'), map_location=args.device).to(args.device)
    sep_net = SeparableNetwork(Es, Et, t_resnet, decoder, args.nt_cond, args.skipco)
    sep_net.eval()
    return sep_net


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
    xp_config.n_object = 1

    ##################################################################################################################
    # Load test data
    ##################################################################################################################
    print('Loading data...')
    test_dataset = load_dataset(xp_config, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,  pin_memory=True)
    swap_dataset = SwapDataset(False, args.data_dir, xp_config.nt_cond, seq_len=xp_config.nt_cond + args.nt_pred)
    swap_loader = DataLoader(swap_dataset, batch_size=args.batch_size, pin_memory=True)
    nc = 3
    size = 64

    ##################################################################################################################
    # Load model
    ##################################################################################################################
    print('Loading model...')
    sep_net = build_model(xp_config)

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
        _, _, s_codes, _ = sep_net.get_forecast(x_cond, nt_test)

        # Content swap
        x_swap_cond, x_swap_target = batch
        x_swap_cond = x_swap_cond.to(device)
        x_swap_target = x_swap_target.to(device)
        x_swap_cond_byte = x_cond.cpu().mul(255).byte()
        x_swap_target_byte = x_swap_target.cpu().mul(255).byte()
        cond_swap.append(x_swap_cond_byte.permute(0, 1, 3, 4, 2))
        target_swap.append(x_swap_target_byte.permute(0, 1, 3, 4, 2))
        x_swap_pred = sep_net.get_forecast(x_swap_cond, nt_test, init_s_code=s_codes[:, 0])[0]
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
        print(name, res.mean(), '+/-', 1.960 * res.std() / np.sqrt(len(res)))

    ##################################################################################################################
    # Save samples
    ##################################################################################################################
    np.savez_compressed(os.path.join(args.xp_dir, 'results_swap.npz'), **results)
    np.savez_compressed(os.path.join(args.xp_dir, 'content_swap_gt.npz'), gt_swap=torch.cat(gt_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'content_swap_test.npz'), content_swap=torch.cat(content_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'cond_swap_test.npz'), cond_swap=torch.cat(cond_swap).numpy())
    np.savez_compressed(os.path.join(args.xp_dir, 'target_swap_test.npz'), target_swap=torch.cat(target_swap).numpy())


if __name__ == '__main__':
    p = argparse.ArgumentParser(prog="PDE-Driven Spatiotemporal Disentanglement (3D Warehouse Chairs content swap testing)")
    p.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the dataset is saved.')
    p.add_argument('--xp_dir', type=str, metavar='DIR', required=True,
                   help='Directory where the model configuration file and checkpoints are saved.')
    p.add_argument('--batch_size', type=int, metavar='BATCH', default=16,
                   help='Batch size used to compute metrics.')
    p.add_argument('--nt_pred', type=int, metavar='PRED', required=True,
                   help='Total of frames to predict.')
    p.add_argument('--device', type=int, metavar='DEVICE', default=None,
                   help='GPU where the model should be placed when testing (if None, put it on the CPU)')
    p.add_argument('--test_seed', type=int, metavar='SEED', default=1,
                   help='Manual seed.')
    args = DotDict(vars(p.parse_args()))
    main(args)
