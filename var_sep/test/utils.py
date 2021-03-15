import os
import torch

from var_sep.networks.model import SeparableNetwork
from var_sep.utils.ssim import ssim_loss


def load_model(args, epoch_number=None):
    append = f'_{epoch_number}' if epoch_number is not None else ''
    Es = torch.load(os.path.join(args.xp_dir, f'ov_Es{append}.pt'), map_location=args.device).to(args.device)
    Et = torch.load(os.path.join(args.xp_dir, f'ov_Et{append}.pt'), map_location=args.device).to(args.device)
    t_resnet = torch.load(os.path.join(args.xp_dir, f't_resnet{append}.pt'), map_location=args.device).to(args.device)
    decoder = torch.load(os.path.join(args.xp_dir, f'decoder{append}.pt'), map_location=args.device).to(args.device)
    sep_net = SeparableNetwork(Es, Et, t_resnet, decoder, args.nt_cond, args.skipco)
    sep_net.eval()
    return sep_net


def _ssim_wrapper(pred, gt):
    bsz, nt_pred = pred.shape[0], pred.shape[1]
    img_shape = pred.shape[2:]
    ssim = ssim_loss(pred.reshape(bsz * nt_pred, *img_shape), gt.reshape(bsz * nt_pred, *img_shape), max_val=1.,
                     reduction='none')
    return ssim.mean(dim=[2, 3]).view(bsz, nt_pred, img_shape[0])
