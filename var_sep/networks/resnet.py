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


import torch.nn as nn

from var_sep.networks.conv import make_conv_block
from var_sep.networks.mlp import MLP


class MLPResBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPResBlock, self).__init__()
        self.mlp = MLP(input_size, hidden_size, input_size, 3)

    def forward(self, x):
        residual = self.mlp(x)
        return x + residual, residual


class MLPResnet(nn.Module):
    def __init__(self, input_size, n_blocks, hidden_size):
        super(MLPResnet, self).__init__()
        self.in_size = input_size
        self.n_blocks = n_blocks
        blocks = []
        for i in range(self.n_blocks):
            blocks += [MLPResBlock(input_size, hidden_size)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, return_res=True):
        residuals = []
        for j in range(self.n_blocks):
            x, res = self.blocks[j].forward(x)
            residuals.append(res)
        if return_res:
            return x, residuals
        else:
            return x


class ConvResBlock(nn.Module):
    def __init__(self, in_c, out_c, nf=64):
        super(ConvResBlock, self).__init__()
        self.conv = nn.Sequential(
            make_conv_block(nn.Conv2d(in_c, nf, 3, padding=1), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf, nf, 3, padding=1), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf, out_c, 3, padding=1), activation='none')
        )
        if in_c != out_c:
            # conv -- bn -- leaky_relu
            self.up = make_conv_block(nn.Conv2d(in_c, out_c, 3, padding=1), activation='none')
        else:
            self.up = nn.Identity()

    def forward(self, x):
        residual = self.conv(x)
        x = self.up(x) + residual
        return x, residual


class ConvResnet(nn.Module):
    def __init__(self, in_c, n_blocks=1, nf=64):
        super(ConvResnet, self).__init__()
        self.n_blocks = n_blocks
        self.resblock_modules = nn.ModuleList()
        for i in range(n_blocks):
            self.resblock_modules.append(ConvResBlock(in_c, in_c, nf=nf))

    def forward(self, x, return_res=True):
        residuals = []
        for i, residual_block in enumerate(self.resblock_modules):
            x, residual = residual_block(x)
            residuals.append(residual)
        if return_res:
            return x, residuals
        return x
