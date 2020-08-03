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
