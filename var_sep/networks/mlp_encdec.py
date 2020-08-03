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


import torch

import numpy as np
import torch.nn as nn

from var_sep.networks.mlp import MLP
from var_sep.networks.utils import activation_factory


class MLPEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nlayers):
        super(MLPEncoder, self).__init__()
        self.mlp = MLP(input_size, hidden_size, output_size, nlayers)

    def forward(self, x, return_skip=False):
        x = x.view(len(x), -1)
        return self.mlp(x)


class MLPDecoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_shape, nlayers, last_activation, mixing):
        super(MLPDecoder, self).__init__()
        self.output_shape = output_shape
        self.mixing = mixing
        self.mlp = MLP(latent_size, hidden_size, np.prod(np.array(output_shape)), nlayers)
        self.last_activation = activation_factory(last_activation)

    def forward(self, z1, z2, skip=None):
        if self.mixing == 'concat':
            z = torch.cat([z1, z2], dim=1)
        else:
            z = z1 * z2
        x = self.mlp(z)
        x = self.last_activation(x)
        return x.view([-1] + self.output_shape)
