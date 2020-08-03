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

import torch.nn as nn


class ConstantS(nn.Module):
    def __init__(self, return_value=1, code_size=1):
        super(ConstantS, self).__init__()
        self.code_size = code_size
        self.return_value = return_value

    def forward(self, x, return_skip=False):
        bt_size = len(x)
        return torch.ones(bt_size, self.code_size).to(x) * self.return_value


# The following code is adapted from SRVP https://github.com/edouardelasalles/srvp; see license notice and copyrights
# below.

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


def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == "identity":
        return nn.Identity()

    raise ValueError(f'Activation function `{name}` not yet implemented')


def init_net(net, init_type='normal', init_gain=0.02):
    """
    Initializes the input module with the given parameters.
    Parameters
    ----------
    m : torch.nn.Module
        Module to initialize.
    init_type : str
        'normal', 'xavier', 'kaiming', or 'orthogonal'. Orthogonal initialization types for convolutions and linear
        operations.
    init_gain : float
        Gain to use for the initialization.
    """
    def init_func(m):  # Define the initialization function
        classname = m.__class__.__name__
        if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname == 'BatchNorm2d':
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # Iterate the initialization function on the modules
