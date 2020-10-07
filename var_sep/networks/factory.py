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


import numpy as np

from var_sep.networks.conv import DCGAN64Encoder, VGG64Encoder, DCGAN64Decoder, VGG64Decoder, ResNet18
from var_sep.networks.mlp_encdec import MLPEncoder, MLPDecoder
from var_sep.networks.resnet import MLPResnet
from var_sep.networks.utils import init_net


def get_encoder(nn_type, shape, output_size, hidden_size, nt_cond, init_type, init_gain):
    nc = shape[0]
    if nn_type == 'dcgan':
        encoder = DCGAN64Encoder(nc * nt_cond, output_size, hidden_size)
    elif nn_type == 'vgg':
        encoder = VGG64Encoder(nc * nt_cond, output_size, hidden_size)
    elif nn_type == 'resnet':
        encoder = ResNet18(output_size, nc * nt_cond)
    elif nn_type in ['mlp', 'large_mlp']:
        input_size = nt_cond * np.prod(np.array(shape))
        encoder = MLPEncoder(input_size, hidden_size, output_size, 3)

    init_net(encoder, init_type=init_type, init_gain=init_gain)

    return encoder


def get_decoder(nn_type, shape, code_size_t, code_size_s, last_activation, hidden_size, mixing, skipco, init_type,
                init_gain):
    assert not skipco or nn_type in ['dcgan', 'vgg']

    if mixing == 'mul':
        assert code_size_t == code_size_s
        input_size = code_size_t
    else:
        input_size = code_size_t + code_size_s

    nc = shape[0]
    if nn_type == 'dcgan':
        decoder = DCGAN64Decoder(nc, input_size, hidden_size, skipco, last_activation, mixing)
    elif nn_type == 'vgg':
        decoder = VGG64Decoder(nc, input_size, hidden_size, skipco, last_activation, mixing)
    elif nn_type == 'mlp':
        decoder = MLPDecoder(input_size, hidden_size, shape, 3, last_activation, mixing)
    elif nn_type == 'large_mlp':
        decoder = MLPDecoder(input_size, hidden_size, shape, 4, last_activation, mixing)

    init_net(decoder, init_type=init_type, init_gain=init_gain)

    return decoder


def get_resnet(latent_size, n_blocks, hidden_size, init_type, gain_res):
    resnet = MLPResnet(latent_size, n_blocks, hidden_size)
    init_net(resnet, init_type=init_type, init_gain=gain_res)
    return resnet
