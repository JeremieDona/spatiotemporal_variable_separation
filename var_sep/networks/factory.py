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

from var_sep.networks.conv import (DCGAN64Encoder, VGG64Encoder, DCGAN64Decoder, VGG64Decoder, ResNet18, EncoderSST,
                                   DecoderSST, DecoderSST_Skip)
from var_sep.networks.mlp_encdec import MLPEncoder, MLPDecoder
from var_sep.networks.resnet import MLPResnet, ConvResnet
from var_sep.networks.utils import init_net


def get_encoder(nn_type, shape, output_size, hidden_size, n_layers, nt_cond, init_type, init_gain):
    nc = shape[0]
    dim = shape[-1]
    if nn_type == 'dcgan':
        assert dim == 64
        encoder = DCGAN64Encoder(nc * nt_cond, output_size, hidden_size)
    elif nn_type == 'vgg':
        assert dim in [32, 64]
        encoder = VGG64Encoder(nc * nt_cond, output_size, hidden_size, vgg32=dim == 32)
    elif nn_type == 'resnet':
        encoder = ResNet18(output_size, nc * nt_cond)
    elif nn_type == 'encoderSST':
        encoder = EncoderSST(nc * nt_cond, output_size)
    elif nn_type == 'mlp':
        input_size = nt_cond * np.prod(np.array(shape))
        encoder = MLPEncoder(input_size, hidden_size, output_size, n_layers)

    init_net(encoder, init_type=init_type, init_gain=init_gain)

    return encoder


def get_decoder(nn_type, shape, code_size_t, code_size_s, last_activation, hidden_size, n_layers, mixing, skipco,
                init_type, init_gain):
    assert not skipco or nn_type in ['dcgan', 'vgg', 'decoderSST']

    if mixing == 'mul':
        assert code_size_t == code_size_s
        input_size = code_size_t
    else:
        input_size = code_size_t + code_size_s

    nc = shape[0]
    dim = shape[-1]
    if nn_type == 'dcgan':
        assert dim == 64
        decoder = DCGAN64Decoder(nc, input_size, hidden_size, skipco, last_activation, mixing)
    elif nn_type == 'vgg':
        assert dim in [32, 64]
        decoder = VGG64Decoder(nc, input_size, hidden_size, skipco, last_activation, mixing, vgg32=dim == 32)
    elif nn_type == 'mlp':
        decoder = MLPDecoder(input_size, hidden_size, shape, n_layers, last_activation, mixing)
    elif nn_type == 'decoderSST':
        assert mixing == 'concat'
        if skipco:
            decoder = DecoderSST_Skip(input_size, nc, last_activation)
        else:
            decoder = DecoderSST(input_size, nc, last_activation)

    init_net(decoder, init_type=init_type, init_gain=init_gain)

    return decoder


def get_resnet(latent_size, n_blocks, hidden_size, init_type, gain_res, fully_conv=False):
    if fully_conv:
        resnet = ConvResnet(latent_size, n_blocks=n_blocks, nf=hidden_size)
    else:
        resnet = MLPResnet(latent_size, n_blocks, hidden_size)

    init_net(resnet, init_type=init_type, init_gain=gain_res)

    return resnet
