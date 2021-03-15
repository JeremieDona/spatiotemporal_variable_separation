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


import torch

import torch.nn as nn

from var_sep.networks.utils import activation_factory


def make_conv_block(conv, activation, bn=True):
    """
    Supplements a convolutional block with activation functions and batch normalization.
    Parameters
    ----------
    conv : torch.nn.Module
        Convolutional block.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function after
        the convolution.
    bn : bool
        Whether to add batch normalization after the activation.
    """
    out_channels = conv.out_channels
    modules = [conv]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if activation != 'none':
        modules.append(activation_factory(activation))
    return nn.Sequential(*modules)


class BaseEncoder(nn.Module):
    """
    Module implementing the encoders forward method.
    Attributes
    ----------
    nh : int
        Number of dimensions of the output flat vector.
    """
    def __init__(self, nh):
        """
        Parameters
        ----------
        nh : int
            Number of dimensions of the output flat vector.
        """
        super(BaseEncoder, self).__init__()
        self.nh = nh

    def forward(self, x, return_skip=False):
        """
        Parameters
        ----------
        x : torch.Tensor
            Encoder input.
        return_skip : bool
            Whether to extract and return, besides the network output, skip connections.
        """
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        skips = []
        h = x
        for layer in self.conv:
            h = layer(h)
            skips.append(h)
        h = self.last_op(h).view(-1, self.nh)
        if return_skip:
            return h, skips[::-1]
        return h


class DCGAN64Encoder(BaseEncoder):
    """
    Module implementing the DCGAN encoder.
    """
    def __init__(self, nc, nh, nf):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        """
        super(DCGAN64Encoder, self).__init__(nh)
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(nc, nf, 4, 2, 1), activation='leaky_relu', bn=False),
            make_conv_block(nn.Conv2d(nf, nf * 2, 4, 2, 1), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1), activation='leaky_relu')
        ])
        self.last_op = nn.Sequential(nn.Flatten(), nn.Linear(nf * 8 * 4 * 4, nh))


class VGG64Encoder(BaseEncoder):
    """
    Module implementing the VGG encoder.
    """
    def __init__(self, nc, nh, nf, vgg32=False):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the input data.
        nh : int
            Number of dimensions of the output flat vector.
        nf : int
            Number of filters per channel of the first convolution.
        vgg32 : bool
            Whether to adapt the architecture for a 32x32 input.
        """
        super(VGG64Encoder, self).__init__(nh)
        self.conv = nn.ModuleList([
            nn.Sequential(
                make_conv_block(nn.Conv2d(nc, nf, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf, nf, 3, 1, 1), activation='leaky_relu')
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf, nf * 2, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), activation='leaky_relu')
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), activation='leaky_relu')
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(nf * 4, nf * 8, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1), activation='leaky_relu')
            )
        ])
        self.last_op = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0) if not vgg32 else nn.Identity(),
            make_conv_block(nn.Conv2d(nf * 8, nh, 4, 1, 0), activation='none')
        )


class BaseDecoder(nn.Module):
    """
    Module implementing the decoders forward method.

    Attributes
    ----------
    ny : int
        Number of dimensions of the output flat vector.
    skip : bool
        Whether to include skip connections into the decoder.
    mixing : str
        'mul' or 'concat'. Whether to multiply both inputs, or concatenate them.
    """
    def __init__(self, ny, skip, last_activation, mixing):
        """
        Parameters
        ----------
        ny : int
            Number of dimensions of the output flat vector.
        skip : bool
            Whether to include skip connections into the decoder.
        last_activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function after
            the last convolution.
        mixing : str
            'mul' or 'concat'. Whether to multiply both inputs, or concatenate them.
        """
        super(BaseDecoder, self).__init__()
        self.ny = ny
        self.skip = skip
        self.mixing = mixing
        self.last_activation = activation_factory(last_activation)

    def forward(self, z1, z2, skip=None):
        """
        Parameters
        ----------
        z1 : torch.Tensor
            First decoder input (S).
        z2 : torch.Tensor
            Second decoder input (S).
        skip : list
            List of tensors representing skip connections.
        """
        assert skip is None and not self.skip or self.skip and skip is not None

        if self.mixing == 'concat':
            z = torch.cat([z1, z2], dim=1)
        else:
            z = z1 * z2

        h = self.first_upconv(z.view(*z.shape, 1, 1))
        for i, layer in enumerate(self.conv):
            if skip is not None:
                h = torch.cat([h, skip[i]], 1)
            h = layer(h)
        return self.last_activation(h)


class DCGAN64Decoder(BaseDecoder):
    """
    Module implementing the DCGAN decoder.
    """
    def __init__(self, nc, ny, nf, skip, last_activation, mixing):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder architecture.
        skip : list
            List of tensors representing skip connections.
        last_activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function after
            the last convolution.
        mixing : str
            'mul' or 'concat'. Whether to multiply both inputs, or concatenate them.
        """
        super(DCGAN64Decoder, self).__init__(ny, skip, last_activation, mixing)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = make_conv_block(nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0), activation='leaky_relu')
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(nf * 8 * coef, nf * 4, 4, 2, 1), activation='leaky_relu'),
            make_conv_block(nn.ConvTranspose2d(nf * 4 * coef, nf * 2, 4, 2, 1), activation='leaky_relu'),
            make_conv_block(nn.ConvTranspose2d(nf * 2 * coef, nf, 4, 2, 1), activation='leaky_relu'),
            nn.ConvTranspose2d(nf * coef, nc, 4, 2, 1)
        ])


class VGG64Decoder(BaseDecoder):
    """
    Module implementing the VGG decoder.
    """
    def __init__(self, nc, ny, nf, skip, last_activation, mixing, vgg32=False):
        """
        Parameters
        ----------
        nc : int
            Number of channels in the output shape.
        ny : int
            Number of dimensions of the input flat vector.
        nf : int
            Number of filters per channel of the first convolution of the mirror encoder architecture.
        skip : list
            List of tensors representing skip connections.
        last_activation : str
            'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function after
            the last convolution.
        mixing : str
            'mul' or 'concat'. Whether to multiply both inputs, or concatenate them.
        vgg32 : bool
            Whether to adapt the architecture for a 32x32 output.
        """
        super(VGG64Decoder, self).__init__(ny, skip, last_activation, mixing)
        # decoder
        coef = 2 if skip else 1
        self.first_upconv = nn.Sequential(
            make_conv_block(nn.ConvTranspose2d(ny, nf * 8, 4, 1, 0), activation='leaky_relu'),
            nn.Upsample(scale_factor=2, mode='nearest') if not vgg32 else nn.Identity()
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 8 * coef, nf * 8, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 8, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 4 * coef, nf * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * 2 * coef, nf * 2, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(nf * 2, nf, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')
            ),
            nn.Sequential(
                make_conv_block(nn.Conv2d(nf * coef, nf, 3, 1, 1), activation='leaky_relu'),
                nn.ConvTranspose2d(nf, nc, 3, 1, 1),
            ),
        ])


class EncoderSST(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderSST, self).__init__()
        self.conv1 = nn.Sequential(
                make_conv_block(nn.Conv2d(in_c, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'))
        self.conv2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(64, 64 * 2, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1), activation='leaky_relu'),
            )
        self.conv3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(64 * 2, 64 * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1), activation='leaky_relu'),
            )
        self.conv4 = nn.Sequential(
                make_conv_block(nn.Conv2d(64 * 4, 64 * 8, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 8, out_c, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(out_c, out_c, 3, 1, 1), activation='none', bn=False),
            )

    def forward(self, x, return_skip=False):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        # skipcos
        h1 = self.conv1(x)  # 64, 64, 64
        h2 = self.conv2(h1)  # 128, 32, 32
        h3 = self.conv3(h2)  # 256, 16, 16
        # code
        h4 = self.conv4(h3)  # 512, 16, 16
        if return_skip:
            return h4, [h3, h2, h1]
        return h4


class DecoderSST_Skip(nn.Module):
    def __init__(self, in_c, out_c, out_f):
        super(DecoderSST_Skip, self).__init__()
        self.conv1 = nn.Sequential(
                make_conv_block(nn.Conv2d(in_c, 256, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(256, 256, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(256, 128, 3, 1, 1), activation='leaky_relu'),
        )
        self.conv2 = nn.Sequential(
                make_conv_block(nn.Conv2d(256+128, 128, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(128, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')  # 64, 64, 64
        )
        self.conv3 = nn.Sequential(
                make_conv_block(nn.Conv2d(128+64, 128, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(128, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')  # 64, 64, 64
        )
        self.conv4 = nn.Sequential(
                make_conv_block(nn.Conv2d(64*2, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, out_c, 3, 1, 1), activation='leaky_relu'),
        )
        self.out_f = activation_factory(out_f)

    def forward(self, s_code, t_code, skip):
        h3, h2, h1 = skip
        x = torch.cat([s_code, t_code], dim=1)
        out = self.conv1(x)  # 128, 16, 16
        out = torch.cat([h3, out], dim=1)
        out = self.conv2(out)  # 64, 32, 32
        out = torch.cat([h2, out], dim=1)
        out = self.conv3(out)  # 64, 64, 64
        out = torch.cat([h1, out], dim=1)
        out = self.conv4(out)
        return self.out_f(out)


class DecoderSST(nn.Module):
    def __init__(self, in_c, out_c, out_f):
        super(DecoderSST, self).__init__()
        self.conv1 = nn.Sequential(
                make_conv_block(nn.Conv2d(in_c, 256, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(256, 256, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(256, 128, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')
        )  # 128, 32, 32

        self.conv2 = nn.Sequential(
                make_conv_block(nn.Conv2d(128, 128, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(128, 128, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(128, 64, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest'),
        )  # 64, 64, 64
        self.conv3 = nn.Sequential(
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, out_c, 3, 1, 1), activation='leaky_relu'))

        self.out_f = activation_factory(out_f)

    def forward(self, s_code, t_code, skip=None):
        x = torch.cat([s_code, t_code], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out_f(x)


# The following implementation of ResNet18 is taken from DrNet (https://github.com/edenton/drnet-py).
# All credit goes to its authors Emily Denton and Vighnesh Birodkar.


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, pose_dim, nc=3, out_f=None):
        block = BasicBlock
        layers = [2, 2, 2, 2, 2]
        self.inplanes = 64
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=5, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.conv_out = nn.Conv2d(512, pose_dim, kernel_size=3)
        self.bn_out = nn.BatchNorm2d(pose_dim)
        self.out_function = activation_factory(out_f)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_skip=False):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv_out(x)
        x = self.out_function(x)

        x = x.view(len(x), -1)

        return x
