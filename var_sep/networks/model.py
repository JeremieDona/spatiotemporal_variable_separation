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


class SeparableNetwork(nn.Module):

    def __init__(self, Es, Et, t_resnet, decoder, nt_cond, skipco):
        super(SeparableNetwork, self).__init__()

        assert isinstance(Es, nn.Module)
        assert isinstance(Et, nn.Module)
        assert isinstance(t_resnet, nn.Module)
        assert isinstance(decoder, nn.Module)

        # Networks
        self.Es = Es
        self.Et = Et
        self.decoder = decoder
        self.t_resnet = t_resnet

        # Attributes
        self.nt_cond = nt_cond
        self.skipco = skipco

        # Gradient-enabling parameter
        self.__grad = True

    @property
    def grad(self):
        return self.__grad

    @grad.setter
    def grad(self, grad):
        assert isinstance(grad, bool)
        self.__grad = grad

    def get_forecast(self, cond, n_forecast, init_t_code=None, init_s_code=None):
        s_codes = []
        t_codes = []
        forecasts = []
        t_residuals = []

        if init_s_code is None:
            s_code = self.Es(cond, return_skip=self.skipco)
        else:
            s_code = init_s_code
        if self.skipco:
            s_code, s_skipco = s_code
        else:
            s_skipco = None

        if init_t_code is None:
            t_code = self.Et(cond)
        else:
            t_code = init_t_code

        s_codes.append(s_code)
        t_codes.append(t_code)

        # Decode first frame
        forecast = self.decoder(s_code, t_code, skip=s_skipco)
        forecasts.append(forecast)

        # Forward prediction
        for t in range(1, n_forecast):
            t_code, t_res = self.t_resnet(t_code)
            t_codes.append(t_code)
            t_residuals.append(t_res)
            forecast = self.decoder(s_code, t_code, skip=s_skipco)
            forecasts.append(forecast)

        # Stack predictions
        forecasts = torch.cat([x.unsqueeze(1) for x in forecasts], dim=1)
        t_codes = torch.cat([x.unsqueeze(1) for x in t_codes], dim=1)
        s_codes = torch.cat([x.unsqueeze(1) for x in s_codes], dim=1)

        return forecasts, t_codes, s_codes, t_residuals
