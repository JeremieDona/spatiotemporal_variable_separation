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
import torch.nn.functional as F

from tqdm import tqdm

from var_sep.utils.helper import save


# Apex
try:
    from apex import amp
    APX_ = 1
except Exception:
    APX_ = 0


def zero_order_loss(s_code_old, s_code_new, skipco):
    if skipco:
        s_code_old = s_code_old[0]
        s_code_new = s_code_new[0]
    return (s_code_old - s_code_new).pow(2).mean()


def ae_loss(cond, target, sep_net, nt_cond, offset, skipco):

    """
    Autoencoding function: we consider in this case that:

    if offset == nt_cond:
        St = S_{t, t+1, ..., t+nt_cond} , Tt = T_{t, t+1, ..., t+nt_cond} is associated to t, i.e
        D(St, Tt) = vt,
    somehow like a backward inference.

    if offset == 0:
        St = S_{t, t+1, ..., t+nt_cond} , Tt = T_{t, t+1, ..., t+nt_cond} is associated to t+nt_cond, i.e
        D(St, Tt) = v(t + nt_cond),
    somehow like estimating how dynamic has moved from t up to t + dt

    This function also returns the result of the application of Es on the first and last seen frames.
    """

    full_data = torch.cat([cond, target], dim=1)
    data_new = full_data[:, -nt_cond:]
    data_old = full_data[:, :nt_cond]

    # Encode spatial information
    s_code_old = sep_net.Es(data_old, return_skip=skipco)
    s_code_new = sep_net.Es(data_new, return_skip=skipco)

    # Encode motion information at a random time
    if offset == 0:
        t_random = np.random.randint(nt_cond, full_data.size(1))
    else:
        t_random = np.random.randint(nt_cond, full_data.size(1) + 1)
    t_code_random = sep_net.Et(full_data[:, t_random - nt_cond:t_random])

    # Decode from S and random T
    if skipco:
        reconstruction = sep_net.decoder(s_code_old[0], t_code_random, skip=s_code_old[1])
    else:
        reconstruction = sep_net.decoder(s_code_old, t_code_random)

    # AE loss
    supervision_data = full_data[:, t_random - offset]
    loss = F.mse_loss(supervision_data, reconstruction, reduction='mean')

    return loss, s_code_new, s_code_old


def train(xp_dir, train_loader, device, sep_net, optimizer, scheduler, apex_amp, epochs, lamb_ae, lamb_s, lamb_t,
          lamb_pred, offset, nt_cond, nt_pred, no_s, skipco):

    if apex_amp and not APX_:
        raise ImportError('Apex not installed (https://github.com/NVIDIA/apex)')

    if apex_amp:
        sep_net, optimizer = amp.initialize(sep_net, optimizer, opt_level='O1')

    if no_s:
        lamb_t = 0
        print("No regularization on T as there is no S")

    assert offset == nt_cond or offset == 0

    try:
        for epoch in range(epochs):

            sep_net.train()
            for bt, (cond, target) in enumerate(tqdm(train_loader)):
                cond, target = cond.to(device), target.to(device)
                total_loss = 0

                optimizer.zero_grad()

                # ##########
                # AUTOENCODE
                # ##########
                ae_loss_value, s_recent, s_old = ae_loss(cond, target, sep_net, nt_cond, offset, skipco)
                total_loss += lamb_ae * ae_loss_value

                # ##################
                # SPATIAL INVARIANCE
                # ##################
                spatial_ode_loss = zero_order_loss(s_old, s_recent, skipco)
                total_loss += lamb_s * spatial_ode_loss

                # #############
                # FORECAST LOSS
                # #############
                full_data = torch.cat([cond, target], dim=1)  # Concatenate all frames
                forecasts, t_codes, s_codes, t_residuals = sep_net.get_forecast(cond, nt_pred + offset, init_s_code=s_old)
                # To make data and target match
                if offset == 0:
                    forecast_offset = nt_cond
                else:
                    forecast_offset = 0
                forecast_loss = F.mse_loss(forecasts, full_data[:, forecast_offset:])
                total_loss += lamb_pred * forecast_loss

                # ################
                # T REGULARIZATION
                # ################
                t_reg = 0.5 * torch.sum(t_codes[:, 0].pow(2), dim=1).mean()
                total_loss += lamb_t * t_reg

                if apex_amp:
                    with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    total_loss.backward()

                optimizer.step()

            if scheduler is not None:
                scheduler.step()

    except KeyboardInterrupt:
        pass

    save(xp_dir, sep_net)
