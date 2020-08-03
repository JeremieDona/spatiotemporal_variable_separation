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


import json
import os
import torch
import yaml


def save(elem_xp_path, sep_net):
    to_save = True
    while to_save:
        try:
            torch.save(sep_net.Et, os.path.join(elem_xp_path, 'ov_Et.pt'))
            torch.save(sep_net.Es, os.path.join(elem_xp_path, 'ov_Es.pt'))
            torch.save(sep_net.decoder, os.path.join(elem_xp_path, 'decoder.pt'))
            torch.save(sep_net.t_resnet, os.path.join(elem_xp_path, 't_resnet.pt'))
            to_save = False
        except:
            print("unable to save all files")


# The following code is adapted from SRVP https://github.com/edouardelasalles/srvp; see license notice and copyrights
# below.

# # Copyright 2020 Mickael Chen, Edouard Delasalles, Jean-Yves Franceschi, Patrick Gallinari, Sylvain Lamprier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class DotDict(dict):
    """
    Dot notation access to dictionary attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    """
    Loads a yaml input file.
    """
    with open(path, 'r') as f:
        opt = yaml.safe_load(f)
    return DotDict(opt)


def load_json(path):
    """
    Loads a json input file.
    """
    with open(path, 'r') as f:
        opt = json.load(f)
    return DotDict(opt)
