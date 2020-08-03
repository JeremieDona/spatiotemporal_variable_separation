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


import argparse
import os
import torch

import numpy as np

from functools import partial
from torchdiffeq import odeint
from tqdm import trange


def decreasing_energy_source(t, invT0, f0):
    return f0 * np.exp(-invT0 * t)


def circle_idx(center=(32, 32), r=5):
    cols, rows = np.meshgrid(range(64), range(64))
    idx = (cols - center[0])**2 + (rows-center[1])**2 < r**2
    return idx


def derivative(t, y, source, c, nx, ny, dx, dz, circular, order):
    """
    The wave equation is defined by w'' = c2 LAP w.

    We generate the data by using the partial differential equation
    on the full state (w,w'):

            (w, w') = (w', c2 LAP w)

    This function implements the mapping:
    (w, w') --> (w', w'')

    via finite difference scheme for data generation to allow integration
    using torchdiffeq further.
    """

    # if a circular source is to be used
    if circular:
        circle_mask = circle_idx().astype(float)
        circle_mask = torch.tensor(circle_mask, dtype=torch.float)
    else:
        circle_mask = circle_idx(r=1).astype(float)
        circle_mask = torch.tensor(circle_mask, dtype=torch.float)

    # State : (2, 64, 64)
    state = y[0]  # (64, 64)
    state_diff = y[1]  # (64, 64)

    state_yy = torch.zeros(state_diff.shape)
    state_xx = torch.zeros(state_diff.shape)

    # Calculate partial derivatives, be careful around the boundaries
    if order == 3:
        # Third order
        for i in range(1, ny - 1):
            state_yy[:, i] = state[:, i + 1] - 2 * state[:, i] + state[:, i - 1]

        for j in range(1, nx - 1):
            state_xx[j, :] = state[j - 1, :] - 2 * state[j, :] + state[j + 1, :]
    elif order == 5:
        # Fifth order
        for i in range(2, nx - 2):
            state_yy[:, i] = (-1 / 12 * state[:, i + 2] + 4 / 3 * state[:, i + 1] - 5 / 2 * state[:, i]
                              + 4 / 3 * state[:, i - 1] - 1 / 12 * state[:, i - 2])
        for j in range(2, ny - 2):
            state_xx[j, :] = (-1 / 12 * state[j + 2, :] + 4 / 3 * state[j + 1, :] - 5 / 2 * state[j, :]
                              + 4 / 3 * state[j - 1, :] - 1 / 12 * state[j - 2, :])

    lap = (c**2) * (state_yy + state_xx) / dx**2

    if source is not None:
        lap = source(t.item()) * circle_mask + lap

    derivative = torch.cat([state_diff.unsqueeze(0), lap.unsqueeze(0)], 0)

    return derivative


def generate(size, frame_size, seq_len, dt, data_dir):
    """
    Generates the WaveEq dataset in folder \'data\' of the given directory as pt files `homogenous_wave${INDEXSEQ}.pt`
    where $INDEXSEQ is the index of the sequence, each containing the following fields:
        - `simul`: float tensor of dimensions (length, width, height) representing a sequence;
        - `c`: celocity coefficient used to create the associated sequence in `simul`.

    Parameters
    ----------
    size : int
        Number of sequences to generate (size of the dataset).
    frame_size : int
        Width and height of the sequences.
    seq_len : int
        Length of generated sequences.
    dt : float
        Step size of ODE solver and time interval between each frame.
    data_dir : str
        Directory where the folder `data` will be created.
    """
    # Create the directory if needed
    data_dir = os.path.join(data_dir, 'data')
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Generate all sequences
    for i in trange(size):
        # Source
        source_init_value = np.random.uniform(1, 30)
        source = partial(decreasing_energy_source, invT0=20, f0=source_init_value)

        # Null initial condition
        initial_condition = torch.zeros(1, frame_size, frame_size)

        # Velocity coefficient
        c = np.random.uniform(300, 400)

        # Numerically solving wave equation
        dF = partial(derivative, source=source, c=c, nx=frame_size, ny=frame_size, dx=1, dz=1, circular=True, order=5)
        t = torch.arange(0, dt * seq_len, dt)
        simul = odeint(dF, y0=initial_condition.expand((2, frame_size, frame_size)), t=t, method="rk4")[:, 0]
        print(simul.size())
        # Save sequences and velocities coefficients
        torch.save({'simul': simul, 'c': c}, os.path.join(data_dir, f'homogenous_wave{i}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='WaveEq preprocessing.',
        description='Generates the WaveEq dataset in folder \'data\' of the given directory as pt files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the data will be saved.')
    parser.add_argument('--seq_len', type=int, metavar='LEN', default=300,
                        help='Length of generated sequences.')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    parser.add_argument('--frame_size', type=int, metavar='SIZE', default=64,
                        help='Size of generated frames.')
    parser.add_argument('--size', type=int, metavar='SIZE', default=300,
                        help='Number of sequences to generate (size of the dataset).')
    parser.add_argument('--dt', type=int, metavar='SIZE', default=0.001,
                        help='Step size of ODE solver and time interval between each frame.')
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(args.seed)

    # Generate dataset
    generate(args.size, args.frame_size, args.seq_len, args.dt, args.data_dir)
