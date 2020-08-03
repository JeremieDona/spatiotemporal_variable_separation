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

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Choice of sample pixels for the WaveEq-100 dataset.',
        description='Generates the pixels used for the WaveEqPartial (WaveEq-100 if 100 pixels are drawn) dataset. \
                     Chosen coordinates are save in an npz file, in `rand_w` and `rand_h` fields.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where the data will be saved.')
    parser.add_argument('--number', type=int, metavar='NUM', default=100,
                        help='Number of pixels to generate.')
    parser.add_argument('--frame_size', type=int, metavar='SIZE', default=64,
                        help='Size of frames (bound on pixel values).')
    parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                        help='Fixed NumPy seed to produce the same dataset at each run.')
    args = parser.parse_args()

    # Fix random seed
    np.random.seed(args.seed)

    # Create the directory if needed
    data_dir = os.path.join(args.data_dir, 'pixels')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    # Generate pixel coordinates
    rand_w = np.random.randint(args.frame_size, size=args.number)
    rand_h = np.random.randint(args.frame_size, size=args.number)

    # Save coordinates
    np.savez_compressed(os.path.join(data_dir, 'pixels.npz'), rand_w=rand_w, rand_h=rand_h)
