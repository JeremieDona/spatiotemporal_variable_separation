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
import tqdm

from PIL import Image


def generate(data_dir, image_size):
    data_dir = os.path.join(data_dir, 'rendered_chairs')
    sequence_folders = os.listdir(data_dir)
    sequence_folders.remove('all_chair_names.mat')
    for sequence_folder in tqdm.tqdm(sequence_folders, ncols=0):
        sequence_dir = os.path.join(data_dir, sequence_folder, 'renders')
        for i, image_file in enumerate(sorted(os.listdir(sequence_dir))):
            image = Image.open(os.path.join(sequence_dir,
                                            image_file)).crop((100, 100, 500, 500)).resize((image_size, image_size),
                                                                                           resample=Image.LANCZOS)
            image.save(os.path.join(sequence_dir, f'{i}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='3D Warehouse chairs preprocessing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, metavar='DIR', required=True,
                        help='Folder where videos from the original dataset are stored.')
    parser.add_argument('--image_size', type=int, metavar='SIZE', default=64,
                        help='Width and height of resulting processed videos.')
    args = parser.parse_args()

    generate(args.data_dir, args.image_size)
