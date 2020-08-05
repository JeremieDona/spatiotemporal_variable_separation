# PDE-Driven Spatiotemporal Disentanglement

Official implementation of the paper *PDE-Driven Spatiotemporal Disentanglement* (Jérémie Donà,* Jean-Yves Franceschi,* Sylvain Lamprier, Patrick Gallinari).


## [Preprint](https://arxiv.org/abs/2008.01352)


## Requirements

All models were trained with Python 3.8.1 and PyTorch 1.4.0 using CUDA 10.1. The `requirements.txt` file lists Python package dependencies.

We obtained all our models thanks to mixed-precision training with Nvidia's [Apex](https://nvidia.github.io/apex/) (v0.1), allowing to accelerate training on the most recent Nvidia GPU architectures. This optimization can be enabled using the command-line options.


## Execution

All scripts should be executed as modules from the root of this folder. For example, the training script can be launched with:
```bash
python -m var_sep.main
```


## Datasets

Preprocessing scripts are located in the `var_sep/preprocessing` folder for the WaveEq, WaveEq-100 and Moving MNIST datasets:
- `var_sep.preprocessing.mnist.make_test_set` creates the Moving MNIST testing set;
- `var_sep.preprocessing.wave.gen_wave` generates the WaveEq dataset;
- `var_sep.preprocessing.wave.gen_pixels` chooses pixels to draw from the WaxeEq dataset to create the WaveEq-100 dataset.

Regarding SST, we refer the reader to the article in which it was introduced ([https://openreview.net/forum?id=By4HsfWAZ](https://openreview.net/forum?id=By4HsfWAZ)) and its authors, as we do not own the preprocessing script to this date.


## Training

Please refer to the help message of `main.py`:
```bash
python -m var_sep.main --help
```
which lists options and hyperparameters to train our model.


## Testing

Evaluation scripts on testing sets are located in the `var_sep/test` folder.
- `var_sep.test.mnist.test` evaluates the prediction PSNR and SSIM of the model on Moving MNIST;
- `var_sep.test.mnist.test_disentanglement` evaluates the disentanglement PSNR and SSIM of the model by swapping contents and digits on Moving MNIST;
- `var_sep.sst.wave.test` computes the prediction MSE of the model after 6 and 10 prediction steps on SST;
- `var_sep.test.wave.test` computes the prediction MSE of the model after 40 prediction steps on WaveEq and WaveEq-100;
Please refer to the corresponding help messages for further information.
