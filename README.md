# PDE-Driven Spatiotemporal Disentanglement

Official implementation of the paper *PDE-Driven Spatiotemporal Disentanglement* (Jérémie Donà,* Jean-Yves Franceschi,* Sylvain Lamprier, Patrick Gallinari), accepted at ICLR 2021.


## [Article](https://openreview.net/forum?id=vLaHRtHvfFp)

## [Preprint](https://arxiv.org/abs/2008.01352)


## Requirements

All models were trained with Python 3.8.1 and PyTorch 1.4.0 using CUDA 10.1.
The `requirements.txt` file lists Python package dependencies.

We obtained all our models thanks to mixed-precision training with Nvidia's [Apex](https://nvidia.github.io/apex/) (v0.1), allowing to accelerate training on the most recent Nvidia GPU architectures (starting from Volta).
This optimization can be enabled using the command-line options.
We also enabled PyTorch's inetrgated [mixed-precision training package](https://pytorch.org/docs/stable/amp.html) as an experimental feature, which should provide similar results.


## Datasets

### Moving MNIST

The training dataset is generated on the fly.
The testing set can be generated as an `.npz` file in the directory `$DIR` with the following command:
```bash
python -m var_sep.preprocessing.mmnist.make_test_set --data_dir $DIR
```

### 3D Warehouse Chairs

The original multi-view dataset can be downloaded at [https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar](https://www.di.ens.fr/willow/research/seeing3Dchairs/data/rendered_chairs.tar).
In order to train and test our model on this dataset, it should be preprocessed to obtain 64x64 cropped images using the following command:

```bash
python -m var_sep.preprocessing.chairs.gen_chairs --data_dir $DIR
```
where `$DIR` is the directory where the dataset was downloaded and extracted.
The preprocessing script will save the processed images in the same location as the original images in the extracted archive.

### TaxiBJ

We used the preprocessed dataset provided by [MIM's authors in their official repository](https://github.com/Yunbo426/MIM).
It consists in four HDF5 files named `BJ${YEAR}_M32x32_T30_InOut.h5` where `$YEAR` is ranges from 13 to 16.

### SST

We refer the reader to the article in which this dataset was introduced ([https://openreview.net/forum?id=By4HsfWAZ](https://openreview.net/forum?id=By4HsfWAZ)) and its authors, as we do not own the preprocessing script to this date.

### WaveEq & WaveEq-100

WaveEq data are generated in the directory `$DIR` by the following command:
```bash
python -m var_sep.preprocessing.wave.gen_wave --data_dir $DIR
```
and sampled pixels are chosen by the following script:
```bash
python -m var_sep.preprocessing.wave.gen_wave --data_dir $DIR
```


## Training

In order to train a model on the GPU indexed by `$NDEVICE` with data directory and save directory respectively given by `$DATA_DIR` and `$XP_DIR`, execute the following command:
```bash
python -m var_sep.main --device $NDEVICE --xp_dir $XP_DIR --data_dir $DATA_DIR
```
Options `--apex_amp` and `--torch_amp` can be used to accelerate training (see [requirements](#Requirements)).

Models presented in the paper can be obtained using the following parameters:
- for Moving MNIST:
```bash
--data mnist --epochs 800 --beta1 0.5 --scheduler
```
- for 3D Warehouse Chairs:
```bash
--data chairs --epochs 120 --gain_resnet 0.71 --code_size_t 10 --architecture resnet --decoder_architecture dcgan --lamb_ae 1 --lamb_s 1
```
- for TaxiBJ:
```bash
--data taxibj --nt_cond 4 --nt_pred 4 --lr 4e-5 --batch_size 100 --epochs 550 --scheduler --scheduler_decay 0.2 --scheduler_milestones 250 300 350 400 450 --offset 4 --gain_resnet 0.71 --architecture vgg --lamb_ae 45 --lamb_s 0.0001
```
- for SST:
```bash
--data sst --nt_cond 4 --nt_pred 6 --epochs 30 --code_size_t 64 --code_size_s 196 --gain_res 0.2 --offset 0 --gain_resnet 0.71 --architecture encoderSST --decoder_architecture decoderSST --lamb_ae 1 --lamb_s 100 --lamb_t 5e-6 --skipco --n_blocks 2
```
- for WaveEq:
```bash
--data wave --nt_cond 5 --nt_pred 20 --epochs 250 --batch_size 128 --code_size_t 32 --code_size_s 32 --gain_resnet 0.71 --offset 5 --n_blocks 3 --mixing mul --architecture mlp --enc_hidden_size 1200 --dec_hidden_size 1200 --dec_n_layers 4 --lamb_ae 1
```
- for WaveEq-100:
```bash
--data wave_partial --nt_cond 5 --nt_pred 20 --epochs 250 --batch_size 128 --code_size_t 32 --code_size_s 32 --gain_resnet 0.71 --offset 5 --n_blocks 3 --mixing mul --architecture mlp --enc_hidden_size 2400 --dec_hidden_size 150 --lamb_ae 1
```

Please also refer the help message of the program:
```bash
python -m var_sep.main --help
```
which lists options and hyperparameters to train our model.


## Testing

Trained models can be tested as follows.
These evaluations can be run on GPU using the `--device`options of each script.
Please also refer to the help message of each script for more information.

### Moving MNIST

Prediction performance (MSE, PSNR and SSIM) on Moving MNIST over a number `$HOR` of predicted frames is assessed using the following command:
```bash
python -m var_sep.test.mnist.test --xp_dir $XP_DIR --data_dir $DATA_DIR --nt_pred $HOR
```
For instance, long-term prediction results in the paper corresponds to setting `$HOR` to 95.

Disentanglement performance can be computed in the sawe way:
```bash
python -m var_sep.test.mnist.test_disentanglement --xp_dir $XP_DIR --data_dir $DATA_DIR --nt_pred $HOR
```

### 3D Warehouse Chairs
Disentanglement performance can be computed using the following command similarly to Moving MNIST:
```bash
python -m var_sep.test.chairs.test_disentanglement --xp_dir $XP_DIR --data_dir $DATA_DIR --nt_pred $HOR
```

### TaxiBJ
Prediction MSE can be computed using the following command:
```bash
python -m var_sep.test.taxibj.test --xp_dir $XP_DIR --data_dir $DATA_DIR
```

### SST
Prediction MSE can be computed using the following command:
```bash
python -m var_sep.test.sst.test --xp_dir $XP_DIR --data_dir $DATA_DIR
```

### WaveEq & WaveEq-100

Prediction MSE on both datasets can be computed using the following command:
```bash
python -m var_sep.test.wave.test --xp_dir $XP_DIR --data_dir $DATA_DIR
```
