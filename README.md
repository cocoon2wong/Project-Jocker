<!--
 * @Author: Conghao Wong
 * @Date: 2022-07-07 21:43:30
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2022-08-02 15:50:42
 * @Description: file content
 * @Github: https://github.com/cocoon2wong
 * Copyright 2022 Conghao Wong, All Rights Reserved.
-->

# Project-Jocker

## Branch `BB` Notes ⚠️

Please note that contents in this branch are still under development and are not the fianl version.

![BB](./figures/bb.jpeg)

## Requirements

The codes are developed with python 3.9.
Additional packages used are included in the `requirements.txt` file.
We recommend installing the above versions of the python packages in a virtual environment (like the `conda` environment), otherwise there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your python  environment:

```bash
pip install -r requirements.txt
```

Then, clone this repository and check out with

```bash
git clone https://github.com/cocoon2wong/Project-Jocker.git
cd Project-Jocker
git checkout bb
```

## Training On Your Datasets

The `$MODEL` contains two main sub-networks, the coarse-level keypoints estimation sub-network, and the fine-level spectrum interpolation sub-network.
`$MODEL` forecast agents' multiple trajectories end-to-end.
Considering that most of the loss function terms used to optimize the model work within one sub-network alone, we divide `$MODEL` into `$MODEL-a` and `$MODEL-b`, and apply gradient descent separately for easier training.
You can train your own `$MODEL` weights on your datasets by training each of these two sub-networks.
After training, you can still use it as a regular end-to-end model.

### Dataset Prepare

Before training `$MODEL` on your own dataset, you should add your dataset information to the `datasets` directory.
See [this document](./docs/datasetFormat.md) (Not Available Now) for details.

### Dataset Process

## Evaluation

You can use the following command to evaluate the `$MODEL` performance end-to-end:

```bash
python main.py \
  --model MKII \
  --loada A_MODEL_PATH \
  --loadb B_MODEL_PATH
```

Where `A_MODEL_PATH` and `B_MODEL_PATH` are the folders of the two sub-networks' weights.

## Pre-Trained Models

We have provided our pre-trained model weights to help you quickly evaluate the `$MODEL` performance.
Our pre-trained models contain model weights trained on `ETH-UCY` by the `leave-one-out` stragety, and model weights trained on `SDD` via the dataset split method from [SimAug](https://github.com/JunweiLiang/Multiverse).

Please note that we do not use dataset split files like trajnet for several reasons.
For example, the frame rate problem in `ETH-eth` sub-dataset, and some of these splits only consider the `pedestrians` in the SDD dataset.
We process the original full-dataset files from these datasets with observations = 3.2 seconds (or 8 frames) and predictions = 4.8 seconds (or 12 frames) to train and test the model.
Detailed process codes are available in `./scripts/add_ethucy_datasets.py`, `./scripts/add_sdd.py`, and `./scripts/sdd_txt2csv.py`.
See deatils in [this issue](https://github.com/cocoon2wong/Vertical/issues/1).

In order to start validating the effects of our pre-trained models, please following these steps to prepare dataset files and model weights:

1. As this repository contains only codes, you may need to download the original dataset files first.
   If you have clone this repository with `git clone` command, you can download the dataset files by the following command:

   ```bash
   git submodule update --init --recursive
   ```

   Or you can just download them from [here](https://github.com/cocoon2wong/Project-Luna), then re-name the folder as `dataset_original` and put it into the repository root path.

2. You need to process these original dataset files so that they are in a format that our code can read and process.
   You can run the following lines to process the `ETH-UCY`, `SDD`, and `SDD_pedestrian` (a sub-dataset from SDD that only contains `"Pedestrian"` agents) dataset files:

   ```bash
   cd Project-Jocker/dataset_original
   python main.py
   ```

3. Create soft link of these folders:

   ```bash
   cd Project-Jocker
   ln -s dataset_original/dataset_processed ./
   ln -s dataset_original/dataset_configs ./
   ```

4. After these steps, you can find and download our model weights file [here](https://github.com/cocoon2wong/Project-Jocker/releases), and put them into the `./weights` folder (optional).

You can start the quick evaluation via the following commands:

### V^2-Net (Our ECCV2022)

```bash
for dataset in eth hotel univ zara1 zara2 sdd
  python main.py \
    --model MKII \
    --loada ./weights/vertical/a_${dataset} \
    --loadb ./weights/vertical/b_${dataset}
```

### $MODEL

```bash
TBA
```

### Linear-Interpolation Models

You can also start testing the fast version of our pre-trained models with the argument `--loadb l` instead of the `--loadb $MODEL_PATH`.
The `--loadb l` will replace the original stage-2 spectrum interpolation sub-network with the simple linear interpolation method.
Although it may reduce the prediction performance, the model will implement much faster.
You can start test these linear-interpolation models like the following command:

```bash
python main.py --model MKII --loada $SOME_MODEL_PATH --loadb l
```

Here, `$SOME_MODEL_PATH` is still the path of model weights of the stage-1 sub-networks.

### Results

All test results will be saved in the `./test.log` file.
You can view this file to check the models' test results.
If the code runs without errors, our pre-trained models will be able to obtain the following metrics:

| Models         | eth       | hotel     | univ      | zara1     | zara2     | sdd (full) |
| -------------- | --------- | --------- | --------- | --------- | --------- | ---------- |
| V^2-Net        | 0.23/0.37 | 0.10/0.16 | 0.21/0.35 | 0.19/0.30 | 0.14/0.24 | 6.82/10.63 |
| V^2-Net-Linear | 0.25/0.38 | 0.11/0.16 | 0.21/0.35 | 0.21/0.31 | 0.15/0.25 | 6.88/10.69 |
| $MODEL         |
| $MODEL-Linear  |

### Visualization

If you have the dataset videos and put them into the `videos` folder, you can draw the visualized results by adding the `--draw_reuslts $SOME_VIDEO_CLIP` argument.
(You must put videos according to the `video_path` item in the clip's `plist` config file in the `./dataset_configs` folder.)
If you want to draw visualized trajectories like what our paper shows, you can add the additional `--draw_distribution 2` argument.
For example, if you have put the video `zara1.mp4` into `./videos/zara1.mp4`, you can draw the V^2-Net results with the following commands:

```bash
python main.py --model MKII \
    --loada ./weights/vertical/a_zara1 \
    --loadb ./weights/vertical/b_zara1 \
    --draw_results zara1 \
    --draw_distribution 2
```

## Args Used

Please specific your customized args when training or testing your model through the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.
Args with `argtype='static'` means that their values can not be changed once after training.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--K_train`, type=`int`, argtype=`'static'`.
  Number of multiple generations when training. This arg only works for `Generative Models`.
  The default value is `10`.
- `--K`, type=`int`, argtype=`'dynamic'`.
  Number of multiple generations when test. This arg only works for `Generative Models`.
  The default value is `20`.
- `--anntype`, type=`str`, argtype=`'static'`.
  Type of annotations in the predicted trajectories. Canbe `'coordinate'` or `'boundingbox'`.
  The default value is `'coordinate'`.
- `--batch_size`, type=`int`, argtype=`'dynamic'`.
  Batch size when implementation.
  The default value is `5000`.
- `--dataset`, type=`str`, argtype=`'static'`.
  Name of the video dataset to train or evaluate. For example, `'ETH-UCY'` or `'SDD'`. DO NOT set this argument manually.
  The default value is `'error'`.
- `--dim`, type=`int`, argtype=`'static'`.
  Dimension of the `trajectory`. For example, (x, y) -> `dim = 2`.
  The default value is `dim`.
- `--draw_distribution`, type=`int`, argtype=`'dynamic'`.
  Conrtols if draw distributions of predictions instead of points.
  The default value is `0`.
- `--draw_results`, type=`str`, argtype=`'dynamic'`.
  Controls if draw visualized results on video frames. Accept the name of one video clip. Make sure that you have put video files into `./videos` according to the specific name way. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_results` if `draw_results != 'null'`.
  The default value is `'null'`.
- `--epochs`, type=`int`, argtype=`'static'`.
  Maximum training epochs.
  The default value is `500`.
- `--force_clip`, type=`str`, argtype=`'dynamic'`.
  Force test video clip. It only works when `test_mode` is `one`.
  The default value is `'null'`.
- `--force_dataset`, type=`str`, argtype=`'dynamic'`.
  Force test dataset.
  The default value is `'null'`.
- `--force_split`, type=`str`, argtype=`'dynamic'`.
  Force test dataset. Only works when evaluating when `test_mode` is `one`.
  The default value is `'null'`.
- `--gpu`, type=`str`, argtype=`'dynamic'`.
  Speed up training or test if you have at least one nvidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`.
  The default value is `'0'`.
- `--interval`, type=`float`, argtype=`'static'`.
  Time interval of each sampled trajectory coordinate.
  The default value is `0.4`.
- `--load`, type=`str`, argtype=`'dynamic'`.
  Folder to load model. If set to `null`, it will start training new models according to other args.
  The default value is `'null'`.
- `--log_dir`, type=`str`, argtype=`'static'`.
  Folder to save training logs and models. If set to `null`, logs will save at `args.save_base_dir/current_model`.
  The default value is `'null'`.
- `--lr`, type=`float`, argtype=`'static'`.
  Learning rate.
  The default value is `0.001`.
- `--model_name`, type=`str`, argtype=`'static'`.
  Customized model name.
  The default value is `'model'`.
- `--model`, type=`str`, argtype=`'static'`.
  Model type used to train or test.
  The default value is `'none'`.
- `--obs_frames`, type=`int`, argtype=`'static'`.
  Observation frames for prediction.
  The default value is `8`.
- `--pred_frames`, type=`int`, argtype=`'static'`.
  Prediction frames.
  The default value is `12`.
- `--restore`, type=`str`, argtype=`'dynamic'`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`.
  The default value is `'null'`.
- `--save_base_dir`, type=`str`, argtype=`'static'`.
  Base folder to save all running logs.
  The default value is `'./logs'`.
- `--save_model`, type=`int`, argtype=`'static'`.
  Controls if save the final model at the end of training.
  The default value is `1`.
- `--split`, type=`str`, argtype=`'static'`.
  Dataset split used when training and evaluating.
  The default value is `'zara1'`.
- `--start_test_percent`, type=`float`, argtype=`'static'`.
  Set when to start validation during training. Range of this arg is `0 <= x <= 1`. Validation will start at `epoch = args.epochs * args.start_test_percent`.
  The default value is `0.0`.
- `--step`, type=`int`, argtype=`'dynamic'`.
  Frame interval for sampling training data.
  The default value is `1`.
- `--test_mode`, type=`str`, argtype=`'dynamic'`.
  Test settings, canbe `'one'` or `'all'` or `'mix'`. When set it to `one`, it will test the model on the `args.force_split` only; When set it to `all`, it will test on each of the test dataset in `args.split`; When set it to `mix`, it will test on all test dataset in `args.split` together.
  The default value is `'mix'`.
- `--test_step`, type=`int`, argtype=`'static'`.
  Epoch interval to run validation during training. """ return self._get('test_step', 3, argtype='static') """ Trajectory Prediction Args 
  The default value is `3`.
- `--use_extra_maps`, type=`int`, argtype=`'dynamic'`.
  Controls if uses the calculated trajectory maps or the given trajectory maps. The model will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this argument to `1`.
  The default value is `0`.
- `--use_maps`, type=`int`, argtype=`'static'`.
  Controls if uses the context maps to model social and physical interactions in the model.
  The default value is `1`.

### Silverballers args

- `--Kc`, type=`int`, argtype=`'static'`.
  Number of style channels in `Agent` model.
  The default value is `20`.
- `--T`, type=`str`, argtype=`'static'`.
  Type of transformations used when encoding or decoding trajectories. It could be: - `none`: no transformations - `fft`: fast fourier transform - `haar`: haar wavelet transform - `db2`: DB2 wavelet transform 
  The default value is `'fft'`.
- `--depth`, type=`int`, argtype=`'static'`.
  Depth of the random contract id.
  The default value is `16`.
- `--feature_dim`, type=`int`, argtype=`'static'`.
  Feature dimension used in most layers.
  The default value is `128`.
- `--key_points`, type=`str`, argtype=`'static'`.
  A list of key-time-steps to be predicted in the agent model. For example, `'0_6_11'`.
  The default value is `'0_6_11'`.
- `--loada`, type=`str`, argtype=`'dynamic'`.
  Path for agent model.
  The default value is `'null'`.
- `--loadb`, type=`str`, argtype=`'dynamic'`.
  Path for handler model.
  The default value is `'null'`.
- `--points`, type=`int`, argtype=`'static'`.
  Controls the number of keypoints accepted in the handler model.
  The default value is `1`.
- `--preprocess`, type=`str`, argtype=`'static'`.
  Controls if running any preprocess before model inference. Accept a 3-bit-like string value (like `'111'`): - the first bit: `MOVE` trajectories to (0, 0); - the second bit: re-`SCALE` trajectories; - the third bit: `ROTATE` trajectories.
  The default value is `'111'`.
<!-- DO NOT CHANGE THIS LINE -->

## Thanks

Codes of the Transformers used in this model comes from [TensorFlow.org](https://www.tensorflow.org/tutorials/text/transformer);  
Dataset csv files of ETH-UCY come from [SR-LSTM (CVPR2019) / E-SR-LSTM (TPAMI2020)](https://github.com/zhangpur/SR-LSTM);  
Original dataset annotation files of SDD come from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/), and its split file comes from [SimAug (ECCV2020)](https://github.com/JunweiLiang/Multiverse);  
All contributors of the repository [Vertical](https://github.com/cocoon2wong/Vertical).

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghao_wong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
