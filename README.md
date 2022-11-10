<!--
 * @Author: Conghao Wong
 * @Date: 2022-07-07 21:43:30
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2022-11-10 19:18:46
 * @Description: file content
 * @Github: https://github.com/cocoon2wong
 * Copyright 2022 Conghao Wong, All Rights Reserved.
-->

# Project-Jocker

## Branch `BB` Notes ⚠️

Please note that the contents in this branch are still under development and are not the final version.

![BB](./figures/bb.jpeg)

## Requirements

The codes are developed with python 3.9.
Additional packages used are included in the `requirements.txt` file.
We recommend installing python packages in a virtual environment (like the `conda` environment).
Otherwise, there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your python environment:

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
Considering that most of the loss function terms used to optimize the model work within one sub-network alone, we divide `$MODEL` into `$MODEL-a` and `$MODEL-b`, and apply gradient descent separately for more accessible training.
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
Our pre-trained models contain model weights trained on `ETH-UCY` by the `leave-one-out` strategy and model weights trained on `SDD` via the dataset split method from [SimAug](https://github.com/JunweiLiang/Multiverse).

Please note that we do not use dataset split files like TrajNet for several reasons.
For example, the frame rate problem in `ETH-eth` sub-dataset, and some of these splits only consider the `pedestrians` in the SDD dataset.
We process the original full-dataset files from these datasets with observations = 3.2 seconds (or 8 frames) and predictions = 4.8 seconds (or 12 frames) to train and test the model.
Detailed process codes are available in `./scripts/add_ethucy_datasets.py`, `./scripts/add_sdd.py`, and `./scripts/sdd_txt2csv.py`.
See details in [this](https://github.com/cocoon2wong/Vertical/issues/1) issue](https://github.com/cocoon2wong/Vertical/issues/1).

In order to start validating the effects of our pre-trained models, please follow these steps to prepare dataset files and model weights:

1. As this repository contains only codes, you may need to download the original dataset files first.
   If you have cloned this repository with `git clone` command, you can download the dataset files by the following command:

   ```bash
   git submodule update --init --recursive
   ```

   Or you can just download them from [here, then rename the folder as `dataset_original` and put it into the repository root path.

2. You need to process these original dataset files so that they are in a format that our code can read and process.
   You can run the following lines to process the `ETH-UCY`, `SDD`, and `SDD_pedestrian` (a sub-dataset from SDD that only contains `"Pedestrian"` agents) dataset files:

   ```bash
   cd Project-Jocker/dataset_original
   python main.py
   ```

3. Create soft links to these folders:

   ```bash
   cd Project-Jocker
   ln -s dataset_original/dataset_processed ./
   ln -s dataset_original/dataset_configs ./
   ```

4. After these steps, you can find and download our model weights file [here](https://github.com/cocoon2wong/Project-Jocker/releases) and put them into the `./weights` folder (optional).

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
You can start testing these linear-interpolation models with the following command:

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
Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 --ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

Instruction for the `argtype`:

- Args with `argtype='STATIC'` indicate that their values can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with `argtype='DYNAMIC'` indicate that their values can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved `json` file.
- Args with `argtype='TEMPORARY'` indicate that these values will not be saved into `json` files.
  The program will parse these args from the terminal at each time.

<!-- DO NOT CHANGE THIS LINE -->
### Basic args

- `--K_train`, type=`int`, argtype=`STATIC`.
  The number of multiple generations when training. This arg only works for multiple-generation models.
  The default value is `10`.
- `--K`, type=`int`, argtype=`DYNAMIC`.
  Number of multiple generations when testing. This arg only works for multiple-generation models.
  The default value is `20`.
- `--anntype`, type=`str`, argtype=`STATIC`.
  Model's predicted annotation type. Can be `'coordinate'` or `'boundingbox'`.
  The default value is `'coordinate'`.
- `--auto_dimension`, type=`int`, argtype=`TEMPORARY`.
  Choose whether to handle the dimension adaptively. It is now only used for silverballers models that are trained with annotation type `coordinate` but want to test on datasets with annotation type `boundingbox`.
  The default value is `0`.
- `--batch_size`, type=`int`, argtype=`DYNAMIC`.
  Batch size when implementation.
  The default value is `5000`.
- `--dataset`, type=`str`, argtype=`STATIC`.
  Name of the video dataset to train or evaluate. For example, `'ETH-UCY'` or `'SDD'`. NOTE: DO NOT set this argument manually.
  The default value is `'error'`.
- `--dim`, type=`int`, argtype=`STATIC`.
  Dimension of the `trajectory`. For example, - coordinate (x, y) -> `dim = 2`; - boundingbox (xl, yl, xr, yr) -> `dim = 4`.
  The default value is `dim`.
- `--draw_distribution`, type=`int`, argtype=`TEMPORARY`.
  Controls if draw distributions of predictions instead of points. If `draw_distribution == 0`, it will draw results as normal coordinates; If `draw_distribution == 1`, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors.
  The default value is `0`.
- `--draw_index`, type=`str`, argtype=`TEMPORARY`.
  Indexes of test agents to visualize. Numbers are split with `_`. For example, `'123_456_789'`.
  The default value is `'all'`.
- `--draw_results`, type=`str`, argtype=`TEMPORARY`.
  Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the `plist` file (saved in `dataset_configs` folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_results` if `draw_results != 'null'`.
  The default value is `'null'`.
- `--draw_videos`, type=`str`, argtype=`TEMPORARY`.
  Controls whether draw visualized results on video frames and save as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the `plist` file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_videos` if `draw_videos != 'null'`.
  The default value is `'null'`.
- `--epochs`, type=`int`, argtype=`STATIC`.
  Maximum training epochs.
  The default value is `500`.
- `--force_clip`, type=`str`, argtype=`TEMPORARY`.
  Force test video clip (ignore the train/test split). It only works when `test_mode` has been set to `one`.
  The default value is `'null'`.
- `--force_dataset`, type=`str`, argtype=`TEMPORARY`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`.
  The default value is `'null'`.
- `--force_split`, type=`str`, argtype=`TEMPORARY`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`.
  The default value is `'null'`.
- `--gpu`, type=`str`, argtype=`TEMPORARY`.
  Speed up training or test if you have at least one NVidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`. NOTE: It only supports training or testing on one GPU.
  The default value is `'0'`.
- `--interval`, type=`float`, argtype=`STATIC`.
  Time interval of each sampled trajectory point.
  The default value is `0.4`.
- `--load`, type=`str`, argtype=`TEMPORARY`.
  Folder to load model (to test). If set to `null`, the training manager will start training new models according to other given args.
  The default value is `'null'`.
- `--log_dir`, type=`str`, argtype=`STATIC`.
  Folder to save training logs and model weights. Logs will save at `args.save_base_dir/current_model`. DO NOT change this arg manually. (You can still change the path by passing the `save_base_dir` arg.) 
  The default value is `'null'`.
- `--lr`, type=`float`, argtype=`STATIC`.
  Learning rate.
  The default value is `0.001`.
- `--model_name`, type=`str`, argtype=`STATIC`.
  Customized model name.
  The default value is `'model'`.
- `--model`, type=`str`, argtype=`STATIC`.
  The model type used to train or test.
  The default value is `'none'`.
- `--obs_frames`, type=`int`, argtype=`STATIC`.
  Observation frames for prediction.
  The default value is `8`.
- `--pmove`, type=`int`, argtype=`STATIC`.
  Index of the reference point when moving trajectories.
  The default value is `-1`.
- `--pred_frames`, type=`int`, argtype=`STATIC`.
  Prediction frames.
  The default value is `12`.
- `--protate`, type=`float`, argtype=`STATIC`.
  Reference degree when rotating trajectories.
  The default value is `0.0`.
- `--pscale`, type=`str`, argtype=`STATIC`.
  Index of the reference point when scaling trajectories.
  The default value is `'autoref'`.
- `--restore`, type=`str`, argtype=`DYNAMIC`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`.
  The default value is `'null'`.
- `--save_base_dir`, type=`str`, argtype=`STATIC`.
  Base folder to save all running logs.
  The default value is `'./logs'`.
- `--split`, type=`str`, argtype=`STATIC`.
  The dataset split that used to train and evaluate.
  The default value is `'zara1'`.
- `--start_test_percent`, type=`float`, argtype=`STATIC`.
  Set when (at which epoch) to start validation during training. The range of this arg should be `0 <= x <= 1`. Validation may start at epoch `args.epochs * args.start_test_percent`.
  The default value is `0.0`.
- `--step`, type=`int`, argtype=`DYNAMIC`.
  Frame interval for sampling training data.
  The default value is `1`.
- `--test_mode`, type=`str`, argtype=`TEMPORARY`.
  Test settings. It can be `'one'`, `'all'`, or `'mix'`. When setting it to `one`, it will test the model on the `args.force_split` only; When setting it to `all`, it will test on each of the test datasets in `args.split`; When setting it to `mix`, it will test on all test datasets in `args.split` together.
  The default value is `'mix'`.
- `--test_step`, type=`int`, argtype=`STATIC`.
  Epoch interval to run validation during training. """ return self._get('test_step', 3, argtype=STATIC) """ Trajectory Prediction Args 
  The default value is `3`.
- `--update_saved_args`, type=`int`, argtype=`TEMPORARY`.
  Choose whether to update (overwrite) the saved arg files or not.
  The default value is `0`.
- `--use_extra_maps`, type=`int`, argtype=`DYNAMIC`.
  Controls if uses the calculated trajectory maps or the given trajectory maps. The training manager will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this argument to `1`.
  The default value is `0`.

### Silverballers args

- `--Kc`, type=`int`, argtype=`STATIC`.
  The number of style channels in `Agent` model.
  The default value is `20`.
- `--T`, type=`str`, argtype=`STATIC`.
  Type of transformations used when encoding or decoding trajectories. It could be: - `none`: no transformations - `fft`: fast Fourier transform - `haar`: haar wavelet transform - `db2`: DB2 wavelet transform 
  The default value is `'fft'`.
- `--depth`, type=`int`, argtype=`STATIC`.
  Depth of the random noise vector.
  The default value is `16`.
- `--feature_dim`, type=`int`, argtype=`STATIC`.
  Feature dimensions that are used in most layers.
  The default value is `128`.
- `--key_points`, type=`str`, argtype=`STATIC`.
  A list of key time steps to be predicted in the agent model. For example, `'0_6_11'`.
  The default value is `'0_6_11'`.
- `--loada`, type=`str`, argtype=`TEMPORARY`.
  Path for agent model.
  The default value is `'null'`.
- `--loadb`, type=`str`, argtype=`TEMPORARY`.
  Path for handler model.
  The default value is `'null'`.
- `--points`, type=`int`, argtype=`STATIC`.
  Controls the number of keypoints accepted in the handler model.
  The default value is `1`.
- `--preprocess`, type=`str`, argtype=`STATIC`.
  Controls if running any preprocess before model inference. Accept a 3-bit-like string value (like `'111'`): - The first bit: `MOVE` trajectories to (0, 0); - The second bit: re-`SCALE` trajectories; - The third bit: `ROTATE` trajectories.
  The default value is `'111'`.
<!-- DO NOT CHANGE THIS LINE -->

## Thanks

Codes of the Transformers used in this model come from [TensorFlow.org](https://www.tensorflow.org/tutorials/text/transformer)](https://www.tensorflow.org/tutorials/text/transformer);  
Dataset CSV files of ETH-UCY come from [SR-LSTM (CVPR2019) / E-SR-LSTM (TPAMI2020)](https://github.com/zhangpur/SR-LSTM);  
Original dataset annotation files of SDD come from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/), and its split file comes from [SimAug (ECCV2020)](https://github.com/JunweiLiang/Multiverse);  
All contributors of the repository [Vertical](https://github.com/cocoon2wong/Vertical).

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
