# SMART: Self-supervised Multi-task pretrAining with contRol Transformers

This is the official codebase for the ICLR 2023 spotlight paper [SMART: Self-supervised Multi-task pretrAining with contRol Transformers](https://openreview.net/forum?id=9piH3Hg8QEf).
If you use this code in an academic context, please use the following citation:

```
@inproceedings{
sun2023smart,
title={{SMART}: Self-supervised Multi-task pretrAining with contRol Transformers},
author={Yanchao Sun and Shuang Ma and Ratnesh Madaan and Rogerio Bonatti and Furong Huang and Ashish Kapoor},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=9piH3Hg8QEf}
}
```

## Setting up

- Using conda

  ```
  # dmc specific
  # create env
  conda env create --file docker/environment.yml

  # activate conda
  conda activate smart
  bash scripts/dmc_setup.sh

  # install this repo
  (smart) $ pip install -e .
  ```

- Using docker

  ```
  # build image
  docker build \
        -f Dockerfile_base_azureml_dmc \
        --build-arg BASE_IMAGE=openmpi4.1.0-cuda11.3-cudnn8-ubuntu20.04:latest \
        -t smart:latest .

  # run image
  docker run -it -d --gpus=all --name=rl_pretrain_dmc_1 -v HOST_PATH:CONTAINER_PATH smart:latest

  # setup the repo (run inside the container)
  pip install -e .
  ```

## Downloading data and pre-trained models download from Azure

- Install azcopy

  ```
  wget https://aka.ms/downloadazcopy-v10-linux
  tar -xvf downloadazcopy-v10-linux
  sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
  rm -rf *azcopy*
  ```

- Downloading the full dataset (1.18TiB)

  ```
  # download to data/ directory
  azcopy copy 'https://smartrelease.blob.core.windows.net/smartrelease/data/dmc_ae' 'data' --recursive
  ```

- Downloading a subset of the full dataset

  ```
  # download to data/ directory
  azcopy copy 'https://smartrelease.blob.core.windows.net/smartrelease/data/dmc_ae/TYPE_DOMAIN_TASK' 'data' --recursive
  ```

  where

  - `TYPE`: `randcollect`, `fullcollect`
    Note: `fullcollect` datasets are ~10x larger than `randcollect` datasets)

  - `DOMAIN_TASK`: `cartpole_balance`, `cartpole_swingup`, `cheetah_run`, `finger_spin`, `hopper_hop`, `hopper_stand`, `pendulum_swingup`, `walker_run`, `walker_stand`, or  `walker_walk` (See Table 2 in the paper)

  Example:

  ```
  # download to data/ directory (~ 9.7 GB each)
  azcopy copy 'https://smartrelease.blob.core.windows.net/smartrelease/data/dmc_ae/randcollect_walker_walk' 'data' --recursive
  azcopy copy 'https://smartrelease.blob.core.windows.net/smartrelease/data/dmc_ae/randcollect_cheetah_run' 'data' --recursive
  ```

- Downloading the pretrained models

  ```
  # download to models/ directory (236.34 MiB)
  azcopy copy 'https://smartrelease.blob.core.windows.net/smartrelease/models' '.' --recursive
  ```

## Running the code

### Testing on small subset of full dataset

Let us run the code on the aforementioned small subset of `randcollect_walker_walk` and `randcollect_cheetah_run` (corresponds to `configs/pretrain_configs/multipretrain_source_test.json`)

```
python src/dmc_pretrain.py \
        --source_data_type rand \
        --train_replay_id 1 \
        --epochs 10 --num_steps 80000 --model_type naive \
        --multi_config configs/pretrain_configs/multipretrain_source_test.json \
        --output_dir ./outputs/pretrain_explore/ \
        --data_dir_prefix data
```

### Pretraining on multiple domains and tasks

The set of pretraining tasks can be specified in the config file as shown below:

- Pretrain with offline data collected by exploratory policies

```
python src/dmc_pretrain.py \
        --epochs 10 --num_steps 80000 --train_replay_id 5 --model_type naive \
        --multi_config configs/pretrain_configs/multipretrain_source_v1.json \
        --output_dir ./outputs/pretrain_explore/ \
        --data_dir_prefix data
```

- Pretrain with offline data collected by random policies

```
python src/dmc_pretrain.py \
        --source_data_type rand \
        --epochs 10 --num_steps 80000 --train_replay_id 5 --model_type naive \
        --multi_config configs/pretrain_configs/multipretrain_source_v1.json \
        --output_dir ./outputs/pretrain_random/ \
        --data_dir_prefix data
```

### Using pretrained model and finetunes the policy on a specific downstream task:

You can also download our pretrained models as reported in the paper, using the `azcopy` command in the previous section.

```
## set the downstream domain and task
DOMAIN=cheetah
TASK=run

## behavior cloning as the learning algorithm
python src/dmc_downstream.py \
        --epochs 30 --num_steps 1000000 --domain ${DOMAIN} --task ${TASK} \
        --model_type naive --no_load_action \
        --load_model_from ./outputs/pretrain_explore/checkpoints/last.ckpt \
        --output_dir ./outputs/${DOMAIN}_${TASK}_bc/

## RTG-conditioned learning as the learning algorithm
python src/dmc_downstream.py \
        --epochs 30 --num_steps 1000000 --domain ${DOMAIN} --task ${TASK} \
        --model_type reward_conditioned --rand_select --no_load_action \
        --load_model_from ./outputs/pretrain_explore/checkpoints/last.ckpt \
        --output_dir ./outputs/${DOMAIN}_${TASK}_rtg/
```

Note that if *--load_model_from* is not specified, the model is trained from scratch.
