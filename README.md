# SMART: Self-supervised Multi-task pretrAining with contRol Transformers 


This is the official codebase for the ICLR 2023 spotlight paper "SMART: Self-supervised Multi-task pretrAining with contRol Transformers". Pretrained models can be downloaded [here](https://link-url-here.org). Dataset can be downloaded [here](https://link-url-here.org).

## Setting up

- Using conda

  ```
  # dmc specific
  # create env
  conda env create --file docker/environment.yml

  # activate conda
  conda activate smart
  bash src/scripts/dmc_setup.sh

  # install this repo
  (smart) $ pip install -e .
  ```

- Using docker

  ```
  # dmc specific
  docker pull PUBLIC_DOCKER_IMAGE

  # run image
  docker run -it -d --gpus=all --name=rl_pretrain_dmc_1 -v HOST_PATH:CONTAINER_PATH commondockerimages.azurecr.io/atari_pretrain:latest-azureml-dmc

  # setup the repo (run inside the container)
  pip install -e .
  ```

## Preparing the dataset

Download dataset to PATH_TO_DATASET, or collect data following this instruction.

## Running the code

**Pretraining on multiple domains and tasks** (selection of pretraining tasks can be specified in the config file as shown below):
```
## pretrain with offline data collected by exploratory policies
python src/dmc_multidomain_train.py \
        --epochs 10 --num_steps 80000 --train_replay_id 5 --model_type naive \
        --multi_config configs/train_configs/multipretrain_source_v1.json \
        --output_dir ./outputs/pretrain_explore/ \
        --data_dir_prefix PATH_TO_DATASET

## pretrain with offline data collected by random policies
python src/dmc_multidomain_train.py \
        --epochs 10 --num_steps 80000 --train_replay_id 5 --model_type naive \
        --multi_config configs/train_configs/multipretrain_source_v1.json --source_data_type rand \
        --output_dir ./outputs/pretrain_random/ \
        --data_dir_prefix PATH_TO_DATASET 
```
You can also download our pretrained models as reported in the paper in this [here](https://link-url-here.org).




The command below **loads the pretrained model and finetunes the policy on a specific downstream task**:
```
## (example) set the downstream domain and task
DOMAIN=cheetah
TASK=run

## behavior cloning as the learning algorithm
python src/dmc_train.py \
        --epochs 30 --num_steps 1000000 --domain ${DOMAIN} --task ${TASK} \
        --model_type naive --no_load_action \
        --load_model_from ./outputs/pretrain_explore/checkpoints/last.ckpt \
        --output_dir ./outputs/${DOMAIN}_${TASK}_bc/

## RTG-conditioned learning as the learning algorithm
python src/dmc_train.py \
        --epochs 30 --num_steps 1000000 --domain ${DOMAIN} --task ${TASK} \
        --model_type reward_conditioned --rand_select --no_load_action \
        --load_model_from ./outputs/pretrain_explore/checkpoints/last.ckpt \ 
        --output_dir ./outputs/${DOMAIN}_${TASK}_rtg/
```

Note that if *--load_model_from* is not specified, the model is trained from scratch.