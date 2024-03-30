#!/bin/sh
env="WarehouseEnv"
algo="mat"
seed=1


echo "env is ${env}, algo is ${algo}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_warehouse.py --env_name ${env} \
  --n_block 1 --algorithm_name ${algo} --seed ${seed} \
  --n_training_threads 16 --n_rollout_threads 25 \
  --num_mini_batch 40 --episode_length 350 --num_env_steps 10000000 \
  --lr 5e-4 --ppo_epoch 15 --clip_param 0.15 \
  --pair_agents 1 --save_interval 100000 --num_objects 2\
  --use_value_active_masks --use_eval 

