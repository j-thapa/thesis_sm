#!/bin/sh
env="WarehouseEnv"
algo="mamamba"
seed=1
exp_name="train"

#default grid size is (10,10), to alter gridsize make changes in train_warehouse.py

echo "env is ${env}, algo is ${algo}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_warehouse.py --env_name ${env} --exp_name ${exp_name}\
  --n_block 1 --algorithm_name ${algo} --seed ${seed} \
  --n_training_threads 1 --n_rollout_threads 1 \
  --num_mini_batch 40 --episode_length 200 --num_env_steps 10000000 \
  --lr 0.005 --ppo_epoch 15 --clip_param 0.05 \
  --pair_agents 1 --save_interval 2000000 --num_objects 4 --max_steps 120 \
  --use_eval 
