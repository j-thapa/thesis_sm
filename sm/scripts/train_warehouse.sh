#!/bin/sh
env="WarehouseEnv"
algo="mat"
seed=1


echo "env is ${env}, algo is ${algo}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=0 python train/train_warehouse.py --env_name ${env} \
  --n_block 1 --algorithm_name ${algo} --seed ${seed} \
  --n_training_threads 18 --n_rollout_threads 35 \
  --num_mini_batch 40 --episode_length 180 --num_env_steps 10000000 \
  --lr 0.0002 --ppo_epoch 10 --clip_param 0.15 \
  --pair_agents 1 --save_interval 100000 --num_objects 1 --entropy_coef 0.001 \
   --use_eval 

