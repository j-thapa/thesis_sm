#!/bin/sh
env="StarCraft2"
map="27m_vs_30m"
algo="marwkv_v4"
exp="speed_inf_"
seed=3


#7e-4 trying for 3s5z
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=3 python train/train_smac.py --env_name ${env} --n_block 3 --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 10000000 --lr 1e-3 --ppo_epoch 5 --clip_param 0.05 --save_interval 100000 --use_value_active_masks --n_eval_rollout_threads 32 --use_eval