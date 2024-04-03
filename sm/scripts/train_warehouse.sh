# #!/bin/sh
# env="WarehouseEnv"
# algo="mat"
# seed=1


# echo "env is ${env}, algo is ${algo}, seed is ${seed}"
# CUDA_VISIBLE_DEVICES=0 python train/train_warehouse.py --env_name ${env} \
#   --n_block 1 --algorithm_name ${algo} --seed ${seed} \
#   --n_training_threads 16 --n_rollout_threads 25 \
#   --num_mini_batch 40 --episode_length 190 --num_env_steps 10000000 \
#   --lr 0.0015 --ppo_epoch 15 --clip_param 0.05 \
#   --pair_agents 1 --save_interval 100000 --num_objects 1 \
#    --use_eval 


# HAPPO and HARTPO 

#!/bin/sh
env="WarehouseEnv"
algo="happo"
seed=1
exp="mlp"
running_max=20
kl_threshold=0.06

echo "env is ${env}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
# for number in `seq ${running_max}`;
# do
  # echo "the ${number}-th running:"
  CUDA_VISIBLE_DEVICES=1 python train/train_warehouse.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp}  \
  --running_id ${number}--n_training_threads 32 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 20000000 --ppo_epoch 5 \
    --pair_agents 1 --save_interval 100000 --num_objects 1  --episode_length 190  --lr 0.0009 \
  --stacked_frames 1 --kl_threshold ${kl_threshold}  --use_eval  --share_policy --separated  --use_popart --image_observation
# done