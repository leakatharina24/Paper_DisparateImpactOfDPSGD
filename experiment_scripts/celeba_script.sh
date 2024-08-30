#!/bin/bash

for seed in {0..4}
do
  dir="iclr_celeba/$seed"
  angles="True"
  hessians="True"

  echo "$seed nonpriv"
  python main.py --dataset=celeba --method=regular --config lr=0.01 --config  max_epochs=30 --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200 --config logdir=$dir/celeba_nonpriv --config sampled_expected_loss=True

  echo "$seed dpsgd"
  python main.py --dataset=celeba --method=dpsgd --config lr=0.01 --config max_epochs=30 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config logdir=$dir/celeba_dpsgd --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200 --config sampled_expected_loss=True


  echo "$seed dpsgd-f"
  python main.py --dataset=celeba --method=dpsgd-f --config lr=0.01 --config max_epochs=30 --config delta=1e-6 --config noise_multiplier=0.8 --config base_max_grad_norm=1 --config counts_noise_multiplier=8 --config logdir=$dir/celeba_dpsgdf --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200 --config sampled_expected_loss=True

  echo "$seed dpsgd-g"
  python main.py --dataset=celeba --method=dpsgd-global --config lr=0.1 --config max_epochs=30 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config strict_max_grad_norm=100 --config logdir=$dir/celeba_dpsgdg --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200 --config sampled_expected_loss=True

  echo "$seed dpsgd-global-adapt"
  python main.py --dataset=celeba --method=dpsgd-global-adapt --config lr=0.1 --config max_epochs=30 --config delta=1e-6 --config noise_multiplier=0.8 --config l2_norm_clip=1 --config strict_max_grad_norm=100 --config lr=0.1 --config logdir=$dir/celeba_dpsgdg_adapt --config threshold=0.7 --config seed=$seed --config evaluate_angles=$angles --config evaluate_hessian=$hessians --config angle_comp_step=200 --config sampled_expected_loss=True
done

