#!/bin/bash

# Define the number of parallel jobs
# parallel_jobs=4

# Define the input arguments
# inputs=("train_TFT" "--pred" "input3" "input4" "input5")

# Loop the Python script in parallel with input arguments
# echo "${inputs[@]}" | tr ' ' '\n' | parallel -j $parallel_jobs python AAChinaTFT-D.py {}
# QuantileLoss

for Yr in 2008 2017 2018; do
   python AAChinaTFT-D.py train_TFT -exp_name 'DQuantileLoss_finder_test' -predicted_year $Yr -batch_size 8 -learning_rate 0.08 -loss_func_metric 'QuantileLoss' &
   # echo $Yr
   # wait 10
done
