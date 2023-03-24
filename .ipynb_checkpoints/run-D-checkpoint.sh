#!/bin/bash

# Define the number of parallel jobs
# parallel_jobs=4

# Define the input arguments
# inputs=("train_TFT" "--pred" "input3" "input4" "input5")

# Loop the Python script in parallel with input arguments
# echo "${inputs[@]}" | tr ' ' '\n' | parallel -j $parallel_jobs python AAChinaTFT-D.py {}

for Yr in 2004 2008; do
   python AAChinaTFT-D.py train_TFT -exp_name 'RMSE_lr_finder_test' -predicted_year $Yr -batch_size 4 -learning_rate 0.05897 -loss_func_metric 'RMSE' &
   # echo $Yr
   # wait 10
done
