#!/bin/bash

# Define the number of parallel jobs
# parallel_jobs=4

# Define the input arguments
# inputs=("train_TFT" "--pred" "input3" "input4" "input5")

# Loop the Python script in parallel with input arguments
# echo "${inputs[@]}" | tr ' ' '\n' | parallel -j $parallel_jobs python AAChinaTFT-D.py {}
# QuantileLoss

for Yr in 2017; do
   python AAB.py train_TFT -exp_name 'AAB_RMSE_CyclicLR' -predicted_year $Yr -batch_size 96 -learning_rate 0.085 -loss_func_metric 'RMSE' -max_epochs 180
   # echo $Yr
   # wait 10
done
