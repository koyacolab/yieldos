#!/bin/bash

# Define the number of parallel jobs
# parallel_jobs=4

# Define the input arguments
# inputs=("train_TFT" "--pred" "input3" "input4" "input5")

# Loop the Python script in parallel with input arguments
# echo "${inputs[@]}" | tr ' ' '\n' | parallel -j $parallel_jobs python AAChinaTFT-D.py {}
# QuantileLoss

python A0Q.py train_TFT -exp_name 'A0Q4_corn' -predicted_years "2005" -batch_size 128 -learning_rate 0.0001 -loss_func_metric 'RMSE' -max_epochs 2550 -crop_name 'corn'

# for Yr in 2017; do
#    python A0.py train_TFT -exp_name 'A0_RMSE_cyclic' -predicted_year $Yr -batch_size 128 -learning_rate 0.01 -loss_func_metric 'RMSE' -max_epochs 240
#    # echo $Yr
#    # wait 10
# done
