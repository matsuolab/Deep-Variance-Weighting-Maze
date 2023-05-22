#!/bin/sh
maze_eps=$1
maze_seed=$2
seed=$3
num_samples_target=$4
weight_mode=$5
maze_size=25
num_pitfalls=8
iteration=20000

# munchausen
poetry run python experiments/run_generative.py --weight_mode $weight_mode --seed $seed --maze_seed $maze_seed --maze_eps $maze_eps --iteration $iteration --maze_size $maze_size --num_pitfall $num_pitfalls --num_samples_target $num_samples_target --is_munchausen

# vanilla
poetry run python experiments/run_generative.py --weight_mode $weight_mode --seed $seed --maze_seed $maze_seed --maze_eps $maze_eps --iteration $iteration --maze_size $maze_size --num_pitfall $num_pitfalls --num_samples_target $num_samples_target
