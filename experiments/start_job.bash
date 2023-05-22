maze_eps=0.1
num_samples_target=3
weight_mode=sigmastar

for weight_mode in dvw sigmastar none; do
    for maze_seed in {1..20}; do
        for seed in {1..3}; do
            bash experiments/run_generative.bash $maze_eps $maze_seed $seed $num_samples_target $weight_mode
        done
    done
done
