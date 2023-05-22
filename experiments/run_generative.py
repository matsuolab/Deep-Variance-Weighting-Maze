import pickle
import numpy as np
import shinrl as srl
import os
WM = srl.DiscreteViSolver.DefaultConfig.WEIGHTMODE

# settings:
discount = 0.995
horizon = int(1 / (1 - discount))


def make_pitfall_maze(size, num_pitfalls, seed):
    np.random.seed(seed)
    maze = np.zeros((size, size))
    maze[0][0] = 3  # start
    maze[-1][-1] = 1  # reward
    # to ensure path to the reward
    maze[0][1] = -1
    maze[1][1] = -1
    maze[1][0] = -1
    maze[-2][0] = -1
    maze[-2][-1] = -1
    maze[-1][-2] = -1
    empty = np.array(np.where(maze == 0))
    pitfall = np.random.choice(len(empty[0]), num_pitfalls)
    maze[empty[0][pitfall], empty[1][pitfall]] = 4  # set pitfall
    maze = np.where(maze == -1, 0, maze)
    return maze


def run_solver(maze_eps, solver_config, seed, maze_seed, maze_size, num_pitfalls):
    import shinrl as srl
    import gym
    from shinrl import DiscreteViSolver
    maze = make_pitfall_maze(maze_size, num_pitfalls, maze_seed)
    env_config = srl.Maze.DefaultConfig(
                        obs_mode=srl.Maze.DefaultConfig.OBS_MODE.random,
                        eps=maze_eps,
                        discount=discount,
                        horizon=horizon,
                    )
    env = gym.make("ShinMaze-v0", maze=maze, config=env_config)
    env.reset()
    mixins = DiscreteViSolver.make_mixins(env, solver_config)
    solver = DiscreteViSolver.factory(env, solver_config, mixins)
    solver.seed(seed)
    solver.run()
    return solver.scalars


def main(args):
    if args.weight_mode == "none":
        wm = WM.none
        if args.is_munchausen:
            algo_name = "M-DQN"
        else:
            algo_name = "DQN"
    elif args.weight_mode == "sigmastar":
        wm = WM.sigma_star
        if args.is_munchausen:
            algo_name = "sigmastar-M-DQN"
        else:
            algo_name = "sigmastar-DQN"
    elif args.weight_mode == "dvw":
        wm = WM.dvw
        if args.is_munchausen:
            algo_name = "dvw-M-DQN"
        else:
            algo_name = "dvw-DQN"
    else:
        raise ValueError

    path = f"experiments/results/{args.maze_eps}-{args.num_samples_target}/{algo_name}"
    print("PATH: ", path)
    if not os.path.exists(path):
        os.makedirs(path)

    path += f"/{args.maze_seed}-{args.seed}.pkl"
    if args.is_munchausen:
        er_coef = 1e-5
        kl_coef = discount * er_coef / (1 - discount)
    else:
        er_coef = 0
        kl_coef = 0

    config = srl.DiscreteViSolver.DefaultConfig(
        approx="nn",
        explore="oracle",
        verbose=False,
        steps_per_epoch=args.iteration,
        eval_interval=int(args.iteration / 100),
        add_interval=int(args.iteration / 100),
        kl_coef=kl_coef,
        er_coef=er_coef,
        weight_mode=wm,
        target_update_interval=100,
        num_samples_target=args.num_samples_target,
        logp_clip=-1,
        lr=0.001,
    )

    scalars = run_solver(args.maze_eps, config, args.seed, args.maze_seed, args.maze_size, args.num_pitfall)
    with open(path, "wb") as f:
        pickle.dump(scalars, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--is_munchausen", action="store_true")
    parser.add_argument("--weight_mode", type=str, default="none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maze_seed", type=int, default=0)
    parser.add_argument("--maze_eps", type=float, default=0.1)
    parser.add_argument("--iteration", type=int, default=1000)
    parser.add_argument("--maze_size", type=int, default=25)
    parser.add_argument("--num_pitfall", type=int, default=8)
    parser.add_argument("--num_samples_target", type=int, default=3)

    args = parser.parse_args()
    main(args)
