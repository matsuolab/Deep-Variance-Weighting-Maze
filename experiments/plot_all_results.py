import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme(style="white")

nonreg_label_color= {
    "sigmastar-DQN": (r"$f=f_*, \tau=\kappa=0$", sns.color_palette()[2]),
    "DQN": (r"$f=\mathbf{1}, \tau=\kappa=0$", sns.color_palette()[3]),
    "dvw-DQN": (r"$f=f^{\mathrm{DVW}}, \tau=\kappa=0$", sns.color_palette()[4]),
}

reg_label_color= {
    "sigmastar-M-DQN": (r"$f=f_*, \tau > 0, \kappa > 0$", sns.color_palette()[0]),
    "M-DQN": (r"$f=\mathbf{1}, \tau > 0, \kappa > 0$", sns.color_palette()[1]),
    "dvw-M-DQN": (r"$f=f^{\mathrm{DVW}}, \tau > 0, \kappa > 0$", sns.color_palette()[6]),
}

 
def to_df(results, label):
    xs = []
    ys = []
    for result in results:
        xs.append(np.array(result[label]["x"], dtype=int) * 100)  # 100 is due to implementation
        ys.append(np.array(result[label]["y"], dtype=float))
    xs, ys = np.hstack(xs), np.hstack(ys)
    return pd.DataFrame({"step": xs, label: ys})


def plot_result(results_scalar, ax, label_color):
    for key, r in results_scalar.items():
        label, color = label_color[key]
        sns.lineplot(data=to_df(r, "NormedGap"), x="step", y="NormedGap", errorbar=("sd", 0.5), label=label, legend=None, estimator=np.mean, ax=ax, color=color)


def get_results_scalar(maze_eps, num_samples_target, label_color):
    _path = results_dir / f"{maze_eps}-{num_samples_target}"
    results_scalar = {key: [] for key in label_color.keys()}

    # load scalars
    for key in label_color.keys():
        path = _path / key
        for p in path.rglob("*.pkl"):
            with open(str(p), "rb") as f:
                results_scalar[key].append(pickle.load(f))

    return results_scalar


def plot_and_save(results_dir, regularized=True):
    fig = plt.figure(figsize=(12, 4))
    sns.set(font_scale=1.4)
    axes = []
    if regularized:
        label_color = reg_label_color
        filename = "reg-optimality-gap"
    else:
        label_color = nonreg_label_color
        filename = "optimality-gap"

    with sns.axes_style("whitegrid"):
        ax = fig.add_subplot(121)
        results_scalar = get_results_scalar(0.1, 3, label_color)
        plot_result(results_scalar, ax, label_color)
        ax.set_ylabel(r"Normalized $\|v_* - v_{\pi_k}\|_\infty$", fontsize=21)
        ax.set_ylim([0., 1.0])
        ax.set_xlabel(r"Iteration", fontsize=21)
        axes.append(ax)
    
        ax = fig.add_subplot(122)
        results_scalar = get_results_scalar(0.1, 10, label_color)
        plot_result(results_scalar, ax, label_color)
        ax.get_yaxis().get_label().set_visible(False)
        ax.set_ylim([0., 1.0])
        ax.set_xlabel(r"Iteration", fontsize=21)
        axes.append(ax)

        agents = [label_color[key][0] for key in results_scalar.keys()]
        handles = [None] * len(agents)
        for ax in axes:
            handle, label = ax.get_legend_handles_labels()
            for h, agent in zip(handle, label):
                handles[agents.index(agent)] = h

        lgd = fig.legend(handles, agents, loc="upper center",
                        bbox_to_anchor=(0.5, 1.2), ncol=4)
        fig.tight_layout()
        plt.savefig(str(results_dir / (filename + ".png")), bbox_extra_artists=(lgd, ), bbox_inches="tight")
        plt.savefig(str(results_dir / (filename + ".pdf")), bbox_extra_artists=(lgd, ), bbox_inches="tight")


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    results_dir = pathlib.Path(__file__).parent / "results"
    plot_and_save(results_dir, regularized=True)
    plot_and_save(results_dir, regularized=False)
