import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N = 100
B_SCALE = 100.0
B_MEAN = 0.5
B_MIN = 0.1
B_MAX = 0.9

G = 1000.0

SIGMAS = [0.0, 0.05, 0.10, 0.15, 0.20]

NUM_TRIALS = 100

NODE_COUNTS = [10, 25, 50, 100, 200]
SWEEP_SIGMAS = [0.0, 0.10, 0.20]

RANDOM_SEED = 42

ALEPH_RNG = np.random.default_rng(RANDOM_SEED + 1)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
IMAGES_DIR = os.path.join(HERE, "images")


def generate_bandwidths(sigma, rng, n=N):
    if sigma == 0:
        return np.full(n, B_MEAN) * B_SCALE

    beta = []

    while len(beta) < n:
        samples = rng.normal(B_MEAN, sigma, size=n * 2)
        samples = samples[(samples >= B_MIN) & (samples <= B_MAX)]
        beta.extend(samples.tolist())

    return np.array(beta[:n]) * B_SCALE


def communication_time(beta):
    return G / beta


def bottleneck_bandwidth(B, degrees):
    linked = degrees > 0

    return np.min(B[linked] / degrees[linked])


def calculate_time(B, degrees, volume=1.0):
    beta = bottleneck_bandwidth(B, degrees)

    return {
        "beta": beta,
        "total_time": volume * communication_time(beta),
    }


def build_ebot(B):
    n = len(B)
    order = np.argsort(-B)

    degrees = np.zeros(n, dtype=int)

    selected = [order[0]]

    for i in order[1:]:

        best_parent = None
        best_cost = np.inf

        for j in selected:

            temp_degrees = degrees.copy()
            temp_degrees[i] += 1
            temp_degrees[j] += 1

            cost = communication_time(bottleneck_bandwidth(B, temp_degrees))

            if cost < best_cost:
                best_cost = cost
                best_parent = j

        degrees[i] += 1
        degrees[best_parent] += 1

        selected.append(i)

    return degrees


def build_ps(B):
    n = len(B)
    root = np.argmax(B)

    degrees = np.ones(n, dtype=int)
    degrees[root] = n - 1

    return degrees


def build_ring(B):
    n = len(B)

    return np.full(n, 2, dtype=int)


def build_aleph(B):
    n = len(B)
    ps = int(np.argmax(B))
    workers = [w for w in range(n) if w != ps]

    b = B[workers].min()
    capacity = {w: int(np.floor(B[w] / b - 1 + 1e-9)) for w in workers}

    unclustered = set(workers)
    clusters = {}

    while unclustered:

        g_bar = {w: min(capacity[w], len(unclustered)) for w in unclustered}
        g_hat = {w: 1 << max(0, int(np.ceil(np.log2(max(g, 1)))))
                 for w, g in g_bar.items()}

        top = max(g_hat.values())
        cas = [w for w in unclustered if g_hat[w] == top and g_bar[w] > 0]

        if not cas:
            break

        alpha = {w: g_bar[w] / len(unclustered) for w in cas}
        m = sum(alpha.values())

        selected = False

        for w in cas:

            if w not in unclustered:
                continue

            if ALEPH_RNG.random() >= min(1.0, 1.0 / m):
                continue

            selected = True
            members = [w]
            unclustered.discard(w)

            for w2 in sorted(unclustered):
                if len(members) - 1 >= capacity[w]:
                    break
                if ALEPH_RNG.random() < alpha[w]:
                    members.append(w2)

            unclustered.difference_update(members)
            clusters[w] = members

        if not selected:
            break

    degrees = np.zeros(n, dtype=int)

    for agg, members in clusters.items():
        degrees[ps] += 1
        degrees[agg] += 1
        for w in members[1:]:
            degrees[w] += 1
            degrees[agg] += 1

    for w in unclustered:
        degrees[ps] += 1
        degrees[w] += 1

    return degrees


ARCHITECTURES = {
    "eBOT": build_ebot,
    "PS": build_ps,
    "Ring": build_ring,
    "ALEPH": build_aleph,
}

VOLUME = {
    "Ring": lambda n: (n - 1) / n,
}


def run_trial(sigma, rng, n=N):
    B = generate_bandwidths(sigma, rng, n)

    return [
        {
            "sigma": sigma,
            "architecture": name,
            **calculate_time(
                B, build(B), volume=VOLUME.get(name, lambda n: 1.0)(n)
            ),
        }
        for name, build in ARCHITECTURES.items()
    ]


def run_simulation(sigmas=SIGMAS, trials=NUM_TRIALS, seed=RANDOM_SEED):
    global ALEPH_RNG
    rng = np.random.default_rng(seed)
    ALEPH_RNG = np.random.default_rng(seed + 1)

    rows = []

    for sigma in sigmas:
        for trial in range(trials):
            for result in run_trial(sigma, rng):
                result["trial"] = trial
                rows.append(result)

    return pd.DataFrame(rows)


def run_node_sweep(
    node_counts=NODE_COUNTS,
    sigmas=SWEEP_SIGMAS,
    trials=NUM_TRIALS,
    seed=RANDOM_SEED
):
    global ALEPH_RNG
    rng = np.random.default_rng(seed)
    ALEPH_RNG = np.random.default_rng(seed + 1)

    rows = []

    for n in node_counts:
        for sigma in sigmas:
            for trial in range(trials):
                for result in run_trial(sigma, rng, n):
                    result["N"] = n
                    result["trial"] = trial
                    rows.append(result)

    return pd.DataFrame(rows)


def summarize(df, by=("sigma",)):
    return (
        df
        .groupby([*by, "architecture"])
        .agg(
            total_time_mean=("total_time", "mean"),
            total_time_std=("total_time", "std"),
            beta_mean=("beta", "mean"),
            beta_std=("beta", "std"),
        )
        .reset_index()
    )


STYLE = {
    "eBOT":  {"color": "#2a78d6", "marker": "o"},
    "PS":    {"color": "#eb6834", "marker": "s"},
    "Ring":  {"color": "#1baf7a", "marker": "^"},
    "ALEPH": {"color": "#eda100", "marker": "D"},
}

HIGHLIGHT = "eBOT"
HIGHLIGHT_STYLE = {"linewidth": 2.6, "markersize": 6.5, "zorder": 5}
BASELINE_STYLE = {"linewidth": 1.4, "markersize": 4.5, "zorder": 3,
                  "linestyle": "--"}

FIGURES = {
    "aggregation_time_vs_sigma": (
        "total_time_mean", "total_time_std",
        "Aggregation time (ms)", True,
    ),
    "effective_bandwidth_vs_sigma": (
        "beta_mean", "beta_std",
        "Min. effective bandwidth (Gbps)", True,
    ),
}


def plot_metric(
    summary, column, ylabel, yerr_column=None, log=False,
    x="sigma", xlabel=r"Bandwidth heterogeneity $\sigma$", logx=False
):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for architecture in ARCHITECTURES:

        data = summary[summary["architecture"] == architecture]

        ax.errorbar(
            data[x],
            data[column],
            yerr=None if yerr_column is None else data[yerr_column],
            label=architecture,
            capsize=3,
            elinewidth=1,
            **STYLE[architecture],
            **(HIGHLIGHT_STYLE if architecture == HIGHLIGHT
               else BASELINE_STYLE),
        )

        if architecture != HIGHLIGHT:
            ax.plot(
                data[x], data[column],
                linestyle="none",
                markersize=BASELINE_STYLE["markersize"],
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=HIGHLIGHT_STYLE["zorder"] + 1,
                **STYLE[architecture],
            )

    if log:
        ax.set_yscale("log")

    if logx:
        ax.set_xscale("log")
        ax.minorticks_off()

    ticks = sorted(summary[x].unique())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(ticks)
    if logx:
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.grid(True, which="major", color="#e4e4e2", linewidth=0.6)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    legend = ax.legend(frameon=False, ncol=2, fontsize=7.5)
    for text in legend.get_texts():
        if text.get_text() == HIGHLIGHT:
            text.set_fontweight("bold")
    fig.tight_layout()

    return fig


def save_figures(summary, outdir=IMAGES_DIR, show=False):
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
    })

    paths = []

    for name, (column, yerr_column, ylabel, log) in FIGURES.items():

        fig = plot_metric(summary, column, ylabel, yerr_column, log)

        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)

        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return paths


def save_node_figures(sweep_summary, outdir=IMAGES_DIR, show=False):
    os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 9,
    })

    paths = []

    for sigma in sorted(sweep_summary["sigma"].unique()):

        data = sweep_summary[sweep_summary["sigma"] == sigma]

        fig = plot_metric(
            data, "total_time_mean", "Aggregation time (ms)",
            "total_time_std", log=True,
            x="N", xlabel="Number of nodes $N$", logx=True,
        )

        name = f"aggregation_time_vs_nodes_sigma{round(sigma * 100):03d}"
        path = os.path.join(outdir, f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)

        if not show:
            plt.close(fig)

    if show:
        plt.show()

    return paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--images-dir", default=IMAGES_DIR)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    df = run_simulation(trials=args.trials, seed=args.seed)
    summary = summarize(df)

    print("\n===== Simulation Results =====\n")
    print(summary.to_string(index=False))

    os.makedirs(args.results_dir, exist_ok=True)

    csv_paths = [
        os.path.join(args.results_dir, "aggregation_simulation_results.csv"),
        os.path.join(args.results_dir, "aggregation_simulation_summary.csv"),
    ]

    df.to_csv(csv_paths[0], index=False)
    summary.to_csv(csv_paths[1], index=False)

    image_paths = save_figures(summary, args.images_dir, show=args.show)

    sweep = run_node_sweep(trials=args.trials, seed=args.seed)
    sweep_summary = summarize(sweep, by=("N", "sigma"))

    print("\n===== Scalability Results =====\n")
    print(sweep_summary.to_string(index=False))

    csv_paths += [
        os.path.join(args.results_dir, "node_sweep_results.csv"),
        os.path.join(args.results_dir, "node_sweep_summary.csv"),
    ]

    sweep.to_csv(csv_paths[2], index=False)
    sweep_summary.to_csv(csv_paths[3], index=False)

    image_paths += save_node_figures(
        sweep_summary, args.images_dir, show=args.show
    )

    print("\nResults saved to:")
    for path in csv_paths + image_paths:
        print(f"  {path}")
