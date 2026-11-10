import argparse
import os
import time

import numpy as np
import pandas as pd
from pyscipopt import Model, quicksum


G = 1000.0

SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)

CONFIGS = [
    (10, 0.00),
    (10, 0.10),
    (20, 0.00),
    (20, 0.10),
    (30, 0.00),
]


def generate_bandwidths(N, sigma, rng):
    if sigma == 0:
        q = np.full(N, 0.5)
    else:
        q = rng.normal(0.5, sigma, N)
        q = np.clip(q, 0.1, 0.9)

    return 100.0 * q


def evaluate_tree(B, edges):
    N = len(B)

    degrees = np.zeros(N, dtype=int)

    for child, parent in edges:
        degrees[child] += 1
        degrees[parent] += 1

    beta = np.min(B / (degrees + 1))

    return {
        "objective": G / beta,
        "beta": beta,
        "degrees": degrees,
    }


def eBOT_with_root(B, root):
    N = len(B)

    V = [root]
    E = []

    degrees = np.zeros(N, dtype=int)

    remaining = sorted(
        [i for i in range(N) if i != root],
        key=lambda i: B[i],
        reverse=True
    )

    for i in remaining:

        best_parent = None
        best_cost = float("inf")

        for j in V:

            temp_degrees = degrees.copy()
            temp_degrees[j] += 1
            temp_degrees[i] = 1

            cost = G / np.min(B / (temp_degrees + 1))

            if cost < best_cost:
                best_cost = cost
                best_parent = j

        E.append((i, best_parent))
        V.append(i)

        degrees[best_parent] += 1
        degrees[i] = 1

    result = evaluate_tree(B, E)
    result.update(edges=E, root=root)

    return result


def eBOT(B):
    start_time = time.time()

    best = min(
        (eBOT_with_root(B, root) for root in range(len(B))),
        key=lambda s: s["objective"]
    )

    best["runtime"] = time.time() - start_time

    return best


def solve_p2_scip(
    B,
    time_limit=300,
    gap_limit=1e-6,
    warm_start=None,
    verbose=False
):
    N = len(B)

    model = Model("P2")

    if not verbose:
        model.hideOutput()

    x = {
        (i, j): model.addVar(vtype="B", name=f"x_{i}_{j}")
        for i in range(N)
        for j in range(N)
        if i != j
    }

    z = {
        i: model.addVar(vtype="B", name=f"root_{i}")
        for i in range(N)
    }

    beta = model.addVar(
        lb=0.0, ub=float(np.max(B)), vtype="C", name="beta"
    )

    u = {
        i: model.addVar(lb=0.0, ub=N - 1, vtype="C", name=f"u_{i}")
        for i in range(N)
    }

    T_comm = model.addVar(lb=0.0, vtype="C", name="T_comm")

    for i in range(N):
        model.addCons(
            quicksum(x[i, j] for j in range(N) if j != i) + z[i] == 1,
            name=f"parent_or_root_{i}"
        )

    model.addCons(
        quicksum(z[i] for i in range(N)) == 1,
        name="one_root"
    )

    for i in range(N):

        degree = (
            quicksum(x[i, j] for j in range(N) if j != i)
            + quicksum(x[j, i] for j in range(N) if j != i)
        )

        model.addCons(
            beta * (degree + 1) <= B[i],
            name=f"normalized_bandwidth_{i}"
        )

    for (i, j), x_ij in x.items():
        model.addCons(
            u[i] >= u[j] + 1 - N * (1 - x_ij),
            name=f"acyclic_{i}_{j}"
        )

    model.addCons(T_comm * beta >= G, name="communication_time")

    model.setObjective(T_comm, "minimize")

    model.setParam("limits/time", time_limit)
    model.setParam("limits/gap", gap_limit)

    if warm_start is not None:

        sol = model.createSol()
        parent_of = dict(warm_start["edges"])
        order = {warm_start["root"]: 0}
        while len(order) < N:
            for child, parent in warm_start["edges"]:
                if parent in order and child not in order:
                    order[child] = order[parent] + 1

        for (i, j), x_ij in x.items():
            model.setSolVal(sol, x_ij, 1.0 if parent_of.get(i) == j else 0.0)

        for i in range(N):
            model.setSolVal(sol, z[i], 1.0 if i == warm_start["root"] else 0.0)
            model.setSolVal(sol, u[i], float(order[i]))

        model.setSolVal(sol, beta, float(warm_start["beta"]))
        model.setSolVal(sol, T_comm, float(warm_start["objective"]))

        model.addSol(sol)

    start_time = time.time()
    model.optimize()
    runtime = time.time() - start_time

    if model.getNSols() == 0:
        return None

    edges = [
        (i, j) for (i, j), x_ij in x.items()
        if model.getVal(x_ij) > 0.5
    ]

    root = next(i for i in range(N) if model.getVal(z[i]) > 0.5)

    result = evaluate_tree(B, edges)
    result.update(
        edges=edges,
        root=root,
        runtime=runtime,
        gap=model.getGap(),
        status=str(model.getStatus()),
    )

    return result


def run_benchmark(
    configs=CONFIGS,
    trials=20,
    time_limit=300,
    seed=SEED,
    verbose=True
):
    rng = np.random.default_rng(seed)

    rows = []

    for N, sigma in configs:

        if verbose:
            print(f"\n=== N = {N}, sigma = {sigma:.2f} ===")

        n_trials = 1 if sigma == 0 else trials

        for trial in range(n_trials):

            B = generate_bandwidths(N, sigma, rng)

            heuristic = eBOT(B)
            reference = solve_p2_scip(
                B, time_limit=time_limit, warm_start=heuristic
            )

            if reference is None:
                if verbose:
                    print(f"Trial {trial:3d}: SCIP found no solution")
                continue

            gap = (
                (heuristic["objective"] - reference["objective"])
                / reference["objective"]
                * 100
            )

            rows.append({
                "N": N,
                "sigma": sigma,
                "trial": trial,
                "ebot_objective": heuristic["objective"],
                "ref_objective": reference["objective"],
                "ebot_gap_percent": gap,
                "ebot_beta": heuristic["beta"],
                "ref_beta": reference["beta"],
                "ebot_runtime": heuristic["runtime"],
                "ref_runtime": reference["runtime"],
                "ref_mip_gap": reference["gap"],
                "ref_status": reference["status"],
            })

            if verbose:
                print(
                    f"Trial {trial:3d}: "
                    f"eBOT={heuristic['objective']:.3f}, "
                    f"ref={reference['objective']:.3f} "
                    f"({reference['status']}, "
                    f"{reference['runtime']:.1f}s), "
                    f"gap={gap:.2f}%"
                )

    return pd.DataFrame(rows)


def summarize(df):
    summary = (
        df
        .groupby(["N", "sigma"], sort=False)
        .agg(
            ebot_objective=("ebot_objective", "mean"),
            ref_objective=("ref_objective", "mean"),
            ebot_gap_percent=("ebot_gap_percent", "mean"),
            ebot_runtime=("ebot_runtime", "mean"),
            ref_runtime=("ref_runtime", "mean"),
            trials=("trial", "count"),
            ref_optimal=(
                "ref_status",
                lambda s: s.isin(["optimal", "gaplimit"]).sum()
            ),
        )
        .reset_index()
    )

    return summary


def to_markdown(summary):
    lines = [
        "| \\(N\\) | \\(\\sigma\\) | eBOT objective "
        "| Reference objective | eBOT gap "
        "| eBOT Time (ms) | Solver Time (s) |",
        "| ----: | ---------: | -------------: "
        "| ------------------: | -------: "
        "| -------------: | --------------: |",
    ]

    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.N:5d} "
            f"| {row.sigma:10.2f} "
            f"| {row.ebot_objective:14.3f} "
            f"| {row.ref_objective:19.3f} "
            f"| {row.ebot_gap_percent:7.2f}% "
            f"| {row.ebot_runtime * 1000:14.2f} "
            f"| {row.ref_runtime:15.2f} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    df = run_benchmark(
        trials=args.trials,
        time_limit=args.time_limit,
        seed=args.seed,
    )

    summary = summarize(df)

    os.makedirs(args.results_dir, exist_ok=True)

    csv_paths = [
        os.path.join(args.results_dir, "ebot_vs_scip.csv"),
        os.path.join(args.results_dir, "ebot_vs_scip_summary.csv"),
    ]

    df.to_csv(csv_paths[0], index=False)
    summary.to_csv(csv_paths[1], index=False)

    print("\n" + to_markdown(summary))

    not_optimal = summary[summary["ref_optimal"] < summary["trials"]]
    if not not_optimal.empty:
        print(
            "\nNote: SCIP hit the time limit on some instances; "
            "their reference is the best incumbent, not a proven optimum:"
        )
        print(not_optimal[["N", "sigma", "trials", "ref_optimal"]]
              .to_string(index=False))

    print("\nSaved:")
    for path in csv_paths:
        print(f"  {path}")
