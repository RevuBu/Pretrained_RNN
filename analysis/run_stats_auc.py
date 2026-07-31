import pickle
import numpy as np
from scipy.stats import chi2, f


def bootstrap_condition_ci(data, n_iter=10000, ci=95, seed=None, func="mean"):
    data = np.asarray(data, dtype=float)
    n = len(data)
    if seed is not None:
        np.random.seed(seed)
    stats = np.empty(n_iter)
    stat_fn = np.mean if func == "mean" else np.median
    for i in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        stats[i] = stat_fn(data[idx])
    alpha = (100 - ci) / 2
    return {
        func: float(stat_fn(stats)),
        "std": float(np.std(stats, ddof=1)),
        "ci_low": float(np.percentile(stats, alpha)),
        "ci_high": float(np.percentile(stats, 100 - alpha)),
        "ci": ci,
        "n_iter": n_iter,
    }


def bootstrap_paired_diff(monkey, model, n_iter=10000, ci=95, seed=None):
    monkey = np.asarray(monkey, dtype=float)
    model = np.asarray(model, dtype=float)
    n = len(monkey)
    assert len(model) == n
    if seed is not None:
        np.random.seed(seed)

    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        diffs[i] = np.mean(model[idx] - monkey[idx])

    obs_diff = float(np.mean(diffs))
    alpha = (100 - ci) / 2
    p_val = float(np.mean(np.sign(diffs) != np.sign(obs_diff)))
    return {
        "diff_mean": obs_diff,
        "std": float(np.std(diffs, ddof=1)),
        "ci_low": float(np.percentile(diffs, alpha)),
        "ci_high": float(np.percentile(diffs, 100 - alpha)),
        "p_value": 2.0 * min(p_val, 1 - p_val, 0.5),
        "ci": ci,
        "n_iter": n_iter,
    }


def friedman_test(data_matrix):
    data = np.asarray(data_matrix, dtype=float)
    n_sessions, n_conditions = data.shape
    ranked = np.array(
        [np.argsort(np.argsort(data[i])) + 1 for i in range(n_sessions)], dtype=float
    )
    avg_ranks = ranked.mean(axis=0)

    chi2_stat = (
        12
        * n_sessions
        / (n_conditions * (n_conditions + 1))
        * np.sum((avg_ranks - (n_conditions + 1) / 2) ** 2)
    )
    df = n_conditions - 1

    f_stat = (n_sessions - 1) * chi2_stat / (
        n_sessions * (n_conditions - 1) - chi2_stat
    )
    p_value = float(1 - f.cdf(f_stat, df, df * (n_sessions - 1)))
    p_chi2 = float(1 - chi2.cdf(chi2_stat, df))

    return {
        "chi2": float(chi2_stat),
        "df": df,
        "p_value": p_value,
        "p_value_chi2": p_chi2,
        "n_sessions": n_sessions,
        "n_conditions": n_conditions,
        "mean_ranks": avg_ranks.tolist(),
    }


def bootstrap_cc_profile(monkey, model, n_components=10, n_iter=10000,
                         ci=95, seed=None):
    monkey = np.asarray(monkey, dtype=float)
    model = np.asarray(model, dtype=float)
    n = monkey.shape[0]
    if seed is not None:
        np.random.seed(seed)

    diffs = np.empty((n_iter, n_components))
    for i in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        diffs[i] = model[idx].mean(0) - monkey[idx].mean(0)

    alpha = (100 - ci) / 2
    result = {
        "mean_monkey": monkey.mean(0).tolist(),
        "mean_model": model.mean(0).tolist(),
        "diff_mean": diffs.mean(0).tolist(),
        "diff_std": diffs.std(0, ddof=1).tolist(),
        "ci_low": [float(np.percentile(diffs[:, j], alpha)) for j in range(n_components)],
        "ci_high": [float(np.percentile(diffs[:, j], 100 - alpha)) for j in range(n_components)],
        "ci": ci,
        "n_iter": n_iter,
    }
    return result


def analyze(monkey_label, auc_file, cca_file):
    with open(auc_file, "rb") as f:
        auc_data = pickle.load(f)
    with open(cca_file, "rb") as f:
        cca_data = pickle.load(f)

    monkey_key = [k for k in auc_data if monkey_label in k][0]
    model_keys = [k for k in auc_data if k != monkey_key]
    n_sessions = len(auc_data[monkey_key])
    n_cc = cca_data[monkey_key].shape[1]

    print(f"Sessions: {n_sessions}, Conditions: {len(auc_data)}, CC: {n_cc}")
    print()

    # 1. Bootstrap CI
    ranking = []
    for key in [monkey_key] + model_keys:
        ci = bootstrap_condition_ci(auc_data[key], n_iter=10000, seed=42)
        ranking.append((ci["mean"], key, ci))
    ranking.sort(key=lambda x: -x[0])
    for i, (mean, key, _) in enumerate(ranking, 1):
        print(f"  {i}. {key:20s} AUC={mean:.4f}")

    # 2. Paired diff vs monkey
    monkey_auc = auc_data[monkey_key]
    pairs = []
    for key in model_keys:
        result = bootstrap_paired_diff(monkey_auc, auc_data[key], n_iter=10000, seed=42)
        pairs.append((key, result))

    sorted_idx = np.argsort([p[1]["p_value"] for p in pairs])
    holm = {}
    for rank_idx, orig_idx in enumerate(sorted_idx):
        key, result = pairs[orig_idx]
        result["holm_adj"] = 0.05 / (len(pairs) - rank_idx)
        result["holm_reject"] = result["p_value"] < result["holm_adj"]
        holm[key] = result

    print(f"\n{'Model':20s} {'Diff':>8s} {'CI_low':>8s} {'CI_high':>8s} {'p':>8s}")
    for key in model_keys:
        r = holm[key]
        print(f"  {key:20s} {r['diff_mean']:+8.4f} {r['ci_low']:8.4f} {r['ci_high']:8.4f} {r['p_value']:8.4f}")

    # 3. Friedman test
    data_matrix = np.column_stack([auc_data[k] for k in [monkey_key] + model_keys])
    ft = friedman_test(data_matrix)
    print(f"\nFriedman: chi2={ft['chi2']:.4f}, p={ft['p_value']:.6f}")

    # 4. CC profile
    monkey_cc = cca_data[monkey_key]
    for key in model_keys:
        profile = bootstrap_cc_profile(monkey_cc, cca_data[key], n_components=n_cc, n_iter=10000, seed=42)
        n_sig = sum(1 for j in range(n_cc) if profile["ci_low"][j] > 0 or profile["ci_high"][j] < 0)
        print(f"  {key:20s} {n_sig}/{n_cc} CC components significant")


if __name__ == "__main__":
    print("=" * 60)
    print("MONKEY C ANALYSIS")
    print("=" * 60)
    auc_file_C = "Data/CO/AUC_C_all.pkl"
    cca_file_C = "Data/CO/CCA_C_all.pkl"
    analyze("Monkey C", auc_file_C, cca_file_C)

    print()
    print("=" * 60)
    print("MONKEY M ANALYSIS")
    print("=" * 60)
    auc_file_M = "Data/CO/AUC_M_all.pkl"
    cca_file_M = "Data/CO/CCA_M_all.pkl"
    analyze("Monkey M", auc_file_M, cca_file_M)
