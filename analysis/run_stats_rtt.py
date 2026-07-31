import numpy as np
from itertools import combinations
from scipy.stats import friedmanchisquare


data = np.load("Data/RTT/rtt_DSA_260505.npy")
labels = ["RTT_251220", "RTT_260330_20", "RTT_260330_25", "RTT_251130_fs"]

np.random.seed(42)
n_iter = 10000
alpha = 2.5

per_model = {}
for i, label in enumerate(labels):
    row = data[i]
    n = len(row)
    means = np.empty(n_iter)
    for j in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        means[j] = np.mean(row[idx])
    per_model[label] = {"mean": np.mean(means), "std": np.std(means, ddof=1),
                        "ci_low": np.percentile(means, alpha),
                        "ci_high": np.percentile(means, 100 - alpha)}

ranked = sorted(labels, key=lambda x: per_model[x]["mean"])
for r, label in enumerate(ranked, 1):
    print(f"  {r}. {label:20s} DSA={per_model[label]['mean']:.6f}")

stat, p_f = friedmanchisquare(*data)
print(f"\nFriedman: chi2={stat:.3f}, p={p_f:.6f}")

pairs = []
for a, b in combinations(range(4), 2):
    d1, d2 = data[a], data[b]
    n = len(d1)
    diffs = np.empty(n_iter)
    for j in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        diffs[j] = np.mean(d1[idx] - d2[idx])
    obs = np.mean(diffs)
    p_val = np.mean(np.sign(diffs) != np.sign(obs))
    pairs.append({
        "pair": f"{labels[a]} vs {labels[b]}",
        "diff": obs,
        "ci_low": np.percentile(diffs, alpha),
        "ci_high": np.percentile(diffs, 100 - alpha),
        "p": 2.0 * min(p_val, 1 - p_val, 0.5),
    })

pairs.sort(key=lambda x: x["p"])
for r, pr in enumerate(pairs):
    pr["adj"] = 0.05 / (len(pairs) - r)
    pr["reject"] = pr["p"] < pr["adj"]
    sig = "***" if pr["p"] < 0.001 else "**" if pr["p"] < 0.01 else "*" if pr["p"] < 0.05 else "n.s."
    print(f"  {pr['pair']:40s} delta={pr['diff']:+8.6f} p={pr['p']:.4f} {sig}")
