import numpy as np


dataset_labels = [
    "MC_agg", "MM_agg",
    "SR_Dense", "SR_Moderate", "SR_Sparse",
    "CO_Single", "CO_Multi",
    "DR_Single", "DR_Multi",
]
file_name = "Data/CO/"

for fname, monkey in [("DSA_C.npy", "Monkey C"), ("DSA_M.npy", "Monkey M")]:
    arr = np.load(file_name+fname)
    print(f"  {monkey}")

    np.random.seed(42)
    n_iter = 10000
    alpha = 2.5

    per_target = {}
    for j, label in enumerate(dataset_labels):
        data = arr[j]
        n = len(data)
        means = np.empty(n_iter)
        for i in range(n_iter):
            idx = np.random.choice(n, n, replace=True)
            means[i] = np.mean(data[idx])
        per_target[label] = {
            "mean": np.mean(means),
            "std": np.std(means, ddof=1),
            "ci_low": np.percentile(means, alpha),
            "ci_high": np.percentile(means, 100 - alpha),
        }

    sorted_targets = sorted(dataset_labels, key=lambda x: per_target[x]["mean"])
    for rank, label in enumerate(sorted_targets, 1):
        print(f"  {rank}. {label:15s} DSA={per_target[label]['mean']:.6f}")

    # Pairwise bootstrap (7 models only)
    models = dataset_labels[2:]
    n_models = len(models)
    pairs = []
    for a in range(n_models):
        for b in range(a + 1, n_models):
            data1 = arr[dataset_labels.index(models[a])]
            data2 = arr[dataset_labels.index(models[b])]
            n = len(data1)
            diffs = np.empty(n_iter)
            for i in range(n_iter):
                idx = np.random.choice(n, n, replace=True)
                diffs[i] = np.mean(data1[idx] - data2[idx])
            obs = np.mean(diffs)
            p_val = np.mean(np.sign(diffs) != np.sign(obs))
            pairs.append({
                "pair": f"{models[a]} vs {models[b]}",
                "diff": obs,
                "ci_low": np.percentile(diffs, alpha),
                "ci_high": np.percentile(diffs, 100 - alpha),
                "p": 2.0 * min(p_val, 1 - p_val, 0.5),
            })

    pairs.sort(key=lambda x: x["p"])
    for r, pr in enumerate(pairs):
        pr["adj"] = 0.05 / (len(pairs) - r)
        pr["reject"] = pr["p"] < pr["adj"]

    sig = [p for p in pairs if p["reject"]]
    print(f"  Sig pairs: {len(sig)}/{len(pairs)}")
    for p in sig:
        print(f"    {p['pair']:40s} delta={p['diff']:+8.6f} p={p['p']:.4f}")
    print()
