# stats_compare_8conditions.py
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata

# ============================
# 使い方（最低限ここだけ編集）
# ============================
# runsフォルダ配下に、各条件の out_dir があり、その中に test_metrics.csv がある想定
RUNS_ROOT = r"C:\Users\orilab\Desktop\masumoto\2dunet\runs"  # 例: r"C:\Users\orilab\Desktop\masumoto\2dunet\runs"
CSV_NAME = "test_metrics.csv"

# 事後検定の対象指標
METRIC_COL = "Dice"

# 片側検定も出したいなら True（「best > others」を統計的に言いたいとき）
DO_ONE_SIDED = True


# ============================
# Holm補正
# ============================
def holm_adjust(pvals, alpha=0.05):
    """
    Holm-Bonferroni adjustment.
    Returns adjusted p-values (same order as input).
    """
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)

    # step-down
    for k, idx in enumerate(order):
        adj[idx] = (m - k) * pvals[idx]

    # monotone non-decreasing in sorted order
    # enforce cumulative max on sorted adjusted pvalues, then map back
    adj_sorted = adj[order]
    adj_sorted = np.maximum.accumulate(adj_sorted)
    adj[order] = np.minimum(adj_sorted, 1.0)

    return adj


# ============================
# Kendall's W (Friedmanの効果量)
# ============================
def kendalls_w(data_2d):
    """
    data_2d: shape (n_subjects, k_conditions), values are scores (Dice).
    Kendall's W for Friedman (effect size): 0..1
    """
    X = np.asarray(data_2d, dtype=float)
    n, k = X.shape

    # ranks within each subject (row); average ranks for ties
    ranks = np.vstack([rankdata(row, method="average") for row in X])
    Rj = ranks.sum(axis=0)
    Rbar = np.mean(Rj)
    S = np.sum((Rj - Rbar) ** 2)
    W = 12 * S / (n**2 * (k**3 - k) + 1e-12)
    return float(W)


# ============================
# CSV収集（test_metrics.csvを全部拾う）
# ============================
csv_paths = glob.glob(os.path.join(RUNS_ROOT, "**", CSV_NAME), recursive=True)
if len(csv_paths) == 0:
    raise FileNotFoundError(f"No {CSV_NAME} found under: {RUNS_ROOT}")

# condition名 = test_metrics.csv が入ってるフォルダ名
cond2df = {}
for p in csv_paths:
    cond = os.path.basename(os.path.dirname(p))
    df = pd.read_csv(p)
    # MEAN行は除外
    df = df[df["case_id"] != "MEAN"].copy()
    df = df[["case_id", METRIC_COL]].copy()
    df.rename(columns={METRIC_COL: cond}, inplace=True)
    cond2df[cond] = df

# ============================
# case_idで内部結合して、全条件共通の症例だけに揃える
# ============================
dfs = list(cond2df.values())
wide = dfs[0]
for d in dfs[1:]:
    wide = pd.merge(wide, d, on="case_id", how="inner")

if wide.shape[0] < 3:
    raise ValueError("Common cases across conditions are too few after merge.")

# 条件列
cond_cols = [c for c in wide.columns if c != "case_id"]

# ============================
# Friedman検定
# ============================
X = wide[cond_cols].to_numpy(dtype=float)  # (n_cases, k_conditions)
friedman_stat, friedman_p = friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
W = kendalls_w(X)

print("=== Friedman test (repeated measures) ===")
print(f"n_cases = {X.shape[0]}, k_conditions = {X.shape[1]}")
print(f"Friedman chi2 = {friedman_stat:.6f}, p = {friedman_p:.6g}")
print(f"Kendall's W (effect size) = {W:.4f}  (0..1, larger = stronger)")

# ============================
# ベスト条件（平均Diceが最大）を自動選択
#   ※中央値で選びたいなら mean() -> median() に変えてOK
# ============================
means = wide[cond_cols].mean(axis=0)
best_cond = means.idxmax()
print("\n=== Best condition by mean Dice ===")
print(f"Best = {best_cond}, mean Dice = {means[best_cond]:.6f}")

# ============================
# 事後検定：best vs others（対応あり）Wilcoxon + Holm補正
# ============================
pvals_two = []
others = [c for c in cond_cols if c != best_cond]
for c in others:
    # wilcoxonは差が全部0だと落ちることがあるので例外処理
    a = wide[best_cond].to_numpy(dtype=float)
    b = wide[c].to_numpy(dtype=float)
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        # 全差0など：p=1扱い
        stat, p = np.nan, 1.0
    pvals_two.append(p)

pvals_two_adj = holm_adjust(pvals_two)

res = pd.DataFrame(
    {
        "compare": [f"{best_cond} vs {c}" for c in others],
        "p_two_sided": pvals_two,
        "p_two_sided_holm": pvals_two_adj,
        "mean_best": [means[best_cond]] * len(others),
        "mean_other": [means[c] for c in others],
        "delta_mean(best-other)": [means[best_cond] - means[c] for c in others],
    }
).sort_values("p_two_sided_holm")

print("\n=== Post-hoc (paired Wilcoxon) best vs others ===")
print(res.to_string(index=False))

# ============================
# 片側（best > other）も欲しい場合
# ============================
if DO_ONE_SIDED:
    pvals_one = []
    for c in others:
        a = wide[best_cond].to_numpy(dtype=float)
        b = wide[c].to_numpy(dtype=float)
        try:
            stat, p = wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        except ValueError:
            stat, p = np.nan, 1.0
        pvals_one.append(p)
    pvals_one_adj = holm_adjust(pvals_one)

    res_one = pd.DataFrame(
        {
            "compare": [f"{best_cond} > {c}" for c in others],
            "p_one_sided": pvals_one,
            "p_one_sided_holm": pvals_one_adj,
        }
    ).sort_values("p_one_sided_holm")

    print("\n=== Post-hoc (paired Wilcoxon, one-sided: best > other) ===")
    print(res_one.to_string(index=False))

# ============================
# 結果をCSV保存（修論用に貼りやすい）
# ============================
outdir = os.path.join(RUNS_ROOT, "_stats")
os.makedirs(outdir, exist_ok=True)
wide.to_csv(os.path.join(outdir, "dice_wide_table.csv"), index=False)
res.to_csv(
    os.path.join(outdir, "posthoc_best_vs_others_wilcoxon_holm.csv"), index=False
)
if DO_ONE_SIDED:
    res_one.to_csv(
        os.path.join(outdir, "posthoc_best_gt_others_wilcoxon_holm.csv"), index=False
    )

print(f"\nSaved tables to: {outdir}")
