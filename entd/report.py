"""Tables and plots from work_dir/results/*.parquet -> work_dir/results/report/."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import summarize

COLORS = {"heldout": "#1f77b4", "train": "#d62728", "null": "#7f7f7f"}


def _p(rd, name):
    return os.path.join(rd, name)


def _save(fig, rd, name):
    fig.tight_layout()
    fig.savefig(_p(rd, name), dpi=130)
    plt.close(fig)


def plot_vs_k(summ, metric, ax, variant=None, label_prefix="", linestyle="-", color_by_set=True):
    s = summ[summ.metric == metric]
    if variant is not None:
        s = s[s.variant == variant]
    for set_name, g in s.groupby("set"):
        g = g.sort_values("k")
        ax.plot(g.k, g["mean"], linestyle, marker="o", ms=3, color=COLORS.get(set_name), label=f"{label_prefix}{set_name}")
        ax.fill_between(g.k, g.lo, g.hi, color=COLORS.get(set_name), alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("k")


def main(cfg):
    rd = os.path.join(cfg["work_dir"], "results")
    out = os.path.join(rd, "report")
    os.makedirs(out, exist_ok=True)
    lines = [f"# Results: {cfg['name']}\n"]

    # ---------------- projected-space metrics
    pm = pd.read_parquet(_p(rd, "proj_metrics.parquet"))
    lam_cols = [c for c in pm.columns if c.startswith("nll_lam_")]
    val_cols = ["js_total", "tv_total", "top1_agree", "null_js_mean", "null_js_pct", "nll_model", "nll_e_smoothed",
                "n_distinct_w", "n_leak", "d2_k"] + lam_cols
    s_full = summarize(pm[pm.variant == "full"], val_cols, ["set", "k"])
    s_full["variant"] = "full"
    s_full.to_csv(_p(out, "summary_proj_full.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    plot_vs_k(s_full, "js_total", axes[0])
    plot_vs_k(s_full, "null_js_mean", axes[0], label_prefix="naive null: ", linestyle="--")
    axes[0].set_ylabel("JS(e_k, p_q)")
    axes[0].set_title("Empirical neighbor histogram vs model prediction")
    axes[0].legend(fontsize=8)
    plot_vs_k(s_full, "top1_agree", axes[1])
    axes[1].set_ylabel("P(argmax e_k == argmax p_q)")
    axes[1].set_title("Top-1 agreement")
    plot_vs_k(s_full, "d2_k", axes[2])
    axes[2].set_ylabel("||s_q - s_k||^2 (projected)")
    axes[2].set_yscale("log")
    axes[2].set_title("Distance of the k-th neighbor")
    _save(fig, out, "divergence_vs_k.png")

    # memorization: model NLL vs neighbor NLL, train vs held-out
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    plot_vs_k(s_full, "nll_e_smoothed", axes[0])
    for set_name, g in s_full[s_full.metric == "nll_model"].groupby("set"):
        axes[0].axhline(g["mean"].iloc[0], color=COLORS[set_name], linestyle=":", label=f"model NLL ({set_name})")
    axes[0].set_ylabel("-log e_k(y_q) (smoothed)")
    axes[0].set_title("NLL of the true next token under the neighbor histogram")
    axes[0].legend(fontsize=8)
    # lambda curve on held-out, best k
    lam = sorted(float(c.split("_")[-1]) for c in lam_cols)
    for set_name in ("heldout", "train"):
        g = s_full[(s_full.set == set_name) & (s_full.metric.isin(lam_cols))]
        for k in sorted(g.k.unique()):
            if k not in (8, 64, 512, max(g.k.unique())):
                continue
            gk = g[g.k == k].copy()
            gk["lam"] = gk.metric.str.split("_").str[-1].astype(float)
            gk = gk.sort_values("lam")
            axes[1].plot(gk.lam, gk["mean"], marker="o", ms=3, color=COLORS[set_name], alpha=0.4 + 0.6 * (k == max(g.k.unique())),
                         label=f"{set_name} k={k}")
    axes[1].set_xlabel("lambda (weight on neighbor histogram)")
    axes[1].set_ylabel("NLL of true next token under mixture")
    axes[1].set_title("kNN-LM style interpolation")
    axes[1].legend(fontsize=7)
    _save(fig, out, "memorization_lambda.png")

    # coverage / untrained-store
    s_var = summarize(pm, ["js_total", "top1_agree", "d2_k"], ["set", "variant", "k"])
    s_var.to_csv(_p(out, "summary_proj_variants.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, set_name in zip(axes, ("heldout", "train")):
        for variant, g in s_var[(s_var.metric == "js_total") & (s_var.set == set_name)].groupby("variant"):
            g = g.sort_values("k")
            ax.plot(g.k, g["mean"], marker="o", ms=3, label=variant)
            ax.fill_between(g.k, g.lo, g.hi, alpha=0.12)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k")
        ax.set_ylabel("JS(e_k, p_q)")
        ax.set_title(f"{set_name} queries: store variants (coverage / untrained)")
        ax.legend(fontsize=8)
    _save(fig, out, "coverage.png")

    # strata (held-out, full store): by offset bin and entropy tercile
    q = pd.read_parquet(_p(rd, "queries.parquet"))
    pmq = pm[pm.variant == "full"].merge(q[["qid", "off_bin", "is_eot", "last_seen", "dup_train"]], on="qid")
    pmq["H_terc"] = pd.qcut(pmq.H_q, 3, labels=False, duplicates="drop")
    s_off = summarize(pmq, ["js_total", "null_js_mean"], ["set", "off_bin", "k"])
    s_H = summarize(pmq, ["js_total", "null_js_mean"], ["set", "H_terc", "k"])
    s_off.to_csv(_p(out, "summary_by_offset.csv"), index=False)
    s_H.to_csv(_p(out, "summary_by_entropy.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (sname, col, lab), ax in zip([(s_off, "off_bin", "context-length bin"), (s_H, "H_terc", "entropy tercile")], axes):
        g0 = sname[(sname.metric == "js_total") & (sname.set == "heldout")]
        for b, g in g0.groupby(col):
            g = g.sort_values("k")
            ax.plot(g.k, g["mean"], marker="o", ms=3, label=f"{lab} {b}")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k")
        ax.set_ylabel("JS(e_k, p_q), held-out")
        ax.legend(fontsize=8)
    _save(fig, out, "strata.png")
    if (pmq.set == "train").any() and (pmq.last_seen >= 0).any():
        tr = pmq[pmq.set == "train"].copy()
        tr["seen_bin"] = pd.qcut(tr.last_seen, 4, labels=False, duplicates="drop")
        s_seen = summarize(tr, ["nll_model", "nll_e_smoothed", "js_total"], ["seen_bin", "k"])
        s_seen.to_csv(_p(out, "summary_by_step_seen.csv"), index=False)

    # ---------------- exact re-rank: decomposition and T statistic
    if os.path.exists(_p(rd, "rerank_metrics.parquet")):
        rm = pd.read_parquet(_p(rd, "rerank_metrics.parquet"))
        rm["T_z"] = rm["T"] / rm["T_se"].replace(0, np.nan)
        cols = ["js_total", "js_geom", "js_calib", "null_js_total_mean", "null_js_calib_mean", "null_js_calib_pct",
                "T", "T_z", "top1_agree", "d2_exact_k", "d2_proj_k", "n_leak", "n_distinct_w"]
        s_rr = summarize(rm, [c for c in cols if c in rm], ["set", "k"])
        s_rr.to_csv(_p(out, "summary_rerank.csv"), index=False)
        # global z per (set, k): sum T / sqrt(sum var)
        gz = rm.groupby(["set", "k"]).apply(lambda g: pd.Series({"global_z": g["T"].sum() / np.sqrt((g["T_se"] ** 2).sum()),
                                                                   "mean_T": g["T"].mean(), "n": len(g)})).reset_index()
        gz.to_csv(_p(out, "global_z.csv"), index=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for set_name in ("heldout", "train"):
            g = s_rr[s_rr.set == set_name]
            for metric, ls, lab in [("js_total", "-", "total d(e,p_q)"), ("js_geom", "--", "geometry d(pbar,p_q)"),
                                    ("js_calib", "-.", "calibration d(e,pbar)"), ("null_js_calib_mean", ":", "null calib")]:
                gg = g[g.metric == metric].sort_values("k")
                axes[0].plot(gg.k, gg["mean"], ls, marker="o", ms=3, color=COLORS[set_name], label=f"{set_name}: {lab}")
        axes[0].set_xscale("log", base=2)
        axes[0].set_ylabel("JS")
        axes[0].set_title("Exact re-rank: decomposition")
        axes[0].legend(fontsize=7)
        for set_name, g in gz.groupby("set"):
            g = g.sort_values("k")
            axes[1].plot(g.k, g.global_z, marker="o", ms=3, color=COLORS[set_name], label=set_name)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_xscale("log", base=2)
        axes[1].set_ylabel("global z of T (log-score calibration test)")
        axes[1].set_title("T_k = mean[-log p_q(y_i)] - mean[H(p_i)+KL(p_i||p_q)]")
        axes[1].legend(fontsize=8)
        for set_name in ("heldout", "train"):
            g = s_rr[(s_rr.set == set_name)]
            for metric, ls in [("null_js_calib_pct", "-")]:
                gg = g[g.metric == metric].sort_values("k")
                axes[2].plot(gg.k, gg["mean"], ls, marker="o", ms=3, color=COLORS[set_name], label=set_name)
        axes[2].axhline(0.5, color="k", lw=0.5)
        axes[2].set_xscale("log", base=2)
        axes[2].set_ylabel("percentile of observed d(e,pbar) in null")
        axes[2].set_title("Consistency resampling (y_i* ~ p_i)")
        axes[2].legend(fontsize=8)
        _save(fig, out, "rerank_decomposition.png")
        # projection fidelity on real queries: exact vs projected rank distances
        rc = pd.read_parquet(_p(rd, "rerank_candidates.parquet"))
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sub = rc.sample(min(20000, len(rc)), random_state=0)
        ax.scatter(sub.d2_exact, sub.d2_proj, s=2, alpha=0.3)
        lim = [0, float(max(sub.d2_exact.max(), sub.d2_proj.max()))]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel("exact ||s_q - s_i||^2")
        ax.set_ylabel("projected")
        ax.set_title("Candidate distances: exact vs projected")
        _save(fig, out, "exact_vs_projected.png")

    # ---------------- radius
    if os.path.exists(_p(rd, "radius_metrics.parquet")):
        rad = pd.read_parquet(_p(rd, "radius_metrics.parquet"))
        s_rad = summarize(rad, ["js_total", "top1_agree", "count_full", "count_sub"], ["set", "radius"])
        s_rad.to_csv(_p(out, "summary_radius.csv"), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for set_name, g in s_rad[s_rad.metric == "js_total"].groupby("set"):
            g = g.sort_values("radius")
            axes[0].plot(g.radius, g["mean"], marker="o", color=COLORS[set_name], label=set_name)
            axes[0].fill_between(g.radius, g.lo, g.hi, color=COLORS[set_name], alpha=0.15)
        axes[0].set_xlabel("radius on ||sqrt p - sqrt q||")
        axes[0].set_ylabel("JS(e_r, p_q)  (sub-store histogram)")
        axes[0].legend(fontsize=8)
        for set_name, g in rad.groupby("set"):
            med = g.groupby("radius").count_full.median()
            axes[1].plot(med.index, med.values + 1, marker="o", color=COLORS[set_name], label=f"{set_name} (median)")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("radius")
        axes[1].set_ylabel("# positions within radius (full store) + 1")
        axes[1].legend(fontsize=8)
        _save(fig, out, "radius.png")

    # ---------------- baselines
    if os.path.exists(_p(rd, "baselines_q.parquet")):
        bq = pd.read_parquet(_p(rd, "baselines_q.parquet"))
        bk = pd.read_parquet(_p(rd, "baselines_knn.parquet"))
        ng_cols = [c for c in bq.columns if c.startswith("ngram")]
        ng_js = [c for c in ng_cols if c.endswith("_js")]
        s_bq = summarize(bq, ["unigram_js", "bucketA_js", "bucketAE_js"] + ng_cols, ["set"])
        s_bk = summarize(bk, ["hknn_js", "dknn_sub_js", "hknn_top1", "dknn_sub_top1"], ["set", "k"])
        s_bq.to_csv(_p(out, "summary_baselines_q.csv"), index=False)
        s_bk.to_csv(_p(out, "summary_baselines_knn.csv"), index=False)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, set_name in zip(axes, ("heldout", "train")):
            g = s_full[(s_full.set == set_name) & (s_full.metric == "js_total")].sort_values("k")
            ax.plot(g.k, g["mean"], marker="o", ms=3, color=COLORS[set_name], label="distribution kNN (full store)")
            for metric, lab in [("hknn_js", "hidden-state kNN (h-subset)"), ("dknn_sub_js", "distribution kNN (h-subset)")]:
                gg = s_bk[(s_bk.set == set_name) & (s_bk.metric == metric)].sort_values("k")
                ax.plot(gg.k, gg["mean"], marker="s", ms=3, label=lab)
            for metric, lab in [("unigram_js", "unigram"), ("bucketA_js", "same-argmax bucket"),
                                ("bucketAE_js", "argmax x entropy bucket")] + [(c, f"longest-suffix n-gram (>= {c[5:-3]} matches)") for c in ng_js]:
                v = s_bq[(s_bq.set == set_name) & (s_bq.metric == metric)]["mean"]
                if len(v):
                    ax.axhline(float(v.iloc[0]), linestyle="--", lw=0.9, label=lab)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k")
            ax.set_ylabel("JS vs p_q")
            ax.set_title(f"{set_name}: baselines")
            ax.legend(fontsize=7)
        _save(fig, out, "baselines.png")

    # ---------------- fidelity + anchors into the summary
    if os.path.exists(_p(rd, "fidelity.json")):
        fid = json.load(open(_p(rd, "fidelity.json")))
        lines.append("## Projection fidelity (recall of exact Hellinger kNN)\n")
        lines.append("| code | " + " | ".join(f"r@{k}" for k in fid["ks"]) + " | r@10 high-H |")
        lines.append("|---|" + "---|" * (len(fid["ks"]) + 1))
        for name, r in fid["recall"].items():
            k10 = str(10) if "10" in r else str(fid["ks"][-1])
            lines.append(f"| {name} | " + " | ".join(f"{r[str(k)]['all']:.3f}" for k in fid["ks"]) + f" | {r[k10].get('H_terc2', float('nan')):.3f} |")
        if "pca_energy_cum" in fid:
            lines.append("\nPCA captured energy: " + ", ".join(f"d={d}: {v:.4f}" for d, v in fid["pca_energy_cum"].items()))
        if "h_decode_max_abs_err" in fid:
            lines.append(f"\nFull-rank h decode max |err| on sqrt p: {fid['h_decode_max_abs_err']:.2e}")
        lines.append("")
    if os.path.exists(_p(rd, "anchors.json")):
        an = json.load(open(_p(rd, "anchors.json")))
        lines.append("## Global anchors (all positions of sampled windows)\n")
        lines.append("| set | mean NLL | mean entropy | top-1 acc | top-1 ECE | n |")
        lines.append("|---|---|---|---|---|---|")
        for s, v in an.items():
            lines.append(f"| {s} | {v['mean_nll']:.4f} | {v['mean_entropy']:.4f} | {v['top1_acc']:.4f} | {v['top1_ece']:.4f} | {v['n_positions']} |")
        lines.append("")

    # key table
    lines.append("## Projected-space kNN (full store): JS(e_k, p_q) with naive null, by set and k\n")
    lines.append("| set | k | JS | [CI] | naive null JS | top-1 agree | NLL model | NLL neighbors | distinct windows | leaks |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    piv = s_full.pivot_table(index=["set", "k"], columns="metric", values="mean")
    pl = s_full[s_full.metric == "js_total"].set_index(["set", "k"])
    for (set_name, k), r in piv.iterrows():
        lo, hi = pl.loc[(set_name, k), ["lo", "hi"]]
        lines.append(f"| {set_name} | {k} | {r.js_total:.4f} | [{lo:.4f}, {hi:.4f}] | {r.get('null_js_mean', float('nan')):.4f} | "
                     f"{r.top1_agree:.3f} | {r.nll_model:.3f} | {r.nll_e_smoothed:.3f} | {r.n_distinct_w:.1f} | {r.n_leak:.2f} |")
    if os.path.exists(_p(out, "global_z.csv")):
        gz = pd.read_csv(_p(out, "global_z.csv"))
        lines.append("\n## Exact re-rank: global z of the log-score statistic T (0 = locally calibrated)\n")
        lines.append("| set | k | mean T | global z | n |")
        lines.append("|---|---|---|---|---|")
        for _, r in gz.iterrows():
            lines.append(f"| {r.set} | {int(r.k)} | {r.mean_T:+.4f} | {r.global_z:+.2f} | {int(r.n)} |")
    if os.path.exists(_p(out, "summary_baselines_q.csv")):
        lines.append("\n## Baselines (JS vs p_q)\n")
        lines.append("| set | unigram | same-argmax | argmax x entropy | " + " | ".join(f"n-gram >={c[5:-3]} (JS, mean n, mean count)" for c in ng_js) + " |")
        lines.append("|---|---|---|---|" + "---|" * len(ng_js))
        for set_name, g in s_bq.groupby("set"):
            v = g.set_index("metric")["mean"]
            ng = " | ".join(f"{v.get(c, np.nan):.4f}, {v.get(c[:-3] + '_n', np.nan):.1f}, {v.get(c[:-3] + '_count', np.nan):.0f}" for c in ng_js)
            lines.append(f"| {set_name} | {v.get('unigram_js', np.nan):.4f} | {v.get('bucketA_js', np.nan):.4f} | "
                         f"{v.get('bucketAE_js', np.nan):.4f} | {ng} |")
    with open(_p(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {out}/summary.md and plots", flush=True)


if __name__ == "__main__":
    import sys
    from .config import load_config
    main(load_config(sys.argv[1]))
