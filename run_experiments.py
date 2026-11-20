"""Expériences du rapport : 2 bases, MDP à 54 états, gamma = 0.95, validation exacte.
Usage : python run_experiments.py   (écrit results/results.json et results/*.png)"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from credit_rl.data import synthetic_base
from credit_rl.mdp54 import (ACTION_NAMES, CreditMDP, Disc54, N_S, behavior_pi, dec, make_log,
                             offline_cql, online_q)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(100, 110))
N_STEPS = 100_000
CQL_GRID = [0.0, 0.5, 1.0, 2.0, 5.0]
N_LOG = 8000

BASES = {"Base A": dict(name="A", n=3000), "Base B": dict(name="B", n=6000)}


def ci95(x):
    x = np.asarray(x)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    res = {}
    for label, cfg in BASES.items():
        df = synthetic_base(cfg["name"], cfg["n"], seed=1)
        idx = np.random.default_rng(0).permutation(len(df))
        n_tr = int(0.7 * len(df))
        tr = df.iloc[idx[:n_tr]].reset_index(drop=True)
        disc = Disc54().fit(tr)
        m = CreditMDP(tr, disc)
        Qs = m.value_iteration()
        pi_star = Qs.argmax(1)
        Vs = Qs.max(1)
        Jstar = m.J(pi_star)
        base_pi, _ = behavior_pi()
        r = dict(
            n=len(df), n_train=n_tr, default_rate=float(df["default"].mean()),
            states_seen=int((m.n_train > 0).sum()),
            Jstar=Jstar, J_refuse=m.J(np.zeros(N_S, int)), J_accept=m.J(np.ones(N_S, int)),
            J_reduced=m.J(2 * np.ones(N_S, int)), J_bank=m.J(base_pi),
            pi_star_counts=np.bincount(pi_star, minlength=3).tolist(),
            threshold_pd=m.rate / (m.rate + m.lgd),
        )
        # exactitude de l'évaluation : comparer avec Monte-Carlo sur 1 politique
        rng = np.random.default_rng(7)
        Rm = []
        for _ in range(400):
            s = int(np.searchsorted(m._cmu, rng.random())); G = 0.0
            for t in range(300):
                u = rng.random(3)
                rr, s = m.step(s, int(pi_star[s]), *u)
                G += (m.gamma ** t) * rr
            Rm.append(G)
        r["J_star_montecarlo"] = float(np.mean(Rm))
        r["J_star_montecarlo_se"] = float(np.std(Rm, ddof=1) / np.sqrt(len(Rm)))

        # --- en ligne : Q-learning vs Double Q-learning
        curves = {"Q-Learning": [], "Double Q-Learning": []}
        final = {"Q-Learning": [], "Double Q-Learning": []}
        bias = {"Q-Learning": [], "Double Q-Learning": []}
        agree = {"Q-Learning": [], "Double Q-Learning": []}
        w = m.mu
        for sd in SEEDS:
            for name, dbl in (("Q-Learning", False), ("Double Q-Learning", True)):
                Q, cv = online_q(m, N_STEPS, alpha=0.05, seed=sd, double=dbl)
                pi = Q.argmax(1)
                final[name].append(m.J(pi))
                curves[name].append([c[1] for c in cv])
                bias[name].append(float(w @ (Q.max(1) - Vs)))
                agree[name].append(float(w @ (pi == pi_star)))
        r["steps"] = [c[0] for c in cv]
        for name in final:
            mean, h = ci95(final[name])
            r[name] = dict(J_mean=mean, J_ci=h, J_sd=float(np.std(final[name], ddof=1)),
                           pct_opt=100 * mean / Jstar,
                           bias_mean=float(np.mean(bias[name])), bias_sd=float(np.std(bias[name], ddof=1)),
                           agree=float(np.mean(agree[name])),
                           curve_mean=np.mean(curves[name], 0).tolist(),
                           curve_sd=np.std(curves[name], 0, ddof=1).tolist())
        diff = np.array(final["Q-Learning"]) - np.array(final["Double Q-Learning"])
        # test de permutation sur la différence de moyennes (échantillons indépendants)
        allv = np.array(final["Q-Learning"] + final["Double Q-Learning"])
        rngp = np.random.default_rng(0); obs = diff.mean(); cnt = 0
        for _ in range(20000):
            p = rngp.permutation(allv)
            cnt += abs(p[:len(SEEDS)].mean() - p[len(SEEDS):].mean()) >= abs(obs) - 1e-12
        r["QL_minus_DQL"] = dict(mean=float(obs), p_perm=cnt / 20000,
                                 ci=[float(np.percentile(
                                     [np.mean(rngp.choice(final["Q-Learning"], len(SEEDS))) -
                                      np.mean(rngp.choice(final["Double Q-Learning"], len(SEEDS)))
                                      for _ in range(5000)], q)) for q in (2.5, 97.5)])

        # --- hors ligne : Q-learning hors-ligne vs CQL (même journal par seed)
        off = {a: [] for a in CQL_GRID}
        coverage = []
        unseen_choice = {a: [] for a in CQL_GRID}
        for sd in range(200, 210):
            S, A, R, S2 = make_log(m, n=N_LOG, eps=0.10, seed=sd)
            cnt_sa = np.bincount(S * 3 + A, minlength=N_S * 3).reshape(N_S, 3)
            coverage.append(float((cnt_sa > 0).mean()))
            for a in CQL_GRID:
                Q = offline_cql(S, A, R, S2, cql_alpha=a)
                pi = Q.argmax(1)
                off[a].append(m.J(pi))
                unseen_choice[a].append(float(w @ (cnt_sa[np.arange(N_S), pi] == 0)))
        r["offline"] = {}
        base_vals = np.array(off[0.0])
        for a in CQL_GRID:
            v = np.array(off[a]); mean, h = ci95(v)
            d = v - base_vals
            dm, dh = ci95(d) if a > 0 else (0.0, 0.0)
            r["offline"][str(a)] = dict(J_mean=mean, J_ci=h, pct_opt=100 * mean / Jstar,
                                        gain=dm, gain_ci=dh, wins=int((d > 0).sum()),
                                        unseen=float(np.mean(unseen_choice[a])))
        r["offline_coverage_sa"] = float(np.mean(coverage))
        res[label] = r
        print(label, "J*", round(Jstar, 3), "QL", round(r["Q-Learning"]["J_mean"], 3),
              "DQL", round(r["Double Q-Learning"]["J_mean"], 3),
              {k: round(v["J_mean"], 2) for k, v in r["offline"].items()}, flush=True)

        # --- figures (noir et gris, simples)
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        st = np.array(r["steps"])
        for name, ls in (("Q-Learning", "-"), ("Double Q-Learning", "--")):
            mu_, sd_ = np.array(r[name]["curve_mean"]), np.array(r[name]["curve_sd"])
            ax.plot(st, mu_, ls, color="black", label=name)
            ax.fill_between(st, mu_ - sd_, mu_ + sd_, color="gray", alpha=0.18)
        ax.axhline(Jstar, color="black", ls=":", label="J* (value iteration)")
        ax.set_xlabel("pas d'interaction"); ax.set_ylabel("J(π glouton), évaluation exacte")
        ax.set_title(f"{label} : convergence (10 seeds, moyenne ± écart-type)", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"conv_{cfg['name']}.png"), dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        xs = np.arange(len(CQL_GRID))
        ms = [r["offline"][str(a)]["J_mean"] for a in CQL_GRID]
        hs = [r["offline"][str(a)]["J_ci"] for a in CQL_GRID]
        ax.bar(xs, ms, yerr=hs, color="0.55", edgecolor="black", capsize=3)
        ax.axhline(Jstar, color="black", ls=":", label="J*")
        ax.axhline(r["J_bank"], color="black", ls="--", label="règles de la banque")
        ax.set_xticks(xs); ax.set_xticklabels(["0 (Q-L hors ligne)"] + [str(a) for a in CQL_GRID[1:]], fontsize=8)
        ax.set_xlabel("α_cql"); ax.set_ylabel("J(π), évaluation exacte")
        ax.set_title(f"{label} : apprentissage hors ligne ({N_LOG} transitions)", fontsize=10)
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"offline_{cfg['name']}.png"), dpi=200); plt.close(fig)

        # politique optimale par état (heatmap simple en texte -> table)
        pol = {}
        for risk in range(3):
            for hist in range(2):
                for debt in range(3):
                    row = []
                    for inc in range(3):
                        s = risk * 18 + inc * 6 + debt * 2 + hist
                        row.append(int(pi_star[s]))
                    pol[f"{risk}-{hist}-{debt}"] = row
        r["pi_star_by_profile"] = pol
        r["d_by_risk"] = [float(np.average(m.d[[s for s in range(N_S) if dec(s)[0] == k]],
                                           weights=m.mu[[s for s in range(N_S) if dec(s)[0] == k]])) for k in range(3)]

    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
