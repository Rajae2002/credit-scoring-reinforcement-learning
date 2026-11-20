"""Pipeline complet : split, discrétisation, journal de la banque, entraînement, évaluation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from .agents import greedy, train_cql, train_double_q, train_q_learning
from .data import CREDIT_TYPES, Discretizer
from .mdp54 import Disc54
from .env import ACTIONS, behavior_policy, break_even_pd, reward_matrix, rule_policy


def sup_features(df):
    X = pd.DataFrame({
        "age": df["age"], "salary": df["salary"], "late": df["late_payments"],
        "burden": df["amount"] / df["salary"], "amount": df["amount"],
    })
    for c in CREDIT_TYPES:
        X["type_" + c] = (df["credit_type"] == c).astype(int)
    return X


def evaluate(actions, R):
    """Métriques d'une politique sur le test (récompense connue pour les 3 actions)."""
    n = len(R)
    r = R[np.arange(n), actions]
    return {
        "Profit moyen / client (k MAD)": r.mean(),
        "Profit total (k MAD)": r.sum(),
        "% acceptés": 100 * np.mean(actions == 1),
        "% montant réduit": 100 * np.mean(actions == 2),
        "% refusés": 100 * np.mean(actions == 0),
    }


def run(df, rate=0.20, lgd=0.60, cut=0.50, rescue_p=0.50,
        epochs=30, alpha=0.05, cql_alpha=1.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_tr = int(0.7 * len(df))
    tr, te = df.iloc[idx[:n_tr]].reset_index(drop=True), df.iloc[idx[n_tr:]].reset_index(drop=True)

    disc = Discretizer().fit(tr)
    S_tr, S_te = disc.transform(tr), disc.transform(te)
    R_tr = reward_matrix(tr, rate, lgd, cut, rescue_p, seed)
    R_te = reward_matrix(te, rate, lgd, cut, rescue_p, seed + 1)

    # journal historique de la banque : on ne voit que la récompense de l'action choisie
    A_log = behavior_policy(tr, seed=seed)
    r_log = R_tr[np.arange(len(tr)), A_log]

    # --- apprentissage + courbes d'apprentissage sur le test
    curves = {"Q-learning": [], "Double Q-learning": [], "CQL": []}

    def cb(name):
        def f(ep, Q):
            a = greedy(Q, S_te)
            curves[name].append(R_te[np.arange(len(te)), a].mean())
        return f

    Q = {
        "Q-learning": train_q_learning(S_tr, A_log, r_log, epochs, alpha, seed=seed, on_epoch=cb("Q-learning")),
        "Double Q-learning": train_double_q(S_tr, A_log, r_log, epochs, alpha, seed=seed, on_epoch=cb("Double Q-learning")),
        "CQL": train_cql(S_tr, A_log, r_log, epochs, alpha, cql_alpha, seed=seed, on_epoch=cb("CQL")),
    }

    # --- modèles supervisés (ce que font la plupart des notebooks Kaggle)
    Xtr, Xte = sup_features(tr), sup_features(te)
    lr = LogisticRegression(max_iter=2000).fit((Xtr - Xtr.mean()) / Xtr.std(), tr["default"])
    mu, sd = Xtr.mean(), Xtr.std()
    pd_lr = lr.predict_proba((Xte - mu) / sd)[:, 1]
    rf = RandomForestClassifier(200, min_samples_leaf=5, random_state=seed).fit(Xtr, tr["default"])
    pd_rf = rf.predict_proba(Xte)[:, 1]
    thr = break_even_pd(rate, lgd)
    sup = pd.DataFrame({
        "Modèle": ["Régression logistique", "Random Forest"],
        "AUC": [roc_auc_score(te["default"], pd_lr), roc_auc_score(te["default"], pd_rf)],
        "Accuracy (seuil 0.5)": [accuracy_score(te["default"], pd_lr > 0.5),
                                 accuracy_score(te["default"], pd_rf > 0.5)],
    })

    # --- comparaison des politiques sur le test
    pols = {
        "Tout accepter": np.ones(len(te), int),
        "Règles de la banque": rule_policy(te),
        "Scoring logistique (seuil rentabilité)": np.where(pd_lr < thr, 1, 0),
        "Q-learning": greedy(Q["Q-learning"], S_te),
        "Double Q-learning": greedy(Q["Double Q-learning"], S_te),
        "CQL": greedy(Q["CQL"], S_te),
        "Oracle (borne max)": R_te.argmax(axis=1),
    }
    table = pd.DataFrame({k: evaluate(v, R_te) for k, v in pols.items()}).T
    # défaut parmi les clients acceptés (montant plein ou réduit)
    dr = {}
    for k, a in pols.items():
        m = a > 0
        dr[k] = 100 * te["default"].values[m].mean() if m.any() else np.nan
    table["% défauts parmi acceptés"] = pd.Series(dr)

    d54 = Disc54().fit(tr)
    return dict(d54=d54, train=tr, test=te, disc=disc, Q=Q, curves=curves, table=table, sup=sup,
                lr=lr, lr_mu=mu, lr_sd=sd, rf=rf, A_log=A_log, S_tr=S_tr, r_log=r_log,
                params=dict(rate=rate, lgd=lgd, cut=cut, rescue_p=rescue_p,
                            epochs=epochs, alpha=alpha, cql_alpha=cql_alpha))


def decide_client(res, age, salary, ctype, late, amount):
    """Décision des 3 agents pour un client saisi dans l'interface."""
    import pandas as pd
    row = pd.DataFrame([dict(age=age, salary=salary, credit_type=ctype,
                             late_payments=late, amount=amount, default=0)])
    s = int(res["disc"].transform(row)[0])
    out = {k: (int(Q[s].argmax()), Q[s].copy()) for k, Q in res["Q"].items()}
    X = sup_features(row)
    p = float(res["lr"].predict_proba((X - res["lr_mu"]) / res["lr_sd"])[:, 1][0])
    lvl, pd54 = res["d54"].risk_level(row)
    risk = dict(level=int(lvl[0]), label=Disc54.RISK_LABELS[int(lvl[0])], pd=float(pd54[0]))
    return s, out, p, risk
