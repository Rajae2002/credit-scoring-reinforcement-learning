"""MDP à 54 états, 3 actions, gamma = 0.95 : modèle exact, simulateur et agents tabulaires.

État  s = (risque 3) x (revenu 3) x (endettement 3) x (historique 2) = 54
        s = risque*18 + revenu*6 + endettement*2 + historique     (historique 0 = bon, 1 = mauvais)
Action 0 = refuser, 1 = accepter, 2 = accepter avec montant réduit.

Dynamique (hypothèse de modélisation, documentée dans le rapport) :
  - refuser            : le client part, l'état suivant est le prochain demandeur (loi mu)
  - accepter / réduire : avec proba RHO le même client revient, avec un historique et un risque
                         mis à jour selon l'issue (remboursé / défaut) ; sinon prochain demandeur ~ mu
Récompenses (k MAD) : profit = +r*M si remboursé, -L*M si défaut (M réduit si action 2).
Les statistiques par état (taux de défaut, montant moyen, mu) sont estimées sur la base d'entraînement.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from .data import CREDIT_TYPES

N_S, N_A = 54, 3
GAMMA = 0.95
ACTION_NAMES = ["Refuser", "Accepter", "Accepter (réduit)"]


def enc(risk, inc, debt, hist):
    return risk * 18 + inc * 6 + debt * 2 + hist


def dec(s):
    return s // 18, (s // 6) % 3, (s // 2) % 3, s % 2


# ---------------------------------------------------------------- discrétisation en 54 états
class Disc54:
    def fit(self, df):
        X = self._x(df)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(max_iter=2000).fit((X - self.mu_) / self.sd_, df["default"])
        pdv = self.lr.predict_proba((X - self.mu_) / self.sd_)[:, 1]
        self.risk_edges = np.quantile(pdv, [1 / 3, 2 / 3])
        self.inc_edges = np.quantile(df["salary"], [1 / 3, 2 / 3])
        self.debt_edges = np.quantile(df["amount"] / df["salary"], [1 / 3, 2 / 3])
        return self

    @staticmethod
    def _x(df):
        # score de risque a priori : âge, type de crédit, montant/salaire, (sans l'historique, qui est une dimension à part)
        cols = [df["age"].values, (df["amount"] / df["salary"]).values]
        cols += [(df["credit_type"] == c).values.astype(float) for c in CREDIT_TYPES[1:]]
        return np.column_stack(cols).astype(float)

    RISK_LABELS = ["Faible", "Moyen", "Élevé"]

    def risk_level(self, df):
        """Niveau de risque (0 faible, 1 moyen, 2 élevé) et probabilité de défaut estimée."""
        X = (self._x(df) - self.mu_) / self.sd_
        pdv = self.lr.predict_proba(X)[:, 1]
        return np.digitize(pdv, self.risk_edges), pdv

    def transform(self, df):
        X = (self._x(df) - self.mu_) / self.sd_
        pdv = self.lr.predict_proba(X)[:, 1]
        risk = np.digitize(pdv, self.risk_edges)
        inc = np.digitize(df["salary"].values, self.inc_edges)
        debt = np.digitize((df["amount"] / df["salary"]).values, self.debt_edges)
        hist = (df["late_payments"].values >= 1).astype(int)
        return enc(risk, inc, debt, hist)


# ---------------------------------------------------------------- modèle exact
class CreditMDP:
    def __init__(self, df, disc, rate=0.20, lgd=0.60, cut=0.50, rescue=0.50, rho=0.5, prior=10):
        s = disc.transform(df)
        n = np.bincount(s, minlength=N_S).astype(float)
        d0 = df["default"].mean()
        dsum = np.bincount(s, weights=df["default"].values, minlength=N_S)
        self.d = (dsum + prior * d0) / (n + prior)                       # taux de défaut lissé
        msum = np.bincount(s, weights=df["amount"].values / 1000, minlength=N_S)
        self.M = np.where(n > 0, msum / np.maximum(n, 1), (df["amount"] / 1000).mean())
        self.mu = (n + 1e-3) / (n + 1e-3).sum()                           # loi du prochain demandeur
        self.n_train = n
        self.rate, self.lgd, self.cut, self.rescue, self.rho = rate, lgd, cut, rescue, rho
        self.gamma = GAMMA

        # défaut effectif et montant effectif par (s, a)
        self.d_eff = np.zeros((N_S, N_A))
        self.M_eff = np.zeros((N_S, N_A))
        for st in range(N_S):
            risk, inc, debt, hist = dec(st)
            self.d_eff[st, 1] = self.d[st]
            self.d_eff[st, 2] = self.d[st] * (1 - rescue * (debt == 2))
            self.M_eff[st, 1] = self.M[st]
            self.M_eff[st, 2] = cut * self.M[st]
        # récompense espérée
        self.R = np.zeros((N_S, N_A))
        self.R[:, 1:] = self.M_eff[:, 1:] * ((1 - self.d_eff[:, 1:]) * rate - self.d_eff[:, 1:] * lgd)
        # états suivants si le même client revient
        self.nxt_ok = np.zeros(N_S, int)
        self.nxt_bad = np.zeros(N_S, int)
        for st in range(N_S):
            risk, inc, debt, hist = dec(st)
            self.nxt_ok[st] = enc(max(risk - 1, 0), inc, debt, 0)
            self.nxt_bad[st] = enc(min(risk + 1, 2), inc, debt, 1)
        # matrice de transition exacte T[s,a,s']
        T = np.zeros((N_S, N_A, N_S))
        T[:, 0, :] = self.mu
        for a in (1, 2):
            for st in range(N_S):
                T[st, a, :] = (1 - rho) * self.mu
                T[st, a, self.nxt_ok[st]] += rho * (1 - self.d_eff[st, a])
                T[st, a, self.nxt_bad[st]] += rho * self.d_eff[st, a]
        self.T = T
        assert np.allclose(T.sum(2), 1)

    # ---- résolution exacte
    def value_iteration(self, tol=1e-12):
        Q = np.zeros((N_S, N_A))
        while True:
            V = Q.max(1)
            Qn = self.R + self.gamma * self.T @ V
            if np.abs(Qn - Q).max() < tol:
                return Qn
            Q = Qn

    def evaluate(self, pi):
        """V^pi = (I - gamma T_pi)^-1 R_pi (évaluation exacte) ; pi : vecteur d'actions."""
        idx = np.arange(N_S)
        Tpi, Rpi = self.T[idx, pi, :], self.R[idx, pi]
        return np.linalg.solve(np.eye(N_S) - self.gamma * Tpi, Rpi)

    def J(self, pi):
        return float(self.mu @ self.evaluate(pi))

    # ---- simulateur (échantillonne exactement la dynamique de T et de R)
    def step(self, s, a, u1, u2, u3):
        if a == 0:
            return 0.0, int(np.searchsorted(self._cmu, u3))
        d, M = self.d_eff[s, a], self.M_eff[s, a]
        default = u1 < d
        r = -self.lgd * M if default else self.rate * M
        if u2 < self.rho:
            s2 = self.nxt_bad[s] if default else self.nxt_ok[s]
        else:
            s2 = int(np.searchsorted(self._cmu, u3))
        return r, s2

    @property
    def _cmu(self):
        if not hasattr(self, "_c"):
            self._c = np.cumsum(self.mu)
            self._c[-1] = 1.0
        return self._c


# ---------------------------------------------------------------- agents en ligne (ε-greedy)
def _eps(t, n, e0=1.0, e1=0.05, frac=0.5):
    return max(e1, e0 - (e0 - e1) * t / (frac * n))


def online_q(mdp, n_steps=100_000, alpha=0.05, seed=0, checkpoints=20, double=False):
    rng = np.random.default_rng(seed)
    U = rng.random((n_steps, 4))
    Q1 = np.zeros((N_S, N_A))
    Q2 = np.zeros((N_S, N_A)) if double else None
    g = mdp.gamma
    s = int(np.searchsorted(mdp._cmu, rng.random()))
    curve = []
    every = n_steps // checkpoints
    for t in range(n_steps):
        u = U[t]
        Qa = (Q1 + Q2) / 2 if double else Q1
        a = int(rng.integers(N_A)) if u[3] < _eps(t, n_steps) else int(Qa[s].argmax())
        r, s2 = mdp.step(s, a, u[0], u[1], u[2])
        if double:
            if rng.random() < 0.5:
                astar = Q1[s2].argmax()
                Q1[s, a] += alpha * (r + g * Q2[s2, astar] - Q1[s, a])
            else:
                astar = Q2[s2].argmax()
                Q2[s, a] += alpha * (r + g * Q1[s2, astar] - Q2[s, a])
        else:
            Q1[s, a] += alpha * (r + g * Q1[s2].max() - Q1[s, a])
        s = s2
        if (t + 1) % every == 0:
            Qa = (Q1 + Q2) / 2 if double else Q1
            curve.append((t + 1, mdp.J(Qa.argmax(1))))
    Qf = (Q1 + Q2) / 2 if double else Q1
    return Qf, curve


# ---------------------------------------------------------------- hors-ligne : journal + CQL
def behavior_pi(eps=0.10):
    """Politique historique de la banque : règles par état + bruit eps."""
    base = np.ones(N_S, int)
    for st in range(N_S):
        risk, inc, debt, hist = dec(st)
        if risk == 2 and hist == 1:
            base[st] = 0
        elif debt == 2:
            base[st] = 2
    return base, eps


def make_log(mdp, n=8000, eps=0.10, seed=0):
    rng = np.random.default_rng(seed)
    base, _ = behavior_pi(eps)
    U = rng.random((n, 4))
    s = int(np.searchsorted(mdp._cmu, rng.random()))
    S, A, R, S2 = np.zeros(n, int), np.zeros(n, int), np.zeros(n), np.zeros(n, int)
    for t in range(n):
        a = int(rng.integers(N_A)) if U[t, 3] < eps else int(base[s])
        r, s2 = mdp.step(s, a, U[t, 0], U[t, 1], U[t, 2])
        S[t], A[t], R[t], S2[t] = s, a, r, s2
        s = s2
    return S, A, R, S2


def offline_cql(S, A, R, S2, gamma=GAMMA, cql_alpha=0.0, iters=400, lr=0.5, eta=0.05):
    """Version tabulaire par lots. cql_alpha = 0 -> Q-learning hors-ligne (baseline)."""
    Q = np.zeros((N_S, N_A))
    cnt = np.bincount(S * N_A + A, minlength=N_S * N_A).reshape(N_S, N_A).astype(float)
    n_s = cnt.sum(1)
    seen = cnt > 0
    pb = np.where(n_s[:, None] > 0, cnt / np.maximum(n_s[:, None], 1), 1 / N_A)   # loi empirique de la banque
    for _ in range(iters):
        y = R + gamma * Q[S2].max(1)
        ybar = np.bincount(S * N_A + A, weights=y, minlength=N_S * N_A).reshape(N_S, N_A) / np.maximum(cnt, 1)
        Q = np.where(seen, Q + lr * (ybar - Q), Q)                           # partie Bellman (paires observées)
        if cql_alpha > 0:
            e = np.exp(Q - Q.max(1, keepdims=True))
            soft = e / e.sum(1, keepdims=True)
            Q = Q - eta * cql_alpha * np.where(n_s[:, None] > 0, soft - pb, 0)   # pénalité conservatrice
    return Q
