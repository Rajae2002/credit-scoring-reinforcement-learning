"""MDP du crédit : actions, récompenses, politique de la banque (données « loguées »).

MDP (épisode d'une seule étape) :
    S = profil discrétisé du client      (324 états)
    A = {0: Refuser, 1: Accepter, 2: Accepter avec montant réduit}
    R = profit (en milliers de MAD) selon l'issue du crédit
    P = transition vers l'état terminal (le client est traité, l'épisode s'arrête)
"""
import numpy as np

ACTIONS = ["Refuser", "Accepter", "Accepter (montant réduit)"]


def reward_matrix(df, rate=0.20, lgd=0.60, cut=0.50, rescue_p=0.50, seed=0):
    """Récompense de chacune des 3 actions pour chaque client (matrice n x 3).

    - Refuser            : 0
    - Accepter           : +rate * montant si bon payeur, -lgd * montant si défaut
    - Accepter réduit    : même chose avec montant * cut. Hypothèse de simulation :
                           un défaut dû au surendettement (montant/salaire élevé)
                           est « sauvé » avec probabilité rescue_p si on réduit le montant.
    """
    A = df["amount"].values / 1000.0
    d = df["default"].values
    burden = df["amount"].values / df["salary"].values
    thr = np.quantile(burden, 0.5)
    u = np.random.default_rng(seed).random(len(df))

    R = np.zeros((len(df), 3))
    R[:, 1] = np.where(d == 1, -lgd * A, rate * A)
    A2 = cut * A
    rescued = (d == 1) & (burden >= thr) & (u < rescue_p)
    R[:, 2] = np.where((d == 1) & ~rescued, -lgd * A2, rate * A2)
    return R


def behavior_policy(df, eps=0.20, seed=0):
    """Politique historique de la banque (règles simples + 20 % de décisions aléatoires)."""
    rng = np.random.default_rng(seed)
    burden = (df["amount"] / df["salary"]).values
    hi = np.quantile(burden, 0.66)
    a = np.where(df["late_payments"].values >= 3, 0, np.where(burden >= hi, 2, 1))
    noise = rng.random(len(df)) < eps
    a = np.where(noise, rng.integers(0, 3, len(df)), a)
    return a


def rule_policy(df):
    """Même règle sans le bruit (sert de comparaison « règles de la banque »)."""
    return behavior_policy(df, eps=0.0)


def break_even_pd(rate, lgd):
    """Seuil de probabilité de défaut : on accepte si p < rate / (rate + lgd)."""
    return rate / (rate + lgd)
