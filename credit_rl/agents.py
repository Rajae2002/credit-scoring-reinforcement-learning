"""Trois agents tabulaires, entraînés hors-ligne (offline) sur le journal de la banque.

Journal = liste de transitions (s, a, r) : on ne connaît la récompense que de l'action
réellement choisie par la banque. C'est ce qui rend CQL intéressant.
"""
import numpy as np
from .data import N_STATES

N_ACTIONS = 3


def _target(r, Q_next_row, gamma, done):
    # Équation de Bellman. Ici chaque épisode a UNE étape -> done = True -> cible = r.
    return r if done else r + gamma * Q_next_row.max()


def train_q_learning(S, A, R, epochs=30, alpha=0.05, gamma=0.0, seed=0, on_epoch=None):
    """Q-learning : Q(s,a) <- Q(s,a) + alpha * (cible - Q(s,a))"""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    for ep in range(epochs):
        for i in rng.permutation(len(S)):
            s, a, r = S[i], A[i], R[i]
            Q[s, a] += alpha * (_target(r, Q[s], gamma, True) - Q[s, a])
        if on_epoch:
            on_epoch(ep, Q)
    return Q


def train_double_q(S, A, R, epochs=30, alpha=0.05, gamma=0.0, seed=0, on_epoch=None):
    """Double Q-learning : deux tables QA et QB.
    On met à jour l'une avec l'argmax de l'une et la valeur de l'autre (évite la surestimation)."""
    rng = np.random.default_rng(seed)
    QA = np.zeros((N_STATES, N_ACTIONS))
    QB = np.zeros((N_STATES, N_ACTIONS))
    for ep in range(epochs):
        for i in rng.permutation(len(S)):
            s, a, r = S[i], A[i], R[i]
            if rng.random() < 0.5:
                QA[s, a] += alpha * (r - QA[s, a])   # étape terminale : pas de terme gamma*Q(s',.)
            else:
                QB[s, a] += alpha * (r - QB[s, a])
        if on_epoch:
            on_epoch(ep, (QA + QB) / 2)
    return (QA + QB) / 2


def train_cql(S, A, R, epochs=30, alpha=0.05, cql_alpha=1.0, seed=0, on_epoch=None):
    """CQL (version tabulaire) : perte = erreur de Bellman + cql_alpha * (logsumexp_a Q(s,a) - Q(s,a_donnée)).
    Le terme en plus baisse la valeur des actions jamais vues dans les données -> décisions prudentes."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    for ep in range(epochs):
        for i in rng.permutation(len(S)):
            s, a, r = S[i], A[i], R[i]
            Q[s, a] += alpha * (r - Q[s, a])                      # partie Bellman
            e = np.exp(Q[s] - Q[s].max())
            soft = e / e.sum()                                    # gradient de logsumexp
            grad = soft.copy()
            grad[a] -= 1.0                                        # - 1[a = action donnée]
            Q[s] -= alpha * cql_alpha * grad                      # partie conservatrice
        if on_epoch:
            on_epoch(ep, Q)
    return Q


def greedy(Q, S):
    return Q[S].argmax(axis=1)
