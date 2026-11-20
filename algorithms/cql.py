"""
cql.py : Conservative Q-Learning (CQL) tabulaire, version discrète « CQL(H) »
(Kumar, Zhou, Tucker, Levine, NeurIPS 2020 ; référence à vérifier avant citation).
Adaptation TABULAIRE (l'article utilise des réseaux de neurones).

Contexte : apprentissage HORS LIGNE sur un jeu de transitions figé. Le max de
Q-Learning peut valoriser des actions jamais vues dans les données. CQL ajoute une
pénalité qui baisse Q des actions peu observées :
    perte = 0.5 (Q(s,a) - y)^2 + alpha_cql * ( logsumexp_a' Q(s,a') - Q(s,a) )
    y = r + gamma * max_a' Q(s',a')     (y = r si l'épisode est terminé)
Gradient par échantillon (s, a, r, s') :
    Q(s,.) <- Q(s,.) - lr * alpha_cql * ( softmax(Q(s,.)) - onehot(a) )
    Q(s,a) <- Q(s,a) + lr * ( y - Q(s,a) )
alpha_cql = 0 redonne exactement le Q-Learning.
Ici `alpha` est le pas d'apprentissage (comme dans les autres agents).
"""
import numpy as np

from config import N_STATES, N_ACTIONS


class CQLAgent:
    def __init__(self, n_states=N_STATES, n_actions=N_ACTIONS,
                 alpha=0.05, gamma=0.95, alpha_cql=1.0):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha doit être dans ]0, 1]")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma doit être dans [0, 1]")
        if alpha_cql < 0:
            raise ValueError("alpha_cql doit être >= 0")
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma, self.alpha_cql = alpha, gamma, alpha_cql
        self.Q = np.zeros((n_states, n_actions))

    def update(self, s, a, r, s_next, done):
        """Une mise à jour CQL. Retourne l'erreur TD. Si done=True, s_next est ignoré."""
        q = self.Q[s].copy()
        y = r if done else r + self.gamma * np.max(self.Q[s_next])
        td = y - q[a]
        z = np.exp(q - q.max())                    # softmax stable
        grad = z / z.sum()
        grad[a] -= 1.0                             # softmax(Q(s,.)) - onehot(a)
        self.Q[s] = q - self.alpha * self.alpha_cql * grad
        self.Q[s, a] += self.alpha * td
        return float(td)

    def greedy_action(self, state):
        return int(np.argmax(self.Q[state]))

    def select_action(self, state, epsilon=0.0, rng=None):
        rng = rng or np.random.default_rng()
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return self.greedy_action(state)

    def q_values(self, state):
        return self.Q[state].copy()

    def policy(self):
        return np.argmax(self.Q, axis=1)

    def save(self, path):
        np.save(path, self.Q)

    def load(self, path):
        Q = np.load(path)
        if Q.shape != self.Q.shape:
            raise ValueError(f"forme incompatible : {Q.shape}")
        self.Q = Q
