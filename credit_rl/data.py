"""Données : génération de bases synthétiques + chargement d'un CSV Kaggle + discrétisation.

Toutes les bases sont ramenées au même format (5 entrées + 1 label) :
    age, salary, credit_type, late_payments, amount, default (1 = défaut de paiement)
"""
import numpy as np
import pandas as pd

CREDIT_TYPES = ["consommation", "commerce", "agriculture", "habitat"]
FEATURES = ["age", "salary", "credit_type", "late_payments", "amount"]

N_AGE, N_SAL, N_TYPE, N_LATE, N_BURDEN = 3, 3, 4, 3, 3
N_STATES = N_AGE * N_SAL * N_TYPE * N_LATE * N_BURDEN  # 324


# ----------------------------------------------------------------------------
# 1) Bases synthétiques (utiles pour tester l'application sans internet)
# ----------------------------------------------------------------------------
def _default_prob(age, salary, ctype, late, amount, b0):
    type_eff = np.select([ctype == "commerce", ctype == "agriculture", ctype == "habitat"],
                         [0.25, 0.40, -0.30], default=0.0)
    logit = (b0 + 0.55 * late + 0.45 * (amount / salary)
             - 0.025 * (age - 40) - 0.15 * np.log(salary / 3000) + type_eff)
    return 1 / (1 + np.exp(-logit))


def synthetic_base(name="A", n=3000, seed=0):
    """Base A : style German Credit (crédits plus gros, ~30% défauts).
       Base B : style microfinance (petits crédits, ~18% défauts)."""
    rng = np.random.default_rng(seed)
    if name == "A":
        salary = rng.lognormal(8.3, 0.40, n)       # ~4000 MAD / mois
        amount = rng.lognormal(8.9, 0.70, n)       # ~7000 MAD
        late = np.clip(rng.poisson(1.2, n), 0, 6)
        target = 0.30
    else:
        salary = rng.lognormal(7.9, 0.40, n)       # ~2700 MAD / mois
        amount = rng.lognormal(7.8, 0.60, n)       # ~2400 MAD
        late = np.clip(rng.poisson(0.8, n), 0, 6)
        target = 0.18
    age = rng.integers(19, 70, n)
    ctype = rng.choice(CREDIT_TYPES, n, p=[0.40, 0.30, 0.15, 0.15])
    # on règle b0 par dichotomie pour obtenir le taux de défaut visé
    lo, hi = -8.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _default_prob(age, salary, ctype, late, amount, mid).mean() > target:
            hi = mid
        else:
            lo = mid
    p = _default_prob(age, salary, ctype, late, amount, (lo + hi) / 2)
    default = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"age": age, "salary": salary.round(0), "credit_type": ctype,
                         "late_payments": late, "amount": amount.round(0), "default": default})


# ----------------------------------------------------------------------------
# 2) CSV Kaggle : on associe les colonnes du fichier aux 5 entrées
# ----------------------------------------------------------------------------
def from_csv(raw, mapping, positive_value):
    """raw : DataFrame brut. mapping : {feature: nom_de_colonne ou None}, + clé 'default'.
    positive_value : valeur de la colonne label qui signifie « mauvais payeur »."""
    n = len(raw)
    df = pd.DataFrame(index=raw.index)
    df["age"] = pd.to_numeric(raw[mapping["age"]], errors="coerce") if mapping.get("age") else 35
    df["amount"] = pd.to_numeric(raw[mapping["amount"]], errors="coerce")
    if mapping.get("salary"):
        df["salary"] = pd.to_numeric(raw[mapping["salary"]], errors="coerce")
    else:                                   # pas de salaire dans la base -> neutre
        df["salary"] = df["amount"].median()
    if mapping.get("late_payments"):
        df["late_payments"] = pd.to_numeric(raw[mapping["late_payments"]], errors="coerce")
    else:
        df["late_payments"] = 0
    if mapping.get("credit_type"):
        top = raw[mapping["credit_type"]].astype(str).value_counts().index[:4]
        lab = raw[mapping["credit_type"]].astype(str)
        df["credit_type"] = [CREDIT_TYPES[list(top).index(v)] if v in list(top) else "consommation"
                             for v in lab]
    else:
        df["credit_type"] = "consommation"
    df["default"] = (raw[mapping["default"]].astype(str) == str(positive_value)).astype(int)
    df = df.dropna()
    df = df[(df["salary"] > 0) & (df["amount"] > 0)]
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3) Discrétisation : client -> état (entier de 0 à 323)
# ----------------------------------------------------------------------------
class Discretizer:
    """Transforme les 5 entrées en un état discret s (Q-learning tabulaire)."""

    def fit(self, df):
        self.sal_edges = np.quantile(df["salary"], [1 / 3, 2 / 3])
        self.burden_edges = np.quantile(df["amount"] / df["salary"], [1 / 3, 2 / 3])
        return self

    def bins(self, df):
        age_b = np.digitize(df["age"], [30, 45])                       # <30 | 30-45 | >45
        sal_b = np.digitize(df["salary"], self.sal_edges)              # bas | moyen | haut
        type_b = df["credit_type"].map({c: i for i, c in enumerate(CREDIT_TYPES)}).values
        late_b = np.digitize(df["late_payments"], [1, 3])              # 0 | 1-2 | 3+
        bur_b = np.digitize(df["amount"] / df["salary"], self.burden_edges)  # montant / salaire
        return np.stack([age_b, sal_b, type_b, late_b, bur_b], axis=1)

    def transform(self, df):
        b = self.bins(df)
        return np.ravel_multi_index(b.T, (N_AGE, N_SAL, N_TYPE, N_LATE, N_BURDEN))
