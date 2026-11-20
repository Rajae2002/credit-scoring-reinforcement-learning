"""Barre latérale commune + exécution du pipeline (mise en cache)."""
import io

import pandas as pd
import streamlit as st

from .data import from_csv, synthetic_base
from .pipeline import run

SOURCES = ["Base A (synthétique, style German Credit)",
           "Base B (synthétique, style microfinance)",
           "Importer un CSV (Kaggle)"]


@st.cache_resource(show_spinner="Entraînement des agents...")
def _compute(df_bytes, params):
    df = pd.read_pickle(io.BytesIO(df_bytes))
    return run(df, **dict(params))


def sidebar():
    st.sidebar.header("Données")
    source = st.sidebar.radio("Base de données", SOURCES)
    df, label = None, source

    if source == SOURCES[0]:
        df = synthetic_base("A", n=3000, seed=1)
    elif source == SOURCES[1]:
        df = synthetic_base("B", n=6000, seed=1)
    else:
        up = st.sidebar.file_uploader("Fichier CSV", type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            cols = ["(aucune)"] + list(raw.columns)
            st.sidebar.caption("Associe les colonnes de ton fichier aux 5 entrées :")
            m = {
                "age": st.sidebar.selectbox("Âge", cols),
                "salary": st.sidebar.selectbox("Salaire / revenu", cols),
                "credit_type": st.sidebar.selectbox("Type de crédit / but", cols),
                "late_payments": st.sidebar.selectbox("Historique de paiement (nb retards)", cols),
                "amount": st.sidebar.selectbox("Montant du crédit", cols[1:]),
                "default": st.sidebar.selectbox("Label (défaut / risque)", cols[1:]),
            }
            m = {k: (None if v == "(aucune)" else v) for k, v in m.items()}
            vals = raw[m["default"]].astype(str).unique().tolist()
            pos = st.sidebar.selectbox("Valeur = mauvais payeur", vals)
            df = from_csv(raw, m, pos)
            label = up.name

    st.sidebar.header("Récompenses (MDP)")
    rate = st.sidebar.slider("Taux d'intérêt gagné", 0.05, 0.40, 0.20, 0.01)
    lgd = st.sidebar.slider("Perte si défaut (part du montant)", 0.20, 1.00, 0.60, 0.05)
    cut = st.sidebar.slider("Montant réduit = x du montant demandé", 0.2, 0.9, 0.5, 0.05)
    rescue = st.sidebar.slider("Défauts « sauvés » par la réduction", 0.0, 1.0, 0.5, 0.05)

    st.sidebar.header("Apprentissage")
    epochs = st.sidebar.slider("Epochs", 5, 60, 30, 5)
    alpha = st.sidebar.select_slider("alpha (learning rate)", [0.01, 0.02, 0.05, 0.1, 0.2], 0.05)
    cql_alpha = st.sidebar.slider("alpha CQL (prudence)", 0.0, 3.0, 1.0, 0.1)
    seed = st.sidebar.number_input("Seed", 0, 999, 0)

    if df is None:
        st.info("Charge un fichier CSV dans la barre latérale pour commencer.")
        st.stop()

    params = tuple(dict(rate=rate, lgd=lgd, cut=cut, rescue_p=rescue, epochs=epochs,
                        alpha=alpha, cql_alpha=cql_alpha, seed=int(seed)).items())
    buf = io.BytesIO()
    df.to_pickle(buf)
    res = _compute(buf.getvalue(), params)
    return dict(df=df, label=label, res=res)
