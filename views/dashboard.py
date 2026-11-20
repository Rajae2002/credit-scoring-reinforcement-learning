import numpy as np
import pandas as pd
import streamlit as st

from credit_rl.data import CREDIT_TYPES
from credit_rl.env import ACTIONS, break_even_pd
from credit_rl.pipeline import decide_client, sup_features

ctx = st.session_state["ctx"]
df, res = ctx["df"], ctx["res"]
p = res["params"]

st.title("Credit scoring en microfinance : Q-learning, Double Q-learning et CQL")
st.caption(f"Base : {ctx['label']}  |  {len(df)} clients  |  "
           f"taux de défaut : {100 * df['default'].mean():.1f} %  |  "
           f"train {len(res['train'])} / test {len(res['test'])}")

tab1, tab2, tab3 = st.tabs(["Résultats", "Tester un client", "Comparaison avec la littérature"])

# ------------------------------------------------------------------ Résultats
with tab1:
    t = res["table"]
    rl = t.loc[["Q-learning", "Double Q-learning", "CQL"], "Profit moyen / client (k MAD)"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meilleur agent RL", rl.idxmax())
    c2.metric("Son profit moyen / client", f"{rl.max():.2f} k MAD")
    c3.metric("Scoring logistique", f"{t.loc['Scoring logistique (seuil rentabilité)', 'Profit moyen / client (k MAD)']:.2f} k MAD")
    c4.metric("Tout accepter", f"{t.loc['Tout accepter', 'Profit moyen / client (k MAD)']:.2f} k MAD")

    st.subheader("Comparaison des politiques (sur le jeu de test)")
    st.dataframe(t.style.format("{:.2f}"), width='stretch')

    left, right = st.columns(2)
    with left:
        st.markdown("**Profit moyen par client (k MAD)**")
        st.bar_chart(t["Profit moyen / client (k MAD)"])
    with right:
        st.markdown("**Courbes d'apprentissage** (profit moyen sur le test, par epoch)")
        st.line_chart(pd.DataFrame(res["curves"]))

    left, right = st.columns(2)
    with left:
        st.markdown("**Répartition des décisions (%)**")
        st.bar_chart(t[["% refusés", "% acceptés", "% montant réduit"]])
    with right:
        st.markdown("**Entrées les plus importantes** (Random Forest)")
        names = {"age": "Âge", "salary": "Salaire", "late": "Retards de paiement",
                 "burden": "Montant / salaire", "amount": "Montant du crédit"}
        X = sup_features(res["train"])
        imp = pd.Series(res["rf"].feature_importances_, index=X.columns)
        types = imp[[c for c in X.columns if c.startswith("type_")]].sum()
        imp = imp[[c for c in X.columns if not c.startswith("type_")]].rename(index=names)
        imp["Type de crédit"] = types
        st.bar_chart(imp.sort_values(ascending=False))

    st.markdown("**Scoring supervisé classique** (ce que font la plupart des notebooks Kaggle)")
    st.dataframe(res["sup"].style.format({"AUC": "{:.3f}", "Accuracy (seuil 0.5)": "{:.3f}"}),
                 hide_index=True)

    with st.expander("Aperçu des données"):
        st.dataframe(df.head(20), width='stretch')

# ------------------------------------------------------------------ Tester un client
with tab2:
    st.subheader("Un client se présente à l'agence")
    a, b, c = st.columns(3)
    age = a.slider("Âge", 18, 75, 35)
    salary = b.number_input("Salaire mensuel (MAD)", 500, 50000, 3000, 100)
    ctype = c.selectbox("Type de crédit", CREDIT_TYPES)
    d, e = st.columns(2)
    late = d.slider("Historique : nombre de retards de paiement", 0, 8, 0)
    amount = e.number_input("Montant demandé (MAD)", 500, 200000, 6000, 500)

    s, out, pd_lr, risk = decide_client(res, age, salary, ctype, late, amount)
    st.caption(f"État discret du client : s = {s}  |  montant / salaire = {amount / salary:.1f} mois de salaire")

    r1, r2 = st.columns(2)
    r1.metric("Niveau de risque", risk["label"])
    r2.metric("Probabilité de défaut estimée", f"{100 * risk['pd']:.1f} %")

    cols = st.columns(4)
    for col, (name, (act, q)) in zip(cols[:3], out.items()):
        col.metric(name, ACTIONS[act])
    thr = break_even_pd(p["rate"], p["lgd"])
    cols[3].metric("Scoring logistique", "Accepter" if pd_lr < thr else "Refuser",
                   f"prob. de défaut {100 * pd_lr:.1f} %", delta_color="off")

    st.markdown("**Valeurs Q(s, a) apprises (profit espéré en k MAD)**")
    q = pd.DataFrame({k: v[1] for k, v in out.items()}, index=ACTIONS).T
    st.dataframe(q.style.format("{:.3f}"), width='stretch')

# ------------------------------------------------------------------ Littérature
with tab3:
    st.subheader("Nos résultats vs résultats publiés")
    st.write("Remplis les lignes vides avec les chiffres des notebooks Kaggle / articles que vous citez "
             "(AUC et accuracy sur la même base). Le tableau est modifiable.")
    ours = res["sup"].copy()
    ours.insert(0, "Source", "Notre travail")
    ours.insert(1, "Base", ctx["label"])
    blank = pd.DataFrame({"Source": ["", "", ""], "Base": ["", "", ""], "Modèle": ["", "", ""],
                          "AUC": [np.nan] * 3, "Accuracy (seuil 0.5)": [np.nan] * 3})
    st.data_editor(pd.concat([ours, blank], ignore_index=True), num_rows="dynamic",
                   width='stretch')
    st.info("Les agents Q-learning / Double Q / CQL optimisent un profit, pas l'AUC. "
            "La comparaison avec les notebooks se fait donc via le scoring supervisé "
            "(AUC, accuracy) et via le profit de la politique « scoring logistique + seuil ».")
