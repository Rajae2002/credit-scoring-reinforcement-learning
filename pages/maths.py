import numpy as np
import pandas as pd
import streamlit as st

from credit_rl.data import N_STATES
from credit_rl.env import ACTIONS, break_even_pd

ctx = st.session_state["ctx"]
res = ctx["res"]
p = res["params"]

st.title("Calculs mathématiques")
st.caption("Tout ce que le modèle calcule, étape par étape, avec les paramètres choisis dans la barre latérale.")

# ------------------------------------------------------------------ 1. MDP
st.header("1. Le crédit comme un MDP")
st.markdown("Un MDP est un quintuplet $(S, A, P, R, \\gamma)$. Ici, le client qui se présente est un **état**, "
            "la banque choisit une **action**, et le résultat du crédit donne la **récompense**.")
st.latex(r"S=\{\text{âge},\ \text{salaire},\ \text{type},\ \text{retards},\ \text{montant/salaire}\}"
         rf"\quad |S|={N_STATES}")
st.latex(r"A=\{0:\text{Refuser},\ 1:\text{Accepter},\ 2:\text{Accepter avec montant réduit}\}")
st.markdown("Chaque épisode a **une seule étape** : la décision est prise, le client est traité, l'épisode se termine "
            "(état terminal). Donc la cible de Bellman se réduit à $r$ ; la forme générale est gardée ci-dessous "
            "pour montrer d'où viennent les algorithmes.")
st.latex(r"Q^*(s,a)=\mathbb{E}\big[\,r+\gamma \max_{a'}Q^*(s',a')\,\big]\ \xrightarrow{\ s'\ \text{terminal}\ }\ \mathbb{E}[r\mid s,a]")

# ------------------------------------------------------------------ 2. Récompenses
st.header("2. Récompenses")
rate, lgd, cut = p["rate"], p["lgd"], p["cut"]
st.latex(rf"R(\text{{Refuser}})=0")
st.latex(rf"R(\text{{Accepter}})=\begin{{cases}}+{rate}\times M & \text{{si le client rembourse}}\\ -{lgd}\times M & \text{{si défaut}}\end{{cases}}")
st.latex(rf"R(\text{{Réduit}})=\text{{idem avec }} M'={cut}\times M")
thr = break_even_pd(rate, lgd)
st.markdown("**Seuil de rentabilité.** Avec une probabilité de défaut $q$, le profit espéré d'un crédit de montant $M$ est :")
st.latex(r"\mathbb{E}[R]=(1-q)\,r\,M-q\,L\,M>0\iff q<\frac{r}{r+L}")
st.latex(rf"q<\frac{{{rate}}}{{{rate}+{lgd}}}={thr:.3f}")
st.markdown(f"Donc on accepte un client si sa probabilité de défaut est inférieure à **{100 * thr:.1f} %** "
            "(c'est le seuil utilisé par le « scoring logistique »).")

# ------------------------------------------------------------------ 3. Algorithmes
st.header("3. Les trois algorithmes")
st.subheader("Q-learning")
st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]")
st.markdown("*Analogie : un agent de crédit qui garde, pour chaque type de client, une moyenne mobile "
            "des profits qu'il a vus quand il a accepté, refusé ou réduit.*")

st.subheader("Double Q-learning")
st.latex(r"Q_A(s,a)\leftarrow Q_A(s,a)+\alpha\Big[r+\gamma\,Q_B\big(s',\arg\max_{a'}Q_A(s',a')\big)-Q_A(s,a)\Big]")
st.markdown("On tient **deux** tables, mise à jour au hasard à tour de rôle : une table choisit l'action, "
            "l'autre l'évalue. Cela évite que le « max » surestime les valeurs à cause du bruit. "
            "*Analogie : deux analystes indépendants, l'un propose, l'autre vérifie.* "
            "La décision finale utilise $(Q_A+Q_B)/2$.")

st.subheader("CQL (Conservative Q-Learning)")
st.latex(r"\mathcal{L}(Q)=\tfrac12\big(Q(s,a)-y\big)^2+\alpha_{CQL}\Big[\log\sum_{b}e^{Q(s,b)}-Q(s,a)\Big]")
st.markdown("Le deuxième terme **baisse** la valeur des actions que la banque n'a jamais (ou rarement) essayées "
            "dans l'historique, et **monte** celle de l'action réellement observée. "
            "*Analogie : on ne fait pas confiance à une action jamais testée tant qu'on n'a pas de preuve.* "
            "Pour la table, le gradient donne :")
st.latex(r"Q(s,b)\leftarrow Q(s,b)-\eta\,\alpha_{CQL}\big(\text{softmax}_b(Q(s,\cdot))-\mathbb{1}[b=a]\big)")

# ------------------------------------------------------------------ 4. Calculateur
st.header("4. Calcul d'une mise à jour, pas à pas")
st.markdown("Entre des valeurs de départ et une observation du journal de la banque : on voit les trois mises à jour.")
c1, c2, c3 = st.columns(3)
q0 = np.array([c1.number_input("Q(s, Refuser)", value=0.0, step=0.1),
               c2.number_input("Q(s, Accepter)", value=0.5, step=0.1),
               c3.number_input("Q(s, Réduit)", value=0.0, step=0.1)])
d1, d2, d3 = st.columns(3)
a_obs = d1.selectbox("Action observée", [0, 1, 2], index=1, format_func=lambda i: ACTIONS[i])
r_obs = d2.number_input("Récompense observée r (k MAD)", value=-1.2, step=0.1)
alpha = d3.number_input("alpha", value=float(p["alpha"]), step=0.01, format="%.2f")
ca = st.number_input("alpha CQL", value=float(p["cql_alpha"]), step=0.1)

# Q-learning
q_new = q0.copy()
td = r_obs - q0[a_obs]
q_new[a_obs] = q0[a_obs] + alpha * td
st.markdown("**Q-learning**")
st.latex(rf"Q_{{new}}={q0[a_obs]:.3f}+{alpha}\times({r_obs:.3f}-{q0[a_obs]:.3f})={q_new[a_obs]:.4f}")

# Double Q (on suppose que QA = q0 est mise à jour, comme Q-learning en étape terminale)
st.markdown("**Double Q-learning** : si la table $Q_A$ est tirée, le calcul est le même (étape terminale), "
            "mais chaque table ne voit en moyenne que la moitié des observations.")
st.latex(rf"Q_{{A,new}}={q_new[a_obs]:.4f}\quad(\text{{une fois sur deux}})")

# CQL
e = np.exp(q0 - q0.max())
soft = e / e.sum()
q_cql = q0.copy()
q_cql[a_obs] += alpha * (r_obs - q0[a_obs])
e2 = np.exp(q_cql - q_cql.max())
soft2 = e2 / e2.sum()
grad = soft2.copy()
grad[a_obs] -= 1.0
q_final = q_cql - alpha * ca * grad
st.markdown("**CQL** : (1) partie Bellman, (2) softmax, (3) partie conservatrice")
st.dataframe(pd.DataFrame({"Q départ": q0, "après Bellman": q_cql, "softmax": soft2,
                           "gradient CQL": grad, "Q final CQL": q_final}, index=ACTIONS).style.format("{:.4f}"),
             width='stretch')
st.caption("Remarque : l'action observée monte (gradient négatif), les actions non observées baissent "
           "proportionnellement à leur softmax.")

# ------------------------------------------------------------------ 5. Table apprise
st.header("5. Q-table apprise pour quelques états")
n_obs = np.bincount(res["S_tr"], minlength=N_STATES)
top = np.argsort(-n_obs)[:8]
rows = []
for s in top:
    row = {"état s": int(s), "nb. observations": int(n_obs[s])}
    for k, Q in res["Q"].items():
        row[f"{k} → décision"] = ACTIONS[int(Q[s].argmax())]
    rows.append(row)
st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
st.caption("États les plus fréquents dans le journal d'entraînement. Un état rare a peu d'observations : "
           "c'est là que Q-learning est instable et que la prudence de CQL aide.")
