import streamlit as st

from credit_rl.ui import sidebar

st.set_page_config(page_title="Credit scoring en microfinance (RL)", layout="wide")

st.session_state["ctx"] = sidebar()

pg = st.navigation([
    st.Page("views/dashboard.py", title="Dashboard"),
    st.Page("views/maths.py", title="Calculs mathématiques"),
])
pg.run()
