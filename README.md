# Credit scoring en microfinance : Q-learning, Double Q-learning, CQL

Projet Processus stochastiques. Auteures : Rajae Boutahri, Zineb Zarda.

## Lancer
    pip install -r requirements.txt
    streamlit run app.py

## Structure
    app.py                  navigation + barre latérale
    views/dashboard.py      résultats, tester un client, comparaison littérature
    views/maths.py          MDP, récompenses, formules, calcul pas à pas
    credit_rl/data.py       bases synthétiques, import CSV, discrétisation (324 états)
    credit_rl/env.py        actions, récompenses, politique historique de la banque
    credit_rl/agents.py     Q-learning, Double Q-learning, CQL (tabulaires)
    credit_rl/pipeline.py   split, entraînement, évaluation, scoring logistique / RF

## Utiliser tes bases Kaggle
Dans l'application : « Importer un CSV », puis associer les colonnes du fichier aux 5 entrées
(âge, salaire, type de crédit, retards de paiement, montant) et au label de défaut.
Si une colonne n'existe pas dans la base (ex. salaire dans German Credit), choisir « (aucune) ».

## Hypothèses à mentionner dans le rapport
- MDP à une étape : la décision termine l'épisode (gamma n'a donc pas d'effet).
- Les bases Kaggle ne contiennent que des crédits accordés. La récompense de « Refuser » (0) et
  de « Montant réduit » est simulée : réduire le montant sauve une part (paramètre) des défauts
  dus au surendettement. Ce paramètre est réglable dans la barre latérale.
- Le journal d'entraînement est une politique « banque » simulée (règles + 20 % de bruit) :
  on ne voit que la récompense de l'action choisie (apprentissage hors-ligne), d'où l'intérêt de CQL.
