"""Application Streamlit : prédiction de souscription à un dépôt à terme.

Lancer depuis un terminal, dans le dossier contenant ce fichier et bank_model.joblib :
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_PATH = Path(__file__).parent / "bank_model.joblib"

LABELS = {
    "age": "Âge",
    "job": "Profession",
    "marital": "Situation maritale",
    "education": "Niveau d'éducation",
    "default": "Défaut de crédit connu",
    "housing": "Prêt immobilier",
    "loan": "Prêt personnel",
    "contact": "Type de contact",
    "month": "Mois du dernier contact",
    "day_of_week": "Jour de la semaine",
    "duration": "Durée de l'appel (secondes)",
    "campaign": "Nombre de contacts durant cette campagne",
    "pdays": "Jours depuis le dernier contact (999 = jamais contacté)",
    "previous": "Nombre de contacts avant cette campagne",
    "poutcome": "Résultat de la campagne précédente",
    "emp.var.rate": "Taux de variation de l'emploi",
    "cons.price.idx": "Indice des prix à la consommation",
    "cons.conf.idx": "Indice de confiance des consommateurs",
    "euribor3m": "Taux Euribor 3 mois",
    "nr.employed": "Nombre d'employés (milliers)",
}
PERSONAL = ["age", "job", "marital", "education", "default", "housing", "loan"]
CAMPAIGN = ["contact", "month", "day_of_week", "campaign", "pdays", "previous", "poutcome", "duration"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
ORDERS = {
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
}

st.set_page_config(page_title="Prédiction bancaire", page_icon="🏦", layout="wide")
st.title("🏦 Prédiction de souscription à un dépôt à terme")


@st.cache_resource
def load_artifact():
    if not ARTIFACT_PATH.exists():
        return None
    return joblib.load(ARTIFACT_PATH)


art = load_artifact()
if art is None:
    st.error(f"Fichier introuvable : {ARTIFACT_PATH.name}")
    st.info("Exécute d'abord le notebook jusqu'à la cellule de sauvegarde, "
            "puis place bank_model.joblib dans le même dossier que app.py.")
    st.stop()

pipeline = art["pipeline"]
threshold = art["threshold"]
columns = art["columns"]
categories = art["categories"]
num_stats = art["numeric_stats"]
int_cols = set(art["int_cols"])


def sorted_options(col):
    options = categories[col]
    if col in ORDERS:
        order = ORDERS[col]
        return sorted(options, key=lambda v: order.index(v) if v in order else len(order))
    return options


def field(col, key_prefix=""):
    """Crée le widget adapté à la colonne (liste déroulante ou nombre)."""
    label = LABELS.get(col, col)
    key = f"{key_prefix}{col}"
    if col in categories:
        options = sorted_options(col)
        return st.selectbox(label, options, index=options.index(art["defaults"][col]), key=key)
    s = num_stats[col]
    if col in int_cols:
        return st.number_input(label, int(s["min"]), int(s["max"]), int(s["median"]), step=1, key=key)
    return st.number_input(label, float(s["min"]), float(s["max"]), float(s["median"]),
                           format="%.3f", key=key)


with st.form("prediction_form"):
    left, right = st.columns(2)
    values = {}
    with left:
        st.subheader("👤 Profil du client")
        for col in PERSONAL:
            if col in columns:
                values[col] = field(col)
    with right:
        st.subheader("📞 Campagne")
        for col in CAMPAIGN:
            if col in columns:
                values[col] = field(col)

    with st.expander("📈 Contexte macro-économique (valeurs médianes par défaut)"):
        st.caption("Ces indicateurs dépendent de la période. Ajuste-les si tu connais le contexte.")
        macro_cols = st.columns(len(MACRO))
        for container, col in zip(macro_cols, MACRO):
            if col in columns:
                with container:
                    values[col] = field(col)

    submitted = st.form_submit_button("🔮 Prédire", use_container_width=True)

if submitted:
    row = pd.DataFrame([values])[columns]
    proba = float(pipeline.predict_proba(row)[0, 1])
    will_subscribe = proba >= threshold

    st.markdown("---")
    st.subheader("📊 Résultat")
    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilité de souscription", f"{proba * 100:.1f} %")
    c2.metric("Seuil de décision", f"{threshold * 100:.1f} %")
    c3.metric("Décision", "Contacter" if will_subscribe else "Ne pas prioriser")
    st.progress(min(max(proba, 0.0), 1.0))

    if will_subscribe:
        st.success("✅ Probabilité au-dessus du seuil : ce client vaut la peine d'être contacté.")
    else:
        st.info("📉 Probabilité sous le seuil : ce client n'est pas prioritaire.")

    if art.get("use_duration"):
        st.warning("Le modèle utilise la durée de l'appel, qui n'est connue qu'après l'appel : "
                   "cette prédiction n'est pas utilisable avant de contacter le client.")

st.markdown("---")
st.caption("Le seuil de décision a été choisi pour minimiser un coût métier (appel inutile vs souscription manquée), pas à 50 %.")
