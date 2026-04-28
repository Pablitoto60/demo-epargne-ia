import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------
st.set_page_config(page_title="Démo IA – Conseil Épargne", layout="wide")

# --------------------------------------------------
# Hypothèses du démonstrateur (illustratives)
# --------------------------------------------------
HORIZON_ANNEES = 5
MOIS = HORIZON_ANNEES * 12
VERSEMENT_MENSUEL = 150  # € par mois

RENDEMENTS = {
    "Sécurisé": 0.015,    # 1,5 % / an
    "Équilibré": 0.030,   # 3,0 % / an
    "Dynamique": 0.050    # 5,0 % / an
}

# --------------------------------------------------
# Fonctions de calcul
# --------------------------------------------------
def projection_epargne(versement_mensuel, rendement_annuel, mois):
    r_mensuel = (1 + rendement_annuel) ** (1 / 12) - 1
    capital = np.zeros(mois + 1)
    for t in range(1, mois + 1):
        capital[t] = capital[t - 1] * (1 + r_mensuel) + versement_mensuel
    return capital

def build_chart(risque):
    x = np.arange(MOIS + 1) / 12
    y = projection_epargne(VERSEMENT_MENSUEL, RENDEMENTS[risque], MOIS)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name="Capital estimé",
            line=dict(width=4)
        )
    )

    fig.update_layout(
        title=f"Évolution projetée de l’épargne – Profil {risque} (illustratif)",
        xaxis_title="Années",
        yaxis_title="Capital estimé (€)",
        template="plotly_white",
        margin=dict(t=60, r=20, l=50, b=50)
    )

    fig.add_annotation(
        text="Hypothèses illustratives – aucune garantie – démonstrateur non contractuel",
        x=0,
        y=-0.2,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=11, color="#666"),
        xanchor="left"
    )

    return fig

# --------------------------------------------------
# Interface
# --------------------------------------------------
st.title("🤖 Démonstrateur – Agent IA de conseil épargne")

col_gauche, col_droite = st.columns([1, 1], gap="large")

# ----- Colonne droite : graphique
with col_droite:
    st.subheader("Projection")

    # Valeurs par défaut pour l'état
    if "can_show_projection" not in st.session_state:
        st.session_state.can_show_projection = False
    if "show_projection" not in st.session_state:
        st.session_state.show_projection = False
    if "risque_ui" not in st.session_state:
        st.session_state.risque_ui = "Équilibré"

    # Tant que la reco n'a pas été faite : on n'affiche pas le graphique
    if not st.session_state.can_show_projection:
        st.info("La projection s’affichera après la recommandation produit.")
    else:
        # Bouton pour afficher la projection au moment de la reco
        if not st.session_state.show_projection:
            if st.button("Afficher la projection", type="primary"):
                st.session_state.show_projection = True

        # Une fois le bouton cliqué : on affiche le risque + le graphique
        if st.session_state.show_projection:
            risque = st.selectbox(
                "Niveau de risque",
                ["Sécurisé", "Équilibré", "Dynamique"],
                index=["Sécurisé", "Équilibré", "Dynamique"].index(st.session_state.risque_ui),
                key="risque_select"
            )
            st.session_state.risque_ui = risque

            st.plotly_chart(build_chart(risque), use_container_width=True)

with col_gauche:
    st.subheader("Conversation")

    # -----------------------------
    # Connaissance déjà disponible (fictif)
    # -----------------------------
    KNOWN_PROFILE = {
        "prenom": "Dimitri",
        "age": 28,
        "revenu_net_mensuel": 2400,
        "livret_a_plein": True
    }

    # -----------------------------
    # Mémoire de conversation
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
        {"role": "user", "content": "Je veux épargner mais je ne sais pas par quoi commencer."},
        {"role": "assistant", "content": (
            "Bonjour Dimitri 👋\n\n"
            "Excellente initiative. Je peux t’aider à continuer à épargner de façon simple et adaptée.\n\n"
            "D’après les informations dont je dispose, ton Livret A est déjà plein — c’est une très bonne chose : "
            "ça signifie que ton épargne de précaution est déjà en place.\n\n"
            "Ce qu’il me manque maintenant, ce sont tes projets : qu’est-ce que tu aimerais préparer grâce à ton épargne ?\n\n"
            "*(Simulation illustrative, non contractuelle.)*"
        )}
    ]
    if "step" not in st.session_state:
        st.session_state.step = 0
    if "client" not in st.session_state:
        st.session_state.client = {
            "objectif": None,
            "horizon": None,
            "mensualite": None,
            "risque": None
        }

    def assistant_say(text: str):
        st.session_state.messages.append({"role": "assistant", "content": text})

    def normalize_risk(text: str):
        t = text.lower()
        if "sécur" in t or "secur" in t or "prudent" in t:
            return "Sécurisé"
        if "dyna" in t or "risqu" in t:
            return "Dynamique"
        if "équil" in t or "equil" in t or "moyen" in t:
            return "Équilibré"
        return None

    def extract_number(text: str):
        import re
        m = re.search(r"(\d+)", text.replace(" ", ""))
        return int(m.group(1)) if m else None

    # -----------------------------
    # Affichage de l'historique
    # -----------------------------
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # -----------------------------
    # Entrée utilisateur
    # -----------------------------
    user_text = st.chat_input("Tape ton message…")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.write(user_text)

        step = st.session_state.step
        client = st.session_state.client

        # Étape 0 : objectif
        if step == 0:
            client["objectif"] = user_text.strip()
            assistant_say(
                "Merci, c’est clair.\n\n"
                "Tu penses à quel horizon, à peu près ? Par exemple : 2 ans, 4–5 ans, 10 ans…"
            )
            st.session_state.step = 1

        # Étape 1 : horizon
        elif step == 1:
            client["horizon"] = user_text.strip()
            assistant_say(
                "Parfait.\n\n"
                "Et quel montant tu pourrais mettre de côté chaque mois, sans te mettre en difficulté ? "
                "Même une estimation suffit (ex : 100€, 150€, 200€…)."
            )
            st.session_state.step = 2

        # Étape 2 : mensualité
        elif step == 2:
            m = extract_number(user_text)
            client["mensualite"] = m if m else 150
            assistant_say(
                "Top.\n\n"
                "Dernier point : le niveau de risque.\n"
                "Si la valeur de ton épargne baisse temporairement, tu préfères plutôt :\n"
                "• très prudent / sécuriser au maximum\n"
                "• accepter de petites variations (équilibré)\n"
                "• accepter plus de variations (dynamique)\n\n"
                "Tu te situes plutôt où ?"
            )
            st.session_state.step = 3

        # Étape 3 : risque + reco + projection + CTA
        elif step == 3:
            r = normalize_risk(user_text) or "Équilibré"
            client["risque"] = r 
            st.session_state.risque_ui = r
            st.session_state.can_show_projection = True
            st.session_state.show_projection = False

            prenom = KNOWN_PROFILE["prenom"]
            mensualite = client.get("mensualite") or 150
	    horizon = client.get("horizon") or "4–5 ans"
            risque = client["risque"]

            assistant_say(
                f"Merci {prenom}. Si je résume :\n"
                f"• Ton Livret A est plein (et on le conserve comme épargne de précaution)\n"
                f"• Ton projet : **{client['objectif']}**\n"
                f"• Horizon : **{horizon}**\n"
                f"• Effort d’épargne : **{mensualite}€ / mois**\n"
                f"• Profil : **{risque}**\n\n"
                "✅ **Recommandation** : je te recommande d’ouvrir une **assurance vie**.\n\n"
                "Pourquoi :\n"
                "1) Ton Livret A joue déjà le rôle de sécurité : je te conseille d’en garder une partie pour les imprévus.\n"
                "2) L’assurance vie est adaptée à un horizon de quelques années et reste flexible.\n"
                "3) Elle permet de viser un potentiel de rendement supérieur à un livret.\n\n"
                "📊 Je t’affiche maintenant une projection avec ce montant mensuel.\n"
                "Tu peux ajuster le niveau de risque dans le menu à droite : la courbe se mettra à jour."
            )

            assistant_say(
                "Souhaites-tu passer à l’étape suivante ?\n"
                "✅ Souscrire en ligne à l’assurance vie recommandée\n"
                "✅ Ou simuler d’autres montants / horizons"
            )

            st.session_state.step = 4

        # Étape 4 : souscription / ajustements
        else:
            txt = user_text.lower()
            if "souscri" in txt or "ouvrir" in txt or "ok" in txt or "oui" in txt:
                assistant_say(
                    "Parfait 👍\n\n"
                    "Je te redirige vers le **parcours de souscription en ligne**.\n"
                    "(Démo : bouton / URL à brancher)\n\n"
                    "Récap : on conserve le Livret A comme épargne de précaution, "
                    "et on met en place une assurance vie pour ton projet."
                )
            else:
                assistant_say(
                    "Très bien. Dis-moi ce que tu veux ajuster :\n"
                    "• le montant mensuel (ex : 200€)\n"
                    "• l’horizon (ex : 3 ans)\n"
                    "• ou le niveau de risque (sécurisé / équilibré / dynamique)\n\n"
                    "Et je mets à jour la projection."
                )

