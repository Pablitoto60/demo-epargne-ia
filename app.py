import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# --------------------------------------------------
# Configuration de la page
# --------------------------------------------------
st.set_page_config(page_title="Démo IA – Conseil Épargne", layout="wide")

# --------------------------------------------------
# Hypothèses du démonstrateur (illustratives)
# --------------------------------------------------
HORIZON_ANNEES = 10
MOIS = HORIZON_ANNEES * 12
VERSEMENT_MENSUEL = 150  # € par mois

RENDEMENTS = {
    "Sécurisé": 0.020,    # 2,0 % / an
    "Équilibré": 0.050,   # 5,0 % / an
    "Dynamique": 0.080    # 8,0 % / an
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

def build_chart(risque, versement_mensuel):
    x = np.arange(MOIS + 1) / 12

    # Total (capital estimé)
    total = projection_epargne(versement_mensuel, RENDEMENTS[risque], MOIS)

    # Versements cumulés (sans performance)
    versements = np.arange(MOIS + 1) * versement_mensuel

    # Gain estimé
    gain = total - versements

    fig = go.Figure()

    # 1) Zone Versements cumulés
    fig.add_trace(
        go.Scatter(
            x=x, y=versements,
            mode="lines",
            name="Versements cumulés",
            line=dict(color="rgba(107,114,128,1)", width=2),
            fill="tozeroy",
            fillcolor="rgba(107,114,128,0.15)",
            hovertemplate="Années: %{x:.1f}<br>Versements: %{y:,.0f} €<extra></extra>".replace(",", " ")
        )
    )

    # 2) Zone Gain estimé (fill entre versements et total)
    fig.add_trace(
        go.Scatter(
            x=x, y=total,
            mode="lines",
            name="Gain estimé",
            line=dict(color="rgba(16,185,129,0)", width=0),  # ligne invisible
            fill="tonexty",
            fillcolor="rgba(16,185,129,0.18)",
            hovertemplate="Années: %{x:.1f}<br>Gain estimé: %{customdata:,.0f} €<extra></extra>".replace(",", " "),
            customdata=gain
        )
    )

    # 3) Ligne Capital estimé (au-dessus)
    fig.add_trace(
        go.Scatter(
            x=x, y=total,
            mode="lines",
            name="Capital estimé",
            line=dict(color="#2563EB", width=4),
            hovertemplate=(
                "Années: %{x:.1f}<br>"
                "Capital: %{y:,.0f} €<br>"
                "Versements: %{customdata[0]:,.0f} €<br>"
                "Gain estimé: %{customdata[1]:,.0f} €"
                "<extra></extra>"
            ).replace(",", " "),
            customdata=np.column_stack([versements, gain])
        )
    )

    fig.update_layout(
        title=f"Évolution projetée — Profil {risque} (illustratif)",
        xaxis_title="Années",
        yaxis_title="Montant (€)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=70, r=20, l=60, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    fig.update_yaxes(tickformat=",.0f", ticksuffix=" €")
    fig.update_xaxes(dtick=1)

    fig.add_annotation(
        text="Hypothèses illustratives – aucune garantie – démonstrateur non contractuel",
        x=0, y=-0.22, xref="paper", yref="paper",
        xanchor="left", showarrow=False,
        font=dict(size=11, color="#6B7280")
    )

    return fig

def render_chat():
    st.subheader("Conversation")

    # -----------------------------
    # Connaissance déjà disponible (fictif)
    # -----------------------------
    KNOWN_PROFILE = {
        "prenom": "Pablo",
        "age": 28,
        "revenu_net_mensuel": 2400,
        "livret_a_plein": True,
        "livret_a_montant": 22950
    }

    # -----------------------------
    # Mémoire de conversation    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
        {"role": "assistant", "content": (
            "Bonjour Pablo 👋\n\n"
            "Je suis **Francis**, le robot de **Banque Populaire**.\n"
            "Je suis là pour t’accompagner pas à pas dans la préparation de ton épargne, en toute simplicité.\n\n"
            "Mon objectif est de t’aider à y voir clair, à comprendre tes options, "
            "et à te proposer des solutions adaptées à ta situation et à tes projets.\n\n"
            "Explique-moi ce que tu aimerais faire ou ce qui t’interroge."
       )}
    ]

        st.session_state.step = 0
        st.session_state.show_subscribe_cta = False
        st.session_state.client = {
            "pourquoi": None,
            "objectif": None,
            "horizon": None,
            "mensualite": None,
            "risque": None
        }

    def assistant_say(text: str):
        st.session_state.messages.append({"role": "assistant", "content": text})
    
    def assistant_type(container, text: str, delay: float = 0.06, chunk_words: int = 2):
        """Affiche le message assistant progressivement DANS container, puis l'ajoute à l'historique."""
        import re

        with container:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                parts = re.split(r'(\s+)', text)
                out = []
                word_count = 0

                for part in parts:
                    out.append(part)
                    if not part.isspace():
                        word_count += 1
                    if word_count > 0 and word_count % chunk_words == 0:
                        placeholder.markdown("".join(out))
                        time.sleep(delay)

                # Affiche le reste du message si nécessaire
                placeholder.markdown("".join(out))

        # Stocke le message complet dans l'historique
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
    # Zone scrollable : historique (style Copilot)
    # -----------------------------
    history_box = st.container(height=520, border=True)

    with history_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # --- CTA bouton projection affiché en bas du chat (UI) ---
    if st.session_state.show_subscribe_cta:
        if st.button("Afficher la projection", type="primary", key="btn_show_projection_chat"):
            st.session_state.show_projection = True
            st.rerun()

    # -----------------------------
    # Entrée utilisateur
    # -----------------------------
    
    # --- Reset discret (en bas à gauche, juste au-dessus de la saisie) ---
    left_reset, _ = st.columns([1, 10])
    with left_reset:
        if st.button("↺", key="btn_reset_demo", help="Recommencer la conversation"):
            keys_to_reset = [
                "messages", "step", "client",
                "can_show_projection", "show_projection", "risque_ui",
                "checkout_open", "checkout_step",
                "right_panel", "show_subscribe_cta"
            ]
            for k in keys_to_reset:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    user_text = st.chat_input("Tape ton message…")
    if user_text:
        # 1) Stocke le message utilisateur
        st.session_state.messages.append({"role": "user", "content": user_text})

        # 2) Affiche immédiatement le message dans l'historique (sans attendre un rerun)
        with history_box:
            with st.chat_message("user"):
                st.markdown(user_text)

        # 3) Puis continue ta logique step (et le bot peut taper dans history_box)
        step = st.session_state.step
        client = st.session_state.client

    # ... if step == 0 / elif ... etc ...
        step = st.session_state.step
        client = st.session_state.client

        # Étape 0 : accueil + demande du projet
        if step == 0:
            assistant_type(history_box,
                "Excellente initiative. Je peux t’aider à continuer à épargner de façon simple et adaptée.\n\n"
                "D’après les informations dont je dispose, voici ta situation actuelle :\n"
                "- **Âge :** 28 ans\n"
                "- **Situation :** salarié chez Decathlon\n"
                "- **Revenu net mensuel :** 2 150 €\n"
                "- **Livret A :** 22 950 €\n\n"
                "C’est une très bonne base : ton épargne de précaution est déjà bien constituée.\n\n"
                "**Qu’est-ce que tu aimerais préparer grâce à ton épargne ?**"
            )
            st.session_state.step = 1
            st.stop()

        # Étape 1 : l'utilisateur répond son objectif/projet
        elif step == 1:
            client["objectif"] = user_text.strip()
            assistant_type(history_box, "Merci, c’est clair.\n\n**Tu penses à quel horizon, à peu près ?** (ex : 2 ans, 4–5 ans, 10 ans)")
            st.session_state.step = 2
            st.stop()

        # Étape 2 : horizon
        elif step == 2:
            client["horizon"] = user_text.strip()
            assistant_type(history_box, "Parfait.\n\n**Quel montant pourrais-tu mettre de côté chaque mois, sans te mettre en difficulté ?** (ex : 150€)")
            st.session_state.step = 3
            st.stop()

        # Étape 3 : mensualité
        elif step == 3:
            m = extract_number(user_text)
            client["mensualite"] = m if m else 150
            assistant_type(history_box,
                "Top.\n\n**Dernier point : le niveau de risque.**\n\n"
                "Si la valeur de ton épargne baisse temporairement, tu préfères plutôt :\n"
                "- **Très prudent** / sécuriser au maximum\n"
                "- **Équilibré** / accepter de petites variations\n"
                "- **Dynamique** / accepter plus de variations\n\n"
                "**Tu te situes plutôt où ?**"
            )
            st.session_state.step = 4
            st.stop()

        # Étape 4 : risque + recommandation + déclenche projection
        elif step == 4:
            r = normalize_risk(user_text) or "Équilibré"
            client["risque"] = r

            st.session_state.risque_ui = r
            st.session_state.can_show_projection = True
            st.session_state.show_projection = False

            prenom = KNOWN_PROFILE["prenom"]
            mensualite = client.get("mensualite") or 150
            horizon = client.get("horizon") or "—"
            objectif = client.get("objectif") or "—"

            reco_text = (
                " ✅ Recommandation\n\n"

                f"**Merci {prenom} !** Voilà ce que je te propose au regard de ton projet.\n\n"
                "**Synthèse**\n"
                f"- **Projet :** {objectif}\n"
                f"- **Horizon :** {horizon}\n"
                f"- **Effort d'epargne :** {mensualite} € / mois\n"
                f"- **Profil :** {client['risque']}\n\n"
                "---\n\n"
                " 🎯 Produit recommandé : **Assurance vie**\n"
                f"Une assurance vie avec un profil **{client['risque'].lower()}** (ajustable dans le temps).\n\n"
                "**Pourquoi c'est adapté :**\n"
                "1) Ton **Livret A** couvre déjà l'épargne de précaution → on le conserve pour les imprévus.\n"
                "2) Pour un horizon de plusieurs années, l'assurance vie est **flexible** (versements libres, retraits possibles).\n"
                "3) Elle permet de viser un **potentiel de rendement** supérieur à un livret, en modulant le risque.\n\n"
                "---\n\n"
                "👉 **Étape suivante :** Clique sur **Afficher la projection** pour visualiser l'évolution estimée de ton épargne.\n"
            )

            # ✅ Animation dans le history_box
            assistant_type(history_box, reco_text, delay=0.06, chunk_words=2)

            st.session_state.show_subscribe_cta = True
            st.session_state.step = 5

            # IMPORTANT : à cette étape, tu veux que le panneau de droite affiche le bouton projection
            st.rerun()

       
        # Étape 5 : après reco (ajustements / souscription)
        else:
            txt = user_text.lower()
            if "souscri" in txt or "ouvrir" in txt or "oui" in txt or "ok" in txt:
                assistant_type("Parfait 👍 Clique sur **Souscrire en ligne**.")
                st.session_state.show_subscribe_cta = True
            else:
                assistant_type(history_box,
                    "Très bien. Dis-moi ce que tu veux ajuster :\n"
                    "• le montant mensuel (ex : 200€)\n"
                    "• l’horizon (ex : 3 ans)\n"
                    "• ou le niveau de risque (sécurisé / équilibré / dynamique)\n\n"
                    "Et je mets à jour la projection."
                    )
            st.stop()

def render_right_panel():
# ----- Colonne droite : graphique
# ----- Colonne droite : projection / souscription
    st.subheader("Projection / Souscription")

    # Valeurs par défaut pour l'état
    if "can_show_projection" not in st.session_state:
        st.session_state.can_show_projection = False
    if "show_projection" not in st.session_state:
        st.session_state.show_projection = False
    if "risque_ui" not in st.session_state:
        st.session_state.risque_ui = "Équilibré"

    if "checkout_open" not in st.session_state:
        st.session_state.checkout_open = False
    if "checkout_step" not in st.session_state:
        st.session_state.checkout_step = 1

    # =========================================================
    # 1) MODE SOUSCRIPTION : remplace le graphique
    # =========================================================
    if st.session_state.checkout_open:
        st.subheader("Souscription en ligne — Démo")
        st.caption("Écran interne simulant un parcours de souscription. Aucune donnée n’est transmise.")

        # Récupérer infos “pré-remplies”
        mensu = 150
        objectif = "—"
        horizon = "—"
        if "client" in st.session_state and isinstance(st.session_state.client, dict):
            mensu = st.session_state.client.get("mensualite") or 150
            mensu = int(mensu)
            objectif = st.session_state.client.get("objectif") or "—"
            horizon = st.session_state.client.get("horizon") or "—"
        risque = st.session_state.get("risque_ui", "Équilibré")

        st.info("Démo : ceci simule le tunnel. Dans un vrai parcours : KYC, documents, signature, etc.")

        # Etape 1
        if st.session_state.checkout_step == 1:
            st.markdown("### Récapitulatif")
            st.write("**Produit :** Assurance vie")
            st.write("**Profil :**", risque)
            st.write("**Versement mensuel :**", f"{mensu} €")
            st.write("**Objectif :**", objectif)
            st.write("**Horizon :**", horizon)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Retour à la projection", key="btn_back_to_projection"):
                    st.session_state.checkout_open = False
                    st.session_state.checkout_step = 1
                    st.rerun()
            with col2:
                if st.button("✅ Continuer", type="primary", key="btn_checkout_continue"):
                    st.session_state.checkout_step = 2
                    st.rerun()

        # Etape 2
        elif st.session_state.checkout_step == 2:
            st.success("Souscription confirmée ✅ (démo)")
            st.caption("Dans un vrai parcours : validation, signature électronique, confirmation…")

            if st.button("Terminer", type="primary", key="btn_checkout_finish"):
                st.session_state.checkout_open = False
                st.session_state.checkout_step = 1
                st.rerun()

    # =========================================================
    # 2) MODE PROJECTION : affiché si checkout_open == False
    # =========================================================
    else:
        # Tant que la reco n'a pas été faite : on n'affiche pas le graphique
        if not st.session_state.can_show_projection:
            st.info("La projection s’affichera après la recommandation produit.")
        else:
            # Bouton pour afficher la projection au moment de la reco
            if not st.session_state.show_projection:
                if st.button("Afficher la projection", type="primary", key="btn_show_projection"):
                    st.session_state.show_projection = True
                    st.rerun()

            # Une fois le bouton cliqué : on affiche le risque + le graphique
            if st.session_state.show_projection:
                risque = st.selectbox(
                    "Niveau de risque",
                    ["Sécurisé", "Équilibré", "Dynamique"],
                    index=["Sécurisé", "Équilibré", "Dynamique"].index(st.session_state.risque_ui),
                    key="risque_select"
                )
                st.session_state.risque_ui = risque
                
                mensu = 150
                if "client" in st.session_state and isinstance(st.session_state.client, dict):
                    mensu = st.session_state.client.get("mensualite") or 150
                
                st.plotly_chart(build_chart(risque, mensu), use_container_width=True)

                # Bouton souscription (démo) -> bascule vers l'écran interne
                if st.button("✅ Souscrire en ligne", type="primary", key="btn_souscrire"):
                    st.session_state.checkout_open = True
                    st.session_state.checkout_step = 1
                    st.rerun()


# --------------------------------------------------
# Interface
# --------------------------------------------------
st.header("Francis - Le robot épargne de Banque Populaire")
mode_mobile = st.toggle("📱 Mode mobile", value=False)

if not mode_mobile:
    # 🖥️ MODE DESKTOP
    col_gauche, col_droite = st.columns([1, 1], gap="large")

    with col_gauche:
        render_chat()

    with col_droite:
        render_right_panel()

else:
    # 📱 MODE MOBILE
    render_chat()

    if st.session_state.get("show_projection", False):
        st.divider()
        render_right_panel()



