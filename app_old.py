import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from openai import OpenAI

# --------------------------------------------------
# OpenAI Client (cached)
# --------------------------------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def call_llm(history, context):
    """
    Appelle OpenAI pour générer une réponse.
    - history: liste de messages [{"role": "system/user/assistant", "content": "..."}]
    - context: string avec infos métier
    Retourne le texte de réponse.
    Lève une exception en cas d'erreur.
    """
    client = get_openai_client()
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    
    messages = [
        {"role": "system", "content": (
            "Tu es Francis, le robot conseiller épargne de Banque Populaire. "
            "Ton ton est rassurant, clair et bienveillant. "
            "Tu es non contractuel : ne promets jamais de rendement. "
            "Pose UNE question à la fois si une info manque. "
            "Toujours poser les questions dans l’ordre : objectif, horizon, mensualité, puis risque. "
            "Ne donne pas de recommandation avant d’avoir toutes ces informations. "
            "Utilise le markdown pour structurer tes réponses : **gras** pour les titres, - pour les puces, etc. "
            "Pour les recommandations, utilise des sections claires avec bullets et invites claires.\n\n"
            "Le profil client a déjà été présenté dans le premier message de l’assistant. Ne répète jamais ces informations dans tes réponses.\n"
            "Si tu as besoin de ces données, considère-les comme déjà connues et passe directement à la question suivante ou à la recommandation.\n\n"
            f"Contexte métier :\n{context}"
        )}
    ] + history
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=600
    )
    return response.choices[0].message.content.strip()

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

    # Debug: check if secrets are loaded
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("❌ Clé API OpenAI manquante dans secrets.toml")
    else:
        st.info("✅ Clé API OpenAI chargée")

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
            "Je suis **Francis**, le robot de **Banque Populaire**.\n\n"
            "Je connais déjà ta situation :\n"
            "- **Âge :** 28 ans\n"
            "- **Situation :** salarié chez Decathlon\n"
            "- **Revenu net mensuel :** 2 150 €\n"
            "- **Livret A :** 22 950 €\n\n"
            "Je suis là pour t’aider à préparer ton épargne. Je vais te poser quelques questions pour te conseiller de manière adaptée.\n\n"
            "Dis-moi ce que tu veux préparer, ou je commence directement par la première question."
       )}
    ]

        st.session_state.step = 0
        st.session_state.show_subscribe_cta = False
        st.session_state.profile_summary_sent = True
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
    if st.session_state.step >= 4 and not st.session_state.get("show_projection", False):
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

        # 3) Update client state based on current step
        step = st.session_state.step
        client = st.session_state.client

        if step == 0:
            client["objectif"] = user_text.strip()
            st.session_state.step = 1
        elif step == 1:
            client["horizon"] = user_text.strip()
            st.session_state.step = 2
        elif step == 2:
            m = extract_number(user_text)
            if m is not None:
                client["mensualite"] = m
                st.session_state.step = 3
            # If no number, stay in step 2 and ask again
        elif step == 3:
            r = normalize_risk(user_text)
            if r is not None:
                client["risque"] = r
                st.session_state.risque_ui = r
                st.session_state.can_show_projection = True
                st.session_state.show_projection = False  # Keep False to show the button
                st.session_state.show_subscribe_cta = True
                st.session_state.step = 4
            # If risk isn't recognized, stay in step 3 and ask again
        # For step 4, no update

        # 4) Build context for LLM
        context = f"""
État client actuel :
- Objectif: {client.get('objectif', 'Non défini')}
- Horizon: {client.get('horizon', 'Non défini')}
- Mensualité: {client.get('mensualite', 'Non défini')} €
- Risque: {client.get('risque', 'Non défini')}

État UI :
- Peut afficher projection: {st.session_state.get('can_show_projection', False)}
- Projection affichée: {st.session_state.get('show_projection', False)}
- Souscription ouverte: {st.session_state.get('checkout_open', False)}
- Synthèse déjà présentée: {st.session_state.get('profile_summary_sent', False)}

Règles :
- Si l'objectif n'est pas défini, demande-le.
- Si l'horizon n'est pas défini, demande-le.
- Si la mensualité n'est pas définie, demande-la.
- Si le niveau de risque n'est pas défini, demande-le.
- Si toutes les infos client sont définies, recommande Assurance vie adaptée au risque et invite à cliquer sur 'Afficher la projection'.
- Ne répète pas la synthèse du profil client si elle a déjà été présentée.
"""

        # 5) Call LLM with fallback
        history = st.session_state.messages
        try:
            assistant_reply = call_llm(history, context)
            st.info("🤖 Mode LLM activé")
        except Exception as e:
            st.error(f"⚠️ Mode fallback activé (erreur LLM: {str(e)})")
            # Fallback to original hardcoded responses
            step = st.session_state.step
            if step == 0:
                assistant_reply = (
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
            elif step == 1:
                assistant_reply = "**Tu penses à quel horizon, à peu près ?** (ex : 2 ans, 4–5 ans, 10 ans)"
                st.session_state.step = 2
            elif step == 2:
                assistant_reply = "**Quel montant pourrais-tu mettre de côté chaque mois, sans te mettre en difficulté ?** (ex : 150€)"
                st.session_state.step = 3
            elif step == 3:
                assistant_reply = (
                    "**Dernier point : le niveau de risque.**\n\n"
                    "Si la valeur de ton épargne baisse temporairement, tu préfères plutôt :\n"
                    "- **Très prudent** / sécuriser au maximum\n"
                    "- **Équilibré** / accepter de petites variations\n"
                    "- **Dynamique** / accepter plus de variations\n\n"
                    "**Tu te situes plutôt où ?**"
                )
                st.session_state.step = 4
            elif step == 4:
                # Use hardcoded reco
                prenom = KNOWN_PROFILE["prenom"]
                mensualite = client.get("mensualite") or 150
                horizon = client.get("horizon") or "—"
                objectif = client.get("objectif") or "—"
                assistant_reply = (
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
                st.session_state.show_projection = False
                st.session_state.can_show_projection = True
                st.session_state.show_subscribe_cta = True
                st.session_state.step = 5
            else:
                # Step 5
                txt = user_text.lower()
                if "souscri" in txt or "ouvrir" in txt or "oui" in txt or "ok" in txt:
                    assistant_reply = "Parfait 👍 Clique sur **Souscrire en ligne**."
                else:
                    assistant_reply = (
                        "Très bien. Dis-moi ce que tu veux ajuster :\n"
                        "• le montant mensuel (ex : 200€)\n"
                        "• l’horizon (ex : 3 ans)\n"
                        "• ou le niveau de risque (sécurisé / équilibré / dynamique)\n\n"
                        "Et je mets à jour la projection."
                    )

        # 6) Ensure the projection button is enabled once we have full client info
        if all([client.get("objectif"), client.get("horizon"), client.get("mensualite"), client.get("risque")]):
            st.session_state.can_show_projection = True
            st.session_state.show_subscribe_cta = True

        # 7) Display with typing effect
        assistant_type(history_box, assistant_reply)

        # 8) If the assistant proposes the projection, force the CTA visible immediately
        if "projection" in assistant_reply.lower() or "afficher la projection" in assistant_reply.lower():
            st.session_state.can_show_projection = True
            st.session_state.show_subscribe_cta = True

        # 9) Mark that the profile summary has been used once
        if not st.session_state.profile_summary_sent:
            st.session_state.profile_summary_sent = True

        # 9) Stop to prevent further execution
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

    if st.session_state.get("can_show_projection", False) or st.session_state.get("show_projection", False):
        st.divider()
        render_right_panel()



