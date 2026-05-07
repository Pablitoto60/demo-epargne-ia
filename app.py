import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import re
from openai import OpenAI


# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(page_title="Démo IA – Conseil Épargne", layout="wide")

KNOWN_PROFILE = {
    "prenom": "Laura",
    "age": 28,
    "situation": "Salariée chez Decathlon",
    "revenu_net_mensuel": 2150,
    "livret_a": 22950
}

HORIZON_ANNEES = 10
MOIS = HORIZON_ANNEES * 12

RENDEMENTS = {
    "Sécurisé": 0.020,
    "Équilibré": 0.050,
    "Dynamique": 0.080
}

# --------------------------------------------------
# OpenAI Client (cached)
# --------------------------------------------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def call_llm(history, context):
    """Appelle OpenAI pour générer une réponse."""
    client = get_openai_client()
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    
    messages = [
        {"role": "system", "content": (
            "Tu es Francis, le robot conseiller épargne de Banque Populaire. "
            "Ton ton est **dynamique, actif et proactif** - tu es un cadre enthousiaste qui croit aux projets de ses clients. "
            "Tu es **descriptif** : explique pourquoi chaque question est importante et donne du contexte. "
            "Tu es non contractuel : ne promets jamais de rendement. "
            "Pose UNE question à la fois si une info manque. "
            "Demande les infos dans cet ordre : objectif, horizon, mensualité, puis risque. "
            "⚠️ **IMPORTANT** : Pose AU MOINS 4 questions avant de faire une recommandation (même si toutes les infos semblent remplies). "
            "Ne saute pas d'étapes, pose chaque question avec enthousiasme. "
            "Quand tu donnes une recommandation, structure-la en sections comme suit :\n"
            "- ✅ Recommandation\n"
            "- 📌 Pourquoi c'est adapté\n"
            "- 💡 Ce qu'il faut retenir\n"
            "- 🧭 Prochaines étapes\n"
            "N'utilise pas de titres markdown (#, ##, ###) — garde la même taille de police que le texte, seulement du gras et des emojis.\n"
            "Utilise le markdown : **gras** pour mettre en avant, - pour les puces, 💡 pour les insights.\n\n"
            "Le profil client a déjà été présenté. Ne le répète jamais. "
            "Si tu as besoin de ces données, considère-les comme déjà connues et passe à la question suivante.\n\n"
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
# Extracteurs d'info
# --------------------------------------------------
def extract_montant(text):
    """Extrait un montant (nombre) du texte."""
    m = re.search(r"(\d+)", text.replace(" ", ""))
    return int(m.group(1)) if m else None

def extract_risque(text):
    """Extrait le niveau de risque du texte."""
    t = text.lower()
    if any(w in t for w in ["sécur", "secur", "prudent", "conservat", "faible"]):
        return "Sécurisé"
    if any(w in t for w in ["dyna", "agressif", "élevé", "elevé", "fort"]):
        return "Dynamique"
    if any(w in t for w in ["équil", "equil", "moyen", "modéré", "moderate"]):
        return "Équilibré"
    return None

def get_missing_slots(client):
    """Retourne la liste des slots manquants."""
    missing = []
    if not client.get("objectif"): missing.append("objectif")
    if not client.get("horizon"): missing.append("horizon")
    if not client.get("mensualite"): missing.append("mensualite")
    if not client.get("risque"): missing.append("risque")
    return missing

def extract_info_from_user_input(user_text, client):
    """Essaye d'extraire plusieurs infos du message utilisateur."""
    # Extraction des nombres (montant mensuel)
    montant = extract_montant(user_text)
    if montant and not client.get("mensualite"):
        client["mensualite"] = montant
    
    # Extraction du risque
    risque = extract_risque(user_text)
    if risque and not client.get("risque"):
        client["risque"] = risque
    
    # Horizon simple (ex: "2 ans", "3 à 5 ans")
    if not client.get("horizon"):
        horizon_patterns = [
            r"(\d+)\s*(?:à|-)\s*(\d+)\s*ans?",
            r"(\d+)\s*ans?"
        ]
        for pattern in horizon_patterns:
            match = re.search(pattern, user_text.lower())
            if match:
                client["horizon"] = user_text[match.start():match.end()]
                break
    
    # Si aucun slot n'a été rempli, le texte entier est considéré comme objectif/réponse libre
    if not montant and not risque and not client.get("horizon") and not client.get("objectif"):
        client["objectif"] = user_text.strip()

# --------------------------------------------------
# Fonctions de calcul
# --------------------------------------------------
def projection_epargne(versement_initial, versement_mensuel, rendement_annuel, mois):
    r_mensuel = (1 + rendement_annuel) ** (1 / 12) - 1
    capital = np.zeros(mois + 1)
    capital[0] = versement_initial

    for t in range(1, mois + 1):
        capital[t] = capital[t - 1] * (1 + r_mensuel) + versement_mensuel
    return capital


def build_chart(risque, versement_initial, versement_mensuel, horizon_annees):
    mois = int(horizon_annees * 12)
    x = np.arange(mois + 1) / 12

    total = projection_epargne(versement_initial, versement_mensuel, RENDEMENTS[risque], mois)
    versements = versement_initial + np.arange(mois + 1) * versement_mensuel
    gain = total - versements

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=versements,
        mode="lines",
        name="Versements cumulés",
        line=dict(color="rgba(107,114,128,1)", width=2),
        fill="tozeroy",
        fillcolor="rgba(107,114,128,0.15)",
        hovertemplate="Années: %{x:.1f}<br>Versements: %{y:,.0f} €<extra></extra>".replace(",", " ")
    ))

    fig.add_trace(go.Scatter(
        x=x, y=total,
        mode="lines",
        name="Gain estimé",
        line=dict(color="rgba(16,185,129,0)", width=0),
        fill="tonexty",
        fillcolor="rgba(16,185,129,0.18)",
        hovertemplate="Années: %{x:.1f}<br>Gain estimé: %{customdata:,.0f} €<extra></extra>".replace(",", " "),
        customdata=gain
    ))

    fig.add_trace(go.Scatter(
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
    ))

    fig.update_layout(
        title=f"Évolution projetée — Profil {risque} (illustratif)",
        xaxis_title="Années",
        yaxis_title="Montant (€)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=60, r=10, l=55, b=55),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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


# --------------------------------------------------
# UI Components
# --------------------------------------------------

def display_client_slots():
    """Affiche les slots du client remplis jusqu'à présent."""
    # Ne rien afficher - les infos manquantes ne doivent pas s'afficher au client
    pass

def assistant_type(container, text: str, delay: float = 0.06, chunk_words: int = 2):
    """Affiche le message assistant progressivement."""
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

            placeholder.markdown("".join(out))

    st.session_state.messages.append({"role": "assistant", "content": text})


def stream_llm_response(history, context, placeholder):
    """Stream the assistant response from OpenAI and update a placeholder progressively."""
    client = get_openai_client()
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": (
            "Tu es Francis, le robot conseiller épargne de Banque Populaire. "
            "Ton ton est **dynamique, actif et proactif** - tu es un cadre enthousiaste qui croit aux projets de ses clients. "
            "Tu es **descriptif** : explique pourquoi chaque question est importante et donne du contexte. "
            "Tu es non contractuel : ne promets jamais de rendement. "
            "Pose UNE question à la fois si une info manque. "
            "Demande les infos dans cet ordre : objectif, horizon, mensualité, puis risque. "
            "⚠️ **IMPORTANT** : Pose AU MOINS 4 questions avant de faire une recommandation (même si toutes les infos semblent remplies). "
            "Ne saute pas d'étapes, pose chaque question avec enthousiasme. "
            "Quand tu donnes une recommandation, structure-la en sections comme suit :\n"
            "- ✅ Recommandation\n"
            "- 📌 Pourquoi c'est adapté\n"
            "- 💡 Ce qu'il faut retenir\n"
            "- 🧭 Prochaines étapes\n"
            "N'utilise pas de titres markdown (#, ##, ###) — garde la même taille de police que le texte, seulement du gras et des emojis.\n"
            "Utilise le markdown : **gras** pour mettre en avant, - pour les puces, 💡 pour les insights.\n\n"
            "Le profil client a déjà été présenté. Ne le répète jamais. "
            "Si tu as besoin de ces données, considère-les comme déjà connues et passe à la question suivante.\n\n"
            f"Contexte métier :\n{context}"
        )}
    ] + history

    response_text = ""
    with client.chat.completions.stream(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    ) as stream:
        for event in stream:
            if getattr(event, "type", None) == "content.delta":
                response_text += event.delta
                placeholder.markdown(response_text)
            elif getattr(event, "type", None) == "content.done":
                placeholder.markdown(response_text)

    return response_text.strip()


def render_desktop_chat():
    """Mode desktop avec 2 colonnes compactes."""
    if st.session_state.get("mode_mobile", False):
        st.subheader("💬 Conversation")
        render_chat_core()
        st.subheader("📊 Projection")
        render_projection_panel()
    else:
        col_left, col_right = st.columns([1, 1], gap="small")
        
        with col_left:
            st.subheader("💬 Conversation")
            render_chat_core()
        
        with col_right:
            st.subheader("📊 Projection")
            render_projection_panel()


def render_chat_core():
    """Cœur du chat (réutilisable desktop/mobile)."""
    # Init session
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Bonjour {KNOWN_PROFILE['prenom']} 👋\n\n"
                "Je suis **Francis**, le robot de **Banque Populaire**. "
                "Je connais déjà ton profil et je vais l’utiliser pour te conseiller au mieux.\n\n"
                "**Profil client** :\n"
                f"- Prénom : {KNOWN_PROFILE['prenom']}\n"
                f"- Âge : {KNOWN_PROFILE['age']} ans\n"
                f"- Situation : {KNOWN_PROFILE['situation']}\n"
                f"- Revenu net mensuel : {KNOWN_PROFILE['revenu_net_mensuel']} €\n"
                f"- Épargne actuelle (Livret A) : {KNOWN_PROFILE['livret_a']} €\n\n"
                "Je vais te poser quelques questions pour affiner ma recommandation et te proposer le meilleur dispositif. Commençons !\n\n"
                "**Qu'est-ce que tu aimerais préparer grâce à ton épargne ?** (ex : achat immobilier, projet de voyage, fonds d'urgence…)")
        }]
    
    if "client" not in st.session_state:
        st.session_state.client = {
            "objectif": None,
            "horizon": None,
            "mensualite": None,
            "risque": None
        }
    
    if "can_show_projection" not in st.session_state:
        st.session_state.can_show_projection = False
    
    if "show_projection" not in st.session_state:
        st.session_state.show_projection = False
    
    if "use_llm" not in st.session_state:
        st.session_state.use_llm = False

    if "generating" not in st.session_state:
        st.session_state.generating = False
    
    # Affichage des messages
    history_box = st.container(height=520, border=True)
    with history_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if st.session_state.get("generating", False):
            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    assistant_reply = stream_llm_response(
                        st.session_state.messages,
                        st.session_state.generation_context,
                        placeholder,
                    )
                except Exception as e:
                    assistant_reply = (
                        "Désolé, je n'arrive pas à générer ma réponse pour le moment."
                        " Réessaie plus tard ou bascule en mode non-LLM."
                    )
                    placeholder.markdown(assistant_reply)

            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
            st.session_state.generating = False
            if "recommand" in assistant_reply.lower() or "assurance vie" in assistant_reply.lower():
                st.session_state.can_show_projection = True

    # Affichage des slots
    display_client_slots()
    
    # Bouton reset
    if st.button("↺ Recommencer", key="btn_reset"):
        for key in ["messages", "client", "can_show_projection", "show_projection", "risque_ui", "checkout_open", "checkout_step"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Input utilisateur
    user_text = st.chat_input("Tape ton message…")
    if user_text:
        handle_user_input(user_text)
        st.rerun()
    

def handle_user_input(user_text):
    """Traite l'entrée utilisateur: extraction, LLM, état."""
    # Ajoute le message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_text})
    
    client = st.session_state.client
    
    # Extraction intelligente
    extract_info_from_user_input(user_text, client)
    
    # Détecte si toutes les infos sont remplies
    if all([client.get("objectif"), client.get("horizon"), client.get("mensualite"), client.get("risque")]):
        st.session_state.can_show_projection = True
    
    # Build context
    context = f"""
État client actuel :
- Objectif: {client.get('objectif', 'Non défini')}
- Horizon: {client.get('horizon', 'Non défini')}
- Mensualité: {client.get('mensualite', 'Non défini')} €
- Risque: {client.get('risque', 'Non défini')}

Slots manquants: {', '.join(get_missing_slots(client)) or 'aucun'}

Règles :
- Pose UNE question à la fois pour les infos manquantes.
- Si toutes les infos sont définies, recommande Assurance vie adaptée au risque.
"""
    
    # Choix du mode conversation
    if st.session_state.get("use_llm", False):
        st.session_state.generating = True
        st.session_state.generation_context = context
        return
    else:
        assistant_reply = None

    if assistant_reply is None:
        missing = get_missing_slots(client)
        if "objectif" in missing:
            assistant_reply = "Qu'est-ce que tu aimerais préparer grâce à ton épargne ?"
        elif "horizon" in missing:
            assistant_reply = "À quel horizon tu envisages cela ? (ex: 2 ans, 5 ans, 10 ans)"
        elif "mensualite" in missing:
            assistant_reply = "Quel montant peux-tu mettre de côté chaque mois ? (ex: 150€)"
        elif "risque" in missing:
            assistant_reply = "Quel est ton profil de risque ?\n- **Sécurisé** (faible risque)\n- **Équilibré** (risque modéré)\n- **Dynamique** (plus de risque)"
        else:
            assistant_reply = (
                "✅ **Recommandation personnalisée**\n\n"
                f"Je te recommande une **assurance vie** structurée autour d'un mix sécurisé et dynamique, adapté à ton âge ({KNOWN_PROFILE['age']} ans) et à ta capacité d'épargne.\n\n"
                "📌 **Pourquoi ce choix ?**\n"
                f"- Ton profil indique une situation stable en tant que {KNOWN_PROFILE['situation']}.\n"
                f"- Tu disposes déjà de {KNOWN_PROFILE['livret_a']} € sur ton Livret A, ce qui montre que tu peux te permettre un placement plus structuré.\n"
                "- Une assurance vie te permet de combiner **sécurité**, **flexibilité** et **potentiel de rendement**, tout en conservant un accès aux fonds.\n\n"
                "💡 **Ce que cela t'apporte** :\n"
                "- une protection adaptative en cas de besoin\n"
                "- un capital investi sur du long terme avec un apport mensuel maîtrisé\n"
                "- la possibilité de diversifier entre fonds en euros et unités de compte selon ton appétence au risque\n\n"
                "🧭 **Prochaines étapes** :\n"
                "- Valider ton objectif et ton horizon\n"
                "- Confirmer ton montant mensuel\n"
                "- Choisir le niveau de risque qui te convient\n\n"
                "Clique sur **Voir la projection** pour visualiser l'évolution estimée."
            )
            st.session_state.can_show_projection = True
    
    # Détection de recommandation
    if "recommand" in assistant_reply.lower() or "assurance vie" in assistant_reply.lower():
        st.session_state.can_show_projection = True
    
    # Enregistre la réponse assistant directement dans le flux de conversation
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

def render_projection_panel():
    """Panel de projection."""
    if not st.session_state.get("can_show_projection", False):
        st.info("ℹ️ Complète d'abord le questionnaire pour voir la projection.")
        return
    
    if not st.session_state.get("show_projection", False):
        if st.button("📊 Afficher la projection", type="primary", use_container_width=True):
            st.session_state.show_projection = True
            st.rerun()

    else:
        client = st.session_state.client

        # --- Valeurs "appliquées" (hypothèses validées) ---
        # Defaults + préremplissage depuis le client si dispo
        if "applied_vi" not in st.session_state:
            st.session_state.applied_vi = int(client.get("versement_initial") or 0)
        if "applied_vp" not in st.session_state:
            st.session_state.applied_vp = int(client.get("mensualite") or 150)
        if "applied_horizon" not in st.session_state:
            # horizon stocké peut être texte ("10 ans"). On prend 10 par défaut si non parsable
            st.session_state.applied_horizon = 10
        if "applied_risque" not in st.session_state:
            st.session_state.applied_risque = client.get("risque") or "Équilibré"

        # --- Layout : graphique gauche / contrôles droite ---
        left, right = st.columns([2.3, 1], gap="large")

        # ✅ Colonne droite : formulaire (VI/VP/Horizon + "carrousel" risque + Valider)
        with right:
            st.subheader("⚙️ Hypothèses")

            risques = ["Sécurisé", "Équilibré", "Dynamique"]

            # "Carrousel" : select_slider (look plus "mobile" que selectbox)
            # Alternative possible: st.radio(..., horizontal=True)
            with st.form("hypotheses_form", clear_on_submit=False):
                risque_new = st.select_slider(
                    "Profil d'investissement",
                    options=risques,
                    value=st.session_state.applied_risque
                )

                horizon_new = st.slider(
                    "Horizon (années)",
                    min_value=1,
                    max_value=30,
                    value=int(st.session_state.applied_horizon),
                    step=1
                )

                vi_new = st.number_input(
                    "VI (versement initial)",
                    min_value=0,
                    step=100,
                    value=int(st.session_state.applied_vi)
                )

                vp_new = st.number_input(
                    "VP (mensualité)",
                    min_value=0,
                    step=10,
                    value=int(st.session_state.applied_vp)
                )

                submitted = st.form_submit_button("✅ Valider ces hypothèses", use_container_width=True)

            if submitted:
                st.session_state.applied_risque = risque_new
                st.session_state.applied_horizon = horizon_new
                st.session_state.applied_vi = int(vi_new)
                st.session_state.applied_vp = int(vp_new)
                st.rerun()

        # ✅ Colonne gauche : métriques + graphique
        with left:
            risque = st.session_state.applied_risque
            horizon = st.session_state.applied_horizon
            vi = st.session_state.applied_vi
            vp = st.session_state.applied_vp

            # Chiffres clés (optionnel mais très utile)
            mois = int(horizon * 12)
            cap = projection_epargne(vi, vp, RENDEMENTS[risque], mois)[-1]
            vers = vi + vp * mois
            gain = cap - vers

            m1, m2, m3 = st.columns(3)
            m1.metric("Capital estimé", f"{cap:,.0f} €".replace(",", " "))
            m2.metric("Versements", f"{vers:,.0f} €".replace(",", " "))
            m3.metric("Gain estimé", f"{gain:,.0f} €".replace(",", " "))

            st.plotly_chart(build_chart(risque, vi, vp, horizon), use_container_width=True)

            st.divider()

            # CTA sous le graphe (Souscrire + Conseiller)
            c1, c2 = st.columns(2, gap="small")
            with c1:
                if st.button("✅ Souscrire en ligne", type="primary", use_container_width=True, key="btn_subscribe"):
                    st.success("Redirection vers le tunnel de souscription... (démo)")
            with c2:
                if st.button("📞 Contacter conseiller", use_container_width=True, key="btn_contact_advisor"):
                    st.info("Un conseiller Banque Populaire va vous recontacter. (démo)")


def render_mobile_chat():
    """Rendu unifié pour desktop et mobile (même UI)."""
    render_chat_core()

# ==================================================
# MAIN UI (Interface principale)
# ==================================================
st.markdown(
    """
    <style>
    h1 {
        margin-top: 1.4rem !important;
        margin-bottom: 0.35rem !important;
        color: #0f62fe !important;
        font-size: 2rem !important;
    }
    .main .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    .stButton button {
        margin: 0 !important;
        padding: 0.45rem 0.8rem !important;
        font-size: 0.78rem !important;
        min-width: 100% !important;
        white-space: nowrap !important;
        border-radius: 999px !important;
        background-color: #e0e7ff !important;
        color: #0f172a !important;
        border: 1px solid #c7d2fe !important;
        box-shadow: none !important;
    }
    .stButton button:active {
        background-color: #c7d2fe !important;
    }
    .stButton button.active {
        background-color: #0f62fe !important;
        color: #fff !important;
        border-color: #0f62fe !important;
    }
    .stButton {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    .stMetric .stMetricValue {
        font-size: 1.2rem !important;
        line-height: 1.1 !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
    }
    .stMetric .stMetricValue > div {
        white-space: normal !important;
    }
    .stCheckbox {
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
    }
    .stCheckbox label {
        font-size: 0.8rem !important;
        color: #0f172a !important;
        line-height: 1.2 !important;
    }
    .stCheckbox input {
        transform: scale(0.95) !important;
        margin-right: 0.25rem !important;
    }
    .stMarkdown p {
        margin: 0.2rem 0 !important;
        font-size: 0.9rem !important;
    }
    """,
    unsafe_allow_html=True
)

if "use_llm" not in st.session_state:
    st.session_state.use_llm = False
if "mode_mobile" not in st.session_state:
    st.session_state.mode_mobile = False

st.header("🤖 Francis - Robot épargne Banque Populaire")

st.checkbox("Activer le mode LLM", value=st.session_state.use_llm, key="use_llm_toggle")
st.session_state.use_llm = st.session_state.use_llm_toggle
st.checkbox("Mode mobile", value=st.session_state.mode_mobile, key="mode_mobile_toggle")
st.session_state.mode_mobile = st.session_state.mode_mobile_toggle

render_desktop_chat()