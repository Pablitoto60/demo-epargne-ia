import html as html_lib
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
import time
import re
from openai import OpenAI
from pathlib import Path

def inject_devices_css():
    css = Path("assets/devices.min.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
# --------------------------------------------------
# Configuration
# --------------------------------------------------
st.set_page_config(page_title="Démo IA – Conseil Épargne", layout="wide")

KNOWN_PROFILE = {
    "prenom": "Pablo",
    "age": 28,
    "situation": "Salarié chez Decathlon",
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
            "Ton ton est rassurant, clair et bienveillant. "
            "Tu es non contractuel : ne promets jamais de rendement. "
            "Pose UNE question à la fois si une info manque. "
            "Demande les infos dans cet ordre : objectif, horizon, mensualité, puis risque. "
            "Ne donne pas de recommandation avant d'avoir toutes ces informations. "
            "Utilise le markdown : **gras** pour les titres, - pour les puces.\n\n"
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
def projection_epargne(versement_mensuel, rendement_annuel, mois):
    r_mensuel = (1 + rendement_annuel) ** (1 / 12) - 1
    capital = np.zeros(mois + 1)
    for t in range(1, mois + 1):
        capital[t] = capital[t - 1] * (1 + r_mensuel) + versement_mensuel
    return capital

def build_chart(risque, versement_mensuel):
    x = np.arange(MOIS + 1) / 12
    total = projection_epargne(versement_mensuel, RENDEMENTS[risque], MOIS)
    versements = np.arange(MOIS + 1) * versement_mensuel
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

# --------------------------------------------------
# UI Components
# --------------------------------------------------

def display_client_slots():
    """Affiche les slots du client remplis jusqu'à présent."""
    client = st.session_state.get("client", {})
    missing = get_missing_slots(client)
    
    if missing:
        st.caption(f"ℹ️ Infos manquantes: {', '.join(missing)}")
    else:
        st.caption("✅ Toutes les infos sont disponibles!")

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

def render_desktop_chat():
    """Mode desktop avec 2 colonnes."""
    col_left, col_right = st.columns([1, 1], gap="large")
    
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
            "content": f"Bonjour {KNOWN_PROFILE['prenom']} 👋\n\nJe suis **Francis**, le robot de **Banque Populaire**.\n\nJe vais te poser quelques questions pour te conseiller de manière adaptée. Commençons!\n\n**Qu'est-ce que tu aimerais préparer grâce à ton épargne ?** (ex: achat immobilier, projet de voyage, fonds d'urgence…)"
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
    
    # Affichage des messages
    history_box = st.container(height=520, border=True)
    with history_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
    
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
    
    # Bouton projection (desktop)
    if st.session_state.get("can_show_projection", False) and not st.session_state.get("show_projection", False):
        if st.button("📊 Afficher la projection", type="primary", use_container_width=True):
            st.session_state.show_projection = True
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
        try:
            assistant_reply = call_llm(st.session_state.messages, context)
        except Exception as e:
            assistant_reply = None
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
                f"✅ **Recommandation**\n\n"
                f"Basé sur ton profil, je te recommande une **Assurance vie** avec un profil **{client.get('risque', 'Équilibré').lower()}**.\n\n"
                "Clique sur **Voir la projection** pour visualiser l'évolution estimée."
            )
            st.session_state.can_show_projection = True
    
    # Détection de recommandation
    if "recommand" in assistant_reply.lower() or "assurance vie" in assistant_reply.lower():
        st.session_state.can_show_projection = True
    
    # Affiche la réponse avec effet de typing
    history_box = st.container(height=520, border=True)
    assistant_type(history_box, assistant_reply)

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
        risque = st.selectbox(
            "Profil de risque",
            ["Sécurisé", "Équilibré", "Dynamique"],
            index=["Sécurisé", "Équilibré", "Dynamique"].index(client.get("risque", "Équilibré")),
            key="risque_select"
        )
        
        mensu = client.get("mensualite", 150)
        st.plotly_chart(build_chart(risque, mensu), use_container_width=True)
        
        if st.button("✅ Souscrire en ligne", type="primary", use_container_width=True):
            st.success("Redirection vers le tunnel de souscription... (démo)")

def render_mobile_iphone():
    """Rendu mobile dans une frame iPhone avec chat HTML à l'intérieur."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Bonjour {KNOWN_PROFILE['prenom']} 👋\n\n"
                "Je suis **Francis**, le robot de **Banque Populaire**.\n\n"
                "Je vais te poser quelques questions pour te conseiller de manière adaptée. Commençons !\n\n"
                "**Qu'est-ce que tu aimerais préparer grâce à ton épargne ?** (ex: achat immobilier, projet de voyage, fonds d'urgence…)")
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

    inject_devices_css()

    def build_chat_html(messages, can_show_projection):
        escaped_messages = []
        for m in messages:
            role = m.get("role", "assistant")
            cls = "user" if role == "user" else "assistant"
            content = html_lib.escape(m.get("content", "")).replace("\n", "<br>")
            escaped_messages.append(
                f"<div class='bubble {cls}'>{content}</div>"
            )

        projection_label = "Voir la projection" if can_show_projection else "Voir la projection"
        projection_style = "button-enabled" if can_show_projection else "button-disabled"

        return f"""
        <style>
          .device-wrap {{ display:flex; justify-content:center; padding:18px 0; }}
          .phone-screen {{ height: 820px; display:flex; flex-direction:column; background:#fff; }}
          .statusbar {{ display:flex; justify-content:space-between; font-size:11px; color:#666; padding:8px 12px; border-bottom:1px solid #eee; }}
          .phone-messages {{ flex:1; overflow-y:auto; padding:12px 14px; background:#f5f7fa; }}
          .bubble {{ max-width:82%; padding:12px 14px; border-radius:20px; margin:8px 0; line-height:1.45; font-size:14px; word-break:break-word; }}
          .bubble.user {{ margin-left:auto; background:#2563eb; color:#fff; border-top-right-radius:6px; }}
          .bubble.assistant {{ margin-right:auto; background:#f3f4f6; color:#111827; border-top-left-radius:6px; }}
          .phone-footer {{ border-top:1px solid #eee; padding:10px 14px; background:#fff; flex-shrink:0; }}
          .phone-footer .hint {{ color:#6b7280; font-size:12px; margin-bottom:8px; }}
          .phone-controls {{ display:flex; gap:8px; margin-top:10px; }}
          .phone-button {{ flex:1; border-radius:14px; padding:11px 0; font-weight:700; text-align:center; font-size:13px; }}
          .button-enabled {{ background:#2563eb; color:#fff; }}
          .button-disabled {{ background:#e5e7eb; color:#9ca3af; }}
          .phone-home {{ height:36px; background:#f7f7f7; border-top:1px solid #eee; display:flex; align-items:center; justify-content:center; }}
          .home-bar {{ width:120px; height:4px; background:#000; border-radius:999px; }}
        </style>
        <div class='device-wrap'>
          <div class='device device-iphone-14-pro device-spaceblack'>
            <div class='device-frame'>
              <div class='device-screen'>
                <div class='phone-screen'>
                  <div class='statusbar'>
                    <span>9:41</span><span>BP</span><span>📶 🔋</span>
                  </div>
                  <div class='phone-messages'>
                    {''.join(escaped_messages)}
                  </div>
                  <div class='phone-footer'>
                    <div class='hint'>Tape ton message dans le champ Streamlit ci-dessous.</div>
                    <div class='phone-controls'>
                      <div class='phone-button {projection_style}'>{projection_label}</div>
                      <div class='phone-button button-enabled'>Contacter</div>
                    </div>
                  </div>
                  <div class='phone-home'><div class='home-bar'></div></div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """

    components.html(
        build_chat_html(st.session_state.messages, st.session_state.get("can_show_projection", False)),
        height=860,
        scrolling=True
    )

    st.markdown("<div style='max-width:390px;margin:18px auto 4px auto;'>", unsafe_allow_html=True)
    user_text = st.text_input("Tape ton message…", key="mobile_input")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_text:
        handle_user_input(user_text)
        st.session_state.mobile_input = ""
        st.rerun()

    can_show = st.session_state.get("can_show_projection", False)
    col1, col2 = st.columns(2, gap="small")
    with col1:
        if st.button("📊 Afficher la projection", disabled=not can_show, use_container_width=True, key="mobile_proj"):
            st.session_state.show_projection = True
            st.rerun()
    with col2:
        if st.button("📞 Contacter mon conseiller", use_container_width=True, key="mobile_advisor"):
            st.toast("Conseiller", icon="✅")

    if st.session_state.get("show_projection", False):
        st.divider()
        st.subheader("📊 Projection")
        client = st.session_state.client
        risque = st.selectbox(
            "Profil de risque",
            ["Sécurisé", "Équilibré", "Dynamique"],
            index=["Sécurisé", "Équilibré", "Dynamique"].index(client.get("risque", "Équilibré")),
            key="mobile_risque"
        )
        mensu = client.get("mensualite", 150)
        st.plotly_chart(build_chart(risque, mensu), use_container_width=True)

# ==================================================
# MAIN UI (Interface principale)
# ==================================================
st.header("Francis - Robot épargne Banque Populaire")

st.divider()

# Mode de conversation (pré-remplissage vs LLM)
use_llm = st.checkbox(
    "Activer le mode LLM",
    value=st.session_state.get("use_llm", False),
    help="Quand activé, les réponses peuvent être générées par l'IA. Sinon, le flux reste en mode conversation pré-remplie."
)
st.session_state.use_llm = use_llm

# Toggle mode mobile
mode_mobile = st.toggle("📱 Mode mobile iPhone", value=False)

if mode_mobile:
    render_mobile_iphone()
else:
    render_desktop_chat()