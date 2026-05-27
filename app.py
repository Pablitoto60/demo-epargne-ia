import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import re
from catalog import PRODUCT_CATALOG
from reco_rules import recommend_products
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
    "livret_a": 13543
}

HORIZON_ANNEES = 10
MOIS = HORIZON_ANNEES * 12

RENDEMENTS = {
    "Sécurisé": 0.020,
    "Équilibré": 0.050,
    "Dynamique": 0.080
}

# Scénarios : multiplicateurs appliqués au rendement base
SCENARIO_MULTIPLIERS = {
    "Pessimiste": 0.6,
    "Base": 1.0,
    "Optimiste": 1.4
}

QUIZ_QUESTIONS = [
    {
        "id": "q1_risk_return",
        "type": "QCF",
        "title": "1/4",
        "question": "Une perspective de gain élevé implique généralement un risque de perte en capital plus important.",
        "choices": ["Vrai", "Faux"],
        "correct": "Vrai",
        "ok": "✅ Exact. Plus de potentiel implique généralement plus de variations possibles.",
        "ko": "❌ Pas tout à fait. Viser plus de rendement implique généralement d’accepter plus de risque."
    },
    {
        "id": "q2_etf",
        "type": "QCF",
        "title": "2/4",
        "question": "Un ETF est un fonds à capital garanti.",
        "choices": ["Vrai", "Faux"],
        "correct": "Faux",
        "ok": "✅ Exact. Un ETF suit un indice : sa valeur peut monter comme baisser.",
        "ko": "❌ Non. Un ETF n’est pas garanti : il réplique un indice et fluctue."
    },
    {
        "id": "q3_drawdown",
        "type": "QFD",
        "title": "3/4",
        "question": "Si ton épargne baisse de -10% sur une année, tu fais quoi ?",
        "choices": [
            "Je vends pour éviter que ça baisse davantage",
            "Je ne touche à rien et j’attends",
            "Je continue / je renforce progressivement si je peux",
        ],
        "feedback": {
            "Je vends pour éviter que ça baisse davantage": "🛡️ OK — on privilégiera stabilité et visibilité.",
            "Je ne touche à rien et j’attends": "⚖️ OK — tu sembles à l’aise avec des variations modérées.",
            "Je continue / je renforce progressivement si je peux": "🚀 OK — tu acceptes mieux la volatilité sur le long terme.",
        }
    },
    {
        "id": "q4_esg",
        "type": "ESG",
        "title": "4/4",
        "question": "Sur une échelle de 1 à 5, quelle importance accordes-tu à l’ESG (environnement/social/gouvernance) ?",
        "scale": [1, 2, 3, 4, 5],
        "feedback": {
            1: "OK — tu privilégies surtout les critères financiers.",
            2: "OK — l’ESG n’est pas prioritaire.",
            3: "OK — tu veux un équilibre.",
            4: "OK — tu veux intégrer l’ESG dans le choix.",
            5: "OK — l’ESG est très important pour toi (filtre secteurs).",
        }
    },
]
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
def extract_montant(text: str):
    t = text.lower()

    # Si l'utilisateur parle explicitement de temps, on évite de prendre ce nombre comme montant
    # Ex: "10 ans", "18 mois", "3-5 ans"
    if re.search(r"\b\d+\s*(ans|an|mois)\b", t) or re.search(r"\b\d+\s*(à|-)\s*\d+\s*ans\b", t):
        # On ne retourne pas de montant sur ce pattern (sinon "10" devient VP)
        # Un montant doit être marqué par € ou /mois etc.
        return None

    # Montant explicite en euros
    m = re.search(r"(\d[\d\s]{0,10})\s*€", t)
    if m:
        return int(m.group(1).replace(" ", ""))

    # Mensualité explicite
    m = re.search(r"(\d[\d\s]{0,10})\s*(€\s*)?(/mois|par mois|mensuel|mensuelle)", t)
    if m:
        return int(m.group(1).replace(" ", ""))

    # Sinon: on NE devine pas un montant à partir d'un simple nombre
    return None

def extract_risque(text: str):
    t = text.lower()

    # Expressions naturelles
    if any(x in t for x in ["peu de risque", "pas de risque", "faible risque", "risque faible", "prudent", "sécurisé", "securise"]):
        return "Sécurisé"

    if any(x in t for x in ["pas trop de risque", "je sais pas", "je ne sais pas", "risque modéré", "risque modere", "modéré", "modere", "équilibré", "equilibre", "moyen"]):
        return "Équilibré"

    if any(x in t for x in ["beaucoup de risque", "risque élevé", "risque eleve", "fort risque", "agressif", "dynamique", "prendre des risques"]):
        return "Dynamique"

    # Mots-clés courts (fallback)
    if any(w in t for w in ["sécur", "secur", "faible"]):
        return "Sécurisé"
    if any(w in t for w in ["équil", "equil"]):
        return "Équilibré"
    if any(w in t for w in ["dyna", "élevé", "elevé", "fort"]):
        return "Dynamique"

    return None

def get_missing_slots(client):
    missing = []
    if not client.get("objectif"):
        missing.append("objectif")
    if not client.get("horizon_annees"):
        missing.append("horizon")
    if client.get("mensualite") is None:
        missing.append("mensualite")
    if not client.get("risque"):
        missing.append("risque")
    return missing

def extract_horizon_years(text: str):
    """
    Extrait un horizon (en années) de manière robuste.
    Gère:
      - "10 ans"
      - "3 à 5 ans" / "3-5 ans" (retient le max)
      - "18 mois" (convertit en années, arrondi supérieur)
    """
    t = text.lower().replace(" ", "")

    # Plage en années : "3-5 ans" ou "3 à 5 ans"
    m = re.search(r"(\d+)(?:à|-)(\d+)ans?", t)
    if m:
        return int(m.group(2))

    # Années simples : "10 ans"
    m = re.search(r"(\d+)ans?", t)
    if m:
        return int(m.group(1))

    # Mois : "18 mois"
    m = re.search(r"(\d+)mois", t)
    if m:
        mois = int(m.group(1))
        return max(1, int(np.ceil(mois / 12)))

    return None

def extract_info_from_user_input(user_text, client):
    """Essaye d'extraire plusieurs infos du message utilisateur."""
    # Extraction des nombres (montant mensuel)
    montant = extract_montant(user_text)
    if montant and not client.get("mensualite"):
        client["mensualite"] = montant
    
    # Fallback mensualité : si l'utilisateur répond juste "200" et qu'on attend la mensualité
    if (client.get("mensualite") is None) and st.session_state.get("expected_slot") == "mensualite":
        m = re.fullmatch(r"\s*(\d{1,5})\s*", user_text)
        if m:
            vp = int(m.group(1))
            if 5 <= vp <= 10000:
                client["mensualite"] = vp
                st.session_state.applied_vp = vp

    # --- HORIZON (prioritaire pour éviter la confusion avec un montant) ---
    h = extract_horizon_years(user_text)
    if h and not client.get("horizon_annees"):
        client["horizon_annees"] = h
        client["horizon"] = f"{h} ans"  # pour affichage

    # Fallback horizon : si l'utilisateur répond juste "10" et qu'on attend l'horizon
    if (not client.get("horizon_annees")) and st.session_state.get("expected_slot") == "horizon":
        m = re.fullmatch(r"\s*(\d{1,2})\s*", user_text)
        if m:
            h2 = int(m.group(1))
            if 1 <= h2 <= 60:
                client["horizon_annees"] = h2
                client["horizon"] = f"{h2} ans"
                st.session_state.applied_horizon = h2

        # préremplissage pour ton panneau d'hypothèses si tu l'utilises
        st.session_state.applied_horizon = h
    
    # Extraction du risque
    risque = extract_risque(user_text)
    if risque and not client.get("risque"):
        client["risque"] = risque
    
    # Si aucun slot n'a été rempli, le texte entier est considéré comme objectif/réponse libre
    if (montant is None) and (risque is None) and (client.get("horizon_annees") is None) and (not client.get("objectif")):
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

    # Base return selon profil
    base_return = RENDEMENTS[risque]

    # Calcul des 3 scénarios
    scenario_returns = {
        "Prudent": base_return * SCENARIO_MULTIPLIERS["Pessimiste"],
        "Central":   base_return * SCENARIO_MULTIPLIERS["Base"],
        "Favorable":  base_return * SCENARIO_MULTIPLIERS["Optimiste"],
    }

    totals = {}
    for scen, r in scenario_returns.items():
        totals[scen] = projection_epargne(versement_initial, versement_mensuel, r, mois)

    versements = versement_initial + np.arange(mois + 1) * versement_mensuel

    # Arrondir pour éviter les décimales partout (hover + affichage)
    versements = np.round(versements, 0)
    for scen in totals:
        totals[scen] = np.round(totals[scen], 0)

    gain_central = totals["Central"] - versements

    fig = go.Figure()

    # 1) Versements cumulés (référence)
    fig.add_trace(go.Scatter(
        x=x, y=versements,
        mode="lines",
        name="Versements",
        line=dict(color="rgba(107,114,128,1)", width=2),
        fill="tozeroy",
        fillcolor="rgba(107,114,128,0.12)",
        hovertemplate="Années: %{x:.0f}<br>Versements: %{y:,.0f} €<extra></extra>".replace(",", " ")
    ))

    # 2) Enveloppe Prudent -> Favorable (effet pro)
    fig.add_trace(go.Scatter(
        x=x, y=totals["Prudent"],
        mode="lines",
        name="Capital Prudent",
        line=dict(color="rgba(239,68,68,0.0)", width=0),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=x, y=totals["Favorable"],
        mode="lines",
        name="Fourchette (Prudent-Favorable)",
        line=dict(color="rgba(16,185,129,0.0)", width=0),
        fill="tonexty",
        fillcolor="rgba(16,185,129,0.15)",
        hoverinfo="skip"
    ))

    # 3) Ligne Centrale (la trajectoire centrale)
    fig.add_trace(go.Scatter(
        x=x, y=totals["Central"],
        mode="lines",
        name="Capital (Central)",
        line=dict(color="#2563EB", width=4),
        customdata=np.column_stack([versements, gain_central]),
        hovertemplate=(
            "Années: %{x:.0f}<br>"
            "Capital: %{y:,.0f} €<br>"
            "Versements: %{customdata[0]:,.0f} €<br>"
            "Gain estimé: %{customdata[1]:,.0f} €"
            "<extra></extra>"
        ).replace(",", " ")
    ))

    # (Option) lignes Prudent/Favorable visibles aussi (si tu veux)
    fig.add_trace(go.Scatter(
        x=x, y=totals["Prudent"],
        mode="lines",
        name="Capital (Prudent)",
        showlegend=False,  # ✅ cache dans la légende
        line=dict(color="rgba(239,68,68,0.9)", width=2, dash="dot"),
        hovertemplate="Années: %{x:.0f}<br>Capital Prudent: %{y:,.0f} €<extra></extra>".replace(",", " ")
    ))
    fig.add_trace(go.Scatter(
        x=x, y=totals["Favorable"],
        mode="lines",
        name="Capital (Favorable)",
        showlegend=False,  # ✅ cache dans la légende
        line=dict(color="rgba(16,185,129,0.9)", width=2, dash="dot"),
        hovertemplate="Années: %{x:.0f}<br>Capital Favorable: %{y:,.0f} €<extra></extra>".replace(",", " ")
    ))

    # Layout clean (et suppression du titre)
    fig.update_layout(
        title_text="",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemwidth=140
        )
    )

    fig.update_yaxes(tickformat=",.0f", ticksuffix=" €", hoverformat=",.0f")
    fig.update_xaxes(dtick=1, hoverformat=".0f")

    # Mention courte uniquement
    fig.add_annotation(
        text="Hypothèses illustratives",
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

def render_quiz_gaming():
    """Affiche le quiz (4 cartes) dans la colonne droite."""
    st.subheader("🎮 Mini‑quiz")
    st.caption("4 questions rapides pour affiner le conseil.")

    total = len(QUIZ_QUESTIONS)
    step = st.session_state.get("quiz_step", 0)
    st.progress(min(step, total) / total)

    if st.session_state.get("quiz_done", False):
        st.success(f"✅ Quiz terminé — score: {st.session_state.get('quiz_score', 0)}/2")
        st.caption("La projection est maintenant disponible ✅")
        return

    q = QUIZ_QUESTIONS[step]
    st.markdown(f"**{q['title']}**")
    st.markdown(q["question"])

    # ✅ ÉTAPE 2 : si on vient de valider, on affiche l'explication et on attend "Suivant"
    if st.session_state.get("quiz_locked", False) and st.session_state.get("quiz_feedback") is not None:
        fb = st.session_state.quiz_feedback
        (st.success if fb["ok"] else st.warning)(fb["text"])

        if st.button("➡️ Suivant", use_container_width=True, key=f"next_{q['id']}"):
            st.session_state.quiz_feedback = None
            st.session_state.quiz_locked = False
            st.session_state.quiz_step += 1
            if st.session_state.quiz_step >= total:
                st.session_state.quiz_done = True
            st.rerun()

        # IMPORTANT : on stoppe ici pour ne pas ré-afficher les radios/boutons
        return

    if q["type"] == "QCF":
        choice = st.radio("Réponse", q["choices"], key=f"quiz_{q['id']}")
        if st.button("✅ Valider", use_container_width=True, key=f"btn_{q['id']}"):
            st.session_state.quiz_answers[q["id"]] = choice

            is_correct = (choice == q["correct"])
            if is_correct:
                st.session_state.quiz_score += 1

            # on stocke l'explication (texte long) et on bloque
            exp = q["ok"] if is_correct else q["ko"]
            st.session_state.quiz_feedback = {"ok": is_correct, "text": exp}
            st.session_state.quiz_locked = True
            st.rerun()


    elif q["type"] == "QFD":
        choice = st.radio("Choisis la réponse qui te ressemble le plus", q["choices"], key=f"quiz_{q['id']}")
        if st.button("✅ Valider", use_container_width=True, key=f"btn_{q['id']}"):
            st.session_state.quiz_answers[q["id"]] = choice
            exp = q["feedback"].get(choice, "OK")
            st.session_state.quiz_feedback = {"ok": True, "text": exp}  # pas de vrai/faux ici
            st.session_state.quiz_locked = True
            st.rerun()

    elif q["type"] == "ESG":
        score = st.select_slider("Ta note ESG", options=q["scale"], value=3, key=f"quiz_{q['id']}")
        if st.button("✅ Valider", use_container_width=True, key=f"btn_{q['id']}"):
            st.session_state.quiz_answers[q["id"]] = score
            exp = q["feedback"].get(score, "OK")
            st.session_state.quiz_feedback = {"ok": True, "text": exp}
            st.session_state.quiz_locked = True
            st.rerun()

    st.divider()
    if st.button("↺ Recommencer le quiz", use_container_width=True):
        st.session_state.quiz_step = 0
        st.session_state.quiz_answers = {}
        st.session_state.quiz_score = 0
        st.session_state.quiz_done = False
        st.session_state.quiz_announced = False
        st.rerun()

def render_desktop_chat():
    """Mode desktop avec 2 colonnes compactes."""
    col_left, col_right = st.columns([1, 1])

    with col_left:
        render_chat_core()

    st.caption(
        f"DEBUG | slots_complete={st.session_state.get('slots_complete')} "
        f"| quiz_done={st.session_state.get('quiz_done')} "
        f"| reco_done={st.session_state.get('reco_done')}"
    )
    st.caption(
        f"DEBUG client | objectif={bool(st.session_state.client.get('objectif'))} "
        f"| horizon_annees={st.session_state.client.get('horizon_annees')} "
        f"| horizon={st.session_state.client.get('horizon')} "
        f"| mensualite={st.session_state.client.get('mensualite')} "
        f"| risque={st.session_state.client.get('risque')}"
    )

    # 👉 MASQUAGE INTELLIGENT DE LA COLONNE DROITE
    if st.session_state.get("slots_complete", False):

        with col_right:

            # 1) QUIZ
            if not st.session_state.get("quiz_done", False):
                render_quiz_gaming()

            # 2) BOUTON RECO
            elif not st.session_state.get("reco_done", False):
                st.subheader("✅ Étape suivante")
                st.success("Quiz terminé !")

                if st.button("🧠 La recommandation de Francis", type="primary", use_container_width=True):
                    generate_and_push_recommendation()
                    st.rerun()

            # 3) PROJECTION
            else:
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
    
    if "expected_slot" not in st.session_state:
        st.session_state.expected_slot = None  # "objectif" | "horizon" | "mensualite" | "risque"
    
    if "can_show_projection" not in st.session_state:
        st.session_state.can_show_projection = False
    
    if "show_projection" not in st.session_state:
        st.session_state.show_projection = False
    
    if "use_llm" not in st.session_state:
        st.session_state.use_llm = True

    if "slots_complete" not in st.session_state:
        st.session_state.slots_complete = False

    if "reco_done" not in st.session_state:
        st.session_state.reco_done = False
    
    if "reco_prompted" not in st.session_state:
        st.session_state.reco_prompted = False

    if "generating" not in st.session_state:
        st.session_state.generating = False
    
    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = None  # dict {type, message, correct?}
    if "quiz_locked" not in st.session_state:
        st.session_state.quiz_locked = False   # bloque la question tant que l'utilisateur n'a pas cliqué "Suivant"
    
    # --- QUIZ state (Étape 1) ---
    if "quiz_step" not in st.session_state:
        st.session_state.quiz_step = 0  # 0..3
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_done" not in st.session_state:
        st.session_state.quiz_done = False
    if "quiz_announced" not in st.session_state:
        st.session_state.quiz_announced = False
        
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
        for key in [ "messages", "client", "can_show_projection", "show_projection", "risque_ui", "checkout_open", "checkout_step", "quiz_step", "quiz_answers", "quiz_score", "quiz_done", "quiz_announced"]:
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
    st.session_state.messages.append({"role": "user", "content": user_text})

    client = st.session_state.client

    # 1) Extraction
    extract_info_from_user_input(user_text, client)
    missing = get_missing_slots(client)

    # 2) Slots complets => débloque la colonne droite (quiz)
    st.session_state.slots_complete = all([
        client.get("objectif"),
        client.get("horizon_annees"),
        client.get("mensualite"),
        client.get("risque"),
    ])

    # 3) Slots complets mais quiz PAS terminé => on bloque la reco
    if st.session_state.slots_complete and not st.session_state.get("quiz_done", False):
        if not st.session_state.get("quiz_announced", False):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🎮 Super, j’ai tout ce qu’il faut. Avant la recommandation, fais le mini‑quiz à droite 👉"
            })
            st.session_state.quiz_announced = True
        return

    # 4) Quiz terminé mais reco PAS encore déclenchée (par bouton à droite) => on guide
    if st.session_state.slots_complete and st.session_state.get("quiz_done", False) and not st.session_state.get("reco_done", False):
        if not st.session_state.get("reco_prompted", False):
            st.session_state.messages.append({
                "role": "assistant",
                "content": "✅ Quiz terminé ! Clique à droite sur **La recommandation de Francis** pour que je te propose la solution 👇"
            })
            st.session_state.reco_prompted = True
        return

    
    # 5) Si slots PAS complets => on continue la conversation (LLM ou fallback)
    if not st.session_state.slots_complete:
        missing = get_missing_slots(client)

        # ✅ ÉTAPE 2 : on indique au système quel slot est attendu maintenant
        st.session_state.expected_slot = missing[0] if missing else None

        context = f"""
    État client :
    - Objectif : {client.get('objectif', 'Non défini')}
    - Horizon : {client.get('horizon', 'Non défini')} (horizon_annees={client.get('horizon_annees', 'Non défini')})
    - Mensualité : {client.get('mensualite', 'Non défini')} €
    - Risque : {client.get('risque', 'Non défini')}

    Slots manquants : {', '.join(missing) or 'aucun'}

    Règles :
    - Pose UNE question à la fois sur le prochain slot manquant.
    - Ne fais AUCUNE recommandation tant que les slots ne sont pas complets.
    """

        if st.session_state.get("use_llm", True):

            st.session_state.generating = True
            st.session_state.generation_context = context
            return

    # ✅ Mode post-recommandation : conversation libre
    if st.session_state.get("reco_done", False):
        # Contexte : produit reco + alternatives
        reco = st.session_state.get("reco") or {}
        recommended = reco.get("recommended") or {}
        alternatives = reco.get("alternatives") or []
        quiz = st.session_state.get("quiz_answers") or {}

        context = f"""
    Tu es Francis, conseiller épargne Banque Populaire.
    Nous sommes en phase post‑recommandation : l’utilisateur pose des questions libres.
    Réponds naturellement, sans guider par étapes, sans reposer les 4 questions.
    Reste factuel et pédagogique. Si l’utilisateur conteste (ex: "PER bloqué"), explique et propose une alternative adaptée.
    Ne promets jamais de rendement. Ne “ré-invente” pas de produits : utilise seulement {recommended.get('name')} et les alternatives si besoin.
    Termine par : "Hypothèses illustratives — démonstrateur non contractuel."

    Profil / contexte :
    - Projet: {client.get('objectif')}
    - Horizon: {client.get('horizon_annees')} ans
    - Mensualité: {client.get('mensualite')} €
    - Risque: {client.get('risque')}
    - Produit recommandé: {recommended.get('name')} ({recommended.get('id')})
    - Alternatives: {', '.join([p.get('name','') for p in alternatives]) or '—'}
    - Quiz: {quiz}
    """

        # Déclenche la génération LLM (stream)
        st.session_state.generating = True
        st.session_state.generation_context = context
        return

    if assistant_reply:
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    
def generate_and_push_recommendation():
    """Calcule la recommandation et l’ajoute dans le chat (messages)."""
    client = st.session_state.client

    # 1) Choix produit via règles (code choisit)
    recommended, alternatives, debug = recommend_products(
        client=client,
        project_text=client.get("objectif", ""),
        catalog=PRODUCT_CATALOG,
        top_k=3
    )

    st.session_state.reco = {
        "recommended": recommended,
        "alternatives": alternatives,
        "debug": debug
    }

    # 2) Rédaction (LLM explique) — ou fallback
    reco_context = {
        "client": {
            "objectif": client.get("objectif"),
            "horizon_annees": client.get("horizon_annees"),
            "mensualite": client.get("mensualite"),
            "risque": client.get("risque"),
        },
        "quiz": st.session_state.get("quiz_answers", {}),
        "recommended": recommended,
        "alternatives": alternatives,
    }

    try:
        text = call_llm(st.session_state.messages, f"RECO_CONTEXT={reco_context}")
    except Exception:
        # fallback si LLM indisponible
        alt_txt = ", ".join([p["name"] for p in alternatives]) if alternatives else "—"
        text = (
            f"✅ **Recommandation** : {recommended['name']}\n\n"
            f"- 📌 **Pourquoi** : cohérent avec ton horizon ({client.get('horizon_annees')} ans) et ton profil ({client.get('risque')}).\n"
            f"- 🔁 **Alternatives** : {alt_txt}\n\n"
            "🧭 Prochaine étape : clique sur **Afficher la projection**.\n\n"
            "Hypothèses illustratives — démonstrateur non contractuel."
        )

    # 3) Push dans le chat
    st.session_state.messages.append({"role": "assistant", "content": text})

    # 4) Flags pour la suite UX
    st.session_state.reco_done = True
    st.session_state.can_show_projection = True

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
            st.markdown("<div class='hyp-title'>⚙️ Hypothèses</div>", unsafe_allow_html=True)
            st.markdown("<div class='hyp-panel'>", unsafe_allow_html=True)
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

                submitted = st.form_submit_button("Valider\u00A0✅", use_container_width=True)

            if submitted:
                st.session_state.applied_risque = risque_new
                st.session_state.applied_horizon = horizon_new
                st.session_state.applied_vi = int(vi_new)
                st.session_state.applied_vp = int(vp_new)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

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
    
    .stButton button {
        margin: 0 !important;
        padding: 0.40rem 0.75rem !important;
        font-size: 0.72rem !important;
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
        font-size: 0.95rem !important;
        line-height: 1.05 !important;
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
    /* Titre "Hypothèses" custom */
    .hyp-title {
        font-size: 1.05rem !important;   /* baisse encore si tu veux (ex: 0.95rem) */
        font-weight: 700 !important;
        margin: 0 0 0.6rem 0 !important;
        letter-spacing: 0.2px;
    }
    /* Bouton "Valider" (form_submit_button) dans le panneau Hypothèses */
    .hyp-panel [data-testid="stFormSubmitButton"] button {
        font-size: 0.60rem !important;     /* ↓ police */
        padding: 0.25rem 0.50rem !important; /* ↓ hauteur/largeur */
        line-height: 1 !important;
        min-height: 28px !important;       /* ↓ hauteur */
        border-radius: 12px !important;    /* coins plus petits */
    }
    /* (Optionnel) éviter que le bouton prenne trop de largeur visuelle */
    .hyp-panel [data-testid="stFormSubmitButton"] {
    margin-top: 0.4rem !important;
    }
    /* =========================
    UNIFORMISER LA POLICE DU CHAT
    ========================= */

    /* Texte standard dans les messages */
    div[data-testid="stChatMessage"] * {
        font-size: 0.95rem !important;     /* Ajuste ici : 0.9 / 0.95 / 1.0 */
        line-height: 1.35 !important;
    }

    /* Empêcher les titres Markdown (h1/h2/h3/...) d'agrandir la police */
    div[data-testid="stChatMessage"] h1,
    div[data-testid="stChatMessage"] h2,
    div[data-testid="stChatMessage"] h3,
    div[data-testid="stChatMessage"] h4,
    div[data-testid="stChatMessage"] h5,
    div[data-testid="stChatMessage"] h6 {
        font-size: 0.95rem !important;     /* même taille que le reste */
        margin: 0.2rem 0 !important;
        font-weight: 700 !important;       /* conserve l’effet “titre” sans changer la taille */
    }

    /* Listes (puces) : même taille et marges réduites */
    div[data-testid="stChatMessage"] ul,
    div[data-testid="stChatMessage"] ol,
    div[data-testid="stChatMessage"] li {
        font-size: 0.95rem !important;
        margin: 0.15rem 0 !important;
    }

    /* Gras : conserve le gras mais sans changer la taille */
    div[data-testid="stChatMessage"] strong {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
    }

    
    /* Remonter le contenu */
    section.main .block-container {
        padding-top: 0.2rem !important;
    }
    section.main h1,
    section.main h2 {
        margin-top: 0.1rem !important;
    }
    section.main [data-testid="stHeading"] + div {
        margin-top: 0.1rem !important;
    }
    
    """,
    unsafe_allow_html=True
)

# MAIN UI
st.header("🤖 Francis – Robot épargne Banque Populaire")

render_desktop_chat()