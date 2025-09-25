# app.py
import os
import textwrap
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
import streamlit.components.v1 as components

def login():
    """Toont login scherm en controleert wachtwoord"""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Login")
        pwd = st.text_input("Voer wachtwoord in", type="password")
        if st.button("Login"):
            if pwd == st.secrets["PASSWORD"]:
                st.session_state.logged_in = True
                st.rerun()  # herlaad app na succesvolle login
            else:
                st.error("Verkeerd wachtwoord")
        return False
    else:
        return True

# ========= Check login =========
if not login():
    st.stop()  # Stop de rest van de app als gebruiker niet ingelogd is


# ========= Basis =========
st.set_page_config(page_title="PDC Chat", page_icon="💬", layout="wide")

st.cache_data.clear()
st.cache_resource.clear()

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

APP_VERSION = "v0.3.6"

EMBED_MODEL = "text-embedding-3-small"
PRIMARY_MODEL = "gpt-5"
FALLBACK_MODEL = "gpt-4o-mini"
MAX_COMPLETION_TOKENS = 600

# ========= Default instellingen =========
compact_mode = True
top_k = 5
temperature = 0.2
score_min = 0.0

# ========= CSS =========
def inject_css(compact: bool):
    # Afmetingen afhankelijk van compact mode
    f_title  = "1.25rem" if compact else "1.4rem"
    pad_btn  = ".55rem .8rem" if compact else ".8rem 1rem"
    pad_inp  = ".55rem .75rem" if compact else ".75rem .9rem"
    pad_bub  = ".55rem .75rem" if compact else ".9rem 1rem"
    radius   = "12px" if compact else "14px"
    sub_op   = ".75" if compact else ".70"
    block_pt = "0.6rem" if compact else "1.0rem"
    block_pb = "0.8rem" if compact else "1.2rem"

    # CSS voor knoppen, headers, chat-bubbels
    st.markdown(f"""
    <style>
    .stButton>button{{padding:{pad_btn}; border-radius:12px;}}
    .stTextInput>div>div>input{{padding:{pad_inp};}}
    :root {{ --radius: {radius}; }}
    .main .block-container {{ padding-top:{block_pt}; padding-bottom:{block_pb}; }}
    .hdr .title {{ font-size:{f_title}; font-weight:700; line-height:1.15; }}
    .hdr .sub {{ opacity:{sub_op}; margin-top:.12rem; font-size:.95rem; }}
    .chat-bubble {{ padding:{pad_bub}; border-radius:var(--radius); margin:.25rem 0;
      border:1px solid rgba(140,140,160,.25); background: rgba(255,255,255,.03); }}
    .user {{ background: rgba(255,255,255,.08); }}
    .assistant {{ background: rgba(125,255,175,.08); }}
    </style>
    """, unsafe_allow_html=True)


inject_css(compact_mode)


# ========= Header =========
st.markdown("""
<div class="hdr">
  <div class="title">💬 PDC Chat</div>
  <div class="sub">PoC chatbot PDC met Pinecone + OpenAI (GPT-5 + fallback)</div>
</div>
""", unsafe_allow_html=True)

# Versie altijd zichtbaar
st.caption(f"Versie: {APP_VERSION}")
if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

# ========= Env check =========
if not OPENAI_API_KEY or not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    st.error("Zet env variabelen: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME")
    st.stop()

# ========= Clients =========
oai = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

names = [i["name"] for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in names:
    st.error(f"Index '{PINECONE_INDEX_NAME}' niet gevonden. Beschikbaar: {names}")
    st.stop()
index = pc.Index(PINECONE_INDEX_NAME)

# ========= Helpers =========
def embed(text: str) -> list[float]:
    e = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return e.data[0].embedding

def retrieve(query: str, k: int):
    vec = embed(query)
    res = index.query(vector=vec, top_k=k, include_metadata=True)
    matches = res.get("matches", []) or []
    return [m for m in matches if (m.get("score") or 0) >= score_min]

def build_context(matches):
    chunks = []
    for m in matches:
        md = m.get("metadata") or {}
        txt = md.get("text") or md.get("content") or ""
        title = md.get("title") or md.get("source") or m.get("id", "document")
        chunks.append(f"[{title}]\n{textwrap.shorten(txt, width=1000, placeholder=' …')}")
    return "\n\n---\n\n".join(chunks)

SYSTEM_PROMPT = (
    "Je bent beknopt en feitelijk. Gebruik uitsluitend de context voor harde claims. "
    "Als iets niet in de context staat, geef dat aan."
)

def answer(query: str, k: int, temperature: float) -> str:
    matches = retrieve(query, k=k)
    context = build_context(matches)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Vraag:\n{query}\n\nContext:\n{context if context else '(geen context)'}"}
    ]
    try:
        r = oai.responses.create(
            model=PRIMARY_MODEL,
            input=messages,
            temperature=temperature,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )
        return r.output[0].content[0].text
    except Exception as e_primary:
        try:
            r = oai.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                temperature=temperature
            )
            return r.choices[0].message.content
        except Exception as e_fallback:
            raise RuntimeError(f"Primary (GPT-5) en fallback faalden:\n- {e_primary}\n- {e_fallback}")

# ========= Chat UI =========
if "history" not in st.session_state:
    st.session_state.history = [("assistant", "Hoi! Waar kan ik mee helpen?")]

for role, content in st.session_state.history:
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='chat-bubble {role}'>{content}</div>", unsafe_allow_html=True)

user_msg = st.chat_input("Stel je vraag…")
if user_msg:
    st.session_state['counter'] += 1
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(f"<div class='chat-bubble user'>{user_msg}</div>", unsafe_allow_html=True)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Zoeken + nadenken…"):
            try:
                out = answer(user_msg, k=top_k, temperature=temperature)
                st.markdown(f"<div class='chat-bubble assistant'>{out}</div>", unsafe_allow_html=True)
                st.session_state.history.append(("assistant", out))
            except Exception as e:
                st.error(f"Fout tijdens beantwoorden: {e}")

# ========= Autofocus op tekstvak =========
st.markdown("""
<script>
window.onload = function() {
  const input = parent.document.querySelector('textarea[data-baseweb="textarea"]');
  if (input) { input.focus(); }
};
</script>
""", unsafe_allow_html=True)

components.html(
    f"""
        <div>some hidden container</div>
        <p>{st.session_state.counter}</p>
        <script>
            var input = window.parent.document.querySelector('textarea[data-testid="stChatInputTextArea"]');
            input.focus();
        </script>
    """,
    height=150
)