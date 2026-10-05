import hashlib
import shutil
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from rag_engine import CHROMA_DIR, SUPPORTED_EXTENSIONS, index_document, stream_question

st.set_page_config(
    page_title="DocuIntelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ASCII-safe internal IDs. Never use display text as provider IDs.
MODEL_AZURE = "azure_gpt41_mini"
MODEL_GEMINI = "gemini_free"
MODEL_OPENROUTER = "openrouter_free"
MODEL_OPTIONS = [MODEL_AZURE, MODEL_GEMINI, MODEL_OPENROUTER]
MODEL_LABELS = {
    MODEL_AZURE: "GPT-4.1 mini · Azure",
    MODEL_GEMINI: "Gemini 2.5 Flash-Lite · Free",
    MODEL_OPENROUTER: "OpenRouter Free",
}


def init_state():
    defaults = {
        "vectorstore": None,
        "indexed_file": None,
        "chat_history": [],
        "selected_model": MODEL_AZURE,
        "uploaded_file_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Repair sessions created by the old mojibake label.
    old_to_new = {
        "ChatGPT · GPT-4.1 mini": MODEL_AZURE,
        "ChatGPT Â· GPT-4.1 mini": MODEL_AZURE,
        "GPT-4.1 mini": MODEL_AZURE,
        "Grok": MODEL_OPENROUTER,
        "grok": MODEL_OPENROUTER,
    }
    current = st.session_state.get("selected_model")
    if current in old_to_new:
        st.session_state.selected_model = old_to_new[current]
    elif current not in MODEL_OPTIONS:
        st.session_state.selected_model = MODEL_AZURE


def file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def auto_index_uploaded_file(uploaded_file) -> None:
    """Index a newly uploaded document exactly once per file content."""
    if uploaded_file is None:
        return

    current_hash = file_hash(uploaded_file)
    if current_hash == st.session_state.get("uploaded_file_hash"):
        return

    path = Path("uploads") / uploaded_file.name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(uploaded_file.getvalue())

    try:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            st.session_state.vectorstore = index_document(str(path))
        st.session_state.indexed_file = uploaded_file.name
        st.session_state.uploaded_file_hash = current_hash
        st.session_state.chat_history = []
        st.session_state.selected_model = MODEL_AZURE
        st.toast("Document indexed successfully", icon=":material/check_circle:")
    except Exception as exc:
        st.session_state.vectorstore = None
        st.session_state.indexed_file = None
        st.error(f"Indexing failed: {exc}")



def preview_text(text: str, limit: int = 420) -> str:
    clean = " ".join(text.split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def render_sources(message: dict) -> None:
    sources = message.get("sources") or []
    if not sources:
        return

    model = escape(message.get("model", "retrieved context"))
    with st.expander(f"Sources · {model}", expanded=False):
        for i, doc in enumerate(sources, 1):
            src = Path(doc.metadata.get("source", "Unknown")).name
            page = doc.metadata.get("page")
            page_text = f" · page {page + 1}" if page is not None else ""
            st.markdown(
                f"""
                <div class="source-item">
                    <div class="source-title">{i}. {escape(src)}{escape(page_text)}</div>
                    <div class="source-preview">{escape(preview_text(doc.page_content))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history() -> None:
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                st.markdown(message["content"])
                render_sources(message)


init_state()

st.markdown(
    """
<style>
:root {
    --bg: #ffffff;
    --sidebar: #f7f7f5;
    --text: #1f1f1f;
    --muted: #6f6b66;
    --border: #e7e4df;
    --hover: #efede9;
    --accent: #7a5f45;
}

#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
html, body, .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                 BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp { overflow-x: hidden; }
.block-container { max-width: 100%; padding: 0 0 8rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    width: 270px !important;
    min-width: 270px !important;
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.1rem .9rem; }
.sidebar-brand { display:flex; align-items:center; gap:.55rem; margin:.15rem 0 1.35rem; font-weight:700; }
.brand-mark {
    display:grid; place-items:center; width:30px; height:30px;
    border:1px solid #ddd8d1; border-radius:9px; background:#fff;
    color:var(--accent); font-weight:800;
}
.sidebar-label {
    margin:1rem 0 .45rem; color:#858078; font-size:.67rem;
    font-weight:750; letter-spacing:.09em; text-transform:uppercase;
}
.sidebar-rule { height:1px; margin:1rem 0; background:var(--border); }
.doc-card { margin-top:.55rem; padding:.7rem .75rem; border:1px solid var(--border); border-radius:9px; background:#fff; }
.doc-name { overflow:hidden; font-size:.82rem; font-weight:650; text-overflow:ellipsis; white-space:nowrap; }
.doc-status { display:flex; align-items:center; gap:.35rem; margin-top:.35rem; color:#637762; font-size:.74rem; }
.status-dot { width:7px; height:7px; border-radius:50%; background:#718b70; }
.sidebar-foot { color:#85817b; font-size:.72rem; line-height:1.55; }
[data-testid="stSidebar"] .stButton > button {
    min-height:2.15rem; justify-content:flex-start; border:1px solid var(--border);
    border-radius:8px; background:#fff; color:#302f2d; font-size:.83rem; box-shadow:none;
}
[data-testid="stSidebar"] .stButton > button:hover { background:var(--hover); }
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    min-height:70px; padding:.65rem; border:1px dashed #d8d3cc;
    border-radius:9px; background:rgba(255,255,255,.55);
}

/* Header */
.topbar {
    position:sticky; top:0; z-index:10; height:54px; display:flex;
    align-items:center; justify-content:center; border-bottom:1px solid var(--border);
    background:rgba(255,255,255,.96); backdrop-filter:blur(10px);
}
.topbar-inner { width:min(820px, calc(100% - 48px)); display:flex; justify-content:space-between; align-items:center; }
.top-title { overflow:hidden; font-size:.94rem; font-weight:680; text-overflow:ellipsis; white-space:nowrap; }
.top-meta { color:var(--muted); font-size:.76rem; }

/* Conversation */
.chat-shell { width:min(820px, calc(100% - 48px)); margin:0 auto; padding-top:.8rem; padding-bottom:1rem; }
[data-testid="stChatMessage"] { width:100%; padding:.65rem 0; background:transparent; border:0; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { color:var(--text); font-size:.97rem; line-height:1.68; }
[data-testid="stChatMessageContent"] { max-width:760px; }
[data-testid="stChatMessageAvatarAssistant"] { border:1px solid var(--border); background:#fff; color:var(--accent); }
[data-testid="stChatMessageAvatarUser"] { background:#f1efeb; }

/* Welcome */
.welcome { min-height:calc(100vh - 260px); display:flex; align-items:center; justify-content:center; text-align:center; }
.welcome-card { max-width:570px; margin-top:-5vh; }
.welcome h1 { margin:0 0 .65rem; font-size:clamp(2rem,4vw,2.8rem); font-weight:720; letter-spacing:-.035em; line-height:1.1; }
.welcome p { margin:0; color:var(--muted); font-size:.96rem; line-height:1.6; }
.welcome-pill { display:inline-flex; margin-top:1rem; padding:.4rem .7rem; border:1px solid var(--border); border-radius:999px; background:#fbfaf8; color:#74685e; font-size:.76rem; }

/* Sources */
.stExpander { margin-top:.55rem; border:1px solid var(--border) !important; border-radius:9px !important; box-shadow:none !important; }
.source-item { padding:.45rem 0; border-bottom:1px solid #f0eeeb; }
.source-item:last-child { border-bottom:0; }
.source-title { font-size:.81rem; font-weight:650; }
.source-preview { margin-top:.16rem; color:var(--muted); font-size:.77rem; line-height:1.5; }

/* Bottom composer */
[data-testid="stBottomBlock"] {
    background:linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,.96) 20%, #fff 48%);
    border-top:0 !important;
}
[data-testid="stBottomBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
    border:1px solid #d8d4ce !important;
    border-radius:20px !important;
    background:#fff !important;
    box-shadow:0 5px 24px rgba(0,0,0,.07) !important;
    padding:.55rem .65rem .5rem !important;
}
[data-testid="stBottomBlock"] textarea {
    border:0 !important;
    box-shadow:none !important;
    resize:none !important;
    background:transparent !important;
    padding:.45rem .5rem !important;
    font-size:.95rem !important;
    line-height:1.45 !important;
}
[data-testid="stBottomBlock"] textarea:focus { border:0 !important; box-shadow:none !important; }
.composer-hint { color:#918b84; font-size:.68rem; padding:.2rem .35rem 0; }
[data-testid="stBottomBlock"] [data-baseweb="select"] > div {
    min-height:31px; border:0 !important; border-radius:999px;
    background:#f5f3ef; color:#3e3934; font-size:.76rem; box-shadow:none;
}
[data-testid="stBottomBlock"] button { border-radius:999px; }

@media (max-width:900px) {
    [data-testid="stSidebar"] { width:240px !important; min-width:240px !important; }
    .topbar-inner, .chat-shell { width:calc(100% - 32px); }
}
@media (max-width:640px) {
    .topbar-inner, .chat-shell { width:calc(100% - 24px); }
    .welcome { min-height:calc(100vh - 235px); }
    .top-meta { display:none; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
header_title = Path(st.session_state.indexed_file).name if st.session_state.indexed_file else "DocuChat"
header_meta = MODEL_LABELS.get(st.session_state.selected_model, "GPT-4.1 mini") if st.session_state.vectorstore else "Private document Q&A"

st.markdown(
    f"""
    <div class="topbar">
        <div class="topbar-inner">
            <div class="top-title">{escape(header_title)}</div>
            <div class="top-meta">{escape(header_meta)}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand"><span class="brand-mark">▣</span><span>DocuChat</span></div>'
        '<div class="sidebar-label">Workspace</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload document",
        type=[x.lstrip(".") for x in sorted(SUPPORTED_EXTENSIONS)],
        label_visibility="collapsed",
    )

    # Uploading a new file automatically indexes it. No second button is needed.
    if uploaded is not None:
        auto_index_uploaded_file(uploaded)

    st.markdown('<div class="sidebar-label">Current document</div>', unsafe_allow_html=True)

    if st.session_state.indexed_file:
        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-name">{escape(st.session_state.indexed_file)}</div>
                <div class="doc-status"><span class="status-dot"></span><span>Indexed</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No document indexed yet.")

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

    if st.button("New chat", icon=":material/add_comment:", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Clear document", icon=":material/delete:", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.indexed_file = None
        st.session_state.chat_history = []
        st.session_state.selected_model = MODEL_AZURE
        st.session_state.uploaded_file_hash = None
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)
        st.rerun()

    st.markdown(
        '<div class="sidebar-rule"></div><div class="sidebar-label">RAG</div>'
        '<div class="sidebar-foot">Chroma · MiniLM<br>Azure · Gemini · OpenRouter</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Main conversation
# ----------------------------------------------------------------------------
st.markdown('<main class="chat-shell">', unsafe_allow_html=True)

if not st.session_state.chat_history:
    st.markdown(
        f"""
        <section class="welcome">
            <div class="welcome-card">
                <h1>What can I help you understand?</h1>
                <p>{'Ask questions about <strong>' + escape(st.session_state.indexed_file) + '</strong>.' if st.session_state.indexed_file else 'Upload a document and ask questions grounded in its content.'}</p>
                <div class="welcome-pill">▣ Private document Q&amp;A</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
else:
    render_history()

st.markdown('</main>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# ONE composer at the bottom.
# st.bottom is the supported Streamlit mechanism for a pinned bottom area.
# ----------------------------------------------------------------------------
with st.bottom:
    left, center, right = st.columns([1, 8, 1], gap="small")

    with center:
        with st.container(border=True):
            with st.form("docuchat_composer", clear_on_submit=True, border=False):
                prompt = st.text_area(
                    "Message",
                    placeholder=(
                        "Ask anything about your document..."
                        if st.session_state.vectorstore
                        else "Upload and index a document first..."
                    ),
                    height=58,
                    label_visibility="collapsed",
                    disabled=st.session_state.vectorstore is None,
                )

                c1, c2, c3 = st.columns([5.0, 3.4, .7], vertical_alignment="center")

                with c1:
                    st.markdown(
                        '<div class="composer-hint">Type your message · click ↑ to send</div>',
                        unsafe_allow_html=True,
                    )

                with c2:
                    selected_model = st.selectbox(
                        "Model",
                        MODEL_OPTIONS,
                        index=MODEL_OPTIONS.index(st.session_state.selected_model),
                        format_func=lambda value: MODEL_LABELS[value],
                        label_visibility="collapsed",
                    )

                with c3:
                    send = st.form_submit_button(
                        "↑",
                        type="primary",
                        use_container_width=True,
                        disabled=st.session_state.vectorstore is None,
                    )

# ----------------------------------------------------------------------------
# Process submitted question.
# ----------------------------------------------------------------------------
if send and prompt and st.session_state.vectorstore:
    question = prompt.strip()

    if question:
        st.session_state.selected_model = selected_model
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        try:
            stream, sources, provider_model = stream_question(
                st.session_state.vectorstore,
                question,
                selected_model,
            )

            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                answer = st.write_stream(stream).strip() or "No response was returned."
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "model": provider_model,
                }
                st.session_state.chat_history.append(assistant_message)
                render_sources(assistant_message)

        except Exception as exc:
            assistant_message = {
                "role": "assistant",
                "content": "I couldn't generate a response.\n\n" + str(exc),
                "sources": [],
            }
            st.session_state.chat_history.append(assistant_message)

            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                st.markdown(assistant_message["content"])

        st.rerun()
