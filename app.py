import os
import sys
import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models.llm import get_chatgroq_model
from utils.rag import build_vectorstore, retrieve_docs
from utils.web_search import web_search

st.set_page_config(
    page_title="AI ChatBot Blueprint",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


#  Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    /* ── global ── */
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    /* ── sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2330;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* ── titles ── */
    h1 { font-weight: 800; letter-spacing: -0.03em; color: #f8fafc; }
    h2 { font-weight: 700; color: #cbd5e1; }
    h3 { color: #94a3b8; }

    /* ── chat bubbles ── */
    [data-testid="stChatMessage"] {
        background: #161b27;
        border: 1px solid #1e2a3a;
        border-radius: 12px;
        margin-bottom: 8px;
        padding: 4px 8px;
    }

    /* ── input box ── */
    [data-testid="stChatInput"] textarea {
        background: #161b27 !important;
        border: 1px solid #2d3a52 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.88rem !important;
    }

    /* ── buttons ── */
    .stButton > button {
        background: #1e3a5f;
        color: #93c5fd;
        border: 1px solid #2563eb;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #2563eb;
        color: #fff;
    }

    /* ── radio pills ── */
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    /* ── info / warning boxes ── */
    .stAlert {
        background: #1a2236 !important;
        border-color: #2d3a52 !important;
        border-radius: 10px;
    }

    /* ── file uploader ── */
    [data-testid="stFileUploader"] {
        background: #161b27;
        border: 1px dashed #2d3a52;
        border-radius: 10px;
        padding: 8px;
    }

    /* ── divider ── */
    hr { border-color: #1e2330; }

    /* ── spinner text ── */
    .stSpinner p { font-family: 'JetBrains Mono', monospace; color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


#  chat function -> RAG + Web Search
def get_chat_response(chat_model, messages: list, system_prompt: str, db) -> str:
    """
    Build a response by:
      1. Retrieving relevant document chunks (RAG) if a vectorstore exists.
      2. Running a live web search when the API key is configured.
      3. Injecting both context blocks into the system prompt.
      4. Calling the LLM with the full conversation history.
    """
    try:
        user_query = messages[-1]["content"]

        # --- RAG context ---
        rag_context = retrieve_docs(user_query, db)

        # --- Web search context ---
        web_context = web_search(user_query)

        # --- Augmented system prompt ---
        augmented_system = f"""{system_prompt}

You have access to two external knowledge sources. Use them when relevant.

=== Document Context (from uploaded files) ===
{rag_context}

=== Web Search Results ===
{web_context}

Instructions:
- Prefer document context when answering questions about uploaded files.
- Use web results for current events or topics not covered in documents.
- If neither source is relevant, answer from your own knowledge.
- Always be truthful; do not hallucinate sources.
"""

        # --- Format conversation history ---
        formatted = [SystemMessage(content=augmented_system)]
        for msg in messages:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted)
        return response.content

    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

#  Sidebar — shared across pages
def render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown("## ChatBot Blueprint")
        st.divider()

        page = st.radio(
            "Navigate",
            ["💬 Chat", "📄 Instructions"],
            index=0,
            label_visibility="collapsed",
        )

        st.divider()

        # Document upload
        st.markdown("### 📂 Upload Documents")
        st.caption("Supported: PDF, TXT")
        uploaded_files = st.file_uploader(
            "Upload files for RAG",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.button("⚙️ Process Documents", use_container_width=True):
                with st.spinner("Building vector store…"):
                    try:
                        # Save uploads to temp files so loaders can read them
                        tmp_paths = []
                        for uf in uploaded_files:
                            suffix = os.path.splitext(uf.name)[-1]
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=suffix
                            ) as tmp:
                                tmp.write(uf.read())
                                tmp_paths.append(tmp.name)

                        db = build_vectorstore(tmp_paths)
                        st.session_state["db"] = db
                        st.session_state["uploaded_names"] = [f.name for f in uploaded_files]
                        st.success(f"✅ Processed {len(uploaded_files)} file(s).")
                    except Exception as e:
                        st.error(f"Failed to process documents: {e}")

        if st.session_state.get("uploaded_names"):
            st.markdown("**Loaded files:**")
            for name in st.session_state["uploaded_names"]:
                st.markdown(f"- `{name}`")

            if st.button("🗑️ Remove Documents", use_container_width=True):
                st.session_state.pop("db", None)
                st.session_state.pop("uploaded_names", None)
                st.rerun()

        st.divider()

        # ── Clear chat ──
        if page == "💬 Chat":
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

    return page

#  Chat page
def chat_page():
    st.title("🤖 AI ChatBot")
    st.caption("Powered by Groq · RAG · Live Web Search")

    # ── Response mode ──
    col1, col2 = st.columns([3, 1])
    with col2:
        mode = st.radio(
            "Response Mode",
            ["Concise", "Detailed"],
            horizontal=False,
            help="Concise: short answers. Detailed: in-depth with examples.",
        )

    if mode == "Concise":
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer every question concisely in 2-4 sentences. "
            "Be direct and avoid unnecessary elaboration."
        )
    else:
        system_prompt = (
            "You are a knowledgeable AI assistant. "
            "Provide detailed, well-structured answers with examples, context, "
            "and step-by-step explanations where appropriate. "
            "Use bullet points and headers to organise longer responses."
        )

    # ── Load model (cached in session to avoid re-init) ──
    if "chat_model" not in st.session_state:
        try:
            st.session_state["chat_model"] = get_chatgroq_model()
        except Exception as e:
            st.error(f"❌ Could not initialise LLM: {e}")
            st.info("Check the **Instructions** page to set up your API keys.")
            return

    chat_model = st.session_state["chat_model"]
    db = st.session_state.get("db", None)

    # ── Initialise history ──
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # ── RAG status indicator ──
    with col1:
        if db:
            files = st.session_state.get("uploaded_names", [])
            st.success(f"RAG active — {len(files)} document(s) loaded")
        else:
            st.info("No documents loaded. Upload files in the sidebar to enable RAG.")

    st.divider()

    # ── Display history ──
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat input ──
    if prompt := st.chat_input("Ask me anything…"):
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = get_chat_response(
                    chat_model,
                    st.session_state["messages"],
                    system_prompt,
                    db,
                )
            st.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})


#  Instructions page
def instructions_page():
    st.title("Setup & Instructions")
    st.caption("Everything you need to get started.")

    st.markdown("""
## Installation

```bash
pip install -r requirements.txt
```

---

## API Key Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
SERPER_API_KEY=your_serper_key_here
```

| Provider | Where to get the key |
|---|---|
| **Groq** | [console.groq.com/keys](https://console.groq.com/keys) |
| **Serper** (web search) | [serper.dev](https://serper.dev) |

---

## Available Models (Groq)

| Model | Speed | Best for |
|---|---|---|
| `llama-3.1-8b-instant` | Fastest | Quick answers, low latency |
| `llama-3.1-70b-versatile` | Smart | Complex reasoning |
| `mixtral-8x7b-32768` | Balanced | Long contexts |

Change the model in `config/config.py` → `MODEL_NAME`.

---

## Features

### RAG — Document Question Answering
1. Upload **PDF** or **TXT** files via the sidebar.
2. Click **Process Documents**.
3. Ask questions — the bot will retrieve relevant excerpts automatically.

### Live Web Search
- Active automatically when `SERPER_API_KEY` is set.
- The bot uses web results for current events and real-time data.

### Response Modes
| Mode | Description |
|---|---|
| **Concise** | 2-4 sentence replies |
| **Detailed** | Structured, in-depth answers with examples |

---

## Project Structure

```
project/
├── config/
│   └── config.py          ← API keys & settings
├── models/
│   ├── llm.py             ← Groq LLM initialisation
│   └── embeddings.py      ← HuggingFace embeddings (RAG)
├── utils/
│   ├── agent.py           ← LangChain agent (tools)
│   ├── rag.py             ← Document loading & retrieval
│   └── web_search.py      ← Serper web search wrapper
├── app.py                 ← Main Streamlit app
├── requirements.txt
└── .env                   ← Your secrets (never commit!)
```

---

## Deploy to Streamlit Cloud

1. Push your project to GitHub (**without** `.env`).
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) → **New app**.
3. Add secrets under **Settings → Secrets**:
```toml
GROQ_API_KEY = "..."
SERPER_API_KEY = "..."
```
4. Deploy! 🎉

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set` | Add it to `.env` or Streamlit secrets |
| `Web search disabled` | Add `SERPER_API_KEY` to `.env` |
| Slow first response | Embedding model downloads on first run (~80 MB) |
| Documents not found | Click **Process Documents** after uploading |
""")


def main():
    page = render_sidebar()

    if page == "📄 Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()