import os
import time

import faiss
import fitz  
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ----------------------------- Config -----------------------------
EMBED_MODEL = "gemini-embedding-001"
# If one model is busy (503) or returns an empty answer, the next one is tried
CHAT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8
BROAD_K = 16  # chunks used for summary-type questions
MAX_OCR_PAGES = 40
TIMEOUT_MS = 60_000

st.set_page_config(page_title="PDF AI Chatbot", page_icon="📚", layout="wide")

# ----------------------------- Styling -----------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 920px;}

.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    padding: 1.8rem 2.2rem; border-radius: 20px; color: white;
    margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(99,102,241,.30);
}
.hero h1 {margin: 0; font-size: 2rem; color: white; letter-spacing: -.5px;}
.hero p {margin: .4rem 0 0 0; opacity: .92; font-size: 1.02rem;}

.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 1rem 0 1.4rem 0;}
.step {
    border-radius: 16px; padding: 1.1rem 1.2rem;
    border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.07);
}
.step .num {
    display: inline-block; width: 28px; height: 28px; line-height: 28px; text-align: center;
    border-radius: 50%; font-weight: 700; color: white; margin-bottom: .5rem;
    background: linear-gradient(135deg, #6366f1, #ec4899);
}
.step h4 {margin: 0 0 .2rem 0; font-size: 1rem;}
.step p {margin: 0; font-size: .88rem; opacity: .75;}

[data-testid="stChatMessage"] {
    border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    border: 1px solid rgba(128,128,128,.18); background: rgba(128,128,128,.06);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(99,102,241,.13); border-color: rgba(99,102,241,.38);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(99,102,241,.12), rgba(236,72,153,.06));
}
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 600; width: 100%;
    border: none; color: white;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    filter: brightness(1.12); color: white; transform: translateY(-1px);
}

.stats {display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: .6rem 0 1rem 0;}
.stat {
    text-align: center; padding: .6rem .2rem; border-radius: 12px;
    background: rgba(99,102,241,.14); border: 1px solid rgba(99,102,241,.30);
}
.stat b {display: block; font-size: 1.25rem;}
.stat span {font-size: .72rem; opacity: .75; text-transform: uppercase; letter-spacing: .5px;}

.chip {
    display: inline-block; padding: 3px 11px; margin: 2px 4px 2px 0;
    border-radius: 999px; font-size: .78rem;
    background: rgba(99,102,241,.15); border: 1px solid rgba(99,102,241,.35);
}
.section-label {font-size: .85rem; opacity: .7; margin: .8rem 0 .3rem 0; font-weight: 600;}
@media (max-width: 700px) {.steps {grid-template-columns: 1fr;}}
</style>
<div class="hero">
  <h1>📚 PDF AI Chatbot</h1>
  <p>Upload your PDFs and ask anything. Get accurate answers with page references, in your own language.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------- Session state -----------------------------
for key, default in {
    "messages": [],
    "index": None,
    "chunks": [],
    "stats": {"files": 0, "pages": 0, "chunks": 0},
    "pending": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------- Helpers -----------------------------
def get_client(api_key: str):
    try:
        return genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)
        )
    except Exception:
        return genai.Client(api_key=api_key)


def retry(fn, tries=5):
    for attempt in range(tries):
        try:
            return fn()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def make_configs(system=None):
    """Try 'low thinking' first (faster), then fall back to the normal config."""
    base = dict(temperature=0.2)
    if system:
        base["system_instruction"] = system
    cfgs = []
    try:
        cfgs.append(
            types.GenerateContentConfig(
                **base, thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
    except Exception:
        pass
    cfgs.append(types.GenerateContentConfig(**base))
    return cfgs


def ocr_page(client, page) -> str:
    png = page.get_pixmap(dpi=150).tobytes("png")
    contents = [
        types.Part.from_bytes(data=png, mime_type="image/png"),
        "Transcribe ALL text in this page exactly as written, in its original "
        "language (Bangla/English/etc). Keep tables as plain text. "
        "Output only the text, nothing else.",
    ]
    last_err = None
    for model in CHAT_MODELS:
        for cfg in make_configs():
            try:
                r = client.models.generate_content(
                    model=model, contents=contents, config=cfg
                )
                if r.text:
                    return r.text
            except Exception as e:
                last_err = e
                time.sleep(1)
    if last_err:
        raise last_err
    return ""


def extract_pages(client, uploaded_file, status):
    """Return list of (file_name, page_no, text)."""
    doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
    pages, ocr_used = [], 0
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) < 30:  
            if ocr_used < MAX_OCR_PAGES:
                status.write(f"🔍 Running OCR: {uploaded_file.name}, page {i + 1}")
                try:
                    text = ocr_page(client, page).strip()
                    ocr_used += 1
                except Exception:
                    text = ""
        if text:
            pages.append((uploaded_file.name, i + 1, text))
    return pages


def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text): 
            cut = max(
                text.rfind("। ", start, end),
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
            )
            if cut > start + size * 0.5:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed(client, texts, task_type):
    vecs = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]

        def call(b=batch):
            r = client.models.embed_content(
                model=EMBED_MODEL,
                contents=b,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [e.values for e in r.embeddings]

        vecs.extend(retry(call))
        time.sleep(0.5)
    arr = np.array(vecs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def build_index(client, files, status):
    chunks, page_count = [], 0
    for f in files:
        status.write(f"📄 Reading {f.name}")
        pages = extract_pages(client, f, status)
        page_count += len(pages)
        for name, page_no, text in pages:
            for c in split_text(text):
                chunks.append({"file": name, "page": page_no, "text": c})
    if not chunks:
        return None, [], {}
    status.write(f"🧠 Creating embeddings for {len(chunks)} chunks")
    vecs = embed(client, [c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    stats = {"files": len(files), "pages": page_count, "chunks": len(chunks)}
    return index, chunks, stats


def retrieve(client, query, k=TOP_K, broad=False):
    """Normal: MMR search (relevant + diverse). Broad: chunks spread across the document."""
    index, chunks = st.session_state.index, st.session_state.chunks

    if broad:  
        n = min(BROAD_K, len(chunks))
        ids = np.unique(np.linspace(0, len(chunks) - 1, n).astype(int))
        return [chunks[i] for i in ids]

    q = embed(client, [query], "RETRIEVAL_QUERY")
    fetch = min(max(k * 3, k), len(chunks))
    scores, ids = index.search(q, fetch)
    ids = [i for i in ids[0] if i >= 0]
    if len(ids) <= k:
        return [chunks[i] for i in ids]
    cand = np.array([index.reconstruct(int(i)) for i in ids])
    rel = cand @ q[0]
    selected, remaining = [int(np.argmax(rel))], list(range(len(ids)))
    remaining.remove(selected[0])
    while len(selected) < k and remaining:
        div = (cand[remaining] @ cand[selected].T).max(axis=1)
        mmr = 0.7 * rel[remaining] - 0.3 * div
        best = remaining[int(np.argmax(mmr))]
        selected.append(best)
        remaining.remove(best)
    return [chunks[ids[i]] for i in selected]


BASE_SYSTEM = """You are a careful assistant that answers questions about the user's PDF documents.
Rules:
1. Answer ONLY using the CONTEXT given. Never invent facts.
2. If the answer is not in the context, say clearly that the PDF does not contain this information.
3. Reply in the same language the user asked in (Bangla, English, or Banglish -> reply in Bangla).
4. Be accurate and well structured. Use bullet points or steps when helpful.
5. When useful, mention the page number like (Page 3)."""

STYLE_HINT = {
    "Concise": "Keep the answer short and to the point (max ~120 words unless more is truly needed).",
    "Detailed": "Give a thorough, well-organized answer with explanations and examples from the context.",
}


def stream_answer(client, question, ctx_chunks, style):
    system = BASE_SYSTEM + "\n6. " + STYLE_HINT[style]
    context = "\n\n".join(
        f"[{c['file']} | Page {c['page']}]\n{c['text']}" for c in ctx_chunks
    )
    history = ""
    for m in st.session_state.messages[-7:-1]:
        who = "User" if m["role"] == "user" else "Assistant"
        history += f"{who}: {m['content']}\n"
    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"CHAT HISTORY:\n{history or '(none)'}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    errors = []
    for model in CHAT_MODELS:
        for cfg in make_configs(system):
            started = False
            try:
                for chunk in client.models.generate_content_stream(
                    model=model, contents=prompt, config=cfg
                ):
                    if chunk.text:
                        started = True
                        yield chunk.text
                if started:
                    return
                errors.append(f"{model}: empty answer")
            except Exception as e:
                if started:
                    raise
                errors.append(f"{model}: {str(e)[:120]}")
                time.sleep(1)
    raise RuntimeError("No model could answer right now.\n" + "\n".join(errors))


def show_sources(sources):
    if not sources:
        return
    with st.expander("📎 Sources"):
        seen = set()
        for s in sources:
            key = (s["file"], s["page"])
            if key in seen:
                continue
            seen.add(key)
            st.markdown(
                f"<span class='chip'>{s['file']} · page {s['page']}</span>",
                unsafe_allow_html=True,
            )
            st.caption(s["text"][:300] + "...")


def set_pending(text, broad=False):
    st.session_state.pending = {"text": text, "broad": broad}


def chat_as_text():
    lines = []
    for m in st.session_state.messages:
        who = "You" if m["role"] == "user" else "AI"
        lines.append(f"{who}: {m['content']}\n")
    return "\n".join(lines)


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Google API Key", type="password")

    style = st.radio("Answer style", ["Concise", "Detailed"], horizontal=True, index=1)

    st.subheader("📂 Upload PDFs")
    files = st.file_uploader(
        "Choose one or more PDF files", type="pdf", accept_multiple_files=True
    )

    if st.button("🚀 Process PDFs"):
        if not api_key:
            st.error("Please enter your API key first.")
        elif not files:
            st.error("Please upload at least one PDF.")
        else:
            client = get_client(api_key)
            with st.status("Processing your PDFs...", expanded=True) as status:
                try:
                    index, chunks, stats = build_index(client, files, status)
                    if index is None:
                        status.update(label="No readable text found", state="error")
                    else:
                        st.session_state.index = index
                        st.session_state.chunks = chunks
                        st.session_state.stats = stats
                        st.session_state.messages = []
                        status.update(
                            label="✅ Ready! Ask your questions",
                            state="complete",
                            expanded=False,
                        )
                except Exception as e:
                    status.update(label="Something went wrong", state="error")
                    st.error(f"Error: {e}")

    if st.session_state.index is not None:
        s = st.session_state.stats
        st.markdown(
            f"""<div class="stats">
            <div class="stat"><b>{s['files']}</b><span>Files</span></div>
            <div class="stat"><b>{s['pages']}</b><span>Pages</span></div>
            <div class="stat"><b>{s['chunks']}</b><span>Chunks</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    if st.session_state.messages:
        if st.button("🗑️ Clear chat"):
            st.session_state.messages = []
            st.rerun()
        st.download_button(
            "⬇️ Download chat",
            data=chat_as_text(),
            file_name="chat_history.txt",
            mime="text/plain",
        )

# ----------------------------- Main area -----------------------------
pending = st.session_state.pop("pending", None)

if st.session_state.index is None:
    st.markdown(
        """
<div class="steps">
  <div class="step"><div class="num">1</div><h4>Upload</h4>
    <p>Add one or more PDF files from the sidebar. Scanned PDFs work too.</p></div>
  <div class="step"><div class="num">2</div><h4>Process</h4>
    <p>Click <b>Process PDFs</b> so the AI can read and index your documents.</p></div>
  <div class="step"><div class="num">3</div><h4>Ask</h4>
    <p>Ask in English or Bangla and get answers with page references.</p></div>
</div>
""",
        unsafe_allow_html=True,
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_sources(m.get("sources"))

if st.session_state.index is not None and not st.session_state.messages and not pending:
    st.markdown("<div class='section-label'>⚡ Quick actions</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    c1.button(
        "📝 Summarize the document",
        on_click=set_pending,
        args=("Give a clear, well-structured summary of the whole document.", True),
    )
    c2.button(
        "🔑 Key points",
        on_click=set_pending,
        args=("List the most important key points of the document as bullet points.", True),
    )
    c3.button(
        "📊 Important numbers & dates",
        on_click=set_pending,
        args=("List all important numbers, dates, deadlines and figures mentioned, with page numbers.", True),
    )
    c4.button(
        "💡 Explain simply",
        on_click=set_pending,
        args=("Explain what this document is about in very simple words, as if to a beginner.", True),
    )

typed = st.chat_input(
    "Ask anything about your PDFs...", disabled=st.session_state.index is None
)

question, broad = None, False
if pending:
    question, broad = pending["text"], pending["broad"]
elif typed:
    question = typed

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            client = get_client(api_key)
            prev = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
            search_q = (prev[-1] + " " + question) if prev and len(question) < 60 and not broad else question

            with st.spinner("🔎 Searching your documents..."):
                ctx = retrieve(client, search_q, broad=broad)

            with st.spinner("✍️ Writing the answer..."):
                answer = st.write_stream(stream_answer(client, question, ctx, style))

            show_sources(ctx)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": ctx}
            )
            if broad:
                st.rerun()  
        except Exception as e:
            st.error(f"Error: {e}")