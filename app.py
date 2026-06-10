import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import time

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ================= EMBEDDINGS =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="LexNavigator ⚖️",
    layout="wide",
    page_icon="⚖️"
)

# ================= 🎨 UNIQUE CINEMATIC COLOR THEME =================
st.markdown("""
<style>

/* 🌌 NEW COLOR SYSTEM (MIDNIGHT + AMBER + CYAN + SAND) */
.stApp {
    background: linear-gradient(135deg, #0a0f1c, #111827, #1b1f2a);
    color: #e5e7eb;
}

/* 🏛 HEADER */
.main-title {
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #fbbf24, #38bdf8, #a78bfa, #f97316);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 35px rgba(251,191,36,0.2);
}

/* 🧾 SIDEBAR MEMORY */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #111827);
}

/* 🧊 GLASS CARD */
.glass {
    background: rgba(17, 24, 39, 0.65);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-radius: 18px;
    padding: 18px;
    backdrop-filter: blur(12px);
}

/* 🔘 BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, #fbbf24, #38bdf8, #a78bfa);
    color: #0b0f1c;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.6rem 1.2rem;
    border: none;
    box-shadow: 0 0 18px rgba(251,191,36,0.2);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* 💬 CHAT BOX */
.ai-box {
    background: rgba(17, 24, 39, 0.85);
    border-left: 3px solid #fbbf24;
    padding: 16px;
    border-radius: 12px;
}

/* INPUT */
input, textarea {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="main-title">⚖️ LEXNAVIGATOR </div>', unsafe_allow_html=True)

# ================= SESSION MEMORY =================
if "docs" not in st.session_state:
    st.session_state.docs = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================= PDF SAFE READ =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += f"[Page {i+1}] {page.extract_text() or ''}"
        return text
    except:
        return ""

# ================= CHUNKING =================
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ================= INDEX =================
def build_index(text):
    chunks = chunk_text(text)
    emb = embedder.encode(chunks).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return chunks, index

def retrieve(query, chunks, index):
    qv = embedder.encode([query]).astype("float32")
    _, I = index.search(qv, 4)
    return [chunks[i] for i in I[0]]

# ================= SIDEBAR MEMORY =================
with st.sidebar:
    st.header("📚 Document Memory")

    for i, doc in enumerate(st.session_state.docs[-5:]):
        st.write(f"📄 {doc}")

# ================= UPLOAD =================
st.markdown('<div class="glass">', unsafe_allow_html=True)

file = st.file_uploader("📄 Upload Legal / GST / Contract PDF")

if file:
    text = extract_pdf(file)

    if len(text) > 50:
        chunks, index = build_index(text)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.docs.append(file.name)

        st.success("⚡ Document Indexed ")

st.markdown("</div>", unsafe_allow_html=True)

# ================= QUERY =================
query = st.text_input("💬 Ask your question")

# ================= STREAMING AI =================
def stream_response(text):
    placeholder = st.empty()
    output = ""

    for char in text:
        output += char
        time.sleep(0.01)
        placeholder.markdown(f"""
        <div class="ai-box">{output}▌</div>
        """, unsafe_allow_html=True)

# ================= ASK AI =================
if st.button("⚡ Ask AI"):

    if st.session_state.index:

        ctx = retrieve(query, st.session_state.chunks, st.session_state.index)
        context = "\n\n".join(ctx)

        prompt = f"""
You are a senior legal AI assistant.

Context:
{context}

Question:
{query}

Give structured legal explanation.
"""

        with st.spinner("Thinking... ⚖️"):
            res = model.generate_content(prompt).text

        st.session_state.chat_history.append((query, res))

        stream_response(res)

# ================= CHAT HISTORY =================
st.markdown("### 🧠 Conversation Memory")

for q, a in st.session_state.chat_history[-5:]:
    st.markdown(f"**🧑 You:** {q}")
    st.markdown(f"**⚖️ AI:** {a}")
    st.markdown("---")

# ================= FEATURES =================
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📌 Summary") and st.session_state.index:
        text = "\n".join(st.session_state.chunks[:10])
        res = model.generate_content(f"Summarize:\n{text}").text
        st.write(res)

with col2:
    if st.button("⚠️ Risk") and st.session_state.index:
        text = "\n".join(st.session_state.chunks[:15])
        res = model.generate_content(f"Classify risks:\n{text}").text
        st.write(res)

with col3:
    if st.button("📊 Compare") and st.session_state.index:
        st.info("Upload second file in sidebar uploader (future upgrade)")
