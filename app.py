import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ================= EMBEDDINGS =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="LexNavigator  ⚖️", layout="wide", page_icon="⚖️")

# ================= CINEMATIC UI =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #050816, #020409, #000000);
    color: #e5e7eb;
}

.main-header {
    text-align:center;
    font-size:3.2rem;
    font-weight:900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.glass {
    background: rgba(10,14,30,0.7);
    padding:18px;
    border-radius:16px;
    border:1px solid rgba(0,212,255,0.2);
}

.stButton>button {
    background: linear-gradient(135deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:12px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚖️ LEXNAVIGATOR </div>', unsafe_allow_html=True)

# ================= SAFE SESSION =================
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "ready" not in st.session_state:
    st.session_state.ready = False

# ================= PDF SAFE =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for i, page in enumerate(reader.pages):
            text += page.extract_text() or ""
        return text
    except:
        return ""

# ================= CHUNK =================
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

# ================= UPLOAD =================
file = st.file_uploader("📄 Upload PDF")

if file:
    text = extract_pdf(file)

    if len(text.strip()) > 50:
        chunks, index = build_index(text)

        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.ready = True

        st.success("⚡ Document Indexed Successfully")
    else:
        st.session_state.ready = False
        st.error("❌ PDF unreadable or empty")

# ================= INPUT =================
query = st.text_input("💬 Ask Legal Question")

# ================= MAIN ASK =================
if st.button("⚡ Ask AI"):

    if not st.session_state.ready:
        st.warning("⚠️ Upload valid PDF first")

    else:
        ctx = retrieve(query, st.session_state.chunks, st.session_state.index)
        context = "\n\n".join(ctx)

        prompt = f"""
You are a legal AI assistant.

Context:
{context}

Question:
{query}

Give structured legal answer.
"""

        res = model.generate_content(prompt).text
        st.markdown("### 🧠 Answer")
        st.write(res)

# ================= SUMMARY =================
if st.button("📌 Summary"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:10])
        res = model.generate_content(f"Summarize:\n{text}").text
        st.write(res)
    else:
        st.warning("Upload PDF first")

# ================= RISK =================
if st.button("⚠️ Risk Analysis"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:15])
        res = model.generate_content(f"Classify risks:\n{text}").text
        st.write(res)
    else:
        st.warning("Upload PDF first")

# ================= CHECKLIST =================
if st.button("✅ Checklist"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:15])
        res = model.generate_content(f"Create checklist:\n{text}").text
        st.write(res)
    else:
        st.warning("Upload PDF first")

# ================= COMPARE =================
file2 = st.file_uploader("📄 Upload second PDF")

if file and file2:

    t1 = extract_pdf(file)
    t2 = extract_pdf(file2)

    if len(t1) < 50 or len(t2) < 50:
        st.error("❌ One PDF is invalid")
    else:
        prompt = f"""
Compare:

OLD:
{t1[:2000]}

NEW:
{t2[:2000]}
"""
        res = model.generate_content(prompt).text
        st.markdown("### 📊 Comparison")
        st.write(res)
