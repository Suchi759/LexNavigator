# ================= INSTALL (Colab only) =================
# !pip install streamlit faiss-cpu sentence-transformers PyPDF2 google-generativeai numpy

import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ================= EMBEDDING MODEL =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

# ================= CINEMATIC UI =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #050816, #020409, #000000);
    color: #e5e7eb;
}

.title {
    text-align:center;
    font-size:3rem;
    font-weight:900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.card {
    background: rgba(10,14,30,0.7);
    padding:15px;
    border-radius:12px;
    border:1px solid rgba(0,212,255,0.2);
    margin-bottom:10px;
}

.stButton>button {
    background: linear-gradient(135deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:10px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">⚖️ LEXNAVIGATOR AI</div>', unsafe_allow_html=True)

# ================= SESSION STATE =================
if "text" not in st.session_state:
    st.session_state.text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "ready" not in st.session_state:
    st.session_state.ready = False

# ================= SAFE PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""

# ================= CHUNK =================
def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]

# ================= BUILD INDEX =================
def build_index(text):
    chunks = chunk_text(text)
    emb = embedder.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    return chunks, index

# ================= RETRIEVE =================
def retrieve(query):
    qv = embedder.encode([query]).astype("float32")
    _, I = st.session_state.index.search(qv, 4)
    return [st.session_state.chunks[i] for i in I[0]]

# ================= UPLOAD =================
file = st.file_uploader("📄 Upload Legal / GST / Contract PDF")

if file:
    text = extract_pdf(file)

    if len(text.strip()) > 50:
        chunks, index = build_index(text)

        st.session_state.text = text
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.ready = True

        st.success("⚡ Document Indexed Successfully")
    else:
        st.error("❌ Invalid or unreadable PDF")

# ================= QUERY =================
query = st.text_input("💬 Ask Legal Question")

# ================= ASK AI =================
if st.button("⚡ Ask AI"):

    if not st.session_state.ready:
        st.warning("⚠️ Please upload PDF first")
    else:
        ctx = retrieve(query)
        prompt = f"""
You are a legal AI assistant.

Context:
{chr(10).join(ctx)}

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
        res = model.generate_content(f"Summarize legal document:\n{text}").text
        st.write(res)
    else:
        st.warning("Upload PDF first")

# ================= RISK =================
if st.button("⚠️ Risk Analysis"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:15])
        res = model.generate_content(f"Classify risk levels:\n{text}").text
        st.write(res)
    else:
        st.warning("Upload PDF first")

# ================= CHECKLIST =================
if st.button("✅ Compliance Checklist"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:15])
        res = model.generate_content(f"Create GST compliance checklist:\n{text}").text
        st.write(res)
    else:
        st.warning("Upload PDF first")

# ================= COMPARE =================
file2 = st.file_uploader("📄 Upload Second PDF (Compare)")

if st.session_state.ready and file2:

    try:
        reader2 = PdfReader(file2)
        text2 = ""
        for page in reader2.pages:
            text2 += page.extract_text() or ""

        prompt = f"""
Compare Documents:

OLD:
{st.session_state.text[:2000]}

NEW:
{text2[:2000]}
"""

        res = model.generate_content(prompt).text

        st.markdown("### 📊 Comparison Result")
        st.write(res)

    except:
        st.error("❌ Error reading second PDF")
