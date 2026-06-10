import streamlit as st
import numpy as np
import faiss
from PyPDF2 import PdfReader
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= UI =================
st.set_page_config(page_title="LexNavigator AI ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #050816, #020409, #000000);
    color: #e5e7eb;
}

.main {
    text-align:center;
    font-size:3rem;
    font-weight:900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.block {
    background: rgba(10,14,30,0.7);
    padding:15px;
    border-radius:12px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">⚖️ LEXNAVIGATOR AI</div>', unsafe_allow_html=True)

# ================= STATE =================
if "text" not in st.session_state:
    st.session_state.text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "ready" not in st.session_state:
    st.session_state.ready = False

# ================= PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for p in reader.pages:
            text += p.extract_text() or ""
        return text
    except:
        return ""

# ================= CHUNK =================
def chunk_text(text):
    words = text.split()
    return [" ".join(words[i:i+400]) for i in range(0, len(words), 400)]

# ================= INDEX =================
def build_index(text):
    chunks = chunk_text(text)
    emb = embedder.encode(chunks).astype("float32")

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)

    return chunks, index

def retrieve(q):
    qv = embedder.encode([q]).astype("float32")
    _, I = st.session_state.index.search(qv, 4)
    return [st.session_state.chunks[i] for i in I[0]]

# ================= UPLOAD =================
file = st.file_uploader("📄 Upload PDF")

if file:
    text = extract_pdf(file)

    if len(text.strip()) > 50:
        chunks, index = build_index(text)

        st.session_state.text = text
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.ready = True

        st.success("⚡ Document Ready")

# ================= QUERY =================
query = st.text_input("💬 Ask Question")

# ================= ASK =================
if st.button("⚡ Ask AI"):

    if not st.session_state.ready:
        st.warning("Upload PDF first")
    else:
        ctx = retrieve(query)
        prompt = f"""
Context:
{chr(10).join(ctx)}

Question:
{query}
"""

        res = model.generate_content(prompt).text
        st.write(res)

# ================= SUMMARY =================
if st.button("📌 Summary"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:10])
        st.write(model.generate_content(f"Summarize:\n{text}").text)
    else:
        st.warning("Upload PDF first")

# ================= RISK =================
if st.button("⚠️ Risk Analysis"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:15])
        st.write(model.generate_content(f"Risk classify:\n{text}").text)
    else:
        st.warning("Upload PDF first")

# ================= CHECKLIST =================
if st.button("✅ Checklist"):

    if st.session_state.ready:
        text = "\n".join(st.session_state.chunks[:15])
        st.write(model.generate_content(f"Checklist:\n{text}").text)
    else:
        st.warning("Upload PDF first")

# ================= COMPARE (FIXED STATE SAFE) =================
file2 = st.file_uploader("📄 Upload Second PDF")

if st.session_state.ready and file2 is not None:

    t2 = extract_pdf(file2)

    if len(st.session_state.text) < 50 or len(t2) < 50:
        st.error("Invalid PDF")
    else:
        prompt = f"""
Compare Documents:

OLD:
{st.session_state.text[:2000]}

NEW:
{t2[:2000]}
"""
        st.write(model.generate_content(prompt).text)
