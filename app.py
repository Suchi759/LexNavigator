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
st.set_page_config(
    page_title="LexNavigator ⚖️",
    layout="wide",
    page_icon="⚖️"
)

# ================= ULTRA CINEMATIC UI =================
st.markdown("""
<style>

/* 🌌 DARK NEON BACKGROUND */
.stApp {
    background: radial-gradient(circle at top, #050816, #020409, #000000);
    color: #e5e7eb;
}

/* ⚖️ HERO TITLE */
.main-header {
    text-align: center;
    font-size: 3.3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #a855f7, #ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 40px rgba(0,212,255,0.25);
}

/* SUBTITLE */
.sub-header {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* 🧊 GLASS CARDS */
.glass {
    background: rgba(10, 14, 30, 0.7);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 18px;
    padding: 18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 0 35px rgba(168,85,247,0.08);
}

/* 🔘 BUTTONS */
.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #a855f7, #ff3d81);
    color: white;
    border-radius: 12px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    border: none;
    box-shadow: 0 0 25px rgba(168,85,247,0.3);
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 40px rgba(0,212,255,0.4);
}

/* 📄 INPUT */
input, textarea {
    background-color: #0b1020 !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #1f2a44 !important;
}

/* 📊 SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020409, #050816);
}

/* AI RESPONSE BOX */
.ai-box {
    background: linear-gradient(145deg, rgba(15,23,42,0.8), rgba(2,6,23,0.8));
    border-left: 3px solid #00d4ff;
    padding: 18px;
    border-radius: 14px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-header">⚖️ LEXNAVIGATOR </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cinematic Legal Intelligence Engine • RAG Document Assistant</div>', unsafe_allow_html=True)

st.markdown("---")

# ================= SAFE PDF READER =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"[Page {i+1}] {page_text}"

        return text if text.strip() else "⚠️ No readable text found in PDF."

    except Exception as e:
        return f"⚠️ PDF ERROR: {str(e)}"

# ================= CHUNKING =================
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ================= FAISS INDEX =================
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

# ================= SESSION STATE =================
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# ================= UPLOAD =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
file = st.file_uploader("📄 Upload Legal / GST / Contract PDF")
st.markdown("</div>", unsafe_allow_html=True)

if file:
    text = extract_pdf(file)
    chunks, index = build_index(text)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("⚡ Document Indexed Successfully")

# ================= INPUT =================
st.markdown('<div class="glass">', unsafe_allow_html=True)

query = st.text_input("💬 Ask your legal question")
lang = st.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])

st.markdown("</div>", unsafe_allow_html=True)

# ================= AI QUERY =================
if st.button("⚡ Analyze with AI"):

    if st.session_state.index:

        ctx = retrieve(query, st.session_state.chunks, st.session_state.index)
        context = "\n\n".join(ctx)

        prompt = f"""
You are a senior legal AI assistant.

Context:
{context}

Question:
{query}

Return:
- Structured Answer
- Legal Sections
- Risk Explanation
"""

        with st.spinner("⚡ Analyzing Legal Document..."):
            res = model.generate_content(prompt).text

            if lang != "English":
                res = model.generate_content(f"Translate to {lang}: {res}").text

        st.markdown("### 🧠 AI LEGAL ANALYSIS")

        st.markdown(f"""
        <div class="ai-box">
        {res}
        </div>
        """, unsafe_allow_html=True)

# ================= SUMMARY =================
if st.button("📌 Generate Summary") and st.session_state.chunks:
    text = "\n".join(st.session_state.chunks[:10])
    prompt = f"Summarize legal document:\n{text}"
    st.write(model.generate_content(prompt).text)

# ================= RISK =================
if st.button("⚠️ Risk Analysis") and st.session_state.chunks:
    text = "\n".join(st.session_state.chunks[:15])
    prompt = f"Classify legal risk:\n{text}"
    st.write(model.generate_content(prompt).text)

# ================= CHECKLIST =================
if st.button("✅ Compliance Checklist") and st.session_state.chunks:
    text = "\n".join(st.session_state.chunks[:15])
    prompt = f"Create GST compliance checklist:\n{text}"
    st.write(model.generate_content(prompt).text)

# ================= COMPARE =================
file2 = st.file_uploader("📄 Upload second document (Compare)", type=["pdf"])

if file and file2:
    t1 = extract_pdf(file)
    t2 = extract_pdf(file2)

    prompt = f"""
Compare Documents:

OLD:
{t1[:2000]}

NEW:
{t2[:2000]}

Show:
- Changes
- New rules
- Removed clauses
"""

    st.markdown("### 📊 Document Comparison")
    st.write(model.generate_content(prompt).text)
