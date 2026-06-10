import streamlit as st
import numpy as np
import faiss
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

# ================= EMBEDDINGS =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= CINEMATIC UI =================
st.set_page_config(
    page_title="LexNavigator AI ⚖️",
    layout="wide",
    page_icon="⚖️"
)

st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: radial-gradient(circle at 20% 20%, #0b1220, #05070f 40%, #02030a);
    color: #e5e7eb;
}

/* HEADER */
.main-header {
    text-align: center;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #a78bfa, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(56,189,248,0.3);
}

/* SUBTITLE */
.sub-header {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* GLASS CARD */
.glass {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 16px;
    padding: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 25px rgba(56,189,248,0.08);
}

/* INPUT */
input, textarea {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    box-shadow: 0 0 20px rgba(139,92,246,0.3);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 30px rgba(14,165,233,0.5);
}

/* AI BOX */
.ai-box {
    background: linear-gradient(145deg, rgba(15,23,42,0.7), rgba(2,6,23,0.7));
    border-left: 3px solid #38bdf8;
    padding: 18px;
    border-radius: 12px;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-header">⚖️ LEXNAVIGATOR AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cinematic Legal Intelligence Engine • RAG Document Assistant</div>', unsafe_allow_html=True)

st.markdown("---")

# ================= PDF FUNCTIONS =================
def extract_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for i, p in enumerate(pdf.pages):
        t = p.extract_text()
        if t:
            text += f"[Page {i+1}] {t}"
    return text

def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

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

# ================= SESSION =================
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# ================= UPLOAD =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
file = st.file_uploader("📄 Upload Legal / GST / Policy Document")
st.markdown("</div>", unsafe_allow_html=True)

if file:
    text = extract_pdf(file)
    chunks, index = build_index(text)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("Document Indexed ⚡")

# ================= INPUT =================
st.markdown('<div class="glass">', unsafe_allow_html=True)

query = st.text_input("💬 Ask your legal question")
lang = st.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])

st.markdown("</div>", unsafe_allow_html=True)

# ================= ASK AI =================
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
- Legal Section References
- Risk Notes
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
    prompt = f"Summarize key legal points:\n{text}"
    st.write(model.generate_content(prompt).text)

# ================= RISK =================
if st.button("⚠️ Risk Analysis") and st.session_state.chunks:
    text = "\n".join(st.session_state.chunks[:15])
    prompt = f"Classify risk levels:\n{text}"
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
Compare documents:

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
