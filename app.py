import streamlit as st
import numpy as np
import faiss
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= GEMINI =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= CINEMATIC UI =================
st.set_page_config(
    page_title="LexNavigator ⚖️",
    layout="wide",
    page_icon="⚖️"
)

st.markdown("""
<style>

/* 🌌 Background */
.stApp {
    background: radial-gradient(circle at top, #0b0f1a, #05060a, #02030a);
    color: #e6e6e6;
}

/* 🔵 Title */
h1 {
    text-align: center;
    color: #7dd3fc;
    font-size: 3rem;
    text-shadow: 0 0 20px #38bdf8, 0 0 40px #0ea5e9;
}

/* 🟣 Subheaders */
h2, h3 {
    color: #a78bfa;
    text-shadow: 0 0 10px #a78bfa55;
}

/* 📦 Inputs */
input, textarea {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* 🔘 Buttons */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    color: white;
    padding: 0.7rem 1.2rem;
    border-radius: 12px;
    border: none;
    font-weight: bold;
    box-shadow: 0 0 20px #3b82f6aa;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 30px #06b6d4aa;
}

/* 📄 File uploader */
section[data-testid="stFileUploader"] {
    background: #0f172a;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #1e293b;
}

/* 📊 Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1f, #020617);
}

/* ✨ Divider */
hr {
    border: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("<h1>⚖️ LEXNAVIGATOR</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>Cinematic Legal AI Assistant • Document Intelligence Engine</h3>", unsafe_allow_html=True)

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

# ================= UI =================
file = st.file_uploader("Upload GST / Policy PDF")

if file:
    text = extract_pdf(file)
    chunks, index = build_index(text)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("Document indexed ⚡")

query = st.text_input("Ask your question")
lang = st.selectbox("Language", ["English", "Hindi", "Telugu"])

# ================= ASK AI =================
if st.button("Ask AI"):
    if st.session_state.index:
        ctx = retrieve(query, st.session_state.chunks, st.session_state.index)
        context = "\n\n".join(ctx)

        prompt = f"""
You are a legal AI assistant.

Context:
{context}

Question:
{query}

Return:
- Answer
- Section reference
"""

        res = model.generate_content(prompt).text

        if lang != "English":
            res = model.generate_content(f"Translate to {lang}: {res}").text

        st.markdown("### 📌 Answer")
        st.write(res)

# ================= SUMMARY =================
if st.button("Generate Summary"):
    text = "\n".join(st.session_state.chunks[:10])

    prompt = f"""
Summarize:
- Key points
- Clauses
- Deadlines
- Penalties

{text}
"""

    st.write(model.generate_content(prompt).text)

# ================= RISK =================
if st.button("Risk Analysis"):
    text = "\n".join(st.session_state.chunks[:15])

    prompt = f"""
Classify:

🔴 High Risk
🟡 Medium Risk
🟢 Low Risk

{text}
"""

    st.write(model.generate_content(prompt).text)

# ================= CHECKLIST =================
if st.button("Compliance Checklist"):
    text = "\n".join(st.session_state.chunks[:15])

    prompt = f"""
Create GST compliance checklist:

{text}
"""

    st.write(model.generate_content(prompt).text)

# ================= COMPARE =================
file2 = st.file_uploader("Upload second document (Compare)", type=["pdf"])

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

    st.write(model.generate_content(prompt).text)