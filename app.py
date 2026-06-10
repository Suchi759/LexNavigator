import streamlit as st
import numpy as np
import faiss
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= GEMINI =================
genai.configure(api_key=st.secrets["AIzaSyD3mX-FAKEKEY1234567890abcdefGhIjKlMnOpQrStUvWxYz
"])
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")
st.title("⚖️ LexNavigator — Legal AI Assistant")

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
    if "index" in st.session_state:
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
