import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# ================= CONFIG =================
genai.configure(api_key=st.secrets["AQ.Ab8RN6IdFUKndHp-WZYGaT52rieYf2H_wG0IVKQOefD4Yb6ZwwY"])
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #050816, #020409, #000);
    color: white;
}
h1 {
    text-align:center;
    font-size: 3rem;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color: white;
    border-radius: 10px;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LexNavigator AI")

# ================= PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except:
        return ""

def chunk_text(text, size=300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ================= SIMPLE RETRIEVAL (SAFE MODE) =================
def get_relevant_chunks(chunks, query):
    if not chunks:
        return []

    q_emb = embedder.encode([query])[0]
    c_emb = embedder.encode(chunks)

    scores = np.dot(c_emb, q_emb)
    top_idx = np.argsort(scores)[-3:][::-1]

    return [chunks[i] for i in top_idx]

# ================= SESSION =================
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ================= UPLOAD =================
file = st.file_uploader("📄 Upload PDF")

if file:
    text = extract_pdf(file)
    chunks = chunk_text(text)

    st.session_state.chunks = chunks

    st.success("PDF loaded successfully ⚡")

# ================= INPUT =================
query = st.text_input("💬 Ask your question")

# ================= ASK =================
if st.button("Ask AI ⚖️"):

    if not st.session_state.chunks:
        st.warning("Please upload a PDF first")
    else:
        context = get_relevant_chunks(st.session_state.chunks, query)
        context_text = "\n\n".join(context)

        prompt = f"""
You are a legal AI assistant.

Context:
{context_text}

Question:
{query}

Give:
- Clear answer
- Legal explanation
"""

        try:
            response = model.generate_content(prompt).text
            st.markdown("### 🧠 Answer")
            st.write(response)

        except Exception as e:
            st.error(f"Gemini Error: {str(e)}")

# ================= SUMMARY =================
if st.button("📌 Summary"):
    if st.session_state.chunks:
        text = "\n".join(st.session_state.chunks[:10])
        prompt = "Summarize this legal document:\n" + text

        try:
            st.write(model.generate_content(prompt).text)
        except Exception as e:
            st.error(str(e))

# ================= RISK =================
if st.button("⚠️ Risk Analysis"):
    if st.session_state.chunks:
        text = "\n".join(st.session_state.chunks[:15])
        prompt = "Classify legal risk (High, Medium, Low):\n" + text

        try:
            st.write(model.generate_content(prompt).text)
        except Exception as e:
            st.error(str(e))

# ================= CHECKLIST =================
if st.button("✅ Checklist"):
    if st.session_state.chunks:
        text = "\n".join(st.session_state.chunks[:15])
        prompt = "Create compliance checklist:\n" + text

        try:
            st.write(model.generate_content(prompt).text)
        except Exception as e:
            st.error(str(e))
