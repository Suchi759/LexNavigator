# =========================
# LEXNAVIGATOR ⚖️ FULL CINEMATIC RAG APP (FIXED ONE CELL)
# =========================

!pip -q install streamlit google-generativeai faiss-cpu sentence-transformers PyPDF2 numpy pyngrok gtts SpeechRecognition pydub

import os
import threading
import time
import numpy as np
import faiss
import PyPDF2
import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pyngrok import ngrok

# =========================
# GEMINI SETUP
# =========================
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# NGROK SETUP (FIXED)
# =========================
NGROK_TOKEN = "YOUR_NGROK_AUTHTOKEN"
ngrok.set_auth_token(NGROK_TOKEN)

# Kill any old sessions
ngrok.kill()
time.sleep(2)

# Force disconnect any lingering tunnels safely
try:
    tunnels = ngrok.get_tunnels()
    for t in tunnels:
        try:
            ngrok.disconnect(t.public_url)
        except:
            pass
except:
    pass

# =========================
# STREAMLIT APP CODE
# =========================
app_code = """
import streamlit as st
import numpy as np
import faiss
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ================= UI =================
st.set_page_config(page_title="LexNavigator ⚖️", layout="wide")

st.markdown('''
<style>
body { background-color:#0b0f1a; color:#e5e7eb; }
.stApp { background: radial-gradient(circle at top, #0b0f1a, #05070d); }
h1,h2,h3 { color:#4da3ff; text-shadow:0 0 12px #4da3ff55; }
.stButton>button {
    background: linear-gradient(90deg,#4da3ff,#a855f7);
    color:white; border-radius:10px; padding:10px; border:none;
}
</style>
''', unsafe_allow_html=True)

# ================= GEMINI =================
API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= SESSION =================
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None

# ================= PDF =================
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

def retrieve(q, chunks, index):
    qv = embedder.encode([q]).astype("float32")
    _, I = index.search(qv, 4)
    return [chunks[i] for i in I[0]]

# ================= UI =================
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

if st.button("Ask AI"):
    if st.session_state.index:
        ctx = retrieve(query, st.session_state.chunks, st.session_state.index)
        context = "\n\n".join(ctx)

        prompt = f\"\"\"You are a legal AI assistant.

Context:
{context}

Question:
{query}

Return:
- Answer
- Section reference
\"\"\"

        res = model.generate_content(prompt).text

        if lang != "English":
            res = model.generate_content(f"Translate to {lang}: {res}").text

        st.markdown("### 📌 Answer")
        st.write(res)

# ================= SUMMARY =================
if st.button("Generate Summary"):
    text = "\n".join(st.session_state.chunks[:10])

    prompt = f\"\"\"Summarize:
- Key points
- Clauses
- Deadlines
- Penalties

{text}
\"\"\"

    st.write(model.generate_content(prompt).text)

# ================= RISK =================
if st.button("Risk Analysis"):
    text = "\n".join(st.session_state.chunks[:15])

    prompt = f\"\"\"Classify:

🔴 High Risk
🟡 Medium Risk
🟢 Low Risk

{text}
\"\"\"

    st.write(model.generate_content(prompt).text)

# ================= CHECKLIST =================
if st.button("Compliance Checklist"):
    text = "\n".join(st.session_state.chunks[:15])

    prompt = f\"\"\"Create GST compliance checklist:

{text}
\"\"\"

    st.write(model.generate_content(prompt).text)

# ================= COMPARE =================
file2 = st.file_uploader("Upload second document (Compare)", type=["pdf"])

if file and file2:
    t1 = extract_pdf(file)
    t2 = extract_pdf(file2)

    prompt = f\"\"\"Compare documents:

OLD:
{t1[:2000]}

NEW:
{t2[:2000]}

Show:
- Changes
- New rules
- Removed clauses
\"\"\"

    st.write(model.generate_content(prompt).text)
"""

# write app
with open("app.py", "w") as f:
    f.write(app_code)

# =========================
# RUN STREAMLIT + NGROK (FIXED)
# =========================
port = 8501

public_url = ngrok.connect(port, bind_tls=True).public_url

print("🔥 OPEN THIS LINK:", public_url)

def run():
    os.system(f"streamlit run app.py --server.port {port}")

threading.Thread(target=run).start() 