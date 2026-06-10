import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from gtts import gTTS
import tempfile
import time
from reportlab.pdfgen import canvas
from io import BytesIO

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= UI =================
st.set_page_config(page_title="LexNavigator  ⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #06142e, #020617, #0b0f1a);
    color: white;
}

.title {
    font-size: 3rem;
    text-align: center;
    font-weight: 900;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.chat-user {
    background:#1e293b;
    padding:10px;
    border-radius:10px;
    margin:5px;
}

.chat-ai {
    background:#0f172a;
    border-left:3px solid #00d4ff;
    padding:10px;
    border-radius:10px;
    margin:5px;
}

.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:10px;
    padding:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>⚖️ LexNavigator</div>", unsafe_allow_html=True)

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "docs" not in st.session_state:
    st.session_state.docs = {}

# ================= LANGUAGE SELECT =================
lang = st.selectbox("🌐 Choose Language", ["English", "Hindi", "Telugu"])

# ================= SAFE PDF =================
def extract_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t
        return text[:20000]
    except:
        return ""

# ================= CHUNK =================
def chunk(text):
    return text.split(". ")

# ================= RETRIEVE =================
def retrieve(query, chunks):
    if not chunks:
        return []

    q = embedder.encode([query])[0]
    c = embedder.encode(chunks)

    scores = np.dot(c, q)
    top = np.argsort(scores)[-4:][::-1]

    return [chunks[i] for i in top]

# ================= STREAM =================
def stream(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(out)
        time.sleep(0.01)

# ================= VOICE OUTPUT =================
def speak(text):
    try:
        tts = gTTS(text[:300], lang="en")
        path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(path)
        st.audio(path)
    except:
        pass

# ================= SIDEBAR =================
st.sidebar.title("📚 Document Vault")

files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    for f in files:
        text = extract_pdf(f)
        st.session_state.docs[f.name] = chunk(text)

st.sidebar.write("Stored Docs:")
for name in st.session_state.docs:
    st.sidebar.write("📄", name)

# ================= CHAT HISTORY =================
for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>⚖️ {msg}</div>", unsafe_allow_html=True)

# ================= INPUT =================
query = st.text_input("Ask question...")

# ================= AI ENGINE (MULTI LANGUAGE) =================
def ask_ai(query):
    all_chunks = []
    for d in st.session_state.docs.values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    prompt = f"""
You are a senior legal AI assistant.

IMPORTANT LANGUAGE RULE:
Respond ONLY in {lang}.

Give:
1. Answer
2. Legal reasoning
3. Risk level
4. Key clauses

Context:
{context}

Question:
{query}
"""

    try:
        return model.generate_content(prompt).text
    except:
        return "⚠️ Gemini error: try again or reduce input."

# ================= SEND =================
if st.button("⚡ Send") and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    stream(response)
    speak(response)

# ================= SUMMARY =================
if st.button("📌 Summary"):
    all_text = "\n".join([" ".join(v) for v in st.session_state.docs.values()])[:3000]
    prompt = f"Summarize this legal document in {lang}:\n{all_text}"

    try:
        res = model.generate_content(prompt).text
        st.write(res)
    except:
        st.warning("Gemini limit reached")

# ================= COMPARE =================
file1 = st.file_uploader("OLD PDF", type=["pdf"])
file2 = st.file_uploader("NEW PDF", type=["pdf"])

if file1 and file2:
    t1 = extract_pdf(file1)
    t2 = extract_pdf(file2)

    if st.button("Compare"):
        prompt = f"""
Compare legal documents in {lang}:

OLD:
{t1[:2000]}

NEW:
{t2[:2000]}

Show:
- Changes
- Added clauses
- Removed clauses
- Risk impact
"""

        try:
            st.write(model.generate_content(prompt).text)
        except:
            st.warning("Too many tokens")
