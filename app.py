import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= UI =================
st.set_page_config(page_title="LexNavigator⚖️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #050816, #020409, #000);
    color: white;
}

/* Title */
h1 {
    text-align:center;
    font-size: 3rem;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* chat bubbles */
.user {
    background:#1e293b;
    padding:10px;
    border-radius:12px;
    margin:5px;
}
.ai {
    background:linear-gradient(135deg,#0f172a,#1e293b);
    padding:10px;
    border-radius:12px;
    border-left:3px solid #00d4ff;
    margin:5px;
}

/* buttons */
.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LexNavigator ")

# ================= SESSION MEMORY =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "docs" not in st.session_state:
    st.session_state.docs = {}

# ================= PDF FUNCTIONS =================
def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""
    return text

def chunk(text):
    return text.split(". ")

def retrieve(query, chunks):
    if not chunks:
        return []
    q = embedder.encode([query])[0]
    c = embedder.encode(chunks)
    scores = np.dot(c, q)
    top = np.argsort(scores)[-4:][::-1]
    return [chunks[i] for i in top]

# ================= STREAMING EFFECT =================
def stream(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(out)
        time.sleep(0.01)

# ================= SIDEBAR =================
st.sidebar.title("📂 Document Vault")

files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if files:
    for f in files:
        st.session_state.docs[f.name] = chunk(extract_pdf(f))

st.sidebar.write("Stored Docs:")
for name in st.session_state.docs:
    st.sidebar.write("📄", name)

# ================= LANGUAGE =================
lang = st.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])

# ================= CHAT UI =================
st.subheader("💬 Chat")

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'>⚖️ {msg}</div>", unsafe_allow_html=True)

query = st.text_input("Ask something legal...")

# ================= AI ENGINE =================
def ask_ai(query):
    all_chunks = []
    for doc in st.session_state.docs.values():
        all_chunks.extend(doc)

    context = retrieve(query, all_chunks)

    prompt = f"""
You are a senior legal AI assistant.

Language: {lang}

Provide:
1. Clear Answer
2. Legal reasoning
3. Risk level
4. Key clauses

Context:
{context}

Question:
{query}
"""

    res = model.generate_content(prompt).text
    return res

# ================= SEND =================
if st.button("⚡ Send") and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    stream(response)

# ================= FILE COMPARISON =================
st.markdown("---")
st.subheader("📄 Compare Documents")

file1 = st.file_uploader("Upload OLD PDF", type=["pdf"])
file2 = st.file_uploader("Upload NEW PDF", type=["pdf"])

if file1 and file2:
    t1 = extract_pdf(file1)
    t2 = extract_pdf(file2)

    compare_prompt = f"""
Compare these legal documents:

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

    if st.button("Compare"):
        res = model.generate_content(compare_prompt).text
        st.write(res)
