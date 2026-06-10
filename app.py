import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import time
from io import BytesIO
from reportlab.pdfgen import canvas

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= PAGE =================
st.set_page_config(page_title="LexNavigator PRO ⚖️", layout="wide")

# ================= CINEMATIC DARK UI =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0b1220, #050814, #02030a);
    color: #e5e7eb;
}

/* TITLE */
h1 {
    text-align:center;
    font-size: 3rem;
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CHAT BUBBLES */
.user {
    background:#1e293b;
    padding:10px;
    border-radius:12px;
    margin:6px;
}
.ai {
    background:linear-gradient(135deg,#0f172a,#1e293b);
    padding:10px;
    border-radius:12px;
    border-left:3px solid #00d4ff;
    margin:6px;
}

/* BUTTONS */
.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color:white;
    border-radius:10px;
    font-weight:bold;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#020617,#050816);
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LexNavigator AI PRO")

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "docs" not in st.session_state:
    st.session_state.docs = {}

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

def chunk(text):
    return [c.strip() for c in text.split(". ") if c.strip()]

def retrieve(query, chunks):
    if not chunks:
        return []
    q = embedder.encode([query])[0]
    c = embedder.encode(chunks)
    scores = np.dot(c, q)
    top = np.argsort(scores)[-4:][::-1]
    return [chunks[i] for i in top]

# ================= STREAMING EFFECT =================
def stream_text(text):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(out)
        time.sleep(0.01)

# ================= SIDEBAR VAULT =================
st.sidebar.title("📂 Document Vault")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        text = extract_pdf(f)
        st.session_state.docs[f.name] = chunk(text)

st.sidebar.write("Stored Docs:")
for name in st.session_state.docs:
    st.sidebar.write("📄", name)

# ================= LANGUAGE =================
lang = st.selectbox("🌐 Language", ["English", "Hindi", "Telugu"])

# ================= AI ENGINE =================
def ask_ai(query):
    all_chunks = []
    for d in st.session_state.docs.values():
        all_chunks.extend(d)

    context = retrieve(query, all_chunks)

    prompt = f"""
You are a senior legal AI assistant.

Language: {lang}

Give:
1. Clear Answer
2. Legal reasoning
3. Risk level
4. Key clauses

Context:
{context}

Question:
{query}
"""

    return model.generate_content(prompt).text

# ================= CHAT UI =================
st.subheader("💬 Chat Assistant")

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai'>⚖️ {msg}</div>", unsafe_allow_html=True)

query = st.text_input("Ask your legal question")

# ================= SEND =================
if st.button("⚡ Send") and query:
    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    stream_text(response)

# ================= COMPARE DOCUMENTS =================
st.markdown("---")
st.subheader("📄 Compare Documents")

file1 = st.file_uploader("Upload OLD PDF", type=["pdf"])
file2 = st.file_uploader("Upload NEW PDF", type=["pdf"])

if file1 and file2:
    old_text = extract_pdf(file1)[:2000]
    new_text = extract_pdf(file2)[:2000]

    compare_prompt = f"""
Compare legal documents:

OLD:
{old_text}

NEW:
{new_text}

Show:
- Changes
- Added clauses
- Removed clauses
- Risk impact
"""

    if st.button("Compare Now"):
        result = model.generate_content(compare_prompt).text
        st.markdown("### 📊 Comparison Result")
        st.write(result)

# ================= PDF REPORT =================
def make_pdf(text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(50, 800, "LexNavigator Legal Report")
    y = 770
    for line in text.split("\n")[:40]:
        c.drawString(50, y, line[:100])
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

if st.session_state.chat:
    full = "\n".join([m[1] for m in st.session_state.chat if m[0] == "ai"])
    pdf = make_pdf(full)

    st.download_button(
        "📄 Download Legal Report",
        pdf,
        file_name="legal_report.pdf",
        mime="application/pdf"
    )
