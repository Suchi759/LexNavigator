import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from io import BytesIO
from reportlab.pdfgen import canvas

# ================= CONFIG =================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
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

/* chat bubbles */
.chat-user {
    background: #1e293b;
    padding: 10px;
    border-radius: 12px;
    margin: 5px;
}

.chat-ai {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    padding: 10px;
    border-radius: 12px;
    margin: 5px;
    border-left: 3px solid #00d4ff;
}

/* buttons */
.stButton>button {
    background: linear-gradient(90deg,#00d4ff,#a855f7,#ff3d81);
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚖️ LexNavigator")

# ================= SESSION =================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "docs" not in st.session_state:
    st.session_state.docs = {}

# ================= PDF =================
def extract_pdf(file):
    reader = PdfReader(file)
    text = ""
    for p in reader.pages:
        text += p.extract_text() or ""
    return text

def chunk(text):
    return text.split(". ")

def embed(text_list):
    return embedder.encode(text_list)

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
    placeholder = st.empty()
    out = ""
    for c in text:
        out += c
        placeholder.markdown(out)
    return out

# ================= SIDEBAR MEMORY =================
st.sidebar.title("📚 Document Memory")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        text = extract_pdf(file)
        st.session_state.docs[file.name] = chunk(text)

st.sidebar.write("Stored Docs:")
for k in st.session_state.docs.keys():
    st.sidebar.write("📄", k)

# ================= CHAT UI =================
st.subheader("💬 Chat Assistant")

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(f"<div class='chat-user'>🧑 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-ai'>⚖️ {msg}</div>", unsafe_allow_html=True)

query = st.text_input("Ask question")

# ================= CLAUSE EXTRACTION =================
def extract_clauses(text):
    prompt = f"""
Extract legal clauses in bullet format:
{text}
"""
    return model.generate_content(prompt).text

# ================= LEGAL RESPONSE =================
def ask_ai(query):
    all_chunks = []
    for doc in st.session_state.docs.values():
        all_chunks.extend(doc)

    context = retrieve(query, all_chunks)

    prompt = f"""
You are a senior legal AI assistant.

Provide:
1. Structured Answer
2. Legal reasoning
3. Risk level (High/Medium/Low)
4. Important clauses

Context:
{context}

Question:
{query}
"""

    return model.generate_content(prompt).text

# ================= ACTION =================
if st.button("⚡ Send") and query:

    st.session_state.chat.append(("user", query))

    response = ask_ai(query)

    st.session_state.chat.append(("ai", response))

    stream_text(response)

# ================= REPORT DOWNLOAD =================
def make_pdf(text):
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(50, 800, "LexNavigator Legal Report")
    y = 760
    for line in text.split("\n")[:40]:
        c.drawString(50, y, line[:100])
        y -= 15
    c.save()
    buffer.seek(0)
    return buffer

if st.session_state.chat:
    full_text = "\n".join([m[1] for m in st.session_state.chat if m[0] == "ai"])
    pdf = make_pdf(full_text)

    st.download_button(
        "📄 Download Legal Report",
        pdf,
        file_name="legal_report.pdf",
        mime="application/pdf"
    )
