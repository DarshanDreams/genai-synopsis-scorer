import os
os.environ["STREAMLIT_WATCHFILES"] = "false"


import streamlit as st
from sentence_transformers import SentenceTransformer, util
import textstat
import fitz  
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Gen AI Synopsis Scorer", layout="centered")
st.title("📄 Gen AI Synopsis Scorer (Privacy Aware)")

# --- Access Control ---
password = st.text_input("🔒 Enter access token to continue", type="password")
if password != "Catallyst123":
    st.warning("Please enter the correct token to use the app.")
    st.stop()

# --- Instructions ---
st.markdown("Upload your **article (.pdf or .txt)** and **synopsis (.txt)** below.")

# --- File Upload ---
article_file = st.file_uploader("Upload Article", type=["pdf", "txt"])
synopsis_file = st.file_uploader("Upload Synopsis", type=["txt"])

def extract_text_from_article(file):
    if file.type == "application/pdf":
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def extract_text_from_synopsis(file):
    return file.read().decode("utf-8")

import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def get_embedding(text):
    return model.encode([text])

def calculate_similarity(article_text, synopsis_text):
    emb_article = get_embedding(article_text)
    emb_synopsis = get_embedding(synopsis_text)
    similarity = cosine_similarity(emb_article, emb_synopsis)[0][0]
    return similarity

if article_file and synopsis_file:
    st.success("✅ Files uploaded successfully!")

    article_text = extract_text_from_article(article_file)
    synopsis_text = extract_text_from_synopsis(synopsis_file)

    # --- Previews ---
    st.subheader("📘 Article Preview")
    st.text_area("Article Text (first 500 chars)", article_text[:500], height=150)

    st.subheader("📝 Synopsis Preview")
    st.text_area("Synopsis Text (first 500 chars)", synopsis_text[:500], height=150)

    # --- Scoring ---

    # 1. Content Coverage (cosine similarity scaled to 50)
    article_embedding = model.encode(article_text, convert_to_tensor=True)
    synopsis_embedding = model.encode(synopsis_text, convert_to_tensor=True)
    cos_sim = util.pytorch_cos_sim(article_embedding, synopsis_embedding).item()
    coverage = int(cos_sim * 50)

    # 2. Clarity (Flesch Reading Ease scaled to 25)
    reading_ease = textstat.flesch_reading_ease(synopsis_text)
    clarity = min(max(int((reading_ease / 100) * 25), 0), 25)

    # 3. Coherence (average cosine similarity between consecutive sentences, scaled to 25)
    sentences = [s.strip() for s in synopsis_text.split('.') if s.strip()]
    if len(sentences) > 1:
        coherence_scores = []
        for i in range(len(sentences) - 1):
            emb1 = model.encode(sentences[i], convert_to_tensor=True)
            emb2 = model.encode(sentences[i + 1], convert_to_tensor=True)
            coherence_scores.append(util.pytorch_cos_sim(emb1, emb2).item())
        coherence = int((sum(coherence_scores) / len(coherence_scores)) * 25)
    else:
        coherence = 25  # Single sentence gets max coherence

    total_score = coverage + clarity + coherence

    # --- Display Score and Feedback ---
    st.subheader("🧮 Scoring Result")
    st.write(f"**Final Score:** {total_score} / 100")

    # Simple qualitative feedback based on total score
    if total_score > 80:
        feedback = "Great synopsis! Very relevant and covers key points."
    elif total_score > 50:
        feedback = "Good effort, but synopsis can be more comprehensive."
    else:
        feedback = "Synopsis needs improvement in capturing the article’s main ideas."
    st.write("**Feedback:**", feedback)

    # --- Visual Score Breakdown ---
    st.subheader("📊 Visual Score Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="📘 Content Coverage", value=f"{coverage}/50")
        st.progress(int((coverage / 50) * 100))
    with col2:
        st.metric(label="🪄 Clarity", value=f"{clarity}/25")
        st.progress(int((clarity / 25) * 100))
    with col3:
        st.metric(label="🔗 Coherence", value=f"{coherence}/25")
        st.progress(int((coherence / 25) * 100))

else:
    st.info("Please upload both files to proceed.")
