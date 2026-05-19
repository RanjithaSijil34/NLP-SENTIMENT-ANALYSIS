"""
Sentiment Analysis — Streamlit Deployment App
Run:  streamlit run app.py
"""

import re
import pickle
import numpy as np
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyser",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main container */
.main { background: #f8f9fb; }

/* Title banner */
.title-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,.18);
}
.title-box h1 { color: #ffffff; font-size: 2.2rem; margin: 0; letter-spacing: -0.5px; }
.title-box p  { color: #94a3b8; margin: .4rem 0 0; font-size: 1rem; }

/* Result card */
.result-card {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,.10);
    animation: fadeIn .4s ease;
}
@keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

.result-positive { background: linear-gradient(135deg,#d4edda,#b8dfc0); border: 2px solid #4CAF50; }
.result-neutral  { background: linear-gradient(135deg,#fff3cd,#ffe8a1); border: 2px solid #FF9800; }
.result-negative { background: linear-gradient(135deg,#f8d7da,#f5b8bc); border: 2px solid #F44336; }

.result-emoji  { font-size: 3.5rem; margin-bottom: .4rem; }
.result-label  { font-size: 1.9rem; font-weight: 800; margin: .2rem 0; letter-spacing: .5px; }
.result-conf   { font-size: 1rem; opacity: .75; margin-top: .3rem; }
.result-positive .result-label { color: #1b5e20; }
.result-neutral  .result-label { color: #e65100; }
.result-negative .result-label { color: #b71c1c; }

/* Confidence bars */
.conf-bar-wrap { margin-top: 1.2rem; }
.conf-row  { display:flex; align-items:center; gap:.8rem; margin-bottom:.5rem; font-size:.88rem; }
.conf-name { width: 72px; text-align: right; font-weight: 600; color: #444; }
.conf-bar  { flex:1; background: #e2e8f0; border-radius: 99px; height: 12px; overflow:hidden; }
.conf-fill { height:100%; border-radius:99px; transition: width .6s ease; }
.conf-pct  { width: 42px; text-align:left; font-size:.82rem; color:#555; }

/* Example chips */
.chip-grid { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.5rem; }
.chip {
    background:#e2e8f0; border-radius:99px; padding:.35rem .85rem;
    font-size:.82rem; cursor:pointer; user-select:none;
    transition: background .2s;
}
.chip:hover { background:#cbd5e1; }

/* Stats pills */
.stat-row { display:flex; gap:1rem; justify-content:center; margin: .8rem 0 .5rem; flex-wrap:wrap; }
.stat-pill {
    background:#ffffff; border: 1.5px solid #e2e8f0;
    border-radius:10px; padding:.45rem 1rem;
    text-align:center; min-width:90px;
    box-shadow: 0 2px 8px rgba(0,0,0,.04);
}
.stat-pill .val { font-size:1.3rem; font-weight:700; color:#1e293b; }
.stat-pill .lbl { font-size:.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.5px; }

/* Info box */
.info-box {
    background:#ffffff; border-left: 4px solid #0f3460;
    border-radius: 0 10px 10px 0;
    padding:.9rem 1.2rem; margin:.8rem 0; font-size:.88rem; color:#334155;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING (must match training)
# ─────────────────────────────────────────────────────────────
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words("english"))
STEMMER    = PorterStemmer()

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        STEMMER.stem(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open("svm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

svm = LinearSVC()
model = CalibratedClassifierCV(svm)

def predict(text: str, model, tfidf):
    clean   = preprocess(text)
    vec     = tfidf.transform([clean])
    label   = model.predict(vec)[0]
    probas  = model.predict_proba(vec)[0]
    classes = model.classes_

    conf_dict = {c: p for c, p in zip(classes, probas)}
    top_conf  = conf_dict[label]

    return label, top_conf, conf_dict


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
# Title
st.markdown("""
<div class="title-box">
  <h1>🎯 Sentiment Analyser</h1>
  <p>Linear SVM · TF-IDF · Positive · Neutral · Negative</p>
</div>
""", unsafe_allow_html=True)

# Model stats panel
st.markdown("""
<div class="stat-row">
  <div class="stat-pill"><div class="val">78.8%</div><div class="lbl">Accuracy</div></div>
  <div class="stat-pill"><div class="val">81.6%</div><div class="lbl">Precision</div></div>
  <div class="stat-pill"><div class="val">78.8%</div><div class="lbl">Recall</div></div>
  <div class="stat-pill"><div class="val">73.8%</div><div class="lbl">F1-Score</div></div>
  <div class="stat-pill"><div class="val">1,440</div><div class="lbl">Train Rows</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Load model
with st.spinner("Loading model …"):
    model, tfidf = load_model()

# Example reviews
EXAMPLES = {
    "Positive 😊": "This phone is absolutely amazing! Battery life is brilliant and the camera takes stunning photos. Totally worth every rupee.",
    "Neutral 😐": "The phone is okay. Camera is average, battery lasts a day. Nothing special but nothing terrible either.",
    "Negative 😞": "Extremely disappointed. Screen cracked after two days, poor build quality and customer service is terrible.",
}

st.subheader("💬 Enter a Review")

# Quick-fill example chips
st.markdown("**Quick examples:**")
cols = st.columns(3)
for i, (chip_label, chip_text) in enumerate(EXAMPLES.items()):
    if cols[i].button(chip_label, use_container_width=True):
        st.session_state["review_input"] = chip_text

# Text area
review_text = st.text_area(
    label="Type or paste a product review:",
    value=st.session_state.get("review_input", ""),
    height=130,
    placeholder="e.g.  The camera quality is outstanding and delivery was fast…",
    key="review_input",
)

# Word / char counter
word_count = len(review_text.split()) if review_text.strip() else 0
char_count = len(review_text)
st.caption(f"📝 {word_count} words · {char_count} characters")

# Predict button
predict_btn = st.button("🔍  Analyse Sentiment", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
if predict_btn:
    if not review_text.strip():
        st.warning("⚠️  Please enter some review text before analysing.")
    elif word_count < 3:
        st.info("ℹ️  Try adding a bit more text for a more reliable prediction (at least 3 words).")
    else:
        with st.spinner("Analysing …"):
            label, confidence, conf_dict = predict(review_text, model, tfidf)

        css_class = f"result-{label.lower()}"
        # Result card
        conf_pct = confidence * 100
        conf_text = (
            "Very High Confidence" if conf_pct >= 80 else
            "High Confidence"      if conf_pct >= 65 else
            "Moderate Confidence"  if conf_pct >= 50 else
            "Low Confidence"
        )

        st.markdown(f"""
        <div class="result-card {css_class}">
          <div class="result-emoji">{emoji}</div>
          <div class="result-label">{label}</div>
          <div class="result-conf">{conf_text} · {conf_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence breakdown bars
        st.markdown("#### Confidence Breakdown")
        bar_html = '<div class="conf-bar-wrap">'
        for cls in CLASS_ORDER:
            pct   = conf_dict.get(cls, 0) * 100
            color = COLOR_MAP[cls]
            bar_html += f"""
            <div class="conf-row">
              <div class="conf-name">{cls}</div>
              <div class="conf-bar">
                <div class="conf-fill" style="width:{pct:.1f}%; background:{color};"></div>
              </div>
              <div class="conf-pct">{pct:.1f}%</div>
            </div>"""
        bar_html += "</div>"
        st.markdown(bar_html, unsafe_allow_html=True)

        # Preprocessing preview
        with st.expander("🔬 See preprocessed text"):
            clean = preprocess(review_text)
            st.markdown(f"""
            <div class="info-box">
              <b>After preprocessing:</b><br><code style="font-size:.85rem;">{clean if clean else '(empty — all words were stopwords)'}</code>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Pipeline: lowercase → remove special chars → remove stopwords → Porter stemming")

# ─────────────────────────────────────────────────────────────
# SIDEBAR — About
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    **Model:** Linear SVC with Platt scaling  
    **Features:** TF-IDF (unigrams + bigrams, 15k features)  
    **Labels:**  
    - ⭐⭐⭐⭐⭐ / ⭐⭐⭐⭐ → Positive  
    - ⭐⭐⭐ → Neutral  
    - ⭐⭐ / ⭐ → Negative  

    **Dataset:** 1,440 product reviews  
    **Train / Test:** 80% / 20% stratified split  

    ---
    **Preprocessing pipeline**  
    1. Lowercase  
    2. Remove special chars & punctuation  
    3. Remove English stopwords  
    4. Porter Stemming  
    """)

    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    st.markdown("""
    | Sentiment | Reviews |
    |-----------|---------|
    | Positive  | 729     |
    | Negative  | 512     |
    | Neutral   | 199     |
    | **Total** | **1,440** |
    """)
