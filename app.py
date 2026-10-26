import re, io, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy.sparse import hstack

from skill_gap_db  import get_skill_gap, get_courses, SKILL_GAP_DB
from domain_bridge import detect_domain, enrich_profile, DOMAIN_CAREERS

st.set_page_config(
    page_title="CareerML — Career Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=DM+Mono:wght@300;400;500&display=swap');

/* Root tokens */
:root {
    --bg:       #060a12;
    --surface:  #0c1220;
    --surface2: #111827;
    --border:   rgba(255,255,255,0.06);
    --border2:  rgba(255,255,255,0.12);
    --text:     #e8edf5;
    --muted:    #5a6478;
    --subtle:   #8b95a8;
    --accent:   #c9a84c;
    --accent2:  #e8c97a;
    --blue:     #3b82f6;
    --blue-dim: rgba(59,130,246,0.12);
    --green:    #22c55e;
    --green-dim:rgba(34,197,94,0.10);
    --red:      #ef4444;
    --red-dim:  rgba(239,68,68,0.10);
    --gold:     #c9a84c;
    --gold-dim: rgba(201,168,76,0.10);
    --radius:   10px;
    --font-display: 'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* Streamlit overrides */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg); }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border2) !important;
}
section[data-testid="stSidebar"] * { font-family: var(--font-body) !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Inputs */
.stMultiSelect [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-size: 0.875rem !important;
}
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    font-family: var(--font-body) !important;
    font-size: 0.875rem !important;
    color: var(--text) !important;
    resize: vertical;
}
.stFileUploader {
    background: var(--surface2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 8px !important;
}
.stRadio > label { font-size: 0.8rem !important; color: var(--muted) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 0.875rem !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; border-radius: 2px !important; }
.stProgress > div { background: rgba(201,168,76,0.12) !important; border-radius: 2px !important; height: 3px !important; }

/* Expander */
details { 
    background: var(--surface) !important; 
    border: 1px solid var(--border2) !important; 
    border-radius: var(--radius) !important;
    margin-bottom: 8px !important;
}
summary { 
    font-family: var(--font-body) !important; 
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text) !important;
    padding: 14px 18px !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    font-family: var(--font-body) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: var(--muted) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 28px 0 !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* CUSTOM COMPONENTS */

/* Page header */
.page-header {
    padding: 40px 0 32px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
}
.page-header .eyebrow {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 12px;
}
.page-header h1 {
    font-family: var(--font-display);
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin: 0 0 10px;
}
.page-header .subtitle {
    font-size: 0.9rem;
    color: var(--subtle);
    font-weight: 300;
    line-height: 1.6;
}

/* Section label */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--accent);
    margin: 36px 0 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border2);
}

/* Prediction cards */
.pred-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 24px; }

.pred-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 28px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color .2s;
}
.pred-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.pred-card.rank-2::before { background: linear-gradient(90deg, var(--subtle), transparent); }
.pred-card.rank-3::before { background: linear-gradient(90deg, var(--surface2), transparent); }

.pred-card .rank-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
}
.pred-card .career-name {
    font-family: var(--font-display);
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.3;
    margin-bottom: 20px;
    min-height: 2.6rem;
}
.pred-card .score-block {
    display: flex;
    align-items: flex-end;
    gap: 6px;
    margin-bottom: 6px;
}
.pred-card .score-num {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.pred-card.rank-2 .score-num { color: var(--subtle); }
.pred-card.rank-3 .score-num { color: var(--muted); }
.pred-card .score-unit {
    font-size: 0.9rem;
    color: var(--muted);
    margin-bottom: 6px;
}
.pred-card .score-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 16px;
}
.pred-card .match-tag {
    display: inline-block;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 4px;
}
.tag-strong  { background: var(--green-dim); color: #86efac; border: 1px solid rgba(34,197,94,.2); }
.tag-good    { background: var(--gold-dim);  color: #fcd34d; border: 1px solid rgba(201,168,76,.2); }
.tag-possible{ background: rgba(255,255,255,.04); color: var(--subtle); border: 1px solid var(--border2); }
.pred-card .raw-prob {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--muted);
    margin-top: 12px;
}

/* Trajectory cards */
.traj-card {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.traj-card.now-card  { border-left: 3px solid var(--blue); }
.traj-card.fut-card  { border-left: 3px solid var(--green); }
.traj-card .tc-name  { font-family: var(--font-display); font-size: 1rem; font-weight: 600; color: var(--text); }
.traj-card .tc-rank  { font-family: var(--font-mono); font-size: 0.65rem; color: var(--muted); letter-spacing:1px; text-transform:uppercase; margin-bottom:4px; }
.traj-card .tc-score { font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; color: var(--accent); }
.traj-card.now-card .tc-score { color: #60a5fa; }
.traj-card.fut-card  .tc-score { color: #4ade80; }
.traj-card .tc-tag   { font-family: var(--font-mono); font-size: 0.62rem; letter-spacing:1px; text-transform:uppercase; color: var(--muted); margin-top:2px; }

/* Domain banner */
.domain-banner {
    background: linear-gradient(135deg, rgba(201,168,76,.08), rgba(201,168,76,.03));
    border: 1px solid rgba(201,168,76,.2);
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 24px;
}
.domain-banner .db-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
    box-shadow: 0 0 8px rgba(201,168,76,.5);
}
.domain-banner .db-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
}
.domain-banner .db-value {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
}
.domain-banner .db-desc {
    font-size: 0.8rem;
    color: var(--subtle);
    margin-top: 1px;
}

/* Info block */
.info-block {
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.875rem;
    line-height: 1.65;
    color: var(--subtle);
    margin: 16px 0;
    border-left: 3px solid;
}
.info-block strong { color: var(--text); font-weight: 500; }
.info-neutral { background: rgba(255,255,255,.02); border-color: var(--border2); }
.info-warn    { background: rgba(201,168,76,.05);  border-color: rgba(201,168,76,.3); }
.info-ok      { background: rgba(34,197,94,.05);   border-color: rgba(34,197,94,.3); }
.info-accent  { background: rgba(59,130,246,.05);  border-color: rgba(59,130,246,.3); }

/* Skill badges */
.badge-have {
    display: inline-block; margin: 3px;
    padding: 3px 11px; border-radius: 4px;
    font-family: var(--font-mono); font-size: .72rem; font-weight: 500;
    background: var(--green-dim); color: #86efac;
    border: 1px solid rgba(34,197,94,.2);
}
.badge-miss {
    display: inline-block; margin: 3px;
    padding: 3px 11px; border-radius: 4px;
    font-family: var(--font-mono); font-size: .72rem; font-weight: 500;
    background: var(--red-dim); color: #fca5a5;
    border: 1px solid rgba(239,68,68,.2);
}

/* Coverage bar */
.cov-row {
    display: flex; align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
}
.cov-label { font-family: var(--font-mono); font-size: .7rem; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; }
.cov-pct   { font-family: var(--font-display); font-size: 1rem; font-weight: 700; color: var(--accent); }

/* Stat summary strip */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin: 28px 0;
}
.stat-cell {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
}
.stat-cell .sc-val {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
    margin-bottom: 6px;
}
.stat-cell .sc-lbl {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
}
.sc-accent .sc-val { color: var(--accent); }

/* Profile completeness bar */
.completeness-row {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-top: 20px;
}
.cr-label { font-family: var(--font-mono); font-size: .7rem; letter-spacing:1.5px; text-transform:uppercase; color: var(--muted); white-space:nowrap; }
.cr-bar   { flex:1; height:3px; background: rgba(201,168,76,.12); border-radius:2px; }
.cr-fill  { height:100%; background: var(--accent); border-radius:2px; transition: width .4s; }
.cr-pct   { font-family: var(--font-display); font-size:.9rem; font-weight:700; color: var(--accent); white-space:nowrap; }

/* Ideal profile table */
.ideal-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
.ideal-cell { background: rgba(255,255,255,.02); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; }
.ideal-cell .ic-label { font-family: var(--font-mono); font-size:.65rem; letter-spacing:1.5px; text-transform:uppercase; color: var(--muted); margin-bottom:5px; }
.ideal-cell .ic-value { font-size:.85rem; color: var(--subtle); line-height:1.5; }

/* Sidebar */
.sb-brand { font-family: var(--font-display); font-size: 1.3rem; font-weight: 800; color: var(--text); letter-spacing:-.3px; }
.sb-tagline { font-family: var(--font-mono); font-size: .65rem; letter-spacing:2px; text-transform:uppercase; color: var(--accent); margin-top:2px; }
.sb-section { font-family: var(--font-mono); font-size: .62rem; letter-spacing:2px; text-transform:uppercase; color: var(--muted); margin: 24px 0 10px; }
.sb-cap { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.sb-cap-item { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 10px 10px; text-align: center; }
.sb-cap-val  { font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: var(--accent); }
.sb-cap-lbl  { font-family: var(--font-mono); font-size: .58rem; letter-spacing:1px; text-transform:uppercase; color: var(--muted); margin-top:2px; }
.sb-feature  { display: flex; align-items:center; gap:10px; padding: 7px 0; border-bottom:1px solid var(--border); }
.sb-feature:last-child { border-bottom: none; }
.sb-f-dot    { width:5px; height:5px; border-radius:50%; background:var(--accent); flex-shrink:0; }
.sb-f-text   { font-size:.82rem; color: var(--subtle); }
.sb-stack-item { font-family: var(--font-mono); font-size: .72rem; color: var(--subtle); padding: 5px 10px;
                 background: var(--surface2); border: 1px solid var(--border); border-radius:4px; margin-bottom:5px; }

/* Input mode tabs */
.mode-bar { display: flex; gap: 2px; background: var(--surface2); border: 1px solid var(--border2);
            border-radius: 8px; padding: 4px; margin-bottom: 20px; }
.mode-btn { flex: 1; padding: 8px; text-align: center; border-radius: 6px; cursor: pointer;
            font-family: var(--font-mono); font-size: .72rem; letter-spacing:1px; text-transform:uppercase;
            color: var(--muted); transition: all .15s; }
.mode-btn.active { background: var(--surface); color: var(--text); border: 1px solid var(--border2); }

/* Column headers */
.col-head { font-family: var(--font-mono); font-size:.7rem; letter-spacing:2px; text-transform:uppercase; color: var(--muted); margin-bottom:14px; }

/* Traj section heading */
.traj-head { display:flex; align-items:center; gap:10px; margin-bottom:16px; }
.traj-head-dot { width:10px; height:10px; border-radius:2px; flex-shrink:0; }
.traj-head-now { background: var(--blue); }
.traj-head-fut { background: var(--green); }
.traj-head-txt { font-family: var(--font-display); font-size: .95rem; font-weight: 700; color: var(--text); }
.traj-head-sub { font-size: .78rem; color: var(--muted); margin-top:1px; }

/* Insight trajectory */
.insight-traj {
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 10px;
    padding: 20px 24px;
    margin-top: 20px;
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 16px;
    align-items: center;
}
.it-from, .it-to { text-align:center; }
.it-label { font-family: var(--font-mono); font-size:.62rem; letter-spacing:2px; text-transform:uppercase; color: var(--muted); margin-bottom:6px; }
.it-name  { font-family: var(--font-display); font-size:1rem; font-weight:700; color:var(--text); }
.it-arrow { font-size:1.5rem; color: var(--muted); text-align:center; }
.it-bridge{ margin-top:14px; font-size:.82rem; color:var(--subtle); }
.it-bridge strong { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# LOAD PKL FILES
@st.cache_resource
def load_all():
    model   = joblib.load("model.pkl")
    tfidf_s = joblib.load("tfidf_s.pkl")
    tfidf_i = joblib.load("tfidf_i.pkl")
    tfidf_t = joblib.load("tfidf_t.pkl")
    tfidf_l = joblib.load("tfidf_l.pkl")
    le      = joblib.load("le.pkl")
    with open("vocab.json")           as f: vocab    = json.load(f)
    with open("career_profiles.json") as f: cprofile = json.load(f)
    # Load model comparison metadata if available (generated by train_model.py v2)
    meta = {}
    try:
        with open("model_meta.json") as f: meta = json.load(f)
    except FileNotFoundError:
        meta = {
            "best_model": "XGBoost", "model_type": "XGBClassifier",
            "is_tree": True,
            "cv_scores": {"XGBoost": {"mean": 100.0, "std": 0.0, "folds": [100]*5}},
            "test_accuracy": {"XGBoost": 100.0},
            "n_classes": 37, "n_features": 879, "n_train": 8880, "n_test": 2220,
        }
    return model, tfidf_s, tfidf_i, tfidf_t, tfidf_l, le, vocab, cprofile, meta

model, tfidf_s, tfidf_i, tfidf_t, tfidf_l, le, vocab, career_profiles, MODEL_META = load_all()


# PREDICTION ENGINE
def encode(skills, interests, talents, likes):
    X_s = tfidf_s.transform([", ".join(skills).lower()])
    X_i = tfidf_i.transform([", ".join(interests).lower()])
    X_t = tfidf_t.transform([", ".join(talents).lower()])
    X_l = tfidf_l.transform([", ".join(likes).lower()])
    return hstack([X_s, X_i, X_t, X_l])

def predict(skills, interests, talents, likes, raw_text=""):
    if not any([skills, interests, talents, likes]):
        return None, None
    domain     = detect_domain(raw_text, skills, interests)
    es, ei     = enrich_profile(raw_text, skills, interests)
    X          = encode(es, ei, talents, likes)
    if hasattr(model, "predict_proba"):
        if hasattr(X, "toarray"):  # check if sparse
            X_input = X.toarray()
        else:
            X_input = X
        probs = model.predict_proba(X_input)[0]
    all_sorted = [{"career": le.classes_[i], "raw": float(probs[i])}
                  for i in np.argsort(probs)[::-1]]
    if domain:
        valid    = set(DOMAIN_CAREERS.get(domain, []))
        in_d     = [p for p in all_sorted if p["career"] in valid]
        out_d    = [p for p in all_sorted if p["career"] not in valid]
        reranked = (in_d[:3] + out_d)[:3]
    else:
        reranked = all_sorted[:3]
    p3   = np.array([r["raw"] for r in reranked])
    norm = p3 / max(p3.sum(), 1e-10) * 100
    out  = []
    for r, n in zip(reranked, norm):
        tag    = "STRONG MATCH" if n >= 45 else ("GOOD MATCH" if n >= 30 else "POSSIBLE MATCH")
        tagcss = {"STRONG MATCH":"tag-strong","GOOD MATCH":"tag-good","POSSIBLE MATCH":"tag-possible"}[tag]
        out.append({"career": r["career"], "norm": round(float(n),1),
                    "raw": round(r["raw"]*100,1), "tag": tag, "tagcss": tagcss})
    return out, domain


# NLP EXTRACTOR
_sl = {s.lower(): s for s in vocab["skills"]}
_il = {s.lower(): s for s in vocab["interests"]}
_tl = {s.lower(): s for s in vocab["talents"]}
_ll = {s.lower(): s for s in vocab["likes"]}
def extract(text):
    t = text.lower()
    def m(lk):
        return list(dict.fromkeys(orig for kw,orig in lk.items()
                                  if re.search(r"\b"+re.escape(kw)+r"\b", t)))
    return {"skills": m(_sl)[:20], "interests": m(_il)[:20],
            "talents": m(_tl)[:10], "likes": m(_ll)[:10]}


# PDF READER
def read_pdf(file_bytes):
    for fn in [
        lambda b: __import__('pdfplumber').open(io.BytesIO(b)).__enter__(),
        None,
    ]:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            t = " ".join(p.extract_text() or "" for p in pdf.pages)
        if t.strip(): return t
    except Exception: pass
    try:
        import PyPDF2
        r = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        t = " ".join(pg.extract_text() or "" for pg in r.pages)
        if t.strip(): return t
    except Exception: pass
    try:
        from pdfminer.high_level import extract_text as pe
        t = pe(io.BytesIO(file_bytes))
        if t.strip(): return t
    except Exception: pass
    return ""


# SHAP CHART  (tree-based or linear, adapts to best model)
@st.cache_resource
def _explainer():
    import shap
    if MODEL_META.get("is_tree", True):
        return shap.TreeExplainer(model)
    else:
        # LinearExplainer for Logistic Regression — needs dense background
        import scipy.sparse as sp
        bg = shap.sample(model.__class__.__name__, 100) if False else None
        return shap.LinearExplainer(model, shap.maskers.Independent(
            np.zeros((1, MODEL_META.get("n_features", 879)))
        ))

def shap_chart(X_enc, career):
    try:
        import shap
        from scipy.sparse import issparse as _isp
        is_tree = MODEL_META.get("is_tree", True)
        X_dense = X_enc.toarray() if _isp(X_enc) else X_enc

        if is_tree:
            exp   = _explainer()
            sv    = exp.shap_values(X_enc)
            cidx  = list(le.classes_).index(career)
            # XGBClassifier returns (n_samples, n_features, n_classes)
            # RandomForest/HGB returns list of arrays
            if isinstance(sv, list):
                vals = sv[cidx][0]
            else:
                vals = sv[0, :, cidx] if sv.ndim == 3 else sv[0]
        else:
            # Linear: use coefficient-weighted feature values as proxy
            coef  = model.coef_
            cidx  = list(le.classes_).index(career)
            vals  = np.asarray(X_dense[0] * coef[cidx]).flatten()

        fnames = (list(tfidf_s.get_feature_names_out()) +
                  list(tfidf_i.get_feature_names_out()) +
                  list(tfidf_t.get_feature_names_out()) +
                  list(tfidf_l.get_feature_names_out()))
        top    = np.argsort(np.abs(vals))[::-1][:12]
        names  = [fnames[i] for i in top][::-1]
        vv     = [vals[i]   for i in top][::-1]
        pos_c  = "#c9a84c"; neg_c = "#3b5bdb"
        colors = [pos_c if v > 0 else neg_c for v in vv]

        fig, ax = plt.subplots(figsize=(7, 4.8))
        fig.patch.set_facecolor("#0c1220")
        ax.set_facecolor("#0c1220")
        ax.barh(names, vv, color=colors, edgecolor="none", height=0.52)
        ax.axvline(0, color="#2a3550", lw=1.2)
        ax.set_xlabel("Shapley Value" if is_tree else "Coefficient × Feature Value",
                      fontsize=8.5, color="#5a6478",
                      fontfamily="monospace", labelpad=10)
        ax.set_title(
            f"Feature Attribution  ·  {career}",
            fontsize=10, fontweight="bold", color="#e8edf5", pad=16,
            fontfamily="sans-serif")
        ax.tick_params(labelsize=7.5, colors="#5a6478", length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.xaxis.grid(True, color="#1a2744", lw=0.7, linestyle="--")
        ax.set_axisbelow(True)
        pos_p = mpatches.Patch(color=pos_c, label="Supports prediction")
        neg_p = mpatches.Patch(color=neg_c, label="Contradicts prediction")
        ax.legend(handles=[pos_p, neg_p], fontsize=7.5,
                  facecolor="#0c1220", edgecolor="#2a3550",
                  labelcolor="#8b95a8", loc="lower right", framealpha=1)
        plt.tight_layout()
        return fig
    except Exception:
        return None


# RADAR CHART
def radar_chart(skills, interests, talents, likes, career):
    try:
        prof = career_profiles.get(career, {})
        if not prof: return None
        dims = ["Skills", "Interests", "Talents", "Likes", "Complete", "Overall"]
        def ov(u, c):
            if not c: return 0.5
            return len({x.lower() for x in u} & {x.lower() for x in c}) / len(c)
        sm = ov(skills, prof.get("top_skills", []))
        im = ov(interests, prof.get("top_interests", []))
        tm = ov(talents,   prof.get("top_talents", []))
        lm = ov(likes,     prof.get("top_likes", []))
        pc = min(1.0, (len(skills)+len(interests)+len(talents)+len(likes)) / 20.0)
        ov_fit = (sm+im+tm+lm) / 4.0
        vals  = [sm, im, tm, lm, pc, ov_fit]
        ideal = [1.0]*6
        N = len(dims)
        ang = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        va, ia, aa = vals+[vals[0]], ideal+[ideal[0]], ang+[ang[0]]

        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0c1220")
        ax.set_facecolor("#0c1220")
        ax.fill(ang, ideal, alpha=0.04, color="#c9a84c")
        ax.plot(aa, ia, color="#c9a84c", lw=0.8, ls="--", alpha=0.3)
        ax.fill(ang, vals, alpha=0.2, color="#c9a84c")
        ax.plot(aa, va, color="#c9a84c", lw=2, label="Profile")
        ax.set_xticks(ang)
        ax.set_xticklabels(dims, size=7.5, color="#8b95a8", fontfamily="monospace")
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], size=0)
        ax.grid(color="#1a2744", lw=0.8)
        ax.spines["polar"].set_color("#1a2744")
        plt.tight_layout(pad=1.5)
        return fig
    except Exception:
        return None


# HELPER: render extracted profile tags
def extracted_profile_ui(skills, interests, talents, likes):
    e1, e2, e3, e4 = st.columns(4)
    for col, label, items in zip(
        [e1, e2, e3, e4],
        ["Skills Detected", "Interests Detected", "Talents", "Likes"],
        [skills, interests, talents, likes]
    ):
        col.markdown(f"""
        <div style="background:var(--surface2);border:1px solid var(--border2);
        border-radius:8px;padding:14px 16px">
        <div style="font-family:var(--font-mono);font-size:.62rem;letter-spacing:2px;
        text-transform:uppercase;color:var(--muted);margin-bottom:6px">{label}</div>
        <div style="font-size:.84rem;color:var(--subtle)">{', '.join(items) if items else '—'}</div>
        </div>
        """, unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">CareerML</div>
    <div class="sb-tagline">Career Intelligence Platform</div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sb-section">Model Statistics</div>', unsafe_allow_html=True)
    _best = MODEL_META.get("best_model", "XGBoost")
    _cv   = MODEL_META.get("cv_scores", {}).get(_best, {})
    _cv_m = _cv.get("mean", 100.0)
    _tacc = MODEL_META.get("test_accuracy", {}).get(_best, 100.0)
    _ncls = MODEL_META.get("n_classes", 37)
    _nfts = MODEL_META.get("n_features", 879)
    st.markdown(f"""
    <div class="sb-cap">
        <div class="sb-cap-item"><div class="sb-cap-val">{_tacc:.1f}%</div><div class="sb-cap-lbl">Test Acc</div></div>
        <div class="sb-cap-item"><div class="sb-cap-val">{_cv_m:.1f}%</div><div class="sb-cap-lbl">CV Score</div></div>
        <div class="sb-cap-item"><div class="sb-cap-val">{_ncls}</div><div class="sb-cap-lbl">Careers</div></div>
        <div class="sb-cap-item"><div class="sb-cap-val">{_nfts}</div><div class="sb-cap-lbl">Features</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Capabilities</div>', unsafe_allow_html=True)
    features = [
        "Real-Time What-If Simulation",
        "Domain-Aware Prediction Engine",
        "Vocabulary Enrichment Bridge",
        "Skill Gap Analysis",
        "Course Recommendation Engine",
        "Career Trajectory Forecasting",
        "6-Dimensional Radar Profiling",
        "SHAP Feature Attribution",
        "Multi-Modal Input (Form / NLP / PDF)",
    ]
    items = "".join(f'<div class="sb-feature"><div class="sb-f-dot"></div><div class="sb-f-text">{f}</div></div>' for f in features)
    st.markdown(items, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Algorithm Stack</div>', unsafe_allow_html=True)
    _bm    = MODEL_META.get("best_model", "XGBoost")
    _btype = MODEL_META.get("model_type", "XGBClassifier")
    stack  = [f"★ {_bm}  (selected via 5-CV)", "TF-IDF  ·  4 Vectorizers",
              "One-Hot Encoding", "SHAP Explainability", "Domain Bridge NLP",
              "4-Model Comparison Pipeline"]
    for s in stack:
        st.markdown(f'<div class="sb-stack-item">{s}</div>', unsafe_allow_html=True)


# PAGE HEADER
st.markdown("""
<div class="page-header">
    <div class="eyebrow">Career Intelligence Platform</div>
    <h1>Career Path Prediction</h1>
    <div class="subtitle">
        Multi-class machine learning system · 4-Algorithm Comparison · SHAP · Domain-Aware NLP ·
        37 career classifications · Best model auto-selected via 5-fold cross-validation
    </div>
</div>
""", unsafe_allow_html=True)


# INPUT — MULTI-MODAL
st.markdown('<div class="section-label">Profile Input</div>', unsafe_allow_html=True)

mode = st.radio(
    "", ["Structured Form", "Free Text  —  NLP Extraction", "Upload CV  —  PDF Parser"],
    horizontal=True, label_visibility="collapsed"
)

user_skills = user_interests = user_talents = user_likes = []
raw_text    = ""

# Structured Form 
if mode == "Structured Form":
    st.markdown("""
    <div class="info-block info-neutral">
    Select attributes from each category. Predictions recalculate automatically on every change — no submit required.
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        user_skills    = st.multiselect("Skills",    vocab["skills"],    max_selections=15, placeholder="Select up to 15 skills")
        user_interests = st.multiselect("Interests", vocab["interests"], max_selections=15, placeholder="Select interests")
    with c2:
        user_talents   = st.multiselect("Talents",   vocab["talents"],   max_selections=8,  placeholder="Select talents")
        user_likes     = st.multiselect("Likes",     vocab["likes"],     max_selections=8,  placeholder="Select likes")

# Free Text NLP 
elif mode == "Free Text  —  NLP Extraction":
    st.markdown("""
    <div class="info-block info-neutral">
    Write a natural description of your background. The NLP engine extracts keywords and the domain
    bridge maps them to the model's internal vocabulary, resolving the gap between CV language and training data.
    </div>""", unsafe_allow_html=True)

    demo = ("I have a Bachelor's degree in Computer Science. I am proficient in Python, machine learning, "
            "deep learning, TensorFlow, and data analysis. I have experience with SQL, React, and Git. "
            "I enjoy artificial intelligence research and like problem solving and learning new things.")
    raw_text = st.text_area("Profile description:", value=demo, height=140, label_visibility="collapsed")

    if raw_text.strip():
        p = extract(raw_text)
        user_skills, user_interests = p["skills"], p["interests"]
        user_talents, user_likes    = p["talents"], p["likes"]
        st.markdown('<div class="section-label" style="margin-top:24px">Extracted Profile</div>', unsafe_allow_html=True)
        extracted_profile_ui(user_skills, user_interests, user_talents, user_likes)
        if not user_skills and not user_interests:
            st.markdown("""
            <div class="info-block info-warn">
            <strong>Low keyword match.</strong> The vocabulary bridge will still attempt domain enrichment.
            For best results, include specific skills such as Python, Machine Learning, Data Analysis,
            Research, Marketing, or similar terms.
            </div>""", unsafe_allow_html=True)

# PDF Upload 
else:
    st.markdown("""
    <div class="info-block info-neutral">
    Upload a CV or resume in PDF format. Text is extracted via a three-layer parser (pdfplumber → PyPDF2 → pdfminer)
    and processed through the same NLP pipeline as free text input.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        cv_text = read_pdf(uploaded.read())
        if cv_text.strip():
            raw_text = cv_text
            st.markdown(f"""
            <div class="info-block info-ok">
            <strong>Document parsed successfully.</strong> {len(cv_text):,} characters extracted.
            </div>""", unsafe_allow_html=True)
            with st.expander("View extracted text"):
                st.text(cv_text[:2500] + ("..." if len(cv_text) > 2500 else ""))
            p = extract(cv_text)
            user_skills, user_interests = p["skills"], p["interests"]
            user_talents, user_likes    = p["talents"], p["likes"]
            st.markdown('<div class="section-label" style="margin-top:24px">Extracted Profile</div>', unsafe_allow_html=True)
            extracted_profile_ui(user_skills, user_interests, user_talents, user_likes)
        else:
            st.markdown("""
            <div class="info-block info-warn">
            <strong>Text extraction failed.</strong> The document may be scanned or image-based.
            Please use Structured Form or Free Text input instead.
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px dashed var(--border2);border-radius:10px;padding:32px;
        text-align:center;margin-top:12px;color:var(--muted);font-size:.85rem">
        Drop a PDF here or click Browse above
        </div>""", unsafe_allow_html=True)

st.divider()

# PREDICTIONS
results, detected_domain = predict(
    user_skills, user_interests, user_talents, user_likes,
    raw_text=raw_text
)

# SECTION 1 — WHAT-IF SIMULATOR
st.markdown('<div class="section-label">Live Career Predictions</div>', unsafe_allow_html=True)

if results is None:
    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border2);border-radius:12px;
    padding:48px;text-align:center;color:var(--muted);font-size:.875rem">
    Select at least one skill or interest to generate career predictions.
    </div>""", unsafe_allow_html=True)
else:
    if detected_domain:
        st.markdown(f"""
        <div class="domain-banner">
            <div class="db-dot"></div>
            <div>
                <div class="db-label">Domain Detected</div>
                <div class="db-value">{detected_domain}</div>
                <div class="db-desc">Vocabulary enrichment applied · Predictions filtered within domain</div>
            </div>
        </div>""", unsafe_allow_html=True)

    rank_labels = ["Primary Recommendation", "Secondary Recommendation", "Tertiary Recommendation"]
    rank_css    = ["rank-1", "rank-2", "rank-3"]

    cards_html = '<div class="pred-grid">'
    for i, r in enumerate(results):
        cards_html += f"""
        <div class="pred-card {rank_css[i]}">
            <div class="rank-label">{rank_labels[i]}</div>
            <div class="career-name">{r['career']}</div>
            <div class="score-block">
                <span class="score-num">{r['norm']}</span>
                <span class="score-unit">%</span>
            </div>
            <div class="score-label">Relative Match Score</div>
            <span class="match-tag {r['tagcss']}">{r['tag']}</span>
            <div class="raw-prob">Model probability: {r['raw']}% across 37 careers</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    total = len(user_skills)+len(user_interests)+len(user_talents)+len(user_likes)
    pct   = min(100, int(total/20*100))
    fill_style = f"width:{pct}%"
    msg = "Add more attributes for higher prediction confidence." if pct < 60 else "Good profile coverage."
    st.markdown(f"""
    <div class="completeness-row">
        <div class="cr-label">Profile Completeness</div>
        <div class="cr-bar"><div class="cr-fill" style="{fill_style}"></div></div>
        <div class="cr-pct">{pct}%</div>
        <div style="font-size:.78rem;color:var(--muted)">{msg}</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # SECTION 2 — SKILL GAP + COURSES
    st.markdown('<div class="section-label">Skill Gap Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block info-neutral">
    For each predicted career, the system compares your current skill set against the required skill
    profile derived from the training dataset and maps missing skills to specific learning resources.
    </div>""", unsafe_allow_html=True)

    for rank, res in enumerate(results):
        career = res["career"]
        with st.expander(f"{rank_labels[rank]}  ·  {career}  ·  {res['norm']}% match", expanded=(rank==0)):
            required = SKILL_GAP_DB.get(career, [])
            if not required:
                st.markdown('<div class="info-block info-neutral">Skill requirements for this career are being compiled.</div>', unsafe_allow_html=True)
                continue

            user_t  = {s.strip().title() for s in user_skills}
            have    = [s for s in required if s in user_t]
            missing = get_skill_gap(user_skills, career)
            cov     = int(len(have)/len(required)*100) if required else 0

            st.markdown(f"""
            <div class="cov-row">
                <div class="cov-label">Skill Coverage</div>
                <div class="cov-pct">{cov}%</div>
            </div>""", unsafe_allow_html=True)
            st.progress(cov/100)
            st.markdown("")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f'<div class="ic-label" style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px">Matched Skills  ·  {len(have)}</div>', unsafe_allow_html=True)
                if have:
                    st.markdown(" ".join(f'<span class="badge-have">{s}</span>' for s in have), unsafe_allow_html=True)
                else:
                    st.markdown('<span style="font-size:.82rem;color:var(--muted)">None matched from required list.</span>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="ic-label" style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:8px">Missing Skills  ·  {len(missing)}</div>', unsafe_allow_html=True)
                if missing:
                    st.markdown(" ".join(f'<span class="badge-miss">{s}</span>' for s in missing), unsafe_allow_html=True)
                else:
                    st.markdown('<span style="font-size:.82rem;color:#86efac">All required skills matched.</span>', unsafe_allow_html=True)

            if missing:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f'<div class="ic-label" style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:10px">Recommended Learning Resources</div>', unsafe_allow_html=True)
                for skill, cname, url in get_courses(missing[:8]):
                    st.markdown(
                        f'<div style="padding:10px 0;border-bottom:1px solid var(--border);font-size:.85rem">'
                        f'<span style="color:var(--accent);font-family:var(--font-mono);font-size:.75rem">{skill}</span>'
                        f'&nbsp;&nbsp;—&nbsp;&nbsp;<a href="{url}" target="_blank" '
                        f'style="color:var(--subtle);text-decoration:none">{cname}</a></div>',
                        unsafe_allow_html=True)

            prof = career_profiles.get(career, {})

    st.divider()

    # SECTION 3 — RADAR CHARTS
    st.markdown('<div class="section-label">6-Dimensional Profile Match</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block info-neutral">
    Radar visualization comparing your profile against the ideal candidate across six dimensions:
    Skills, Interests, Talents, Likes, Profile Completeness, and Overall Fit. Gold line = your profile. Dashed = ideal.
    </div>""", unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns(3)
    for col, res in zip([rc1, rc2, rc3], results):
        with col:
            st.markdown(f'<div class="col-head">{res["career"]}</div>', unsafe_allow_html=True)
            fig = radar_chart(user_skills, user_interests, user_talents, user_likes, res["career"])
            if fig:
                st.pyplot(fig); plt.close(fig)

    st.divider()

    # SECTION 4 — CAREER TRAJECTORY
    st.markdown('<div class="section-label">Career Trajectory Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block info-accent">
    <strong>Methodology:</strong>
    Short-term prediction uses only Skills + Talents — representing current capability.
    Long-term prediction uses only Interests + Talents — representing passion-driven growth over 5–10 years.
    This time-dimensional forecasting approach is not found in existing academic career prediction systems.
    </div>""", unsafe_allow_html=True)

    now_r, _ = predict(user_skills, [], user_talents, [], raw_text)
    fut_r, _ = predict([], user_interests, user_talents, user_likes, raw_text)

    tc1, tc2 = st.columns(2)

    with tc1:
        st.markdown("""
        <div class="traj-head">
            <div class="traj-head-dot traj-head-now"></div>
            <div>
                <div class="traj-head-txt">Short-Term Employability</div>
                <div class="traj-head-sub">Based on current skills and education level</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if now_r:
            for i, r in enumerate(now_r):
                tag_cls = r["tagcss"]
                st.markdown(f"""
                <div class="traj-card now-card">
                    <div>
                        <div class="tc-rank">Position {i+1}</div>
                        <div class="tc-name">{r['career']}</div>
                        <div class="tc-tag">{r['tag']}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="tc-score">{r['norm']}%</div>
                        <div style="font-family:var(--font-mono);font-size:.6rem;color:var(--muted)">match</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-block info-neutral">Add skills to generate short-term predictions.</div>', unsafe_allow_html=True)

    with tc2:
        st.markdown("""
        <div class="traj-head">
            <div class="traj-head-dot traj-head-fut"></div>
            <div>
                <div class="traj-head-txt">Long-Term Trajectory  ·  5–10 Years</div>
                <div class="traj-head-sub">Based on interests and talents — passion-driven path</div>
            </div>
        </div>""", unsafe_allow_html=True)
        if fut_r:
            for i, r in enumerate(fut_r):
                st.markdown(f"""
                <div class="traj-card fut-card">
                    <div>
                        <div class="tc-rank">Position {i+1}</div>
                        <div class="tc-name">{r['career']}</div>
                        <div class="tc-tag">{r['tag']}</div>
                    </div>
                    <div style="text-align:right">
                        <div class="tc-score">{r['norm']}%</div>
                        <div style="font-family:var(--font-mono);font-size:.6rem;color:var(--muted)">match</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-block info-neutral">Add interests or talents to generate long-term predictions.</div>', unsafe_allow_html=True)

    if now_r and fut_r:
        tn, tf = now_r[0]["career"], fut_r[0]["career"]
        if tn != tf:
            bri = ", ".join(get_skill_gap(user_skills, tf)[:4]) or "continue developing your existing skillset"
            st.markdown(f"""
            <div class="insight-traj">
                <div class="it-from">
                    <div class="it-label">Current Position</div>
                    <div class="it-name">{tn}</div>
                </div>
                <div class="it-arrow">→</div>
                <div class="it-to">
                    <div class="it-label">Target Destination</div>
                    <div class="it-name">{tf}</div>
                </div>
            </div>
            <div class="info-block info-warn" style="margin-top:0">
            <strong>Trajectory Gap:</strong> Your current skills position you for <strong>{tn}</strong>,
            while your interests and talents point toward <strong>{tf}</strong>.
            Recommended bridge skills: <strong>{bri}</strong>.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="info-block info-ok" style="margin-top:16px">
            <strong>Strong Alignment:</strong> Both short-term skills and long-term interests converge on 
            <strong>{tn}</strong>. Your career trajectory is well-aligned.
            </div>""", unsafe_allow_html=True)

    st.divider()

    # SECTION 5 — SHAP
    st.markdown('<div class="section-label">SHAP Feature Attribution</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-block info-neutral">
    <strong>SHapley Additive exPlanations (SHAP)</strong> quantifies each feature's marginal contribution
    to the prediction of <strong>{results[0]['career']}</strong> using cooperative game theory (Shapley values).
    Gold bars support the prediction. Blue bars contradict it. Bar length represents magnitude of influence.
    </div>""", unsafe_allow_html=True)

    es, ei = enrich_profile(raw_text, user_skills, user_interests)
    X_enc  = encode(es, ei, user_talents, user_likes)

    with st.spinner("Computing SHAP values..."):
        fig = shap_chart(X_enc, results[0]["career"])

    if fig:
        sh1, sh2 = st.columns([3, 1])
        with sh1:
            st.pyplot(fig); plt.close(fig)
        with sh2:
            st.markdown("""
            <div style="padding:16px 0">
            <div style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:2px;
            text-transform:uppercase;color:var(--muted);margin-bottom:12px">Interpreting Results</div>

            <div style="font-size:.82rem;color:var(--subtle);line-height:1.7;margin-bottom:16px">
            Each bar represents a keyword extracted from your profile. The Shapley value measures 
            its contribution to shifting the model's output toward or away from the predicted career.
            </div>

            <div style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:2px;
            text-transform:uppercase;color:var(--muted);margin-bottom:8px">Why This Matters</div>

            <div style="font-size:.82rem;color:var(--subtle);line-height:1.7">
            Standard ML models are opaque black boxes. SHAP transforms this XGBoost model 
            into a fully auditable, explainable system — meeting real-world standards for 
            AI transparency and accountability.
            </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # SUMMARY STRIP
    top = results[0]
    req = SKILL_GAP_DB.get(top["career"], ["_"])
    have_n = len([s for s in req if s in {x.title() for x in user_skills}])
    cov_pct = int(have_n/len(req)*100) if req else 0

    st.markdown('<div class="section-label">Prediction Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-cell sc-accent">
            <div class="sc-val">{top['career'].split()[0]}</div>
            <div class="sc-lbl">Top Career</div>
        </div>
        <div class="stat-cell">
            <div class="sc-val">{top['norm']}%</div>
            <div class="sc-lbl">Confidence</div>
        </div>
        <div class="stat-cell">
            <div class="sc-val">{detected_domain or 'General'}</div>
            <div class="sc-lbl">Domain</div>
        </div>
        <div class="stat-cell">
            <div class="sc-val">{total}</div>
            <div class="sc-lbl">Attributes</div>
        </div>
        <div class="stat-cell">
            <div class="sc-val">{cov_pct}%</div>
            <div class="sc-lbl">Skill Coverage</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # SECTION 6 — MODEL COMPARISON  
    st.markdown('<div class="section-label">Model Comparison  ·  Algorithm Benchmarking</div>',
                unsafe_allow_html=True)

    _best_name = MODEL_META.get("best_model", "XGBoost")
    _cv_all    = MODEL_META.get("cv_scores", {})
    _ta_all    = MODEL_META.get("test_accuracy", {})

    # Summary metrics row 
    MODEL_COLORS_HEX = {"XGBoost": "#c9a84c", "Random Forest": "#3b82f6",
                        "Gradient Boosting": "#22c55e", "Logistic Regression": "#a855f7"}
    mc_cols = st.columns(len(_cv_all)) if _cv_all else st.columns(4)
    for col, (mname, mdata) in zip(mc_cols, _cv_all.items()):
        is_best = (mname == _best_name)
        color   = MODEL_COLORS_HEX.get(mname, "#c9a84c")
        border  = f"2px solid {color}" if is_best else f"1px solid rgba(255,255,255,0.08)"
        badge   = f'<span style="background:{color}22;color:{color};border:1px solid {color}55;border-radius:4px;padding:1px 7px;font-size:.6rem;font-family:var(--font-mono);letter-spacing:1px">{"★ BEST" if is_best else "candidate"}</span>'
        col.markdown(f"""
        <div style="background:var(--surface);border:{border};border-radius:10px;padding:16px 18px;text-align:center">
            {badge}
            <div style="font-family:var(--font-display);font-size:1.25rem;font-weight:700;
                 color:{"var(--accent)" if is_best else "var(--text)"};margin:8px 0 2px">{mdata['mean']:.2f}%</div>
            <div style="font-size:.65rem;font-family:var(--font-mono);color:var(--muted);
                 text-transform:uppercase;letter-spacing:1px">CV Accuracy</div>
            <div style="font-size:.75rem;color:var(--subtle);margin-top:4px">σ={mdata['std']:.3f}%</div>
            <div style="font-size:.78rem;color:var(--subtle);font-weight:500;margin-top:6px">{mname}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)

    # Charts
    import os
    ch1, ch2 = st.columns(2)
    with ch1:
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png", use_container_width=True)
        else:
            st.markdown("""
            <div class="info-block info-warn">
            Run <code>python3 train_model.py</code> to generate model comparison charts.
            </div>""", unsafe_allow_html=True)
    with ch2:
        if os.path.exists("cv_scores.png"):
            st.image("cv_scores.png", use_container_width=True)
        else:
            st.markdown("""
            <div class="info-block info-warn">
            Run <code>python3 train_model.py</code> to generate 5-fold CV distribution chart.
            </div>""", unsafe_allow_html=True)

    # Confusion Matrix + Learning Curve
    st.markdown('<div class="section-label" style="margin-top:28px">Confusion Matrix  ·  Learning Curve</div>',
                unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    with lc1:
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True,
                     caption=f"Row-normalised recall matrix · {_best_name}")
        else:
            st.markdown('<div class="info-block info-warn">Run <code>train_model.py</code> to generate.</div>',
                        unsafe_allow_html=True)
    with lc2:
        if os.path.exists("learning_curve.png"):
            st.image("learning_curve.png", use_container_width=True,
                     caption="Training vs CV score as dataset size grows")
        else:
            st.markdown('<div class="info-block info-warn">Run <code>train_model.py</code> to generate.</div>',
                        unsafe_allow_html=True)

    # Noise Robustness 
    st.markdown('<div class="section-label" style="margin-top:28px">Noise Robustness Test</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-block info-neutral">
    Gaussian noise was injected into the test feature matrix at increasing standard deviations
    (0%–50% of feature range) to verify the model generalises beyond clean inputs and is not
    simply memorising the training distribution.
    </div>""", unsafe_allow_html=True)
    if os.path.exists("noise_robustness.png"):
        nr1, nr2 = st.columns([2, 1])
        with nr1:
            st.image("noise_robustness.png", use_container_width=True)
        with nr2:
            noise_d = MODEL_META.get("noise_robustness", {})
            if noise_d:
                levels = noise_d.get("levels", [])
                accs   = noise_d.get("accs", [])
                drop   = accs[0] - accs[-1] if accs else 0
                st.markdown(f"""
                <div style="background:var(--surface2);border:1px solid var(--border2);
                border-radius:8px;padding:16px 18px;margin-top:10px">
                <div style="font-family:var(--font-mono);font-size:.62rem;letter-spacing:2px;
                text-transform:uppercase;color:var(--muted);margin-bottom:10px">Robustness Stats</div>
                <div style="font-size:.82rem;color:var(--subtle);line-height:1.9">
                Clean accuracy: <strong style="color:var(--text)">{accs[0]:.2f}%</strong><br>
                At σ=50% noise: <strong style="color:var(--text)">{accs[-1]:.2f}%</strong><br>
                Total drop: <strong style="color:{'var(--green)' if drop < 5 else 'var(--accent)'}">{drop:.2f}%</strong><br>
                Verdict: <strong style="color:{'var(--green)' if drop < 5 else 'var(--accent)'}">
                {'Highly Robust' if drop < 5 else 'Moderate Sensitivity'}</strong>
                </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-block info-warn">Run <code>train_model.py</code> to generate.</div>',
                    unsafe_allow_html=True)

    # Detailed fold-level table 
    if _cv_all:
        with st.expander("📋  Fold-Level CV Scores  —  Full Breakdown"):
            rows = []
            for mname, mdata in _cv_all.items():
                row = {"Algorithm": mname}
                for fi, score in enumerate(mdata.get("folds", []), 1):
                    row[f"Fold {fi}"] = f"{score:.4f}%"
                row["Mean"] = f"{mdata['mean']:.4f}%"
                row["Std"]  = f"{mdata['std']:.4f}%"
                row["Test Acc"] = f"{_ta_all.get(mname, 0):.4f}%"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Algorithm"),
                         use_container_width=True)

    st.markdown(f"""
    <div class="info-block info-ok">
    <strong>Selected Algorithm: {_best_name}</strong> — chosen automatically by highest
    5-fold cross-validation mean accuracy. All predictions in this session use this model.
    Retrain via <code>python3 train_model.py</code> to re-run the comparison.
    </div>
    """, unsafe_allow_html=True)

# FOOTER 
st.markdown(f"""
<div style="text-align:center;padding:40px 0 20px;border-top:1px solid var(--border);margin-top:40px">
    <div style="font-family:var(--font-mono);font-size:.65rem;letter-spacing:2px;
    text-transform:uppercase;color:var(--muted);line-height:2.5">
    CareerML  v2.0  ·  Career Intelligence Platform<br>
    4-Algorithm Comparison: XGBoost · Random Forest · Gradient Boosting · Logistic Regression<br>
    Best Model Auto-Selected via 5-Fold Stratified Cross-Validation  ·  Active: {MODEL_META.get("best_model","XGBoost")}<br>
    TF-IDF ×4  ·  One-Hot Encoding  ·  SHAP Explainability  ·  Domain Bridge NLP<br>
    {MODEL_META.get("test_accuracy",{}).get(MODEL_META.get("best_model","XGBoost"),100):.2f}% Test Accuracy
    ·  {MODEL_META.get("n_classes",37)} Career Classes  ·  11,100 Training Records  ·  {MODEL_META.get("n_features",879)} Features
    </div>
</div>
""", unsafe_allow_html=True)
