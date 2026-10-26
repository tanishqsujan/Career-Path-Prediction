# CareerML 🎯  v2.0
### A Multi-Class ML Framework with 4-Algorithm Comparison + 5-Fold Cross-Validation

**Team:** Tanishq Sujan · Sujal · Salman Mohammad  
**Pipeline:** XGBoost · Random Forest · Gradient Boosting · Logistic Regression  
**Selection:** Best model auto-chosen via 5-Fold Stratified CV  
**Accuracy:** 100% · 37 Career Classes · 11,100 Records

---

## 🚀 Setup — Do This Exactly Once

### Step 1 — Install packages
```bash
pip install -r requirements.txt
```

### Step 2 — Train & compare models (run once, ~3–5 minutes)
```bash
python3 train_model.py
```
This trains **4 algorithms**, runs **5-fold cross-validation** on each, picks the best, and generates:

| Output File | Purpose |
|---|---|
| `model.pkl` | Best-performing model (auto-selected) |
| `model_meta.json` | CV scores, test accuracy, training times for all 4 models |
| `model_comparison.png` | Bar chart: CV accuracy + test accuracy side-by-side |
| `cv_scores.png` | Box plot: fold-level score distribution for all 4 models |
| `tfidf_s/i/t/l.pkl` | TF-IDF vectorizers (Skills, Interests, Talents, Likes) |
| `ohe.pkl` | One-Hot Encoder (Education, Specialization) |
| `le.pkl` | Label Encoder (37 career classes) |
| `vocab.json` | Dropdown vocabulary for the UI |

### Step 3 — Launch the app
```bash
streamlit run app.py
```
Open **http://localhost:8501**

---

## 🤖 Algorithm Comparison

| Algorithm | Type | Key Strengths for this Task |
|---|---|---|
| **XGBoost** | Gradient Boosted Trees | Native sparse support, fast, SHAP-compatible |
| **Random Forest** | Bagged Trees | Low variance, robust, parallel |
| **Gradient Boosting** | HistGradientBoosting (sklearn) | Fastest sklearn GBM, handles large feature sets |
| **Logistic Regression** | Linear | Excellent TF-IDF baseline, highly interpretable |

The best model is selected automatically by **highest 5-fold CV mean accuracy**.  
If accuracy is tied, the first in the list above wins (tie-break by insertion order).

---

## 📊 What train_model.py v2.0 Produces

### Phase 1 — 5-Fold Stratified Cross-Validation
- Each model is evaluated on 5 shuffled, stratified splits
- Reports fold-level scores, mean, and standard deviation

### Phase 2 — Full Train → Test Evaluation  
- Each model trained on 80% of data, tested on held-out 20%
- Detailed classification report printed for the best model

### Charts Generated
- **model_comparison.png** — Left: CV accuracy with error bars (mean ± std). Right: Test accuracy. Best model highlighted in gold.
- **cv_scores.png** — Box plot with individual fold dots. μ/σ annotations per model. Best model starred.

---

## ✨ App Features (unchanged)

### 1 ⚡ What-If Career Simulator
Real-time Top 3 career predictions — updates instantly on every widget change.

### 2 📚 Skill Gap Analysis + Course Recommendations
Missing skills per predicted career + direct Coursera/edX/Udemy links.

### 3 🛤️ Career Trajectory — Now vs 5–10 Years
Dual prediction using different feature subsets (Skills+Education vs Interests+Talents).

### 4 🎙️ Multi-Modal Input
Three input modes: Structured Form / Free Text NLP / CV PDF Upload.

### 5 📊 Model Comparison Dashboard *(NEW in v2.0)*
Full benchmark: metric cards per model, comparison chart, CV distribution chart, fold-level score table.

### 🔍 SHAP Explainability
Adapts to the selected best model — TreeExplainer for tree-based models, coefficient attribution for linear models.

---

## 🏗️ Architecture v2.0

```
                  train_model.py
          ┌────────────────────────────┐
          │  4 Candidate Models        │
          │  ┌──────────────────────┐  │
          │  │ XGBoost              │  │
          │  │ Random Forest        │  │
          │  │ Gradient Boosting    │  │
          │  │ Logistic Regression  │  │
          │  └──────────────────────┘  │
          │  5-Fold CV ──► Best Model  │
          │  Charts: comparison + cv   │
          └──────────┬─────────────────┘
                     │  model.pkl + model_meta.json
                     ▼
              app.py (Streamlit)
     ┌─────────────────────────────────────────┐
     │  Encoding Pipeline                      │
     │  TF-IDF ×4  +  OneHotEncoder            │
     │               ↓  879+ features          │
     │         Best Model (auto)               │
     │               ↓                         │
     │  Top 3 Careers  +  Skill Gap            │
     │  Trajectory      +  SHAP                │
     │  Model Comparison Dashboard             │
     └─────────────────────────────────────────┘
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| CV Accuracy (best model) | **100%** |
| Test Accuracy | **100%** |
| CV Std Dev | **0.000%** |
| Macro F1 | **1.00** |
| Test samples | 2,220 |
| Train samples | 8,880 |
| Feature dimensions | 879+ |

---

## 🙋 Q&A Cheatsheet

**Q: Why compare 4 algorithms instead of just using XGBoost?**  
A: XGBoost was a fixed choice. v2.0 lets the data decide — all 4 are trained and evaluated under identical conditions (same splits, same 5-fold CV seed), and the winner is used for predictions automatically.

**Q: What if two models tie on CV accuracy?**  
A: Python's `max()` preserves insertion order on ties, so XGBoost > Random Forest > Gradient Boosting > Logistic Regression in a tie.

**Q: Does SHAP still work if Logistic Regression wins?**  
A: Yes. The app detects `is_tree` from `model_meta.json` and uses coefficient × feature-value attribution for linear models instead of TreeExplainer.

**Q: Can I add more algorithms?**  
A: Yes — add entries to the `candidates` dict in `train_model.py`. No changes to `app.py` needed.
