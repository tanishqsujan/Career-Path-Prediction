import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing          import LabelEncoder, OneHotEncoder
from sklearn.model_selection        import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics                import accuracy_score, classification_report
from sklearn.ensemble               import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model           import LogisticRegression
from xgboost                        import XGBClassifier
from scipy.sparse                   import hstack, issparse
import joblib, json, time, warnings

warnings.filterwarnings("ignore")

# Color palette 
DARK         = "#060a12"
SURF         = "#0c1220"
SURF2        = "#111827"
BORDER       = "#1a2744"
MUTED        = "#5a6478"
SUBTLE       = "#8b95a8"
TEXT         = "#e8edf5"
ACCENT       = "#c9a84c"
BLUE         = "#3b82f6"
GREEN        = "#22c55e"
RED          = "#ef4444"
PURPLE       = "#a855f7"
MODEL_COLORS = [ACCENT, BLUE, GREEN, PURPLE]

print("=" * 60)
print("  CareerML — Multi-Model Training Pipeline  v2.0")
print("=" * 60)

# 1. LOAD DATASET
df = pd.read_csv("career_dataset_fixed.csv")
print(f"\n✅  Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")

for col in ["Skills", "Interests", "Talents", "Likes"]:
    df[col] = df[col].fillna("").str.lower()

# 2. VOCABULARY FOR UI DROPDOWNS
def unique_terms(series):
    terms = set()
    for row in series:
        for t in row.split(","):
            t = t.strip()
            if t:
                terms.add(t.title())
    return sorted(terms)

vocab = {
    "skills":    unique_terms(df["Skills"]),
    "interests": unique_terms(df["Interests"]),
    "talents":   unique_terms(df["Talents"]),
    "likes":     unique_terms(df["Likes"]),
}
print(f"📚  Vocab — Skills: {len(vocab['skills'])}  |  "
      f"Interests: {len(vocab['interests'])}  |  "
      f"Talents: {len(vocab['talents'])}  |  Likes: {len(vocab['likes'])}")


# 3. FEATURE ENGINEERING 
le = LabelEncoder()
y  = le.fit_transform(df["Recommended Career"])
print(f"  Target classes: {len(le.classes_)}")

# Split RAW TEXT 
X_train_df, X_test_df, y_train, y_test = train_test_split(
    df[["Skills", "Interests", "Talents", "Likes"]],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Initialize TF-IDF
tfidf_s = TfidfVectorizer(max_features=400, ngram_range=(1, 2))
tfidf_i = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
tfidf_t = TfidfVectorizer(max_features=200, ngram_range=(1, 2))
tfidf_l = TfidfVectorizer(max_features=200, ngram_range=(1, 2))

# FIT only on TRAIN
X_s_train = tfidf_s.fit_transform(X_train_df["Skills"])
X_i_train = tfidf_i.fit_transform(X_train_df["Interests"])
X_t_train = tfidf_t.fit_transform(X_train_df["Talents"])
X_l_train = tfidf_l.fit_transform(X_train_df["Likes"])

# TRANSFORM test
X_s_test = tfidf_s.transform(X_test_df["Skills"])
X_i_test = tfidf_i.transform(X_test_df["Interests"])
X_t_test = tfidf_t.transform(X_test_df["Talents"])
X_l_test = tfidf_l.transform(X_test_df["Likes"])

# Combine
X_train = hstack([X_s_train, X_i_train, X_t_train, X_l_train])
X_test  = hstack([X_s_test,  X_i_test,  X_t_test,  X_l_test])

# Also create FULL X (for CV only — using TRAIN-FITTED vectorizers)
X_full_s = tfidf_s.transform(df["Skills"])
X_full_i = tfidf_i.transform(df["Interests"])
X_full_t = tfidf_t.transform(df["Talents"])
X_full_l = tfidf_l.transform(df["Likes"])
X = hstack([X_full_s, X_full_i, X_full_t, X_full_l])

print(f"  Feature matrix: {X.shape}")
print(f"   Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}\n")

# 4. DEFINE 4 MODELS
candidates = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="saga",
        multi_class="multinomial",
        random_state=42,
        n_jobs=-1,
    ),
}

# HistGradientBoosting requires dense input
def _to_input(name, mat):
    if name == "Gradient Boosting":
        return mat.toarray() if issparse(mat) else mat
    return mat

# 5. 5-FOLD CROSS-VALIDATION
print("-" * 60)
print("  Phase 1 — 5-Fold Stratified Cross-Validation")
print("-" * 60)

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
cv_times  = {}

for name, clf in candidates.items():
    X_cv = _to_input(name, X_train)
    print(f"\n⏳  [{name}] running 5-fold CV …", flush=True)
    t0     = time.time()
    scores = cross_val_score(clf, X_cv, y_train, cv=cv, scoring="accuracy", n_jobs=1)
    elapsed = time.time() - t0
    cv_scores[name] = scores
    cv_times[name]  = round(elapsed, 1)
    print(f"    Fold scores : {np.round(scores * 100, 2)}")
    print(f"    Mean ± Std  : {scores.mean()*100:.2f}% ± {scores.std()*100:.3f}%  [{elapsed:.1f}s]")

# 6. TRAIN ON FULL TRAIN SET + TEST SET EVALUATION
print("\n" + "-" * 60)
print("  Phase 2 — Full Train → Test Set Evaluation")
print("-" * 60)

test_acc   = {}
train_time = {}
models     = {}

for name, clf in candidates.items():
    X_tr = _to_input(name, X_train)
    X_te = _to_input(name, X_test)
    print(f"\n⏳  [{name}] training …", flush=True)
    t0 = time.time()
    clf.fit(X_tr, y_train)
    elapsed = time.time() - t0
    acc = accuracy_score(y_test, clf.predict(X_te))
    test_acc[name]   = round(acc * 100, 4)
    train_time[name] = round(elapsed, 1)
    models[name]     = clf
    print(f"    Test Accuracy : {acc*100:.2f}%  [{elapsed:.1f}s]")

# 7. PICK BEST MODEL  (by 5-fold CV mean accuracy)
best_name  = max(cv_scores, key=lambda n: cv_scores[n].mean())
best_model = models[best_name]

print("\n" + "=" * 60)
print(f"  BEST MODEL: {best_name}")
print(f"      CV Mean Accuracy : {cv_scores[best_name].mean()*100:.2f}%")
print(f"      Test Accuracy    : {test_acc[best_name]:.2f}%")
print("=" * 60)

# 8. REPORT FOR BEST MODEL
X_te_best = _to_input(best_name, X_test)
y_pred    = best_model.predict(X_te_best)
print(f"\n📋  Classification Report — {best_name}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9. CHARTS
names     = list(candidates.keys())

def _fig_style(fig, ax_list):
    fig.patch.set_facecolor(DARK)
    for ax in ax_list:
        ax.set_facecolor(SURF)
        ax.tick_params(colors=SUBTLE, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)

# Chart 1: Model Comparison Bar Chart 
cv_means  = [cv_scores[n].mean() * 100 for n in names]
cv_stds   = [cv_scores[n].std()  * 100 for n in names]
test_accs = [test_acc[n]              for n in names]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
_fig_style(fig, axes)
fig.suptitle("Model Comparison  ·  CareerML", fontsize=13, fontweight="bold",
             color=TEXT, fontfamily="sans-serif", y=1.01)

ax = axes[0]
bars = ax.bar(names, cv_means, color=MODEL_COLORS, width=0.5,
              edgecolor="none", zorder=3)
ax.errorbar(names, cv_means, yerr=cv_stds, fmt="none",
            ecolor=TEXT, elinewidth=1.5, capsize=5, capthick=1.5, zorder=4)
for bar, val, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + std + 0.3,
            f"{val:.2f}%", ha="center", va="bottom",
            fontsize=8, color=TEXT, fontfamily="monospace", fontweight="bold")

best_idx = names.index(best_name)
bars[best_idx].set_edgecolor(ACCENT)
bars[best_idx].set_linewidth(2.5)

y_min = max(0, min(cv_means) - 3)
ax.set_ylim(y_min, max(cv_means) + 4)
ax.set_ylabel("Accuracy (%)", fontsize=8.5)
ax.set_title("5-Fold CV Accuracy  (mean ± std)", fontsize=9.5, pad=12)
ax.tick_params(axis="x", labelsize=8)
ax.yaxis.grid(True, color=BORDER, lw=0.7, linestyle="--", zorder=0)
ax.set_axisbelow(True)

star = mpatches.Patch(facecolor=MODEL_COLORS[best_idx], edgecolor=ACCENT,
                      linewidth=2, label=f"★ Best: {best_name}")
ax.legend(handles=[star], fontsize=7.5, facecolor=SURF, edgecolor=BORDER,
          labelcolor=TEXT, framealpha=1, loc="lower right")

ax2 = axes[1]
bars2 = ax2.bar(names, test_accs, color=MODEL_COLORS, width=0.5,
                edgecolor="none", zorder=3)
for bar, val in zip(bars2, test_accs):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             val + 0.1,
             f"{val:.2f}%", ha="center", va="bottom",
             fontsize=8, color=TEXT, fontfamily="monospace", fontweight="bold")
bars2[best_idx].set_edgecolor(ACCENT)
bars2[best_idx].set_linewidth(2.5)

y_min2 = max(0, min(test_accs) - 3)
ax2.set_ylim(y_min2, max(test_accs) + 4)
ax2.set_ylabel("Accuracy (%)", fontsize=8.5)
ax2.set_title("Test Set Accuracy", fontsize=9.5, pad=12)
ax2.tick_params(axis="x", labelsize=8)
ax2.yaxis.grid(True, color=BORDER, lw=0.7, linestyle="--", zorder=0)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=DARK, edgecolor="none")
plt.close()
print("\n  Saved: model_comparison.png")

# Chart 2: 5-Fold CV Score Distribution (Box Plot) 
fig2, ax3 = plt.subplots(figsize=(11, 5.5))
_fig_style(fig2, [ax3])
fig2.suptitle("5-Fold Cross-Validation Score Distribution  ·  CareerML",
              fontsize=12, fontweight="bold", color=TEXT, y=1.01)

score_data = [cv_scores[n] * 100 for n in names]

bp = ax3.boxplot(score_data, patch_artist=True, notch=False,
                 widths=0.45, showfliers=True,
                 medianprops=dict(color=TEXT, linewidth=2.2),
                 whiskerprops=dict(color=SUBTLE, linewidth=1.2),
                 capprops=dict(color=SUBTLE, linewidth=1.2),
                 flierprops=dict(marker="o", color=SUBTLE, markersize=5,
                                 markerfacecolor=SURF2, linewidth=0.8))

for patch, color in zip(bp["boxes"], MODEL_COLORS):
    patch.set_facecolor(color + "33")
    patch.set_edgecolor(color)
    patch.set_linewidth(1.8)

for i, (scores, color) in enumerate(zip(score_data, MODEL_COLORS), start=1):
    x = np.random.normal(i, 0.06, size=len(scores))
    ax3.scatter(x, scores, color=color, s=50, zorder=4, alpha=0.95,
                edgecolors="none")

ax3.set_xticks(range(1, len(names) + 1))
ax3.set_xticklabels(names, fontsize=9.5)
ax3.set_ylabel("Accuracy (%)", fontsize=8.5)
ax3.set_title("Each dot = one fold  ·  Box = IQR  ·  Centre line = median",
              fontsize=8, color=SUBTLE, pad=10)

y_lo = max(0,   min(min(s) for s in score_data) - 2)
y_hi = min(102, max(max(s) for s in score_data) + 4)
ax3.set_ylim(y_lo, y_hi)
ax3.yaxis.grid(True, color=BORDER, lw=0.7, linestyle="--", zorder=0)
ax3.set_axisbelow(True)

for i, (n, color) in enumerate(zip(names, MODEL_COLORS), start=1):
    m = cv_scores[n].mean() * 100
    s = cv_scores[n].std()  * 100
    is_best = (n == best_name)
    label_text = f"{'★ ' if is_best else ''}μ={m:.2f}%\nσ={s:.3f}%"
    ec = ACCENT if is_best else color + "55"
    lw = 1.5 if is_best else 0.8
    ax3.text(i, y_hi - 0.6,
             label_text,
             ha="center", va="top", fontsize=7.5,
             color=ACCENT if is_best else color,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.35", facecolor=SURF,
                       edgecolor=ec, linewidth=lw, alpha=0.95))

plt.tight_layout()
plt.savefig("cv_scores.png", dpi=150, bbox_inches="tight",
            facecolor=DARK, edgecolor="none")
plt.close()
print("  Saved: cv_scores.png")

# Chart 3: Confusion Matrix (best model) 
from sklearn.metrics import confusion_matrix
print("\n  Generating confusion matrix …", flush=True)
X_te_b  = _to_input(best_name, X_test)
y_pred  = best_model.predict(X_te_b)
cm      = confusion_matrix(y_test, y_pred)
n_cls   = len(le.classes_)
fig3, ax4 = plt.subplots(figsize=(14, 12))
_fig_style(fig3, [ax4])
fig3.suptitle(f"Confusion Matrix  ·  {best_name}  ·  CareerML",
              fontsize=12, fontweight="bold", color=TEXT, y=1.005)
# Normalise to 0-1 per row so colour is readable even at 100%
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
im = ax4.imshow(cm_norm, cmap="YlOrBr", aspect="auto", vmin=0, vmax=1)
cb = fig3.colorbar(im, ax=ax4, fraction=0.025, pad=0.02)
cb.ax.yaxis.set_tick_params(color=SUBTLE, labelsize=7)
cb.set_label("Recall (row-normalised)", color=MUTED, fontsize=8)
for r in range(n_cls):
    for c in range(n_cls):
        v   = cm[r, c]
        txt = str(v) if v > 0 else ""
        col = DARK if cm_norm[r, c] > 0.5 else TEXT
        if txt:
            ax4.text(c, r, txt, ha="center", va="center",
                     fontsize=5.5, color=col, fontfamily="monospace")
short_labels = [lbl[:14] for lbl in le.classes_]
ax4.set_xticks(range(n_cls)); ax4.set_xticklabels(short_labels, rotation=90, fontsize=6.5, color=SUBTLE)
ax4.set_yticks(range(n_cls)); ax4.set_yticklabels(short_labels, fontsize=6.5, color=SUBTLE)
ax4.set_xlabel("Predicted Label", fontsize=8.5, color=MUTED, labelpad=8)
ax4.set_ylabel("True Label",      fontsize=8.5, color=MUTED, labelpad=8)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight",
            facecolor=DARK, edgecolor="none")
plt.close()
print("  Saved: confusion_matrix.png")

# Chart 4: Learning Curve (best model) 
from sklearn.model_selection import learning_curve
print("  Generating learning curve …", flush=True)
train_sizes_pct = np.linspace(0.1, 1.0, 8)
X_lc = _to_input(best_name, X)
ts, tr_scores, val_scores = learning_curve(
    candidates[best_name], X_lc, y,
    train_sizes=train_sizes_pct,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1
)
tr_mean  = tr_scores.mean(axis=1)  * 100
tr_std   = tr_scores.std(axis=1)   * 100
val_mean = val_scores.mean(axis=1) * 100
val_std  = val_scores.std(axis=1)  * 100

fig4, ax5 = plt.subplots(figsize=(9, 5))
_fig_style(fig4, [ax5])
fig4.suptitle(f"Learning Curve  ·  {best_name}  ·  CareerML",
              fontsize=12, fontweight="bold", color=TEXT, y=1.01)
ax5.fill_between(ts, tr_mean - tr_std,  tr_mean + tr_std,  alpha=0.15, color=ACCENT)
ax5.fill_between(ts, val_mean - val_std, val_mean + val_std, alpha=0.15, color=BLUE)
ax5.plot(ts, tr_mean,  "o-", color=ACCENT, lw=2,   label="Training Score",   markersize=6)
ax5.plot(ts, val_mean, "s-", color=BLUE,   lw=2,   label="CV Score (3-fold)", markersize=6)
ax5.set_xlabel("Training Set Size", fontsize=9, color=MUTED, labelpad=8)
ax5.set_ylabel("Accuracy (%)",       fontsize=9, color=MUTED, labelpad=8)
ax5.yaxis.grid(True, color=BORDER, lw=0.7, linestyle="--", zorder=0)
ax5.set_axisbelow(True)
ax5.legend(fontsize=8.5, facecolor=SURF, edgecolor=BORDER, labelcolor=TEXT,
           framealpha=1, loc="lower right")
gap = abs(tr_mean[-1] - val_mean[-1])
note = (f"Train–CV gap at full data: {gap:.2f}%\n"
        f"{'Minimal gap — model generalises well.' if gap < 2 else 'Gap detected — consider regularisation.'}")
ax5.text(0.02, 0.05, note, transform=ax5.transAxes,
         fontsize=7.5, color=SUBTLE, fontfamily="monospace",
         verticalalignment="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=SURF, edgecolor=BORDER, alpha=0.9))
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150, bbox_inches="tight",
            facecolor=DARK, edgecolor="none")
plt.close()
print("  Saved: learning_curve.png")

# Chart 5: Noise Robustness Test 
print("  Running noise robustness test …", flush=True)
from scipy.sparse import csr_matrix, issparse as _isp2

noise_levels = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
noise_accs   = []
X_te_dense   = X_test.toarray() if _isp2(X_test) else X_test
rng = np.random.default_rng(42)
for noise in noise_levels:
    Xn = X_te_dense + rng.normal(0, noise, X_te_dense.shape)
    Xn = csr_matrix(Xn) if not (best_name == "Gradient Boosting") else Xn
    Xn_inp = _to_input(best_name, Xn)
    acc = accuracy_score(y_test, best_model.predict(Xn_inp))
    noise_accs.append(acc * 100)
    print(f"    Noise σ={noise:.2f} → {acc*100:.2f}%")

fig5, ax6 = plt.subplots(figsize=(9, 5))
_fig_style(fig5, [ax6])
fig5.suptitle(f"Noise Robustness  ·  {best_name}  ·  CareerML",
              fontsize=12, fontweight="bold", color=TEXT, y=1.01)
ax6.plot([n*100 for n in noise_levels], noise_accs,
         "D-", color=GREEN, lw=2.2, markersize=7, markerfacecolor=GREEN,
         markeredgecolor=DARK, markeredgewidth=1)
ax6.fill_between([n*100 for n in noise_levels], noise_accs,
                 min(noise_accs) - 2, alpha=0.12, color=GREEN)
for nx, ny in zip(noise_levels, noise_accs):
    ax6.text(nx*100, ny + 0.3, f"{ny:.1f}%", ha="center", va="bottom",
             fontsize=8, color=TEXT, fontfamily="monospace")
ax6.set_xlabel("Gaussian Noise Standard Deviation (% of feature range)",
               fontsize=9, color=MUTED, labelpad=8)
ax6.set_ylabel("Test Accuracy (%)", fontsize=9, color=MUTED, labelpad=8)
ax6.yaxis.grid(True, color=BORDER, lw=0.7, linestyle="--", zorder=0)
ax6.set_axisbelow(True)
drop = noise_accs[0] - noise_accs[-1]
robustness_note = (f"Accuracy drop at max noise: {drop:.2f}%\n"
                   f"{'Highly robust — tree ensemble handles noise well.' if drop < 5 else 'Moderate sensitivity to feature perturbation.'}")
ax6.text(0.98, 0.95, robustness_note, transform=ax6.transAxes,
         fontsize=7.5, color=SUBTLE, fontfamily="monospace",
         ha="right", va="top",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=SURF, edgecolor=BORDER, alpha=0.9))
plt.tight_layout()
plt.savefig("noise_robustness.png", dpi=150, bbox_inches="tight",
            facecolor=DARK, edgecolor="none")
plt.close()
print("  Saved: noise_robustness.png")

# Store noise results in meta for app display
noise_data = {"levels": [n*100 for n in noise_levels], "accs": noise_accs}

# 10. SAVE ALL ARTIFACTS
joblib.dump(best_model, "model.pkl")
joblib.dump(tfidf_s,    "tfidf_s.pkl")
joblib.dump(tfidf_i,    "tfidf_i.pkl")
joblib.dump(tfidf_t,    "tfidf_t.pkl")
joblib.dump(tfidf_l,    "tfidf_l.pkl")
joblib.dump(le,         "le.pkl")

with open("vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

meta = {
    "best_model":  best_name,
    "model_type":  best_model.__class__.__name__,
    "is_tree":     best_name in ("XGBoost", "Random Forest", "Gradient Boosting"),
    "cv_scores": {
        n: {
            "mean":  round(float(cv_scores[n].mean() * 100), 4),
            "std":   round(float(cv_scores[n].std()  * 100), 4),
            "folds": [round(float(v * 100), 4) for v in cv_scores[n]],
        }
        for n in names
    },
    "test_accuracy": test_acc,
    "train_time_s":  train_time,
    "n_classes":     int(len(le.classes_)),
    "n_features":    int(X.shape[1]),
    "n_train":       int(X_train.shape[0]),
    "n_test":        int(X_test.shape[0]),
    "noise_robustness": noise_data,
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n  All saved:")
print("    model.pkl  |  model_meta.json")
print("    tfidf_s/i/t/l.pkl  |  le.pkl  |  vocab.json")
print("    model_comparison.png  |  cv_scores.png")
print("    confusion_matrix.png  |  learning_curve.png  |  noise_robustness.png")
print("\n" + "=" * 60)
print(f"  BEST ALGORITHM  :  {best_name}")
print(f"  CV Accuracy     :  {cv_scores[best_name].mean()*100:.2f}% ± "
      f"{cv_scores[best_name].std()*100:.3f}%")
print(f"  Test Accuracy   :  {test_acc[best_name]:.2f}%")
print("=" * 60)
print("\n  Now run:  streamlit run app.py\n")
