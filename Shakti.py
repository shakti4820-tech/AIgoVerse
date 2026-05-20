import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, ConfusionMatrixDisplay
)
import plotly.express as px

# Import from modular files
from style import inject_style
from data_loader import DATASETS, load_dataset
from algos import CLASSIFIERS, ALGO_INFO, FEATURES
from plot_utils import set_plot_style, plot_decision_boundary, create_3d_scatter
from theory import render_theory

import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classification Visualizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
inject_style()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 CLASSIFIER LAB")
    st.markdown("---")

    st.markdown("**DATASET**")
    ds_name  = st.selectbox("", list(DATASETS.keys()), label_visibility="collapsed")
    n_samples = st.slider("Sample Size", 100, 800, 300, 50)
    noise_val = st.slider("Noise Level", 0.0, 0.5, 0.2, 0.05)

    st.markdown("---")
    st.markdown("**ALGORITHM**")
    clf_name = st.selectbox("", list(CLASSIFIERS.keys()), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**TRAIN / TEST SPLIT**")
    test_size = st.slider("Test Size %", 10, 40, 20, 5)

    st.markdown("---")
    run_btn = st.button("▶  RUN CLASSIFICATION", use_container_width=True)

# ─── Load & Train ────────────────────────────────────────────────────────────
ds_key = DATASETS[ds_name]
X, y, feature_names = load_dataset(ds_key, n_samples, noise_val)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size/100, random_state=42, stratify=y
)
clf = CLASSIFIERS[clf_name]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
n_classes = len(np.unique(y))
info = ALGO_INFO[clf_name]

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <p class="hero-title">🧠 Classification Visualizer</p>
  <p class="hero-subtitle">
    Explore decision boundaries · Compare algorithms · Understand metrics
    <br><span style="color:#00d4ff88;">Dataset: {ds_name} &nbsp;|&nbsp; Algorithm: {clf_name}</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Top Metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Accuracy", f"{acc*100:.1f}%", delta=f"{(acc-0.5)*100:+.1f}% vs random")
with c2: st.metric("Train Samples", len(X_train))
with c3: st.metric("Test Samples",  len(X_test))
with c4: st.metric("Classes",       n_classes)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️  Decision Boundary",
    "🧊  3D Visualization",
    "📚  Theory Notes",
    "📊  Confusion Matrix",
    "📈  ROC Curve",
    "📋  Class Report",
    "📖  About Features"
])

# ── TAB 1: Decision Boundary ──────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        set_plot_style(fig, axes)

        plot_decision_boundary(clf, X_train, y_train, axes[0], feature_names, "Train Set — Decision Boundary")
        plot_decision_boundary(clf, X_test,  y_test,  axes[1], feature_names, "Test Set — Decision Boundary")

        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Algorithm Info</p>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="feature-card">
  <p class="feature-title">{clf_name}</p>
  <p class="feature-desc">{info["desc"]}</p>
</div>
<div class="feature-card" style="border-left-color:#ffd166;">
  <p class="feature-title" style="color:#ffd166;">💡 Simple Explanation</p>
  <p class="feature-desc">{info.get("easy", "")}</p>
</div>
<div class="feature-card" style="border-left-color:#a855f7;">
  <p class="feature-title" style="color:#a855f7;">🌍 Real Life Example</p>
  <p class="feature-desc">{info.get("example", "")}</p>
</div>
<div class="feature-card" style="border-left-color:#06d6a0;">
  <p class="feature-title" style="color:#06d6a0;">✓ Pros</p>
  <p class="feature-desc">{info["pros"]}</p>
</div>
<div class="feature-card" style="border-left-color:#ff6b6b;">
  <p class="feature-title" style="color:#ff6b6b;">✗ Cons</p>
  <p class="feature-desc">{info["cons"]}</p>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<span class="algo-tag">{info['type']}</span>
<span class="algo-tag">Complexity: {info['complexity']}</span>
""", unsafe_allow_html=True)

# ── TAB 2: 3D Visualization ───────────────────────────────────────────────────
with tab2:
    st.markdown('<p class="section-header">Interactive 3D Feature Space</p>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_3d = create_3d_scatter(X_scaled, y, feature_names, f"{ds_name} - Complete Dataset (Scaled)")
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_b:
        st.markdown(f"""
<div class="feature-card">
  <p class="feature-title">🧊 Why 3D?</p>
  <p class="feature-desc">Real-world datasets rarely have just 2 features. Visualizing in 3D gives a better intuition of how data points are separated in higher-dimensional space.</p>
</div>
<div class="feature-card" style="border-left-color:#ffd166;">
  <p class="feature-title" style="color:#ffd166;">🎯 Instructions</p>
  <p class="feature-desc">Use your mouse to rotate the 3D plot. Scroll to zoom in and out. Hover over data points to see their exact feature values.</p>
</div>
""", unsafe_allow_html=True)

# ── TAB 3: Theory Notes ──────────────────────────────────────────────────────
with tab3:
    st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d1b2a,#1a2d4a);border:1px solid #00d4ff33;
border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
  <p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0;">📚 {clf_name}</p>
  <p style="color:#8899aa;font-size:0.9rem;margin:0.4rem 0 0;">Deep dive — theory, intuition, math & real-world examples</p>
</div>
""", unsafe_allow_html=True)
    render_theory(clf_name)

# ── TAB 4: Confusion Matrix ───────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        set_plot_style(fig, [ax])
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title("Confusion Matrix")
        for text in disp.text_.ravel():
            text.set_color('white')
            text.set_fontsize(13)
        st.pyplot(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">What is a Confusion Matrix?</p>', unsafe_allow_html=True)
        st.markdown("""
<div class="feature-card">
  <p class="feature-title">True Positive (TP)</p>
  <p class="feature-desc">Correctly predicted positive class. Model said YES and it was YES.</p>
</div>
<div class="feature-card" style="border-left-color:#ff6b6b;">
  <p class="feature-title" style="color:#ff6b6b;">False Positive (FP) — Type I Error</p>
  <p class="feature-desc">Model said YES but it was actually NO. Also called a false alarm.</p>
</div>
<div class="feature-card" style="border-left-color:#ffd166;">
  <p class="feature-title" style="color:#ffd166;">False Negative (FN) — Type II Error</p>
  <p class="feature-desc">Model said NO but it was actually YES. A missed detection.</p>
</div>
<div class="feature-card" style="border-left-color:#06d6a0;">
  <p class="feature-title" style="color:#06d6a0;">True Negative (TN)</p>
  <p class="feature-desc">Correctly predicted negative class. Model said NO and it was NO.</p>
</div>
""", unsafe_allow_html=True)

# ── TAB 5: ROC Curve ──────────────────────────────────────────────────────────
with tab5:
    fig, ax = plt.subplots(figsize=(7, 5))
    set_plot_style(fig, [ax])
    palette = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#a855f7']

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='#00d4ff', lw=2, label=f"AUC = {roc_auc:.3f}")
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_test, classes=np.unique(y))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=palette[i], lw=2, label=f"Class {i} AUC={roc_auc:.2f}")
    else:
        ax.text(0.5, 0.5, "ROC not available\n(no probability output)",
                ha='center', va='center', color='#8899aa', fontsize=12)

    ax.plot([0, 1], [0, 1], color='#1e3a5f', lw=1.5, linestyle='--', label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Receiver Operating Characteristic")
    ax.legend(facecolor='#0d1522', edgecolor='#1e3a5f', labelcolor='#c8d8e8', fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    st.pyplot(fig, use_container_width=True)

    st.info("📌 **AUC (Area Under Curve):** Closer to 1.0 = better. 0.5 = random guessing. The ROC curve shows the trade-off between sensitivity (TPR) and specificity (1-FPR) across all thresholds.")

# ── TAB 6: Classification Report ─────────────────────────────────────────────
with tab6:
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T.round(3)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    set_plot_style(fig, axes)

    metrics = ['precision', 'recall', 'f1-score']
    colors  = ['#00d4ff', '#06d6a0', '#ffd166']
    titles  = ['Precision', 'Recall', 'F1-Score']

    class_rows = [r for r in df_report.index if r not in ['accuracy', 'macro avg', 'weighted avg']]

    for i, (metric, color, title) in enumerate(zip(metrics, colors, titles)):
        vals = df_report.loc[class_rows, metric].values
        bars = axes[i].bar([f"C{c}" for c in class_rows], vals,
                           color=color + '99', edgecolor=color, linewidth=1.2)
        axes[i].set_ylim(0, 1.15)
        axes[i].set_title(title)
        for bar, v in zip(bars, vals):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f"{v:.2f}", ha='center', color=color, fontsize=9,
                         fontfamily='monospace')

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)

    st.dataframe(
        df_report.style
            .background_gradient(cmap='Blues', subset=['precision','recall','f1-score'])
            .format("{:.3f}"),
        use_container_width=True
    )

# ── TAB 7: About / Feature Explainer ─────────────────────────────────────────
with tab7:
    st.markdown('<p class="section-header">What This App Does</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    for i, (title, desc) in enumerate(FEATURES):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
<div class="feature-card">
  <p class="feature-title">{title}</p>
  <p class="feature-desc">{desc}</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">All 7 Algorithms at a Glance</p>', unsafe_allow_html=True)
    cols = st.columns(len(ALGO_INFO))
    for i, (name, a_info) in enumerate(ALGO_INFO.items()):
        with cols[i]:
            st.markdown(f"""
<div style="background:#0f1c2e;border:1px solid #1e3a5f;border-top:3px solid #00d4ff;
border-radius:10px;padding:0.9rem;text-align:center;">
  <p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:0.72rem;margin:0 0 0.4rem;">{name}</p>
  <span class="algo-tag" style="font-size:0.65rem;">{a_info['type']}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("💡 **Tip:** Start with Moons or Circles dataset and switch algorithms to see how each one carves the decision boundary differently. Then increase noise to test robustness!")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem;text-align:center;
border-top:1px solid #1e3a5f;color:#3a4a5a;font-size:0.78rem;
font-family:'Space Mono',monospace;">
  Classification Visualizer — Built with Streamlit · Scikit-learn · Matplotlib · Plotly
</div>
""", unsafe_allow_html=True)