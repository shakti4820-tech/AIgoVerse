import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from style import inject_style
from clustering_page import render_clustering_page
from regression_page import render_regression_page

# ── Classification imports ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from data_loader import DATASETS, load_dataset
from algos import CLASSIFIERS, ALGO_INFO, FEATURES
from plot_utils import set_plot_style, plot_decision_boundary, create_3d_scatter
from theory import render_theory

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlgoVerse — ML Universe",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_style()

# ─── Extra CSS for nav & home ─────────────────────────────────────────────────
st.markdown("""
<style>
.av-nav-btn {
    display:block;width:100%;text-align:left;
    background:linear-gradient(135deg,#0f1c2e,#111f30);
    border:1px solid #1e3a5f;border-radius:10px;
    padding:0.8rem 1.2rem;margin-bottom:0.5rem;
    color:#8899aa;font-family:'Space Mono',monospace;
    font-size:0.82rem;cursor:pointer;transition:all 0.25s;
}
.av-nav-btn:hover,.av-nav-btn.active {
    border-color:#00d4ff;color:#00d4ff;
    background:linear-gradient(135deg,#0d1b2a,#1a2d4a);
    box-shadow:0 0 14px #00d4ff22;
}
.home-card {
    background:linear-gradient(135deg,#0f1c2e,#162030);
    border:1px solid #1e3a5f;border-top:3px solid #00d4ff;
    border-radius:14px;padding:1.6rem;text-align:center;
    transition:border-color 0.3s;
}
.home-card:hover { border-color:#00d4ff55; }
.home-card-icon { font-size:2.2rem;margin-bottom:0.5rem; }
.home-card-title {
    font-family:'Space Mono',monospace;color:#00d4ff;
    font-size:0.95rem;font-weight:700;margin-bottom:0.4rem;
}
.home-card-desc { color:#8899aa;font-size:0.83rem;line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar Navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem;">
  <p style="font-family:'Space Mono',monospace;font-size:1.4rem;font-weight:700;
     color:#00d4ff;margin:0;text-shadow:0 0 20px #00d4ff44;">🌌 AlgoVerse</p>
  <p style="color:#3a5a7a;font-size:0.72rem;margin:0.2rem 0 0;
     font-family:'Space Mono',monospace;">A Universe of ML Algorithms</p>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Home", "📊  Classification", "🔵  Clustering", "📈  Regression"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
<div style="color:#3a5a7a;font-size:0.7rem;font-family:'Space Mono',monospace;text-align:center;">
Built with Streamlit · Scikit-learn<br>Plotly · Matplotlib
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🏠  HOME PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
<div class="hero-banner" style="padding:3rem;">
  <p class="hero-title" style="font-size:2.8rem;">🌌 AlgoVerse</p>
  <p class="hero-subtitle" style="font-size:1.15rem;">
    Your interactive universe of Machine Learning algorithms.<br>
    <span style="color:#00d4ff88;">
      Visualize · Learn Theory · Explore Real-World Examples · Compare Performance
    </span>
  </p>
</div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📊", "Classification", c1, "#00d4ff",
         "Learn how algorithms draw boundaries to separate different classes.<br><br>"
         "7 algorithms: Logistic Regression, SVM, Random Forest, KNN, Decision Tree, Naive Bayes, Gradient Boosting.<br><br>"
         "Features: Decision boundaries, confusion matrix, ROC curves, theory notes, GFG links."),
        ("🔵", "Clustering", c2, "#a855f7",
         "Discover hidden groups in unlabelled data — no class labels needed.<br><br>"
         "4 algorithms: K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models.<br><br>"
         "Features: Cluster visualization, 3D scatter, elbow curve, silhouette score, theory notes."),
        ("📈", "Regression", c3, "#06d6a0",
         "Predict continuous numerical values from input features.<br><br>"
         "7 algorithms: Linear, Ridge, Lasso, Polynomial, SVR, Random Forest, Gradient Boosting.<br><br>"
         "Features: Fit plot, residual analysis, 3D surface, R² score, theory notes, GFG links."),
    ]
    for icon, title, col, color, desc in cards:
        with col:
            st.markdown(f"""
<div class="home-card">
  <div class="home-card-icon">{icon}</div>
  <div class="home-card-title" style="color:{color};">{title}</div>
  <div class="home-card-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">🎯 What is AlgoVerse?</p>', unsafe_allow_html=True)
    st.markdown("""
<div style="background:linear-gradient(135deg,#0f1c2e,#111f30);border:1px solid #1e3a5f;
border-left:4px solid #00d4ff;border-radius:12px;padding:1.5rem 2rem;margin-bottom:1rem;">
<p style="color:#b0c4d8;font-size:0.95rem;line-height:2;margin:0;">
<b style="color:#00d4ff;">AlgoVerse</b> is a fully interactive Machine Learning education platform built for students,
developers, and data scientists who want to go beyond theory and <i>see</i> how algorithms work in real-time.<br><br>
Select a dataset, tune hyperparameters with sliders, and watch visualizations update instantly.
Each algorithm comes with <b>rich theory notes</b> written in simple English — just like GeeksForGeeks —
plus curated <b>external reading links</b> so you can dive deeper whenever you're ready.<br><br>
Whether you're preparing for an interview, completing a course, or just curious about ML —
<b>AlgoVerse has you covered.</b>
</p></div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">📦 Algorithms at a Glance</p>', unsafe_allow_html=True)
    rows = {
        "📊 Classification (7)": ["Logistic Regression","Decision Tree","Random Forest","SVM (RBF)","KNN","Naive Bayes","Gradient Boosting"],
        "🔵 Clustering (4)":     ["K-Means","DBSCAN","Hierarchical","Gaussian Mixture"],
        "📈 Regression (7)":     ["Linear","Ridge","Lasso","Polynomial","SVR","RF Regressor","GB Regressor"],
    }
    for section, algos in rows.items():
        st.markdown(f"**{section}**")
        tags = "".join([f'<span class="algo-tag">{a}</span>' for a in algos])
        st.markdown(tags, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 📊  CLASSIFICATION PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Classification":
    with st.sidebar:
        st.markdown("---")
        st.markdown("**📊 CLASSIFICATION**")
        clf_name   = st.selectbox("Algorithm", list(CLASSIFIERS.keys()), key="cf_algo")
        ds_name   = st.selectbox("Dataset",   list(DATASETS.keys()),    key="cf_ds")
        n_samples  = st.slider("Samples", 50, 1000, 300, 50,            key="cf_n")
        noise_val  = st.slider("Noise", 0.0, 1.0, 0.2, 0.05,            key="cf_noise")
        test_size  = st.slider("Test Size %", 10, 40, 20, 5,            key="cf_ts")
        
        st.markdown("---")
        st.markdown("**🎛️ HYPERPARAMETERS**")
        if clf_name == "Logistic Regression":
            cf_c = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0, 0.05, key="cf_lr_c")
            cf_penalty = st.selectbox("Penalty type", ["l2", "none"], key="cf_lr_penalty")
            penalty_val = cf_penalty if cf_penalty != "none" else None
            clf = LogisticRegression(C=cf_c, penalty=penalty_val, max_iter=1000, random_state=42)
            
        elif clf_name == "Decision Tree":
            cf_depth = st.slider("Max Tree Depth", 1, 15, 5, key="cf_dt_depth")
            cf_min_split = st.slider("Min Samples to Split", 2, 10, 2, key="cf_dt_min_split")
            cf_criterion = st.selectbox("Criterion", ["gini", "entropy"], key="cf_dt_criterion")
            clf = DecisionTreeClassifier(max_depth=cf_depth, min_samples_split=cf_min_split, criterion=cf_criterion, random_state=42)
            
        elif clf_name == "Random Forest":
            cf_estimators = st.slider("Estimators (Trees count)", 10, 200, 100, 10, key="cf_rf_est")
            cf_depth = st.slider("Max Tree Depth", 1, 15, 6, key="cf_rf_depth")
            cf_features = st.selectbox("Max Split Features", ["sqrt", "log2"], key="cf_rf_feat")
            clf = RandomForestClassifier(n_estimators=cf_estimators, max_depth=cf_depth, max_features=cf_features, random_state=42)
            
        elif clf_name == "SVM (RBF Kernel)":
            cf_c = st.slider("Cost (Regularization C)", 0.1, 50.0, 1.0, 0.5, key="cf_svm_c")
            cf_gamma = st.slider("Gamma (Kernel Width)", 0.01, 5.0, 0.1, 0.05, key="cf_svm_gamma")
            clf = SVC(kernel='rbf', C=cf_c, gamma=cf_gamma, probability=True, random_state=42)
            
        elif clf_name == "K-Nearest Neighbors":
            cf_k = st.slider("K (Neighbors count)", 1, 20, 5, key="cf_knn_k")
            cf_weights = st.selectbox("Weights", ["uniform", "distance"], key="cf_knn_weights")
            cf_metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"], key="cf_knn_metric")
            clf = KNeighborsClassifier(n_neighbors=cf_k, weights=cf_weights, metric=cf_metric)
            
        elif clf_name == "Naive Bayes":
            cf_smoothing = st.slider("Variance Smoothing", 1e-10, 1e-6, 1e-9, 1e-10, format="%.2e", key="cf_nb_smooth")
            clf = GaussianNB(var_smoothing=cf_smoothing)
            
        elif clf_name == "Gradient Boosting":
            cf_estimators = st.slider("Estimators (Boosting rounds)", 10, 150, 80, 10, key="cf_gb_est")
            cf_lr = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="cf_gb_lr")
            cf_depth = st.slider("Max Tree Depth", 1, 10, 3, key="cf_gb_depth")
            clf = GradientBoostingClassifier(n_estimators=cf_estimators, learning_rate=cf_lr, max_depth=cf_depth, random_state=42)

    ds_key = DATASETS[ds_name]
    X, y, feature_names = load_dataset(ds_key, n_samples, noise_val)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size/100, random_state=42, stratify=y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    n_classes = len(np.unique(y))
    info = ALGO_INFO[clf_name]

    st.markdown(f"""<div class="hero-banner">
<p class="hero-title">📊 Classification Lab</p>
<p class="hero-subtitle">Supervised learning · Predict class labels · Explore decision boundaries<br>
<span style="color:#00d4ff88;">Algorithm: {clf_name} &nbsp;|&nbsp; Dataset: {ds_name}</span></p></div>""",
    unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Accuracy",     f"{acc*100:.1f}%", delta=f"{(acc-0.5)*100:+.1f}% vs random")
    with c2: st.metric("Train Samples", len(X_train))
    with c3: st.metric("Test Samples",  len(X_test))
    with c4: st.metric("Classes",       n_classes)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
        "🗺️ Decision Boundary","🧊 3D View","📚 Theory Notes","🧠 Brain Challenge",
        "📊 Confusion Matrix","📈 ROC Curve","📋 Class Report","📖 About"])

    with tab1:
        ca, cb = st.columns([2,1])
        with ca:
            fig, axes = plt.subplots(1,2,figsize=(11,4.5))
            set_plot_style(fig, axes)
            plot_decision_boundary(clf, X_train, y_train, axes[0], feature_names, "Train Set")
            plot_decision_boundary(clf, X_test,  y_test,  axes[1], feature_names, "Test Set")
            plt.tight_layout(pad=2)
            st.pyplot(fig, use_container_width=True)
        with cb:
            st.markdown('<p class="section-header">Algorithm Info</p>', unsafe_allow_html=True)
            for title, color, body in [
                (clf_name, "#00d4ff", info["desc"]),
                ("💡 Simple Explanation","#ffd166", info.get("easy","")),
                ("🌍 Real Life Example","#a855f7", info.get("example","")),
                ("✓ Pros","#06d6a0", info["pros"]),
                ("✗ Cons","#ff6b6b", info["cons"]),
            ]:
                st.markdown(f"""<div class="feature-card" style="border-left-color:{color};">
<p class="feature-title" style="color:{color};">{title}</p>
<p class="feature-desc">{body}</p></div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<p style="color: var(--cyan); font-family: \'Space Mono\', monospace; font-size: 0.95rem; font-weight: 700; margin: 0 0 1rem 0;">🛠️ Customize 3D Perspectives</p>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            cf_3d_color = st.selectbox("Color Mapping Perspective", ["Actual Labels", "Predicted Labels", "Match/Mismatch (Errors)", "Prediction Confidence"], index=0, key="cf_3d_col")
        with col_c2:
            cf_3d_scale = st.selectbox("Color Scheme / Palette", ["Vibrant Space", "Plasma", "Viridis", "Tealrose", "Rainbow"], index=0, key="cf_3d_scale")
        
        col_x, col_y, col_z = st.columns([1, 1, 1])
        with col_x:
            cf_3d_x = st.selectbox("X-Axis Feature", feature_names, index=0, key="cf_3d_x")
            x_idx = feature_names.index(cf_3d_x)
        with col_y:
            cf_3d_y = st.selectbox("Y-Axis Feature", feature_names, index=1, key="cf_3d_y")
            y_idx = feature_names.index(cf_3d_y)
        with col_z:
            cf_3d_z = st.selectbox("Z-Axis Feature", feature_names, index=2, key="cf_3d_z")
            z_idx = feature_names.index(cf_3d_z)
            
        fig3 = create_3d_scatter(
            X_scaled, y, feature_names, 
            title=f"{clf_name} — Dynamic 3D Feature Projection", 
            clf=clf,
            x_idx=x_idx, y_idx=y_idx, z_idx=z_idx,
            color_by=cf_3d_color, colorscale_name=cf_3d_scale
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown(f"""<div style="background:linear-gradient(135deg,#0d1b2a,#1a2d4a);
border:1px solid #00d4ff33;border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
<p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0;">📚 {clf_name}</p>
<p style="color:#8899aa;font-size:0.9rem;margin:0.4rem 0 0;">Theory · Math · Intuition · Real-world examples</p></div>""",
        unsafe_allow_html=True)
        render_theory(clf_name)

    with tab4:
        from theory import render_quiz
        render_quiz(clf_name)

    with tab5:
        ca, cb = st.columns([1,1])
        with ca:
            fig, ax = plt.subplots(figsize=(5.5,4.5))
            set_plot_style(fig, [ax])
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, colorbar=False, cmap='Blues')
            ax.set_title("Confusion Matrix")
            for t in disp.text_.ravel(): t.set_color('white'); t.set_fontsize(13)
            st.pyplot(fig, use_container_width=True)
        with cb:
            st.markdown('<p class="section-header">Reading the Matrix</p>', unsafe_allow_html=True)
            for title, color, desc in [
                ("True Positive (TP)","#06d6a0","Correctly predicted positive. Model said YES and was right."),
                ("False Positive (FP) — Type I","#ff6b6b","Model said YES but was wrong. A false alarm."),
                ("False Negative (FN) — Type II","#ffd166","Model said NO but was wrong. A missed detection."),
                ("True Negative (TN)","#00d4ff","Correctly predicted negative. Model said NO and was right."),
            ]:
                st.markdown(f"""<div class="feature-card" style="border-left-color:{color};">
<p class="feature-title" style="color:{color};">{title}</p>
<p class="feature-desc">{desc}</p></div>""", unsafe_allow_html=True)

    with tab6:
        fig, ax = plt.subplots(figsize=(7,5))
        set_plot_style(fig,[ax])
        palette = ['#00d4ff','#ff6b6b','#ffd166','#06d6a0','#a855f7']
        if hasattr(clf,"predict_proba"):
            y_prob = clf.predict_proba(X_test)
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
                ax.plot(fpr, tpr, color='#00d4ff', lw=2, label=f"AUC={auc(fpr,tpr):.3f}")
            else:
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_test, classes=np.unique(y))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
                    ax.plot(fpr, tpr, color=palette[i], lw=2, label=f"C{i} AUC={auc(fpr,tpr):.2f}")
        else:
            ax.text(0.5,0.5,"ROC not available",ha='center',va='center',color='#8899aa',fontsize=12)
        ax.plot([0,1],[0,1],'--',color='#1e3a5f',lw=1.5,label="Random")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
        ax.legend(facecolor='#0d1522',edgecolor='#1e3a5f',labelcolor='#c8d8e8',fontsize=9)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        st.pyplot(fig, use_container_width=True)
        st.info("📌 AUC closer to 1.0 = better. 0.5 = random guessing.")

    with tab7:
        report = classification_report(y_test, y_pred, output_dict=True)
        df_r = pd.DataFrame(report).T.round(3)
        fig, axes = plt.subplots(1,3,figsize=(13,4))
        set_plot_style(fig,axes)
        class_rows = [r for r in df_r.index if r not in ['accuracy','macro avg','weighted avg']]
        for i,(metric,color,title) in enumerate(zip(
            ['precision','recall','f1-score'],
            ['#00d4ff','#06d6a0','#ffd166'],
            ['Precision','Recall','F1-Score']
        )):
            vals = df_r.loc[class_rows, metric].values
            bars = axes[i].bar([f"C{c}" for c in class_rows], vals, color=color+'99', edgecolor=color, linewidth=1.2)
            axes[i].set_ylim(0,1.15); axes[i].set_title(title)
            for bar,v in zip(bars,vals):
                axes[i].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{v:.2f}",
                             ha='center', color=color, fontsize=9, fontfamily='monospace')
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        st.dataframe(df_r.style.background_gradient(cmap='Blues',subset=['precision','recall','f1-score']).format("{:.3f}"),
                     use_container_width=True)

    with tab8:
        c1, c2 = st.columns(2)
        for i,(title,desc) in enumerate(FEATURES):
            with (c1 if i%2==0 else c2):
                st.markdown(f"""<div class="feature-card">
<p class="feature-title">{title}</p><p class="feature-desc">{desc}</p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔵  CLUSTERING PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  Clustering":
    render_clustering_page()


# ═══════════════════════════════════════════════════════════════════════════════
# 📈  REGRESSION PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Regression":
    render_regression_page()


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1.5rem;text-align:center;
border-top:1px solid #1e3a5f;color:#3a4a5a;font-size:0.78rem;
font-family:'Space Mono',monospace;">
  🌌 AlgoVerse — A Universe of Machine Learning Algorithms
</div>""", unsafe_allow_html=True)
