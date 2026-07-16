import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

PALETTE = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#a855f7', '#f97316', '#ec4899']

# ─── Notes, Lectures, Quizzes Config ───────────────────────────────────────────
REDUCTION_INFO = {
    "PCA": {
        "desc": "Unsupervised linear transformation that projects data along the directions of maximal variance (Principal Components).",
        "easy": "Imagine taking a photo of a 3D object from the best angle to capture as much of its shape and details as possible in a 2D picture.",
        "example": "📷 Image compression: reducing the size of high-resolution images by keeping only the most important color variance patterns.",
        "pros": ["Highly efficient, works on large datasets.", "Unsupervised, does not require labels.", "Preserves global structure of the data."],
        "cons": ["Cannot capture non-linear relationships.", "Principal components can be hard to interpret.", "Assumes features are linearly correlated."],
        "complexity": "O(d²n + d³)",
        "paper": "https://doi.org/10.1080/14786440109462720",
        "doc": "https://scikit-learn.org/stable/modules/decomposition.html#pca",
        "video": "https://www.youtube.com/embed/FgakZw6K1QQ",
        "quizzes": [
            {
                "question": "What mathematical property does PCA maximize when projecting data?",
                "options": ["Accuracy of the target labels", "The variance of the projected data points", "The entropy of the feature space", "The margin between class clusters"],
                "correct": 1,
                "explanation": "PCA projects data onto orthogonal axes (principal components) that maximize the variance (spread) of the data, thereby preserving the most information possible."
            },
            {
                "question": "Principal Components are mathematically guaranteed to be:",
                "options": ["Highly correlated with each other", "Linearly dependent on class labels", "Orthogonal (perpendicular) to each other", "Equal to the original features"],
                "correct": 2,
                "explanation": "Principal components are eigenvectors of the covariance matrix, making them mathematically orthogonal (uncorrelated) to each other."
            },
            {
                "question": "Which matrix is decomposed to find the principal components of a dataset?",
                "options": ["The confusion matrix", "The within-class scatter matrix", "The covariance matrix of standardized features", "The distance matrix"],
                "correct": 2,
                "explanation": "PCA performs eigen-decomposition on the covariance matrix (or SVD on the data matrix) to find the directions of maximum variance."
            },
            {
                "question": "Why is standardizing/scaling data (e.g., StandardScaler) critical before PCA?",
                "options": ["To convert non-numeric columns into numbers", "To prevent features with larger scales from dominating the principal components", "To reduce the dataset size automatically", "To force the variance of all features to be zero"],
                "correct": 1,
                "explanation": "PCA is highly sensitive to the absolute scale of features. If one feature is measured in thousands and another in decimals, the first will dominate variance calculations unless standardized."
            },
            {
                "question": "What is a major limitation of standard PCA?",
                "options": ["It is supervised and requires expensive class annotations", "It is extremely slow and memory intensive on small datasets", "It cannot capture non-linear relationships in data structures", "It fails to output continuous values"],
                "correct": 2,
                "explanation": "PCA is a strictly linear technique. If the underlying data structure is highly curved or non-linear (like Moons or Circles), PCA cannot capture this topology properly."
            }
        ]
    },
    "LDA": {
        "desc": "Supervised linear method that projects data to maximize class separability by finding the optimal decision boundary.",
        "easy": "Like sorting beads by color: it squashes the space so that beads of the same color are tightly packed and different colors are far apart.",
        "example": "🗣️ Speech recognition: projecting audio signals to maximize the distance between different vocal phonemes/words.",
        "pros": ["Supervised, meaning it directly utilizes class labels.", "Acts as a classification boundary finder.", "Reduces dimensions while maximizing class separation."],
        "cons": ["Cannot exceed C - 1 dimensions (where C is the number of classes).", "Assumes normal distribution of features.", "Assumes classes have identical covariance matrices."],
        "complexity": "O(d²n + d³)",
        "paper": "https://doi.org/10.1111/j.1469-1809.1936.tb02137.x",
        "doc": "https://scikit-learn.org/stable/modules/lda_qda.html",
        "video": "https://www.youtube.com/embed/azXCzI57Yfc",
        "quizzes": [
            {
                "question": "What is the primary optimization objective of Linear Discriminant Analysis (LDA)?",
                "options": ["Maximize the variance of the complete dataset", "Maximize the ratio of between-class variance to within-class variance", "Minimize the number of outliers in the dataset", "Map data into an infinite-dimensional space"],
                "correct": 1,
                "explanation": "LDA seeks to project data into a lower-dimensional space where classes are separated as much as possible (maximizing between-class scatter $S_b$) while keeping individual classes compact (minimizing within-class scatter $S_w$)."
            },
            {
                "question": "For a dataset with C distinct target classes, what is the maximum number of dimensions LDA can project to?",
                "options": ["C - 1", "C", "C + 1", "Unlimited"],
                "correct": 0,
                "explanation": "The rank of the between-class scatter matrix is at most $C - 1$. Therefore, LDA is mathematically limited to projecting onto at most $C - 1$ dimensions."
            },
            {
                "question": "How does LDA differ fundamentally from PCA?",
                "options": ["LDA is unsupervised, whereas PCA is supervised", "LDA is supervised (uses class labels), whereas PCA is unsupervised", "LDA only works on 3D datasets, whereas PCA works on any dimension", "LDA has no mathematical constraints on dimensionality"],
                "correct": 1,
                "explanation": "LDA utilizes class labels to maximize group separation (supervised), whereas PCA looks purely at overall variance ignoring any group associations (unsupervised)."
            },
            {
                "question": "Which of the following is a core assumption of LDA?",
                "options": ["Features follow a normal (Gaussian) distribution within each class", "Features are strictly binary (0 or 1)", "There is only one target class in the dataset", "All features must be uncorrelated"],
                "correct": 0,
                "explanation": "LDA assumes that the features within each class are normally distributed and share the same covariance matrix."
            },
            {
                "question": "When would LDA be preferred over PCA?",
                "options": ["When you do not have class labels for the data", "When your goal is classification and you want to maximize group separability", "When you want to project to more dimensions than target classes", "When the data structure is non-linear"],
                "correct": 1,
                "explanation": "LDA is designed to find projection axes that make classification easier, making it preferred for supervised classification pipelines."
            }
        ]
    },
    "t-SNE": {
        "desc": "Unsupervised non-linear method that converts similarities between data points to joint probabilities and minimizes KL divergence.",
        "easy": "Like making a map of friends: it places people close to each other in 2D if they are closely related in high-dimensional space.",
        "example": "🧬 Single-cell genomics: mapping thousands of cells based on gene expressions to identify distinct cell types.",
        "pros": ["Captures complex non-linear manifolds.", "Excellent at preserving local clusters and neighborhoods.", "Highly customizable through perplexity controls."],
        "cons": ["Computationally expensive O(n²) and slow on large datasets.", "Does not preserve global distances (far away points are meaningless).", "Non-deterministic; results can vary between runs."],
        "complexity": "O(n²)",
        "paper": "https://jmlr.org/papers/v9/vandermaaten08a.html",
        "doc": "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
        "video": "https://www.youtube.com/embed/NEaUSP4YerM",
        "quizzes": [
            {
                "question": "What is the role of the 'perplexity' hyperparameter in t-SNE?",
                "options": ["It sets the learning rate of the optimization", "It controls the number of projection dimensions", "It acts as a target number of neighbors, balancing attention between local and global structure", "It defines the maximum iteration count"],
                "correct": 2,
                "explanation": "Perplexity is loosely related to the number of nearest neighbors that each point considers. It controls the scale of the Gaussian kernel used to measure local densities."
            },
            {
                "question": "Why is t-SNE considered non-deterministic?",
                "options": ["Because it does not use a loss function", "Because it uses random initializations, and its cost function is optimized using gradient descent with local minima", "Because it requires target labels to train", "Because it only runs on Linux systems"],
                "correct": 1,
                "explanation": "t-SNE uses a random initialization (unless explicitly seeded) and its objective function (Kullback-Leibler divergence) is non-convex, meaning gradient descent can converge to different local minima."
            },
            {
                "question": "Which loss function does t-SNE minimize during optimization?",
                "options": ["Mean Squared Error (MSE)", "Hinge Loss", "Kullback-Leibler (KL) Divergence", "Binary Cross-Entropy"],
                "correct": 2,
                "explanation": "t-SNE minimizes the Kullback-Leibler (KL) divergence between the high-dimensional probability distribution and the low-dimensional probability distribution of point similarities."
            },
            {
                "question": "Can you use a trained t-SNE model to transform new, unseen data points directly?",
                "options": ["Yes, using standard transform() methods", "No, t-SNE does not learn a parametric mapping function; it must re-run on the entire combined dataset", "Yes, but only if the features are scaled", "Yes, if the perplexity is set below 10"],
                "correct": 1,
                "explanation": "t-SNE is a non-parametric embedding method. It optimizes the coordinates of the specific input points directly and does not learn a formula or weights that can be applied to new points."
            },
            {
                "question": "What is a major limitation of t-SNE regarding global structure?",
                "options": ["It forces all clusters to overlap", "It ignores local structures completely", "Distances between far-off clusters are not preserved and are largely meaningless", "It requires labels to determine global layouts"],
                "correct": 2,
                "explanation": "t-SNE focuses heavily on local structures (neighbor relationships). As a result, the relative distances between distant clusters in the low-dimensional space are not mathematically reliable."
            }
        ]
    },
    "UMAP": {
        "desc": "Unsupervised non-linear method based on Riemannian geometry and fuzzy simplicial sets. Highly scalable and preserves global structure.",
        "easy": "Like pinning a stretchy fabric onto a board: UMAP stretches and bends the dataset smoothly to fit in 2D while keeping both local neighbors and global layouts.",
        "example": "🩺 Disease progression mapping: tracking how patient profiles drift over time in high-dimensional medical records.",
        "pros": ["Faster and much more scalable than t-SNE.", "Preserves both local and global layout structure.", "Can transform new, unseen data points (parametric mapping)."],
        "cons": ["Relies on complex topological mathematics (harder to explain).", "Can be sensitive to hyperparameter combinations.", "Requires installing additional external libraries."],
        "complexity": "O(n log n)",
        "paper": "https://arxiv.org/abs/1802.03426",
        "doc": "https://umap-learn.readthedocs.io/",
        "video": "https://www.youtube.com/embed/eN0wFzBA4Sc",
        "quizzes": [
            {
                "question": "How does UMAP compare to t-SNE regarding global structure preservation?",
                "options": ["UMAP ignores global structure completely, while t-SNE preserves it", "UMAP preserves both local and global structures better than t-SNE by utilizing topological mathematical constructs", "They both ignore global structure completely", "UMAP only works on linear boundaries"],
                "correct": 1,
                "explanation": "UMAP is built on Riemannian geometry and fuzzy simplicial sets, which enables it to capture global layouts much better than t-SNE, which focuses almost exclusively on local distances."
            },
            {
                "question": "Which parameter in UMAP balances focus between local and global structures?",
                "options": ["min_dist", "n_neighbors", "metric", "n_epochs"],
                "correct": 1,
                "explanation": "The `n_neighbors` parameter controls how UMAP looks at local vs global structure. Lower values focus on local detail, while higher values incorporate larger structures."
            },
            {
                "question": "What does the 'min_dist' parameter in UMAP control?",
                "options": ["The learning rate of the optimizer", "The minimum distance between points allowed in the low-dimensional representation, controlling clumpiness", "The size of target clusters", "The distance metric formula"],
                "correct": 1,
                "explanation": "`min_dist` controls how tightly UMAP packs points together in the low-dimensional space. Smaller values lead to tight, clumpy cluster formations."
            },
            {
                "question": "Unlike t-SNE, UMAP supports:",
                "options": ["Supervised dimensionality reduction and transforming new, unseen data points", "Only 2D mappings", "Ignoring features completely", "Faster execution on small datasets only"],
                "correct": 0,
                "explanation": "UMAP supports supervised dimensionality reduction (incorporating label data) and can learn a projection model that can transform new, unseen data points."
            },
            {
                "question": "What is the time complexity scaling of UMAP, making it faster than t-SNE on large datasets?",
                "options": ["O(N³)", "O(N²)", "O(N log N)", "O(1)"],
                "correct": 2,
                "explanation": "UMAP utilizes approximate nearest neighbor algorithms, scaling at approximately $O(N \\log N)$, which is vastly faster than t-SNE's $O(N^2)$ scaling."
            }
        ]
    }
}

def set_style(fig, axes=None):
    fig.patch.set_facecolor('#0a0e1a')
    if axes is not None:
        for ax in (axes if hasattr(axes, '__iter__') else [axes]):
            ax.set_facecolor('#0d1522')
            ax.tick_params(colors='#6b7a8d', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1e3a5f')
            ax.xaxis.label.set_color('#8899aa')
            ax.yaxis.label.set_color('#8899aa')
            ax.title.set_color('#00d4ff')

def render_quiz(algo_name: str):
    st.markdown('<p class="section-header">🧠 Brain Challenge Quiz</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(168, 85, 247, 0.05); border: 1px dashed rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        <p style="color: var(--purple); font-family: 'Space Mono', monospace; font-size: 0.88rem; font-weight: 700; margin: 0;">
            ⚡ CHALLENGE YOUR UNDERSTANDING
        </p>
        <p style="color: var(--text-secondary); font-size: 0.83rem; margin: 0.2rem 0 0;">
            Test your knowledge of the mathematical and structural concepts behind <b>{algo_name}</b>. Correct answers receive instant detailed breakdown feedback!
        </p>
    </div>
    """, unsafe_allow_html=True)

    info = REDUCTION_INFO.get(algo_name, {})
    questions = info.get("quizzes", [])
    
    score_key = f"quiz_score_dr_{algo_name}"
    submitted_key = f"quiz_submitted_dr_{algo_name}"

    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False

    user_choices = []

    for idx, q in enumerate(questions):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        choice = st.radio(
            "Select the correct option:",
            q["options"],
            key=f"q_dr_{algo_name}_{idx}",
            index=None,
            label_visibility="collapsed"
        )
        user_choices.append(choice)
        st.markdown("<br>", unsafe_allow_html=True)

    c_btn, c_score = st.columns([1, 2])
    with c_btn:
        submit = st.button("Submit Quiz", key=f"submit_btn_dr_{algo_name}", disabled=st.session_state[submitted_key])

    if submit:
        if any(c is None for c in user_choices):
            st.error("⚠️ Please answer all questions before submitting!")
            return

        st.session_state[submitted_key] = True
        st.rerun()

    if st.session_state[submitted_key]:
        correct_count = 0
        st.markdown("---")
        st.markdown("### 📊 Quiz Results & Conceptual Analysis")

        for idx, q in enumerate(questions):
            user_choice = user_choices[idx]
            correct_ans_str = q["options"][q["correct"]]
            is_correct = (user_choice == correct_ans_str)

            if is_correct:
                correct_count += 1
                st.success(f"**Question {idx+1}: Correct!** ✅\n\nYour Answer: *{user_choice}*")
            else:
                st.error(f"**Question {idx+1}: Incorrect** ❌\n\nYour Answer: *{user_choice}*\n\nCorrect Answer: *{correct_ans_str}*")

            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 0.8rem 1.2rem; margin-bottom: 1.5rem;">
                <p style="color: var(--gold); font-size: 0.8rem; font-family: 'Space Mono', monospace; font-weight: 700; margin: 0 0 0.3rem 0;">💡 Concept Breakdown</p>
                <p style="color: var(--text-secondary); font-size: 0.84rem; line-height: 1.6; margin: 0;">{q['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.session_state[score_key] = correct_count

        badge_color = "var(--rose)"
        badge_text = "Keep Reviewing! 📚"
        if correct_count >= 5:
            badge_color = "var(--emerald)"
            badge_text = "Grandmaster! 🏆"
        elif correct_count >= 3:
            badge_color = "var(--cyan)"
            badge_text = "Proficient! 🏅"

        st.markdown(f"""
        <div style="background: rgba(11, 15, 25, 0.8); border: 1px solid rgba(255,255,255,0.05); border-left: 5px solid {badge_color}; border-radius: 12px; padding: 1.2rem; text-align: center; margin-top: 1rem;">
            <p style="color: #94a3b8; font-family: 'Space Mono', monospace; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">
                FINAL SCORE
            </p>
            <p style="font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: {badge_color}; margin: 0.3rem 0;">
                {correct_count} / 5
            </p>
            <span class="algo-tag" style="background: {badge_color}1a; border-color: {badge_color}44; color: {badge_color}; font-size: 0.85rem; font-weight: 700; padding: 0.3rem 1.2rem;">
                {badge_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Retry Quiz", key=f"retry_btn_dr_{algo_name}"):
            st.session_state[submitted_key] = False
            for idx in range(len(questions)):
                if f"q_dr_{algo_name}_{idx}" in st.session_state:
                    del st.session_state[f"q_dr_{algo_name}_{idx}"]
            st.rerun()

def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload custom CSV dataset", type=["csv"], key="dr_uploader")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 3:
                st.sidebar.error("CSV needs at least 3 numeric columns!")
                # Fallback to Iris
                d = load_iris()
                X = d.data
                y = d.target
                feature_names = d.feature_names
                target_names = [d.target_names[val] for val in y]
                df_disp = pd.DataFrame(X, columns=feature_names)
                df_disp["target"] = target_names
                return X, y, feature_names, df_disp, target_names
            
            # Find possible label column (last object or integer column)
            non_numeric = [c for c in df.columns if c not in numeric_cols]
            if len(non_numeric) > 0:
                label_col = non_numeric[0]
            else:
                label_col = numeric_cols[-1]
                numeric_cols.remove(label_col)
            
            X = df[numeric_cols].values
            y_raw = df[label_col].values
            
            # Map labels to integers for classification coloring
            unique_labels = np.unique(y_raw)
            label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
            y = np.array([label_map[lbl] for lbl in y_raw])
            
            feature_names = numeric_cols
            target_names = [str(val) for val in y_raw]
            
            # Keep original display df
            df_disp = df[numeric_cols + [label_col]].copy()
            return X, y, feature_names, df_disp, target_names
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
            
    # Default load Iris
    d = load_iris()
    X = d.data
    y = d.target
    feature_names = d.feature_names
    target_names = [d.target_names[val] for val in y]
    df_disp = pd.DataFrame(X, columns=feature_names)
    df_disp["target_label"] = target_names
    return X, y, feature_names, df_disp, target_names

def render_dim_reduction_page():
    st.markdown("""
    <div class="hero-banner">
      <p class="hero-title">📉 Dimensionality Reduction Lab</p>
      <p class="hero-subtitle">
        Unsupervised Learning · Compress feature spaces · Project high-dimensional structures into 2D/3D
      </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">D. Unsupervised Learning — Dimensionality Reduction</p>', unsafe_allow_html=True)
    
    # ── Sidebar Controls ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**📉 DIMENSIONALITY REDUCTION**")
        algo = st.selectbox("Choose Algorithm", ["PCA", "LDA", "t-SNE", "UMAP"], key="dr_algo")
        
        st.markdown("---")
        st.markdown("**🎛️ HYPERPARAMETERS**")
        
        if algo == "PCA":
            pca_components = st.slider("n_components", 2, 3, 2, key="dr_pca_nc")
            whiten = st.checkbox("Whiten components", value=False, key="dr_pca_white")
        elif algo == "LDA":
            lda_components = st.slider("n_components", 2, 3, 2, key="dr_lda_nc")
        elif algo == "t-SNE":
            perplexity = st.slider("Perplexity", 5, 50, 30, key="dr_tsne_perp")
            learning_rate = st.select_slider("Learning Rate", options=[10, 50, 100, 200, 500, 1000], value=200, key="dr_tsne_lr")
            n_iter = st.slider("Iterations", 250, 2000, 1000, 250, key="dr_tsne_iter")
        elif algo == "UMAP":
            n_neighbors = st.slider("n_neighbors", 2, 100, 15, key="dr_umap_neigh")
            min_dist = st.slider("min_dist", 0.0, 0.99, 0.1, 0.05, key="dr_umap_dist")
            metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine", "correlation"], index=0, key="dr_umap_metric")
            
    # Load dataset
    X, y, feature_names, df_disp, target_names = load_data()
    
    # Show dataset overview expander
    with st.expander("📂 Active Dataset Explorer"):
        st.markdown(f"**Shape:** `{X.shape[0]}` rows, `{X.shape[1]}` dimensions/features")
        st.dataframe(df_disp.head(8), use_container_width=True)

    # Process and Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compute dimensionality reduction inside a spinner
    with st.spinner(f"Running {algo} optimization projection..."):
        if algo == "PCA":
            # For PCA, compute both 2D and 3D representations dynamically so tab switching is instantaneous
            pca_2d = PCA(n_components=2, whiten=whiten, random_state=42)
            X_2d = pca_2d.fit_transform(X_scaled)
            exp_var_2d = pca_2d.explained_variance_ratio_
            
            pca_3d = PCA(n_components=3, whiten=whiten, random_state=42)
            X_3d = pca_3d.fit_transform(X_scaled)
            exp_var_3d = pca_3d.explained_variance_ratio_
            
            # Bind active output based on slider
            X_reduced = X_2d if pca_components == 2 else X_3d
            active_components = pca_components
            
        elif algo == "LDA":
            # LDA is limited by target classes: n_components <= C - 1
            max_lda_comp = min(X.shape[1], len(np.unique(y)) - 1)
            active_lda_comp = min(lda_components, max_lda_comp)
            
            lda_model_2d = LDA(n_components=min(2, max_lda_comp))
            X_2d = lda_model_2d.fit_transform(X_scaled, y)
            
            if max_lda_comp >= 3:
                lda_model_3d = LDA(n_components=3)
                X_3d = lda_model_3d.fit_transform(X_scaled, y)
            else:
                # If only 2 components are possible, fill 3rd component with zeroes
                X_3d = np.column_stack([X_2d, np.zeros(len(X_2d))])
                
            X_reduced = X_2d if lda_components == 2 else X_3d
            active_components = lda_components
            
        elif algo == "t-SNE":
            # t-SNE fits representations separately
            tsne_2d = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
            X_2d = tsne_2d.fit_transform(X_scaled)
            
            tsne_3d = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
            X_3d = tsne_3d.fit_transform(X_scaled)
            
            X_reduced = X_2d
            active_components = 2
            
        elif algo == "UMAP":
            # UMAP fits representations separately
            # Cap n_neighbors at dataset size
            act_neighbors = min(n_neighbors, len(X_scaled) - 1)
            umap_2d = umap.UMAP(n_components=2, n_neighbors=act_neighbors, min_dist=min_dist, metric=metric, random_state=42)
            X_2d = umap_2d.fit_transform(X_scaled)
            
            umap_3d = umap.UMAP(n_components=3, n_neighbors=act_neighbors, min_dist=min_dist, metric=metric, random_state=42)
            X_3d = umap_3d.fit_transform(X_scaled)
            
            X_reduced = X_2d
            active_components = 2

    # Metric summary layout
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Model Algorithm", algo)
    with c2:
        st.metric("Input Dimensions", f"{X.shape[1]} Features")
    with c3:
        if algo == "PCA":
            tot_var = np.sum(exp_var_2d) * 100 if active_components == 2 else np.sum(exp_var_3d) * 100
            st.metric("Explained Variance Ratio", f"{tot_var:.2f}%")
        elif algo == "LDA":
            st.metric("Class Separability", "Supervised (Labels used)")
        elif algo == "t-SNE":
            st.metric("KL Divergence Cost", f"{tsne_2d.kl_divergence_:.3f}")
        elif algo == "UMAP":
            st.metric("Projection Topology", "Fuzzy Simplicial Set")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs Configuration ───────────────────────────────────────────────────────
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Plot (2D)", "🧊 3D View", "📝 Notes & Concepts", 
        "🎓 Lecture & How it Works", "❓ Quiz Challenge", "🔗 Resources"
    ])

    # ── TAB 1: 2D Plot ──────────────────────────────────────────────────────────
    with t1:
        st.markdown('<p class="section-header">2D Dimension Reduction Projection</p>', unsafe_allow_html=True)
        col_la, col_lb = st.columns([2, 1])
        
        with col_la:
            df_2d = pd.DataFrame({
                "Component 1": X_2d[:, 0],
                "Component 2": X_2d[:, 1],
                "Class Label": target_names
            })
            
            fig_2d = px.scatter(
                df_2d, x="Component 1", y="Component 2",
                color="Class Label",
                color_discrete_sequence=PALETTE,
                title=f"{algo} Projection (2D space)",
                hover_data=["Class Label"]
            )
            fig_2d.update_layout(
                paper_bgcolor='#0a0e1a',
                plot_bgcolor='#0d1522',
                font_color='#c8d8e8',
                legend_title_text='Class Labels',
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_2d, use_container_width=True)
            
        with col_lb:
            if algo == "PCA":
                st.markdown("**Explained Variance per component**")
                # Show barchart of components
                df_var = pd.DataFrame({
                    "PC": [f"PC{i+1}" for i in range(len(exp_var_2d))],
                    "Variance %": exp_var_2d * 100
                })
                fig_var = px.bar(
                    df_var, x="PC", y="Variance %",
                    color="PC", color_discrete_sequence=['#00d4ff', '#ff6b6b'],
                    height=260
                )
                fig_var.update_layout(
                    paper_bgcolor='#0a0e1a',
                    plot_bgcolor='#0d1522',
                    font_color='#c8d8e8',
                    showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_var, use_container_width=True)
                st.info("💡 **Variance Interpretation:** PC1 points in the direction of the highest variance. The larger the bar, the more information is preserved by that component.")
            elif algo == "LDA":
                st.markdown("**Supervised Separability**")
                st.info("💡 LDA maximizes the distance between the mean vectors of different classes while minimizing the spread of points around each mean. This creates distinct clusters even if classes overlapped in the original space.")
            else:
                st.markdown("**Local Neighborhoods**")
                st.info("💡 Non-linear algorithms focus on preserving the structure of points relative to their nearest neighbors, creating tight localized clusters in 2D space.")

    # ── TAB 2: 3D View ──────────────────────────────────────────────────────────
    with t2:
        st.markdown('<p class="section-header">3D Component Space Visualization</p>', unsafe_allow_html=True)
        col_3da, col_3db = st.columns([2, 1])
        
        with col_3da:
            df_3d = pd.DataFrame({
                "Comp 1": X_3d[:, 0],
                "Comp 2": X_3d[:, 1],
                "Comp 3": X_3d[:, 2],
                "Class Label": target_names
            })
            
            fig_3d = px.scatter_3d(
                df_3d, x="Comp 1", y="Comp 2", z="Comp 3",
                color="Class Label",
                color_discrete_sequence=PALETTE,
                title=f"{algo} 3D Projection Space",
                hover_data=["Class Label"]
            )
            fig_3d.update_layout(
                paper_bgcolor='#0a0e1a',
                font_color='#c8d8e8',
                margin=dict(l=0, r=0, t=40, b=0)
            )
            fig_3d.update_scenes(
                xaxis=dict(title="Component 1", backgroundcolor='#0d1522', gridcolor='#1e3a5f', showbackground=True),
                yaxis=dict(title="Component 2", backgroundcolor='#0d1522', gridcolor='#1e3a5f', showbackground=True),
                zaxis=dict(title="Component 3", backgroundcolor='#0d1522', gridcolor='#1e3a5f', showbackground=True)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with col_3db:
            if algo == "PCA":
                st.markdown("**3D Biplot Concept**")
                st.info("💡 Rotating the 3D scatter plot helps you see how the Iris clusters are separated along the third principal component direction. In PCA, these components represent orthogonal linear equations of the original attributes.")
                
                # Dynamic variance distribution metrics
                st.metric("PC1 Variance Ratio", f"{exp_var_3d[0]*100:.1f}%")
                st.metric("PC2 Variance Ratio", f"{exp_var_3d[1]*100:.1f}%")
                st.metric("PC3 Variance Ratio", f"{exp_var_3d[2]*100:.1f}%")
            else:
                st.markdown("**3D Topologies**")
                st.info("💡 Projecting non-linear manifolds into 3D allows the algorithm to untangle intersecting loops or curved sheets of data points that would overlap if squeezed into 2D.")

    # ── TAB 3: Notes & Concepts ─────────────────────────────────────────────────
    with t3:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#1a2d4a);border:1px solid #00d4ff33;
        border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
          <p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0;">📝 {algo} Mathematical Notes</p>
          <p style="color:#8899aa;font-size:0.9rem;margin:0.4rem 0 0;">Formulas, parameters, use-cases, and complexity metrics</p>
        </div>
        """, unsafe_allow_html=True)
        
        info = REDUCTION_INFO.get(algo, {})
        
        # 1. Core Idea
        st.markdown("### 🎯 Core Idea")
        st.write(info["desc"])
        
        # 2. Math Formula
        st.markdown("### 🧮 Mathematical Intuition")
        if algo == "PCA":
            st.write("We compute the **Covariance Matrix** $C$ of the standardized data matrix $X$:")
            st.latex(r"C = \frac{1}{n} X^T X")
            st.write("Then, we calculate eigenvectors $\mathbf{v}_i$ and eigenvalues $\lambda_i$ of $C$ by solving:")
            st.latex(r"C \mathbf{v}_i = \lambda_i \mathbf{v}_i")
            st.write("The projected representation $Y$ is obtained by multiplying the original data with the top eigenvectors:")
            st.latex(r"Y = X V_k")
        elif algo == "LDA":
            st.write("LDA maximizes Fisher's criterion, defined as the ratio of between-class scatter $S_b$ to within-class scatter $S_w$:")
            st.latex(r"J(\mathbf{w}) = \frac{\mathbf{w}^T S_b \mathbf{w}}{\mathbf{w}^T S_w \mathbf{w}}")
            st.write("Solving this generalized eigenvalue problem leads to the projection weight matrix $W$:")
            st.latex(r"W = S_w^{-1} S_b")
        elif algo == "t-SNE":
            st.write("t-SNE converts high-dimensional Euclidean distances between points $x_i$ and $x_j$ into conditional probabilities $p_{j|i}$:")
            st.latex(r"p_{j|i} = \frac{\exp(-\|\mathbf{x}_i - \mathbf{x}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{x}_i - \mathbf{x}_k\|^2 / 2\sigma_i^2)}")
            st.write("In the low-dimensional space, the similarity of $y_i$ and $y_j$ is modeled using a Student-t distribution:")
            st.latex(r"q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}")
            st.write("t-SNE minimizes the **Kullback-Leibler (KL) divergence** between the two distributions:")
            st.latex(r"KL(P \parallel Q) = \sum_i \sum_{j} p_{ij} \log \frac{p_{ij}}{q_{ij}}")
        elif algo == "UMAP":
            st.write("UMAP models the high-dimensional data using local fuzzy simplicial sets. In high dimensions, the fuzzy set membership is modeled as:")
            st.latex(r"p_{i|j} = \exp\left(-\frac{\max(0, d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i)}{\sigma_i}\right)")
            st.write("where $\rho_i$ is the distance to the nearest neighbor of $x_i$. In the low-dimensional space, UMAP uses a fuzzy set structure:")
            st.latex(r"q_{ij} = \left(1 + a \|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\right)^{-1}")
            st.write("UMAP minimizes the **Fuzzy Set Cross-Entropy** rather than KL-divergence, preserving both local and global layout structures:")
            st.latex(r"\mathcal{L}_{CE}(P, Q) = \sum_{i \neq j} \left[ p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}} \right]")
            
        # 3. Hyperparameters Table
        st.markdown("### 🎛️ Hyperparameters Table")
        if algo == "PCA":
            st.markdown("""
            | Parameter | Default | Effect |
            |---|---|---|
            | `n_components` | `2` | Number of dimensions to project to (2D or 3D). |
            | `whiten` | `False` | Scales components to unit variance; removes noise but changes scale. |
            | `svd_solver` | `'auto'` | Algorithm to perform singular value decomposition. |
            """)
        elif algo == "LDA":
            st.markdown("""
            | Parameter | Default | Effect |
            |---|---|---|
            | `n_components` | `2` | Target dimensions (Max is number of classes - 1). |
            | `solver` | `'svd'` | Solver algorithm ('svd', 'lsqr', or 'eigen'). |
            """)
        elif algo == "t-SNE":
            st.markdown("""
            | Parameter | Default | Effect |
            |---|---|---|
            | `perplexity` | `30` | Number of local neighbors considered. Balances local vs global features. |
            | `learning_rate` | `200` | Controls step size during gradient descent optimization. |
            | `n_iter` | `1000` | Max iterations for gradient optimization. |
            """)
        elif algo == "UMAP":
            st.markdown("""
            | Parameter | Default | Effect |
            |---|---|---|
            | `n_neighbors` | `15` | Target number of neighbors to consider. Low values capture local structure; high values capture global layouts. |
            | `min_dist` | `0.1` | Minimum distance allowed between points in 2D/3D space, controlling cluster density. |
            | `metric` | `'euclidean'` | Method to calculate distance (Euclidean, Manhattan, Cosine, etc.). |
            """)
            
        # 4. Use / Do not use
        st.markdown("### 🧭 Usage Guidelines")
        col_use, col_nuse = st.columns(2)
        with col_use:
            st.success("✅ **When to Use:**")
            if algo == "PCA":
                st.markdown("- You need quick, linear dimension reduction.\n- You want to eliminate multicollinearity for downstream models.\n- You don't have target class labels.")
            elif algo == "LDA":
                st.markdown("- You have class labels and want to maximize separation.\n- You are preprocessing for linear classification models.")
            elif algo == "t-SNE":
                st.markdown("- You want to visualize highly complex non-linear structures in 2D.\n- Your goal is strictly visualization/clustering inspection.")
            elif algo == "UMAP":
                st.markdown("- You have a massive dataset and need fast non-linear mapping.\n- You want to project new test samples onto the learned embedding.\n- You want to preserve both local neighborhoods and global shapes.")
        with col_nuse:
            st.error("❌ **When NOT to Use:**")
            if algo == "PCA":
                st.markdown("- The relationship between features is strongly non-linear.\n- You want to preserve non-linear structures (use t-SNE or UMAP instead).")
            elif algo == "LDA":
                st.markdown("- Your target class label is continuous (regression).\n- Your dataset features are strongly non-Gaussian.")
            elif algo == "t-SNE":
                st.markdown("- You want to run a model on the reduced coordinates (non-parametric).\n- You need to preserve distance comparisons between distant clusters.")
            elif algo == "UMAP":
                st.markdown("- You want mathematically simple, highly explainable coefficients.\n- You don't have the external libraries installed.")
                
        # 5. Time Complexity
        st.markdown("### ⏱️ Computational Complexity")
        st.markdown(f"**Time Complexity:** <span class=\"algo-tag\">{info['complexity']}</span>", unsafe_allow_html=True)

    # ── TAB 4: Lecture & Walkthrough ─────────────────────────────────────────────
    with t4:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#1a2d4a);border:1px solid #00d4ff33;
        border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
          <p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0;">🎓 Lecture: How {algo} Works</p>
          <p style="color:#8899aa;font-size:0.9rem;margin:0.4rem 0 0;">Intuitive analogies, step-by-step algorithms, and comparative analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        info = REDUCTION_INFO.get(algo, {})
        
        # Analogy
        st.markdown("### 💡 Real-World Analogy")
        st.write(info["easy"])
        
        # Steps
        st.markdown("### ⚙️ Step-by-Step Algorithm Walkthrough")
        if algo == "PCA":
            st.markdown("""
            1. **Standardize data:** Subtract the mean and divide by standard deviation for each feature.
            2. **Compute Covariance:** Calculate the covariance matrix representing feature relationships.
            3. **Eigendecomposition:** Find the eigenvectors (directions) and eigenvalues (variance sizes).
            4. **Sort and select:** Sort eigenvalues from largest to smallest and choose the top $k$ eigenvectors.
            5. **Project:** Matrix-multiply the standardized data by the selected eigenvectors.
            """)
        elif algo == "LDA":
            st.markdown("""
            1. **Compute class means:** Calculate the mean vector for each target class.
            2. **Compute scatter matrices:** Compute Within-class scatter ($S_w$) and Between-class scatter ($S_b$).
            3. **Solve Eigensystem:** Solve generalized eigenvalues for $S_w^{-1} S_b$.
            4. **Select discriminants:** Sort eigenvectors by eigenvalues and select the top $k$ directions.
            5. **Transform data:** Project original data onto the discriminant weight vectors.
            """)
        elif algo == "t-SNE":
            st.markdown("""
            1. **High-dim probabilities:** Calculate pairwise similarities as conditional probabilities using Gaussian kernels.
            2. **Force symmetry:** Average the conditional probabilities: $p_{ij} = (p_{j|i} + p_{i|j}) / 2N$.
            3. **Initialize output:** Place target points randomly in 2D or 3D space.
            4. **Low-dim probabilities:** Model low-dimensional similarities using the heavy-tailed Student-t distribution.
            5. **Gradient Descent:** Optimize low-dimensional coordinates by minimizing KL-divergence via gradient updates.
            """)
        elif algo == "UMAP":
            st.markdown("""
            1. **Simplicial complexes:** Model high-dimensional layout structures using fuzzy simplicial sets.
            2. **Optimize distances:** Calculate geodesic distances to find local metrics around each point.
            3. **Low-dim mapping:** Initialize low-dimensional coordinates using spectral embedding.
            4. **Fuzzy optimization:** Minimize fuzzy cross-entropy using stochastic gradient descent.
            5. **Coordinate shift:** Shift coordinates to match target configurations, retaining local/global manifolds.
            """)
            
        # ASCII Flowchart
        st.markdown("### ⛓️ Data Flow Pipeline")
        if algo == "PCA":
            st.code("""
[High-Dim Data (n x d)] ──► [Scale (Mean=0, Var=1)] ──► [Covariance Matrix] ──► [SVD / Eigenvectors] ──► [Project to PC1, PC2]
            """, language="text")
        elif algo == "LDA":
            st.code("""
[High-Dim Data + Labels] ──► [Mean Vectors] ──► [Compute Sw & Sb Matrices] ──► [Solve Sw⁻¹Sb] ──► [Linear Discriminant Plane]
            """, language="text")
        elif algo == "t-SNE":
            st.code("""
[High-Dim Points] ──► [Gaussian Pairwise Sim] ──► [Student-t Low-Dim Sim] ──► [Gradient Descent (KL Minimization)] ──► [2D Clusters]
            """, language="text")
        elif algo == "UMAP":
            st.code("""
[High-Dim Points] ──► [ Riemannian Fuzzy Complexes ] ──► [Spectral Initialization] ──► [Cross-Entropy SGD Optimization] ──► [Manifold Plane]
            """, language="text")

        # Pros and Cons
        st.markdown("### 🏆 Advantages & Limitations")
        col_pro, col_con = st.columns(2)
        with col_pro:
            st.markdown("**Advantages:**")
            for p in info["pros"]:
                st.markdown(f"- {p}")
        with col_con:
            st.markdown("**Disadvantages:**")
            for c in info["cons"]:
                st.markdown(f"- {c}")

        # Comparison table
        st.markdown("### 📊 Dimension Reduction Comparison")
        st.markdown("""
        | Algorithm | Supervised? | Type | Complexity | Preserves |
        |---|---|---|---|---|
        | **PCA** | No | Linear | O(d²n + d³) | Global Variance |
        | **LDA** | Yes | Linear | O(d²n + d³) | Class Boundaries |
        | **t-SNE** | No | Non-Linear | O(n²) | Local Neighbors |
        | **UMAP** | No / Yes | Non-Linear | O(n log n) | Local + Global Manifolds |
        """)

        # Real-World Applications
        st.markdown("### 🌍 Real-World Applications")
        st.markdown("- 🧬 **Genomics:** Visualizing complex cellular properties and genetic maps.")
        st.markdown("- 🖼️ **Computer Vision:** Compressing feature descriptors (like SIFT/SURF) before model matching.")
        st.markdown("- 💳 **Anomaly Detection:** Simplifying multi-dimensional transaction rows to spot outliers in 2D clusters.")
        st.markdown("- 🔍 **Semantic Search:** Projecting word or document embeddings into 2D to map contextual semantic spaces.")

    # ── TAB 5: Quiz Challenge ───────────────────────────────────────────────────
    with t5:
        render_quiz(algo)

    # ── TAB 6: Resources ────────────────────────────────────────────────────────
    with t6:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#0d1b2a,#1a2d4a);border:1px solid #00d4ff33;
        border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
          <p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0;">🔗 Learn More — {algo} Resources</p>
          <p style="color:#8899aa;font-size:0.9rem;margin:0.4rem 0 0;">Academic publications, manuals, lectures, and guides</p>
        </div>
        """, unsafe_allow_html=True)
        
        info = REDUCTION_INFO.get(algo, {})
        
        # Video embed if available
        if info.get("video"):
            st.markdown('<p class="section-header">🎥 Masterclass Video Lecture</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="video-container">
                <iframe src="{info['video']}" title="{algo} Lecture" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown('<p class="section-header">📘 Reference Links</p>', unsafe_allow_html=True)
        st.markdown(f"- [Scikit-learn Documentation]({info['doc']})")
        st.markdown(f"- [Original Research Paper]({info['paper']})")
        st.markdown("- [GeeksforGeeks tutorial](https://www.geeksforgeeks.org/dimensionality-reduction/)")
        st.markdown("- [Interactive Playground / Visual explanation](https://distill.pub/2016/misread-tsne/)")
