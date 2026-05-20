import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

PALETTE = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#a855f7', '#f97316', '#ec4899']

CLUSTER_INFO = {
    "K-Means": {
        "type": "Centroid-Based", "complexity": "Low",
        "desc": "Partitions data into K clusters by minimizing intra-cluster variance (sum of squared distances to centroid).",
        "easy": "Imagine sorting coloured balls into K buckets — each ball goes to the nearest bucket centre. Repeat until stable.",
        "example": "🛒 Customer segmentation: group customers into Budget, Mid-Range, Premium based on spending & frequency.",
        "pros": "Simple, scalable, fast on large datasets.",
        "cons": "Must choose K in advance; fails on non-spherical clusters.",
        "video": {"url": "https://www.youtube.com/embed/4b5d3muPQmA", "title": "K-means clustering explained by StatQuest"},
        "gfg": [
            {"label": "K-Means Clustering", "url": "https://www.geeksforgeeks.org/k-means-clustering-introduction/"},
            {"label": "Elbow Method to find K", "url": "https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/"},
        ]
    },
    "DBSCAN": {
        "type": "Density-Based", "complexity": "Medium",
        "desc": "Groups densely packed points together and marks low-density points as noise/outliers.",
        "easy": "Like finding crowds at a concert — packed groups are clusters, isolated people are outliers.",
        "example": "🚨 Fraud detection: genuine transactions form dense clusters; fraudulent ones appear as isolated outliers.",
        "pros": "Finds arbitrary shapes; detects outliers automatically.",
        "cons": "Two parameters (eps, min_samples) are tricky to tune.",
        "video": {"url": "https://www.youtube.com/embed/RDZUdRSDOok", "title": "DBSCAN explained by StatQuest"},
        "gfg": [
            {"label": "DBSCAN Clustering", "url": "https://www.geeksforgeeks.org/dbscan-clustering-in-ml-density-based-clustering/"},
            {"label": "DBSCAN vs K-Means", "url": "https://www.geeksforgeeks.org/difference-between-k-means-and-dbscan-clustering/"},
        ]
    },
    "Hierarchical": {
        "type": "Hierarchical", "complexity": "High",
        "desc": "Builds a tree (dendrogram) by merging nearest data points or clusters bottom-up.",
        "easy": "Like a family tree — start with individuals, merge into families, then clans, then tribes.",
        "example": "🧬 Gene analysis: genes with similar expression patterns merge step by step into functional groups.",
        "pros": "No need to set K in advance; produces an intuitive dendrogram.",
        "cons": "Slow on large datasets O(n²); no cluster reassignment after merging.",
        "video": {"url": "https://www.youtube.com/embed/7xHsRkOdVwo", "title": "Hierarchical Clustering explained by StatQuest"},
        "gfg": [
            {"label": "Hierarchical Clustering", "url": "https://www.geeksforgeeks.org/hierarchical-clustering-in-machine-learning/"},
            {"label": "Dendrogram Explained", "url": "https://www.geeksforgeeks.org/scipy-cluster-hierarchy-dendrogram/"},
        ]
    },
    "Gaussian Mixture": {
        "type": "Probabilistic", "complexity": "Medium",
        "desc": "Models data as a mixture of Gaussian distributions. Each point gets a soft probability score for each cluster.",
        "easy": "Imagine your data is a blend of several bell curves. GMM figures out how many, their shapes, and where they are.",
        "example": "🎙️ Speaker recognition: different speakers form different Gaussian blobs in audio feature space.",
        "pros": "Soft assignments (probabilities); handles elliptical clusters.",
        "cons": "Assumes Gaussian shape; needs K; may diverge.",
        "video": {"url": "https://www.youtube.com/embed/JNlHZ_S1Ttc", "title": "Gaussian Mixture Models explained by StatQuest"},
        "gfg": [
            {"label": "Gaussian Mixture Models", "url": "https://www.geeksforgeeks.org/gaussian-mixture-model/"},
            {"label": "EM Algorithm", "url": "https://www.geeksforgeeks.org/expectation-maximization-algorithm/"},
        ]
    },
}

QUIZZES = {
    "K-Means": [
        {
            "question": "What is the primary objective of the standard K-Means algorithm?",
            "options": [
                "Maximize the silhouette score across all clusters.",
                "Minimize the Within-Cluster Sum of Squares (WCSS / Inertia).",
                "Maximize the distances between all cluster centroids.",
                "Automatically estimate the optimal number of clusters."
            ],
            "correct": 1,
            "explanation": "K-Means iteratively minimizes the Within-Cluster Sum of Squares (WCSS) or Inertia, which is the sum of squared Euclidean distances of points to their closest centroid."
        },
        {
            "question": "Which of the following describes a key limitation of the K-Means algorithm?",
            "options": [
                "It is computationally too heavy for standard tabular datasets.",
                "It requires setting centroid shapes manually in advance.",
                "It assumes spherical cluster shapes of equal variance, failing on highly non-spherical clusters.",
                "It is a supervised classifier and requires class target labels to function."
            ],
            "correct": 2,
            "explanation": "Because K-Means uses standard Euclidean distance, it is structurally biased toward finding spherical, compact clusters of similar sizes. It fails on complex structures like moons or concentric circles."
        },
        {
            "question": "How does KMeans++ improve standard K-Means?",
            "options": [
                "It selects the number of clusters (K) automatically.",
                "It uses Manhattan distance instead of Euclidean distance.",
                "It initializes centroids sequentially, picking each new center with probability proportional to its distance from existing centers.",
                "It maps the dataset into an infinite-dimensional space first."
            ],
            "correct": 2,
            "explanation": "KMeans++ optimizes centroid initialization by placing starting centers far away from each other. This reduces convergence time and avoids bad local minima."
        }
    ],
    "DBSCAN": [
        {
            "question": "In DBSCAN, how is a Core Point mathematically defined?",
            "options": [
                "A point that lies exactly at the center of the dataset.",
                "A point that contains at least min_samples points within its eps-neighborhood radius.",
                "A point that has the lowest silhouette coefficient.",
                "A point that lies on the boundary of two clusters."
            ],
            "correct": 1,
            "explanation": "A point is a Core Point if its $\\epsilon$-neighborhood contains at least the user-defined `min_samples` (including the point itself)."
        },
        {
            "question": "How does DBSCAN handle points that are neither Core Points nor density-reachable from any Core Point?",
            "options": [
                "It assigns them to the nearest centroid.",
                "It clusters them into a single auxiliary cluster.",
                "It labels them as noise (-1) and excludes them from active clusters.",
                "It halts execution and returns an error."
            ],
            "correct": 2,
            "explanation": "Points that are not Core Points and cannot be reached from any Core Point are categorized as noise/outliers, labeled as $-1$, and are excluded from clusters."
        },
        {
            "question": "What is the primary advantage of DBSCAN over K-Means?",
            "options": [
                "It always scales linearly and is faster on millions of points.",
                "It assigns soft probabilities to cluster assignments.",
                "It discovers clusters of arbitrary shapes and does not require pre-specifying the number of clusters (K).",
                "It works well regardless of highly varying density regions."
            ],
            "correct": 2,
            "explanation": "DBSCAN clusters points by local density, enabling it to find arbitrary geometries (like concentric rings or moons) without requiring the user to guess $K$ beforehand."
        }
    ],
    "Hierarchical": [
        {
            "question": "What is a Dendrogram in Hierarchical Clustering?",
            "options": [
                "A scatter plot showing 3D cluster centroids.",
                "A tree-structured diagram illustrating the sequence and distance of cluster mergers/splits.",
                "A boundary mapping showing predicted classifications.",
                "A curve that determines the variance ratio of clusters."
            ],
            "correct": 1,
            "explanation": "A dendrogram is a tree diagram showing how clusters are iteratively merged (agglomerative) or split (divisive) and the distance levels at which these splits happen."
        },
        {
            "question": "What is defined by 'Complete Linkage' in Agglomerative Hierarchical Clustering?",
            "options": [
                "The minimum distance between any point in Cluster A and any point in Cluster B.",
                "The maximum distance (furthest neighbors) between any point in Cluster A and any point in Cluster B.",
                "The distance between the centroids of Cluster A and Cluster B.",
                "The average of all pairwise distances between points in Cluster A and B."
            ],
            "correct": 1,
            "explanation": "Complete linkage defines the cluster distance as the maximum possible distance between a member of Cluster A and a member of Cluster B (furthest-neighbor method)."
        },
        {
            "question": "What is the main bottleneck of hierarchical clustering on massive datasets?",
            "options": [
                "It underfits the training data severely.",
                "It requires labeled inputs to calculate boundaries.",
                "It has O(N²) space and O(N³) time complexity, making it extremely slow and memory intensive.",
                "It is highly sensitive to the initial random centroids."
            ],
            "correct": 2,
            "explanation": "Agglomerative hierarchical clustering requires calculating and storing an $N \\times N$ distance matrix between all pairs of data points. This leads to high memory and time complexity, which is impractical for large datasets."
        }
    ],
    "Gaussian Mixture": [
        {
            "question": "What is the core difference between the 'soft assignment' of GMM and the 'hard assignment' of K-Means?",
            "options": [
                "GMM only clusters positive values.",
                "K-Means groups points using probability distributions, whereas GMM uses linear boundaries.",
                "GMM assigns each point a probability membership score for each cluster rather than a single rigid label.",
                "K-Means allows points to belong to multiple clusters simultaneously."
            ],
            "correct": 2,
            "explanation": "GMM is a probabilistic framework. Instead of assigning a point rigidly to one cluster (hard assignment), it computes posterior probabilities (responsibilities) of the point belonging to each Gaussian component."
        },
        {
            "question": "Which algorithm is used to train and estimate the parameters (means, covariances, weights) of a GMM?",
            "options": [
                "Gradient Descent with backpropagation.",
                "Expectation-Maximization (EM) Algorithm.",
                "Support Vector optimization.",
                "Levenberg-Marquardt optimizer."
            ],
            "correct": 1,
            "explanation": "GMM uses the Expectation-Maximization (EM) algorithm. In the E-step, it calculates responsibilities (membership weights); in the M-step, it updates the weights, means, and covariances of the Gaussians to maximize likelihood."
        },
        {
            "question": "How does fitting a 'full' covariance matrix in GMM benefit cluster boundaries compared to K-Means?",
            "options": [
                "It forces all clusters to be perfectly circular.",
                "It restricts the clusters to have equal volume.",
                "It allows clusters to adopt elliptical shapes with arbitrary scaling, rotation, and orientation.",
                "It speeds up parameter calculation significantly."
            ],
            "correct": 2,
            "explanation": "A full covariance matrix allows the multivariate Gaussian distributions to have arbitrary shape, rotation, and size. This lets GMM fit complex elliptical, stretched clusters that K-Means cannot model."
        }
    ]
}

DS_OPTIONS = ["Blobs", "Moons", "Circles", "Anisotropic"]

def set_style(fig, axes):
    fig.patch.set_facecolor('#0a0e1a')
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor('#0d1522')
        ax.tick_params(colors='#6b7a8d', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')
        ax.xaxis.label.set_color('#8899aa')
        ax.yaxis.label.set_color('#8899aa')
        ax.title.set_color('#00d4ff')

def make_dataset(name, n, noise):
    if name == "Blobs":
        X, _ = make_blobs(n_samples=n, n_features=3, centers=3, cluster_std=noise*1.5+0.5, random_state=42)
    elif name == "Moons":
        X, _ = make_moons(n_samples=n, noise=noise, random_state=42)
        third_feature = X[:, 0] * 0.7 + X[:, 1] * 0.5 + np.random.normal(0, noise * 0.15, len(X))
        X = np.column_stack([X, third_feature])
    elif name == "Circles":
        X, _ = make_circles(n_samples=n, noise=noise, factor=0.5, random_state=42)
        third_feature = X[:, 0] * 0.5 - X[:, 1] * 0.3 + np.random.normal(0, noise * 0.15, len(X))
        X = np.column_stack([X, third_feature])
    else:  # Anisotropic
        X, _ = make_blobs(n_samples=n, n_features=3, centers=3, cluster_std=noise*1.5+0.5, random_state=42)
        transform = [[0.6, -0.6, 0.2], [-0.4, 0.8, -0.1], [0.3, -0.2, 0.5]]
        X = X @ transform
    return StandardScaler().fit_transform(X)

def run_clustering(name, X, k, eps, min_s, gmm_cov="full"):
    if name == "K-Means":
        m = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = m.fit_predict(X)
        centers = m.cluster_centers_
    elif name == "DBSCAN":
        m = DBSCAN(eps=eps, min_samples=min_s)
        labels = m.fit_predict(X)
        centers = None
    elif name == "Hierarchical":
        m = AgglomerativeClustering(n_clusters=k)
        labels = m.fit_predict(X)
        centers = None
    else:
        m = GaussianMixture(n_components=k, covariance_type=gmm_cov, random_state=42)
        labels = m.fit_predict(X)
        centers = m.means_
    return labels, centers

def card(title, body, color="#00d4ff"):
    return f"""<div class="feature-card" style="border-left-color: {color};">
<p class="feature-title" style="color:{color};">{title}</p>
<p class="feature-desc">{body}</p></div>"""

def render_clustering_quiz(algo_name: str):
    """Renders the 3-question conceptual quiz for clustering."""
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

    questions = QUIZZES.get(algo_name, [])
    if not questions:
        st.warning("Quiz questions are not yet defined for this algorithm.")
        return

    score_key = f"quiz_score_cl_{algo_name}"
    submitted_key = f"quiz_submitted_cl_{algo_name}"

    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False

    user_choices = []

    for idx, q in enumerate(questions):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        choice = st.radio(
            "Select the correct option:",
            q["options"],
            key=f"q_cl_{algo_name}_{idx}",
            index=None,
            label_visibility="collapsed"
        )
        user_choices.append(choice)
        st.markdown("<br>", unsafe_allow_html=True)

    c_btn, c_score = st.columns([1, 2])
    with c_btn:
        submit = st.button("Submit Quiz", key=f"submit_btn_cl_{algo_name}", disabled=st.session_state[submitted_key])

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
        if correct_count == 3:
            badge_color = "var(--emerald)"
            badge_text = "Grandmaster! 🏆"
        elif correct_count == 2:
            badge_color = "var(--cyan)"
            badge_text = "Proficient! 🏅"

        st.markdown(f"""
        <div style="background: rgba(11, 15, 25, 0.8); border: 1px solid rgba(255,255,255,0.05); border-left: 5px solid {badge_color}; border-radius: 12px; padding: 1.2rem; text-align: center; margin-top: 1rem;">
            <p style="color: #94a3b8; font-family: 'Space Mono', monospace; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; margin: 0;">
                FINAL SCORE
            </p>
            <p style="font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: {badge_color}; margin: 0.3rem 0;">
                {correct_count} / 3
            </p>
            <span class="algo-tag" style="background: {badge_color}1a; border-color: {badge_color}44; color: {badge_color}; font-size: 0.85rem; font-weight: 700; padding: 0.3rem 1.2rem;">
                {badge_text}
            </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Retry Quiz", key=f"retry_btn_cl_{algo_name}"):
            st.session_state[submitted_key] = False
            for idx in range(len(questions)):
                if f"q_cl_{algo_name}_{idx}" in st.session_state:
                    del st.session_state[f"q_cl_{algo_name}_{idx}"]
            st.rerun()

def render_clustering_page():
    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**🔵 CLUSTERING**")
        algo = st.selectbox("Algorithm", list(CLUSTER_INFO.keys()), key="cl_algo")
        ds   = st.selectbox("Dataset",   DS_OPTIONS,               key="cl_ds")
        n    = st.slider("Samples",  50, 1000, 300, 50,             key="cl_n")
        noise= st.slider("Noise",    0.0, 1.0, 0.2, 0.05,          key="cl_noise")
        
        st.markdown("---")
        st.markdown("**🎛️ PARAMETERS**")
        
        # Dynamic visibility of parameters
        if algo in ["K-Means", "Gaussian Mixture", "Hierarchical"]:
            k = st.slider("Clusters (K)", 2, 6, 3, key="cl_k")
        else:
            k = 3
            
        if algo == "Gaussian Mixture":
            gmm_cov = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"], key="cl_gmm_cov")
        else:
            gmm_cov = "full"
            
        if algo == "DBSCAN":
            eps = st.slider("DBSCAN eps (Radius)", 0.05, 1.0, 0.3, 0.05, key="cl_eps")
            mins = st.slider("DBSCAN min_samples (Density)", 2, 15, 5, key="cl_mins")
        else:
            eps = 0.3
            mins = 5

    X = make_dataset(ds, n, noise)
    labels, centers = run_clustering(algo, X, k, eps, mins, gmm_cov)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    sil = silhouette_score(X, labels) if n_clusters > 1 and len(set(labels)) > 1 else 0.0
    noise_pts = int(np.sum(labels == -1))

    info = CLUSTER_INFO[algo]

    # Hero
    st.markdown(f"""<div class="hero-banner">
<p class="hero-title">🔵 Clustering Lab</p>
<p class="hero-subtitle">Unsupervised learning · Find hidden patterns · No labels needed<br>
<span style="color:#00d4ff88;">Algorithm: {algo} &nbsp;|&nbsp; Dataset: {ds}</span></p></div>""",
    unsafe_allow_html=True)

    # Metrics
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Clusters Found", n_clusters)
    with c2: st.metric("Silhouette Score", f"{sil:.3f}")
    with c3: st.metric("Total Points", n)
    with c4: st.metric("Noise Points", noise_pts)

    st.markdown("<br>", unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.tabs(["🗺️ Cluster Plot", "🧊 3D View", "📖 Theory Notes", "🧠 Brain Challenge", "📊 Elbow Curve"])

    # ── Tab 1: Cluster Plot ───────────────────────────────────────────────────
    with t1:
        ca, cb = st.columns([2,1])
        with ca:
            fig, ax = plt.subplots(figsize=(8,5))
            set_style(fig, [ax])
            unique = sorted(set(labels))
            for i, lbl in enumerate(unique):
                mask = labels == lbl
                color = '#555566' if lbl == -1 else PALETTE[i % len(PALETTE)]
                name_lbl = "Noise" if lbl == -1 else f"Cluster {lbl}"
                ax.scatter(X[mask,0], X[mask,1], c=color, label=name_lbl,
                           edgecolors='white', linewidths=0.3, s=35, alpha=0.85)
            if centers is not None:
                ax.scatter(centers[:,0], centers[:,1], c='white', marker='*',
                           s=250, zorder=5, edgecolors='#00d4ff', linewidths=1.5, label='Centers')
            ax.set_title(f"{algo} — {ds} Dataset")
            ax.legend(facecolor='#0d1522', edgecolor='#1e3a5f', labelcolor='#c8d8e8', fontsize=8)
            st.pyplot(fig, use_container_width=True)
        with cb:
            st.markdown(card(algo, info["desc"]), unsafe_allow_html=True)
            st.markdown(card("💡 Simple Explanation", info["easy"], "var(--gold)"), unsafe_allow_html=True)
            st.markdown(card("🌍 Real-Life Example", info["example"], "var(--purple)"), unsafe_allow_html=True)

    # ── Tab 2: 3D View ────────────────────────────────────────────────────────
    with t2:
        st.markdown('<p style="color: var(--cyan); font-family: \'Space Mono\', monospace; font-size: 0.95rem; font-weight: 700; margin: 0 0 1rem 0;">🛠️ Customize 3D Perspectives</p>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            cl_3d_color = st.selectbox("Color Mapping Perspective", ["Cluster Assignment", "Original Spatial Structure", "Centroid Proximity (Density)"], index=0, key="cl_3d_col")
        with col_c2:
            cl_3d_scale = st.selectbox("Color Scheme / Palette", ["Vibrant Space", "Plasma", "Viridis", "Tealrose", "Sunset"], index=0, key="cl_3d_scale")
        
        feature_options = ["Feature 1", "Feature 2", "Feature 3"]
        col_x, col_y, col_z = st.columns([1, 1, 1])
        with col_x:
            cl_3d_x = st.selectbox("X-Axis Feature", feature_options, index=0, key="cl_3d_x")
            x_idx = feature_options.index(cl_3d_x)
        with col_y:
            cl_3d_y = st.selectbox("Y-Axis Feature", feature_options, index=1, key="cl_3d_y")
            y_idx = feature_options.index(cl_3d_y)
        with col_z:
            cl_3d_z = st.selectbox("Z-Axis Feature", feature_options, index=2, key="cl_3d_z")
            z_idx = feature_options.index(cl_3d_z)
            
        # Extract dynamic coordinates
        x_coords = X[:, x_idx]
        y_coords = X[:, y_idx]
        z_coords = X[:, z_idx]
        
        # Setup colors/palette
        vibrant_sequence = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#a855f7', '#f97316', '#ec4899']
        plasma_sequence = ['#0d0887', '#9c179e', '#ed7953', '#f0f921']
        viridis_sequence = ['#440154', '#31688e', '#35b779', '#fde725']
        tealrose_sequence = ['#009B9E', '#C4EC74', '#F1B6DA', '#D01C8B']
        sunset_sequence = ['#fdca26', '#f07f4f', '#b83b7e', '#6b118b', '#0d0887']
        
        palettes = {
            "Vibrant Space": vibrant_sequence,
            "Plasma": plasma_sequence,
            "Viridis": viridis_sequence,
            "Tealrose": tealrose_sequence,
            "Sunset": sunset_sequence
        }
        discrete_seq = palettes.get(cl_3d_scale, vibrant_sequence)
        
        # Color perspective setup
        color_val = []
        is_continuous = False
        hover_names = []
        
        if cl_3d_color == "Cluster Assignment":
            color_val = labels.astype(str)
            hover_names = [f"Cluster: {lbl}" if lbl != -1 else "Noise/Outlier" for lbl in labels]
        elif cl_3d_color == "Original Spatial Structure":
            is_continuous = True
            color_val = np.linalg.norm(X, axis=1)
            hover_names = [f"Dist from Origin: {val:.2f}" for val in color_val]
        elif cl_3d_color == "Centroid Proximity (Density)":
            is_continuous = True
            # Calculate distance to nearest cluster representative
            proximity = np.zeros(len(X))
            for lbl in np.unique(labels):
                mask = labels == lbl
                if lbl == -1:
                    # Noise points have low proximity (default to high distance/noise color)
                    proximity[mask] = 2.0
                    continue
                cluster_pts = X[mask]
                if centers is not None and lbl < len(centers):
                    center = centers[lbl]
                else:
                    center = np.mean(cluster_pts, axis=0)
                dists = np.linalg.norm(cluster_pts - center, axis=1)
                proximity[mask] = dists
            color_val = proximity
            hover_names = [f"Centroid Distance: {val:.2f}" for val in color_val]
            
        df3 = pd.DataFrame({
            "x": x_coords,
            "y": y_coords,
            "z": z_coords,
            "color": color_val,
            "hover": hover_names
        })
        
        if is_continuous:
            fig3 = px.scatter_3d(df3, x="x", y="y", z="z",
                                 color="color",
                                 hover_name="hover",
                                 color_continuous_scale=cl_3d_scale if cl_3d_scale in ["Plasma", "Viridis", "Tealrose", "Sunset"] else "Viridis",
                                 title=f"3D Density & Structure Projection — {algo}", height=550)
        else:
            fig3 = px.scatter_3d(df3, x="x", y="y", z="z",
                                 color="color",
                                 hover_name="hover",
                                 color_discrete_sequence=discrete_seq,
                                 title=f"3D Cluster Space — {algo}", height=550)
                                 
        fig3.update_layout(paper_bgcolor='#0a0e1a', font_color='#c8d8e8',
                           margin=dict(l=0,r=0,t=40,b=0))
        fig3.update_scenes(
            xaxis=dict(title=cl_3d_x, backgroundcolor='#0d1522',gridcolor='#1e3a5f',showbackground=True),
            yaxis=dict(title=cl_3d_y, backgroundcolor='#0d1522',gridcolor='#1e3a5f',showbackground=True),
            zaxis=dict(title=cl_3d_z, backgroundcolor='#0d1522',gridcolor='#1e3a5f',showbackground=True))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 3: Theory Notes ───────────────────────────────────────────────────
    with t3:
        st.markdown(f"""<div style="background:linear-gradient(135deg,#0d1b2a,#1a2d4a);
border:1px solid #00d4ff33;border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
<p style="font-family:'Space Mono',monospace;color:#00d4ff;font-size:1.3rem;font-weight:700;margin:0;">📚 {algo} Theory</p>
<p style="color:#8899aa;font-size:0.9rem;margin:0.4rem 0 0;">Rigorous Mathematical Formulation · Intuition · Real-world applications</p></div>""",
        unsafe_allow_html=True)

        # Video Embed
        video = info.get("video")
        if video:
            st.markdown('<p class="section-header">🎥 Masterclass Video Lecture</p>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="video-container">
                <iframe src="{video['url']}" title="{video['title']}" 
                        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                        allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
            st.info("💡 **Learning Tip:** Watch the video to build visual intuition, then explore the mathematical equations below!")

        st.markdown('<p class="section-header">📘 Conceptual & Mathematical Core</p>', unsafe_allow_html=True)

        if algo == "K-Means":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is K-Means Clustering?</p>
                <p class="feature-desc">
                K-Means is the most popular <b>unsupervised centroid-based clustering</b> algorithm. It divides $N$ data points into $K$ disjoint groups (clusters) 
                so that points within each cluster are as close to each other as possible.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Mathematical Formulation")
            st.write("K-Means minimizes the **Within-Cluster Sum of Squares (WCSS / Inertia)**, defined as:")
            st.latex(r"J = \sum_{j=1}^K \sum_{\mathbf{x}_i \in S_j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2")
            st.write("where $S_j$ represents the set of points in Cluster $j$, and $\mathbf{\mu}_j$ is the **mean centroid** of that cluster:")
            st.latex(r"\mathbf{\mu}_j = \frac{1}{|S_j|} \sum_{\mathbf{x}_i \in S_j} \mathbf{x}_i")

            st.write("### ⚙️ The Lloyd's Algorithm Cycle")
            st.write("1. **Initialize:** Choose $K$ starting centroids randomly (or via KMeans++).")
            st.write("2. **Assignment Step:** Assign each data point $\mathbf{x}_i$ to its nearest centroid:")
            st.latex(r"S_j^{(t)} = \left\{ \mathbf{x}_i : \|\mathbf{x}_i - \mathbf{\mu}_j^{(t)}\|^2 \le \|\mathbf{x}_i - \mathbf{\mu}_{l}^{(t)}\|^2 \; \forall l = 1, \dots, K \right\}")
            st.write("3. **Update Step:** Recalculate each centroid to be the average of its assigned points:")
            st.latex(r"\mathbf{\mu}_j^{(t+1)} = \frac{1}{|S_j^{(t)}|} \sum_{\mathbf{x}_i \in S_j^{(t)}} \mathbf{x}_i")
            st.write("4. **Repeat:** Repeat steps 2 and 3 until the centroids no longer shift.")

        elif algo == "DBSCAN":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is DBSCAN?</p>
                <p class="feature-desc">
                <b>DBSCAN</b> (Density-Based Spatial Clustering of Applications with Noise) clusters points by defining areas of high local point density. 
                Unlike K-Means, it automatically spots outliers as noise and discovers clusters of arbitrary non-spherical shapes.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Theoretical Framework")
            st.write("DBSCAN requires two parameters: a radius **Epsilon** ($\\epsilon$) and **Min Samples** ($M$).")
            st.write("For a point $\\mathbf{p}$ in dataset $D$, its **$\\epsilon$-Neighborhood** is defined as:")
            st.latex(r"N_\epsilon(\mathbf{p}) = \{ \mathbf{q} \in D \mid d(\mathbf{p}, \mathbf{q}) \le \epsilon \}")

            st.write("Points are categorized into three distinct roles:")
            st.write("1. **Core Point:** A point $\\mathbf{p}$ is a Core Point if it contains at least $M$ points in its neighborhood:")
            st.latex(r"|N_\epsilon(\mathbf{p})| \ge M")
            st.write("2. **Border Point:** A point $\\mathbf{q}$ that is not a Core Point, but is reachable from some Core Point (within its $\\epsilon$ boundary).")
            st.write("3. **Noise (Outlier):** Any point that is neither a Core Point nor reachable from a Core Point:")
            st.latex(r"\mathbf{o} \in D \text{ is noise if } |N_\epsilon(\mathbf{o})| < M \text{ and } \mathbf{o} \notin N_\epsilon(\mathbf{c}) \; \forall \text{ core points } \mathbf{c}")

            st.write("### 🔗 Density-Reachability")
            st.write("A point $\\mathbf{q}$ is **Directly Density-Reachable** from $\\mathbf{p}$ if $\\mathbf{q} \\in N_\\epsilon(\\mathbf{p})$ and $\\mathbf{p}$ is a Core Point. ")
            st.write("A point $\\mathbf{q}$ is **Density-Reachable** from $\\mathbf{p}$ if there is a chain of intermediate core points linking them together.")

        elif algo == "Hierarchical":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Hierarchical Clustering?</p>
                <p class="feature-desc">
                Hierarchical clustering builds a tree-like hierarchy of clusters. 
                The dominant approach is **Agglomerative (Bottom-Up)**, where each point starts as a single cluster and pairs are merged sequentially as we move up the tree.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Linkage Metrics (How Clusters Merge)")
            st.write("At each step, the two closest clusters $A$ and $B$ are merged. The distance between clusters can be defined in multiple ways:")

            st.write("**1. Single Linkage (Minimum Distance):** merges clusters based on their closest points.")
            st.latex(r"d(A, B) = \min \left\{ d(\mathbf{a}, \mathbf{b}) : \mathbf{a} \in A, \mathbf{b} \in B \right\}")

            st.write("**2. Complete Linkage (Maximum Distance):** merges based on their furthest points, creating compact clusters.")
            st.latex(r"d(A, B) = \max \left\{ d(\mathbf{a}, \mathbf{b}) : \mathbf{a} \in A, \mathbf{b} \in B \right\}")

            st.write("**3. Average Linkage:** merges based on average distance between all pairs.")
            st.latex(r"d(A, B) = \frac{1}{|A| \cdot |B|} \sum_{\mathbf{a} \in A} \sum_{\mathbf{b} \in B} d(\mathbf{a}, \mathbf{b})")

            st.write("**4. Ward's Linkage:** merges the two clusters that minimize the increase in total Within-Cluster Variance:")
            st.latex(r"\Delta(A, B) = \frac{|A| \cdot |B|}{|A| + |B|} \|\mathbf{\mu}_A - \mathbf{\mu}_B\|^2")

        elif algo == "Gaussian Mixture":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Gaussian Mixture Model?</p>
                <p class="feature-desc">
                A <b>Gaussian Mixture Model (GMM)</b> is a soft-probabilistic algorithm. 
                It assumes that the dataset is generated from a mixture of $K$ multi-dimensional Gaussian (Normal) distributions with separate shapes, means, and rotations.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Probability Density Formulation")
            st.write("The probability distribution of a data point $\\mathbf{x}$ is defined as a weighted sum of $K$ multivariate Gaussian densities:")
            st.latex(r"p(\mathbf{x}) = \sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x} \mid \mathbf{\mu}_j, \mathbf{\Sigma}_j)")
            st.write("where $\\pi_j$ is the **mixing weight** for component $j$, satisfying $\\sum \\pi_j = 1$, and $\\mathcal{N}$ represents the Gaussian distribution:")
            st.latex(r"\mathcal{N}(\mathbf{x} \mid \mathbf{\mu}_j, \mathbf{\Sigma}_j) = \frac{1}{(2\pi)^{d/2} |\mathbf{\Sigma}_j|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu}_j)^T \mathbf{\Sigma}_j^{-1} (\mathbf{x} - \mathbf{\mu}_j) \right)")

            st.write("### ⚙️ The Expectation-Maximization (EM) Optimization")
            st.write("GMM is trained using the **EM Algorithm** to find parameters that maximize data likelihood:")
            st.write("1. **Expectation (E-Step):** Calculate the probability (responsibility $\\gamma_{ij}$) that Gaussian $j$ generated point $\\mathbf{x}_i$:")
            st.latex(r"\gamma_{ij} = \frac{\pi_j \mathcal{N}(\mathbf{x}_i \mid \mathbf{\mu}_j, \mathbf{\Sigma}_j)}{\sum_{l=1}^K \pi_l \mathcal{N}(\mathbf{x}_i \mid \mathbf{\mu}_l, \mathbf{\Sigma}_l)}")
            st.write("2. **Maximization (M-Step):** Update the weights $\\pi_j$, means $\\mathbf{\mu}_j$, and covariance matrices $\\mathbf{\Sigma}_j$ using the weights $\\gamma_{ij}$ calculated in the E-step:")
            st.latex(r"\mathbf{\mu}_j^{\text{new}} = \frac{\sum_i \gamma_{ij} \mathbf{x}_i}{\sum_i \gamma_{ij}} \quad , \quad \mathbf{\Sigma}_j^{\text{new}} = \frac{\sum_i \gamma_{ij} (\mathbf{x}_i - \mathbf{\mu}_j^{\text{new}})(\mathbf{x}_i - \mathbf{\mu}_j^{\text{new}})^T}{\sum_i \gamma_{ij}}")

        # GFG Resources
        st.markdown('<p class="section-header">📎 External Academic Resources</p>', unsafe_allow_html=True)
        st.markdown('<div class="link-grid">', unsafe_allow_html=True)
        for lnk in info["gfg"]:
            st.markdown(f"""
            <a class="gfg-link-card" href="{lnk['url']}" target="_blank">
                🔗 {lnk['label']}
            </a>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Tab 4: Brain Challenge Quiz ───────────────────────────────────────────
    with t4:
        render_clustering_quiz(algo)

    # ── Tab 5: Elbow Curve ────────────────────────────────────────────────────
    with t5:
        if algo == "K-Means":
            inertias = []
            K_range = range(1, 10)
            for ki in K_range:
                km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)
            fig, ax = plt.subplots(figsize=(8,4))
            set_style(fig, [ax])
            ax.plot(list(K_range), inertias, 'o-', color='#00d4ff', lw=2, ms=7)
            ax.set_xlabel("Number of Clusters K")
            ax.set_ylabel("Inertia (WCSS)")
            ax.set_title("Elbow Method — Find Optimal K")
            ax.axvline(x=k, color='#ff6b6b', lw=1.5, linestyle='--', label=f"Current K={k}")
            ax.legend(facecolor='#0d1522', edgecolor='#1e3a5f', labelcolor='#c8d8e8')
            st.pyplot(fig, use_container_width=True)
            st.info("📌 The **Elbow Point** is where the curve bends sharply — that's the optimal K. Beyond it, adding more clusters gives diminishing returns.")
        else:
            st.info(f"ℹ️ The Elbow Curve is specific to K-Means. Select **K-Means** from the sidebar to view it.")
