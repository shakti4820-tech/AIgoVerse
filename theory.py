import streamlit as st

VIDEO_LINKS = {
    "Logistic Regression": {"url": "https://www.youtube.com/embed/yIYKR4sgzI8", "title": "Logistic Regression explained by StatQuest"},
    "Decision Tree": {"url": "https://www.youtube.com/embed/7VeUPuFGJHk", "title": "Decision Trees explained by StatQuest"},
    "Random Forest": {"url": "https://www.youtube.com/embed/J4Wdy0Wc_xQ", "title": "Random Forests explained by StatQuest"},
    "SVM (RBF Kernel)": {"url": "https://www.youtube.com/embed/efR1C6CvhmE", "title": "Support Vector Machines explained by StatQuest"},
    "K-Nearest Neighbors": {"url": "https://www.youtube.com/embed/4Ws0oPH350U", "title": "K-Nearest Neighbors explained by StatQuest"},
    "Naive Bayes": {"url": "https://www.youtube.com/embed/CPyugSLBLWQ", "title": "Naive Bayes explained by StatQuest"},
    "Gradient Boosting": {"url": "https://www.youtube.com/embed/3CC4N4z3GJc", "title": "Gradient Boosting explained by StatQuest"},
}

GFG_LINKS = {
    "Logistic Regression": [
        {"label": "Understanding Logistic Regression", "url": "https://www.geeksforgeeks.org/understanding-logistic-regression/"},
        {"label": "Logistic Regression in ML", "url": "https://www.geeksforgeeks.org/logistic-regression-in-machine-learning/"},
        {"label": "Sigmoid Function Details", "url": "https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/"},
    ],
    "Decision Tree": [
        {"label": "Decision Tree — Introduction", "url": "https://www.geeksforgeeks.org/decision-tree/"},
        {"label": "Decision Tree Classification", "url": "https://www.geeksforgeeks.org/decision-tree-implementation-python/"},
        {"label": "Gini Impurity vs Entropy", "url": "https://www.geeksforgeeks.org/gini-impurity-and-entropy-in-decision-tree-ml/"},
    ],
    "Random Forest": [
        {"label": "Random Forest Algorithm", "url": "https://www.geeksforgeeks.org/random-forest-algorithm-in-machine-learning/"},
        {"label": "Bagging vs Boosting", "url": "https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/"},
        {"label": "Feature Importance", "url": "https://www.geeksforgeeks.org/feature-importance-in-random-forest/"},
    ],
    "SVM (RBF Kernel)": [
        {"label": "Support Vector Machine", "url": "https://www.geeksforgeeks.org/support-vector-machine-algorithm/"},
        {"label": "Kernel Trick in SVM", "url": "https://www.geeksforgeeks.org/kernel-trick-in-machine-learning/"},
        {"label": "Hyperparameters C and Gamma", "url": "https://www.geeksforgeeks.org/role-of-c-in-svm/"},
    ],
    "K-Nearest Neighbors": [
        {"label": "K-Nearest Neighbours Algorithm", "url": "https://www.geeksforgeeks.org/k-nearest-neighbours/"},
        {"label": "Choosing the right K", "url": "https://www.geeksforgeeks.org/how-to-choose-the-value-of-k-in-k-nearest-neighbors/"},
        {"label": "Curse of Dimensionality", "url": "https://www.geeksforgeeks.org/curse-of-dimensionality-in-machine-learning/"},
    ],
    "Naive Bayes": [
        {"label": "Naive Bayes Classifiers", "url": "https://www.geeksforgeeks.org/naive-bayes-classifiers/"},
        {"label": "Bayes Theorem", "url": "https://www.geeksforgeeks.org/bayes-theorem/"},
        {"label": "Gaussian Naive Bayes", "url": "https://www.geeksforgeeks.org/gaussian-naive-bayes/"},
    ],
    "Gradient Boosting": [
        {"label": "Gradient Boosting Algorithm", "url": "https://www.geeksforgeeks.org/ml-gradient-boosting/"},
        {"label": "XGBoost vs Gradient Boosting", "url": "https://www.geeksforgeeks.org/xgboost-vs-gradient-boosting/"},
        {"label": "Boosting in Ensemble Learning", "url": "https://www.geeksforgeeks.org/boosting-in-machine-learning-boosting-and-adaboost/"},
    ],
}

QUIZZES = {
    "Logistic Regression": [
        {
            "question": "What is the primary role of the Sigmoid function in Logistic Regression?",
            "options": [
                "To draw a linear decision boundary.",
                "To map any real-valued number into a probability range [0, 1].",
                "To calculate the model accuracy.",
                "To perform gradient descent updates."
            ],
            "correct": 1,
            "explanation": "The Sigmoid function $\\sigma(z) = \\frac{1}{1+e^{-z}}$ maps any real number into the interval $[0, 1]$, which can be interpreted as a probability score."
        },
        {
            "question": "Which loss function is minimized during the training of Logistic Regression?",
            "options": [
                "Mean Squared Error (MSE)",
                "Hinge Loss",
                "Binary Cross-Entropy Loss (Log Loss)",
                "Absolute Error (MAE)"
            ],
            "correct": 2,
            "explanation": "Logistic Regression uses Binary Cross-Entropy Loss (also known as Log Loss) which penalizes confident but incorrect classifications using negative logarithms."
        },
        {
            "question": "How does increasing the regularization parameter C (which is the inverse of regularization strength) affect the model?",
            "options": [
                "Stronger regularization, leading to high bias / underfitting.",
                "Weaker regularization, allowing the model to fit training data more closely (potential overfitting).",
                "It has no effect on regularization.",
                "It forces all weights to become exactly zero."
            ],
            "correct": 1,
            "explanation": "In Scikit-Learn, $C = \\frac{1}{\\lambda}$. Therefore, increasing $C$ decreases the regularization strength $\\lambda$, making the model more complex and prone to overfitting."
        }
    ],
    "Decision Tree": [
        {
            "question": "Which of the following metrics measures the level of impurity or disorder in a node?",
            "options": [
                "R-squared",
                "Gini Impurity & Entropy",
                "Euclidean Distance",
                "Log Loss"
            ],
            "correct": 1,
            "explanation": "Gini Impurity and Entropy are used in Decision Trees to quantify the homogeneity (purity) of a node. A node is completely pure (Gini/Entropy = 0) if all samples belong to one class."
        },
        {
            "question": "What is a major risk of growing a Decision Tree to its maximum depth without pruning?",
            "options": [
                "Underfitting",
                "Overfitting",
                "High Bias",
                "Slow Training Speed"
            ],
            "correct": 1,
            "explanation": "An unconstrained decision tree will grow until every leaf is pure, fitting noise in the training set and causing high variance (overfitting)."
        },
        {
            "question": "If a split yields a child node with 10 samples of Class A and 0 samples of Class B, what is its Gini Impurity?",
            "options": [
                "0.0",
                "0.5",
                "1.0",
                "0.25"
            ],
            "correct": 0,
            "explanation": "The Gini Impurity formula is $1 - \\sum p_i^2$. Here, $p_A = 1$ and $p_B = 0$. Gini = $1 - (1^2 + 0^2) = 0.0$ (perfectly pure)."
        }
    ],
    "Random Forest": [
        {
            "question": "What is the key mechanism of Bootstrap Aggregating (Bagging) in Random Forests?",
            "options": [
                "Training trees sequentially to correct the mistakes of the previous tree.",
                "Training each tree on a random sample of the training data drawn with replacement.",
                "Using only a single feature to make predictions.",
                "Eliminating trees that perform below 50% accuracy."
            ],
            "correct": 1,
            "explanation": "Bagging builds independent trees on bootstrap samples (drawn with replacement). The predictions are aggregated by majority vote, reducing variance."
        },
        {
            "question": "Why does a Random Forest only consider a random subset of features at each node split?",
            "options": [
                "To speed up training only.",
                "To decorrelate the individual trees so that a dominant feature doesn't make all trees look identical.",
                "To make the model linear.",
                "To reduce the number of trees needed."
            ],
            "correct": 1,
            "explanation": "By forcing each split to consider a subset of features (typically $\\sqrt{d}$), the trees become highly diverse and decorrelated, maximizing ensemble performance."
        },
        {
            "question": "How does Random Forest make a final prediction for classification tasks?",
            "options": [
                "By averaging all predictions.",
                "By taking a majority vote among all trees.",
                "By picking the tree with the lowest depth.",
                "By selecting the first tree's prediction."
            ],
            "correct": 1,
            "explanation": "For classification, Random Forest aggregates individual tree decisions by taking a majority vote. For regression, it averages the predictions."
        }
    ],
    "SVM (RBF Kernel)": [
        {
            "question": "What are 'Support Vectors' in Support Vector Machines?",
            "options": [
                "All the data points in the dataset.",
                "The training points that lie closest to the decision boundary and define the margin.",
                "Vectors pointing to the origin.",
                "Weights assigned to regularized features."
            ],
            "correct": 1,
            "explanation": "Support Vectors are the crucial data points that lie directly on or inside the margin boundaries. The optimal separating hyperplane is determined solely by these points."
        },
        {
            "question": "What is the role of the 'Kernel Trick' in SVM?",
            "options": [
                "It reduces the dataset size to save memory.",
                "It implicitly maps data into a higher-dimensional space where a linear boundary can separate the classes.",
                "It automatically removes outliers from the data.",
                "It computes probabilities using Bayes' Theorem."
            ],
            "correct": 1,
            "explanation": "The Kernel Trick allows SVM to solve highly non-linear classification problems by computing high-dimensional inner products without explicitly projecting the data."
        },
        {
            "question": "In SVM with an RBF kernel, how does a high 'Gamma' value affect the decision boundary?",
            "options": [
                "It creates a smoother, more linear boundary.",
                "It makes the boundary highly complex, fitting closely around individual support vectors (potential overfitting).",
                "It disables regularization completely.",
                "It causes underfitting."
            ],
            "correct": 1,
            "explanation": "Gamma controls the radius of influence of individual support vectors. High gamma means a tight, localized influence, causing a very complex, wiggly boundary."
        }
    ],
    "K-Nearest Neighbors": [
        {
            "question": "How does a very small value of K (e.g., K = 1) affect the KNN model?",
            "options": [
                "It leads to a highly robust model with high bias (underfitting).",
                "It creates a highly complex, sensitive decision boundary that is prone to overfitting.",
                "It reduces the classification speed.",
                "It ignores all feature distances."
            ],
            "correct": 1,
            "explanation": "With $K=1$, the model copies the label of the single closest point, making it highly sensitive to noise and outliers (overfitting)."
        },
        {
            "question": "Why is feature scaling (like normalization or standardization) critical for KNN?",
            "options": [
                "Because KNN cannot handle raw floating-point numbers.",
                "Because KNN relies on distance metrics, so features with larger absolute scales would dominate the distance calculation.",
                "To speed up the training process.",
                "To prevent the model from underfitting."
            ],
            "correct": 1,
            "explanation": "KNN calculates geometric distances (e.g., Euclidean). If one feature has values in thousands and another in decimals, the first feature will dominate the distance metric completely."
        },
        {
            "question": "What is the computational complexity of predicting a label for a new point in a vanilla KNN?",
            "options": [
                "$O(1)$",
                "$O(N)$ where $N$ is the number of training samples.",
                "$O(\\log N)$",
                "$O(N^2)$"
            ],
            "correct": 1,
            "explanation": "KNN is a 'lazy learner'. To predict a label, it must compute the distance to every single training point ($O(N)$ operations), making prediction slow on large datasets."
        }
    ],
    "Naive Bayes": [
        {
            "question": "Why is Naive Bayes called 'Naive'?",
            "options": [
                "Because it is very simple to implement.",
                "Because it assumes that all features are conditionally independent given the class label.",
                "Because it cannot handle multi-class datasets.",
                "Because it ignores the prior probability of the classes."
            ],
            "correct": 1,
            "explanation": "It is called 'Naive' because it makes the strong (and often unrealistic) assumption that features are conditionally independent of each other, given the target class."
        },
        {
            "question": "Which mathematical theorem forms the foundation of Naive Bayes?",
            "options": [
                "Pythagorean Theorem",
                "Bayes' Theorem",
                "Central Limit Theorem",
                "Taylor's Theorem"
            ],
            "correct": 1,
            "explanation": "Naive Bayes is directly based on Bayes' Theorem: $P(C|X) = \\frac{P(X|C)P(C)}{P(X)}$."
        },
        {
            "question": "What type of distribution does Gaussian Naive Bayes assume for continuous features?",
            "options": [
                "Uniform Distribution",
                "Normal (Gaussian) Distribution",
                "Bernoulli Distribution",
                "Poisson Distribution"
            ],
            "correct": 1,
            "explanation": "Gaussian Naive Bayes assumes that the continuous features associated with each class follow a Normal (or Gaussian) distribution (bell curve)."
        }
    ],
    "Gradient Boosting": [
        {
            "question": "How are decision trees built in Gradient Boosting compared to Random Forest?",
            "options": [
                "Trees are built completely independently in parallel.",
                "Trees are built sequentially, with each tree trained to correct the errors of the preceding ensemble.",
                "Trees are combined by taking a simple average.",
                "Gradient Boosting does not use decision trees."
            ],
            "correct": 1,
            "explanation": "Unlike Random Forest which grows trees in parallel (bagging), Boosting grows trees sequentially. Each tree predicts the residuals (errors) of the current model."
        },
        {
            "question": "What is the purpose of the 'learning rate' parameter in Gradient Boosting?",
            "options": [
                "To speed up the model training time.",
                "To scale the contribution of each individual tree, acting as a shrinkage factor to prevent overfitting.",
                "To automatically select the best features.",
                "To change the split criterion."
            ],
            "correct": 1,
            "explanation": "The learning rate $\\eta$ scales the step size of gradient updates. Smaller learning rates require more trees but lead to highly robust, well-generalized models."
        },
        {
            "question": "What targets do the weak learners (decision trees) fit during each stage of Gradient Boosting?",
            "options": [
                "The original target labels $y$.",
                "The residuals (negative gradients of the loss function).",
                "The average prediction of the previous trees.",
                "Randomly shuffled target labels."
            ],
            "correct": 1,
            "explanation": "At each step, Gradient Boosting fits a new tree to predict the residuals (errors) of the existing ensemble, moving closer to the true target values."
        }
    ]
}

def render_theory(clf_name: str):
    """Render high-end theory notes, video embeds, and GFG references."""
    import streamlit as st
    
    # ── 1. Video Tutorial Section ─────────────────────────────────────────────
    video = VIDEO_LINKS.get(clf_name)
    if video:
        st.markdown('<p class="section-header">🎥 Masterclass Video Tutorial</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="video-container">
            <iframe src="{video['url']}" title="{video['title']}" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen></iframe>
        </div>
        """, unsafe_allow_html=True)
        st.info("💡 **Interactive Learning Tip:** Watch this curated masterclass to build strong intuition, then tune the hyperparameters in the sidebar to see the math in action!")

    # ── 2. Dynamic Rich Text Notes with LaTeX ───────────────────────────────
    st.markdown('<p class="section-header">📘 Conceptual & Mathematical Core</p>', unsafe_allow_html=True)

    if clf_name == "Logistic Regression":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is Logistic Regression?</p>
            <p class="feature-desc">
            Despite its name, Logistic Regression is a fundamental <b>classification</b> algorithm. 
            It models the probability that an input $X$ belongs to a binary class ($Y \\in \\{0, 1\\}$). 
            It outputs a continuous probability score between 0 and 1, which is then mapped to discrete class labels using a threshold (typically 0.5).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Mathematical Formulation")
        st.write("We compute the linear combination of inputs:")
        st.latex(r"z = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b")
        
        st.write("To map $z$ into the probability space $[0,1]$, we apply the **Sigmoid (Logistic) Function**:")
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        
        st.write("Hence, the predicted probability is:")
        st.latex(r"P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b)")

        st.write("During training, we minimize the **Binary Cross-Entropy Loss (Log Loss)** function using gradient descent:")
        st.latex(r"\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]")

    elif clf_name == "Decision Tree":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is a Decision Tree?</p>
            <p class="feature-desc">
            A Decision Tree classifies data by building a tree flowchart of feature splits. 
            Starting at the root, the algorithm splits the data recursively based on conditions that maximize the purity of the resulting subsets.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Split Impurity Metrics")
        st.write("At each node, the best feature and split point are chosen by minimizing node **impurity**. The two primary metrics are:")
        
        st.write("**1. Gini Impurity**:")
        st.latex(r"I_G(p) = 1 - \sum_{i=1}^C p_i^2")
        st.caption("Gini represents the probability of a randomly chosen element being incorrectly classified.")
        
        st.write("**2. Entropy (Information Theory)**:")
        st.latex(r"H(p) = -\sum_{i=1}^C p_i \log_2(p_i)")
        st.caption("Entropy measures the level of disorder or information uncertainty.")
        
        st.write("The algorithm selects splits that maximize the **Information Gain (IG)**:")
        st.latex(r"IG(D, A) = \text{Impurity}(D) - \sum_{v \in \text{splits}} \frac{|D_v|}{|D|} \text{Impurity}(D_v)")

    elif clf_name == "Random Forest":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is Random Forest?</p>
            <p class="feature-desc">
            Random Forest is an <b>ensemble bagging algorithm</b>. Instead of relying on a single complex decision tree (which overfits), 
            it builds an ensemble of hundreds of diverse decision trees in parallel, making final classifications via <b>majority vote</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Why it Works: Bagging and Decorrelation")
        st.write("If we have $T$ independent trees, each with variance $\sigma^2$, the variance of the average prediction is:")
        st.latex(r"\text{Variance}_{\text{ensemble}} = \frac{\sigma^2}{T}")
        st.write("To achieve tree independence, Random Forest applies two core strategies:")
        st.write("1. **Bootstrap Aggregating (Bagging):** Each tree trains on a random subset of data sampled *with replacement*.")
        st.write("2. **Feature Decorrelation:** At each split, only a random subset of $m$ features is considered:")
        st.latex(r"m \approx \sqrt{\text{total features}}")

    elif clf_name == "SVM (RBF Kernel)":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is Support Vector Machine?</p>
            <p class="feature-desc">
            SVM is a margin-maximization classifier. It fits the <b>widest possible road (margin)</b> that separates class clusters. 
            When data is non-linearly separable, it uses the <b>Kernel Trick</b> to map data into higher dimensions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Math of the Margin & Kernel")
        st.write("We maximize the margin subject to correct class boundaries, solving the dual optimization problem:")
        st.latex(r"\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)")
        st.caption("Subject to: $\\sum \\alpha_i y_i = 0$ and $0 \\le \\alpha_i \\le C$")
        
        st.write("The **Radial Basis Function (RBF) Kernel** computes similarities in infinite-dimensional Hilbert space:")
        st.latex(r"K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)")
        st.write("Here, $\\gamma$ controls the radius of influence of the Support Vectors.")

    elif clf_name == "K-Nearest Neighbors":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is K-Nearest Neighbors?</p>
            <p class="feature-desc">
            KNN is an <b>instance-based, non-parametric</b> classifier. It has no training phase ('lazy learner'). 
            To classify a new query point, it locates the $K$ closest training points in high-dimensional feature space and assigns their majority label.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Distance Calculations")
        st.write("Distance between two points $\\mathbf{p}$ and $\\mathbf{q}$ can be computed using different metrics:")
        
        st.write("**1. Euclidean Distance ($L_2$ Norm):**")
        st.latex(r"d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}")
        
        st.write("**2. Manhattan Distance ($L_1$ Norm):**")
        st.latex(r"d(\mathbf{p}, \mathbf{q}) = \sum_{i=1}^n |p_i - q_i|")
        
        st.write("When weights are distance-based, the weight $w_i$ of neighbor $i$ is defined as:")
        st.latex(r"w_i = \frac{1}{d(\mathbf{p}, \mathbf{x}_i) + \epsilon}")

    elif clf_name == "Naive Bayes":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is Naive Bayes?</p>
            <p class="feature-desc">
            Naive Bayes is a highly scalable <b>probabilistic</b> model. It applies <b>Bayes' Theorem</b> to estimate class probabilities, 
            assuming that all features are <i>conditionally independent</i> given the class label.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Bayes' Theorem Formulation")
        st.write("The posterior probability of class $C_k$ given features $\\mathbf{x}$ is:")
        st.latex(r"P(C_k \mid \mathbf{x}) = \frac{P(C_k) P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}")
        
        st.write("Applying the **conditional independence assumption**, the likelihood is factorized:")
        st.latex(r"P(\mathbf{x} \mid C_k) = P(x_1 \mid C_k) \times P(x_2 \mid C_k) \times \dots \times P(x_n \mid C_k) = \prod_{i=1}^n P(x_i \mid C_k)")
        
        st.write("In **Gaussian Naive Bayes**, continuous features are modeled using the normal distribution:")
        st.latex(r"P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)")

    elif clf_name == "Gradient Boosting":
        st.markdown("""
        <div class="feature-card" style="border-left-color: var(--cyan);">
            <p class="feature-title">What is Gradient Boosting?</p>
            <p class="feature-desc">
            Gradient Boosting is an <b>ensemble boosting</b> method that builds small decision trees sequentially. 
            Rather than independent bagging, each tree is trained to fit the <b>residuals (gradients of loss)</b> of the aggregate model.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("### 🧮 Boosting Formulation")
        st.write("The ensemble starts with a base prediction, typically the mean target:")
        st.latex(r"F_0(\mathbf{x}) = \arg\min_{\gamma} \sum_{i=1}^N \mathcal{L}(y_i, \gamma)")
        
        st.write("For each step $m \\ge 1$, we compute the pseudo-residuals (residuals / gradients):")
        st.latex(r"r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)} \right]")
        
        st.write("We train a tree $h_m(\\mathbf{x})$ to predict $r_{im}$, and add it to the ensemble scaled by learning rate $\\eta$:")
        st.latex(r"F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \sum_{j} \gamma_{jm} I(\mathbf{x} \in R_{jm})")

    # ── 3. GFG References Section ─────────────────────────────────────────────
    links = GFG_LINKS.get(clf_name, [])
    if links:
        st.markdown('<p class="section-header">📎 External Academic Resources</p>', unsafe_allow_html=True)
        st.markdown('<div class="link-grid">', unsafe_allow_html=True)
        for link in links:
            st.markdown(f"""
            <a class="gfg-link-card" href="{link['url']}" target="_blank">
                🔗 {link['label']}
            </a>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_quiz(algo_name: str):
    """Renders an interactive 3-question conceptual/math quiz for the selected algorithm."""
    import streamlit as st
    
    st.markdown('<p class="section-header">🧠 Brain Challenge Quiz</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(168, 85, 247, 0.05); border: 1px dashed rgba(168, 85, 247, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem;">
        <p style="color: var(--purple); font-family: 'Space Mono', monospace; font-size: 0.88rem; font-weight: 700; margin: 0;">
            ⚡ CHALLENGE YOUR UNDERSTANDING
        </p>
        <p style="color: var(--text-secondary); font-size: 0.83rem; margin: 0.2rem 0 0;">
            Test your knowledge of the mathematical and structural concepts behind <b>{algo_name}</b>. Correct answers receive instant detailed breakdown feedback!
        </p>
    </div>
    """.format(algo_name=algo_name), unsafe_allow_html=True)
    
    questions = QUIZZES.get(algo_name, [])
    if not questions:
        st.warning("Quiz questions are not yet defined for this algorithm.")
        return
        
    # Standardize session state keys
    score_key = f"quiz_score_{algo_name}"
    submitted_key = f"quiz_submitted_{algo_name}"
    
    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False
        
    user_choices = []
    
    for idx, q in enumerate(questions):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        choice = st.radio(
            "Select the correct option:",
            q["options"],
            key=f"q_{algo_name}_{idx}",
            index=None,
            label_visibility="collapsed"
        )
        user_choices.append(choice)
        st.markdown("<br>", unsafe_allow_html=True)
        
    # Form submission controls
    c_btn, c_score = st.columns([1, 2])
    with c_btn:
        submit = st.button("Submit Quiz", key=f"submit_btn_{algo_name}", disabled=st.session_state[submitted_key])
        
    if submit:
        # Check if all questions are answered
        if any(c is None for c in user_choices):
            st.error("⚠️ Please answer all questions before submitting!")
            return
            
        st.session_state[submitted_key] = True
        st.rerun()
        
    if st.session_state[submitted_key]:
        # Calculate score and show explanations
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
        
        # Display customized badge
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
        
        # Add a reset button
        if st.button("Retry Quiz", key=f"retry_btn_{algo_name}"):
            st.session_state[submitted_key] = False
            # Clear answers
            for idx in range(len(questions)):
                if f"q_{algo_name}_{idx}" in st.session_state:
                    del st.session_state[f"q_{algo_name}_{idx}"]
            st.rerun()
