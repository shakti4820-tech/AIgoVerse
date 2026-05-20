import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression
import warnings
warnings.filterwarnings('ignore')

PALETTE = ['#00d4ff','#ff6b6b','#ffd166','#06d6a0','#a855f7']

REGR_INFO = {
    "Linear Regression": {
        "type":"Linear","complexity":"Very Low",
        "desc":"Fits a straight line (or hyperplane) through data by minimizing the sum of squared residuals.",
        "easy":"Draw the best-fit straight line through a cloud of points so the distances from points to line are smallest.",
        "example":"🏠 House Price Prediction: predict price from size (sq ft). Bigger house → higher price, linearly.",
        "pros":"Fast, highly interpretable, coefficients show exact feature weights.",
        "cons":"Fails completely on non-linear or curved patterns; sensitive to outliers.",
        "video": {"url": "https://www.youtube.com/embed/PaFPbb66DxQ", "title": "Linear Regression explained by StatQuest"},
        "gfg":[
            {"label":"Linear Regression in ML","url":"https://www.geeksforgeeks.org/ml-linear-regression/"},
            {"label":"Gradient Descent for LR","url":"https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/"},
        ]
    },
    "Ridge Regression": {
        "type":"Regularized","complexity":"Low",
        "desc":"Linear regression with L2 regularization — penalizes large coefficients to prevent overfitting.",
        "easy":"Same as linear regression but adds a penalty for big weights, so the model stays simpler.",
        "example":"📊 Predicting employee salary with many correlated features — Ridge prevents any one feature from dominating.",
        "pros":"Reduces overfitting; handles multicollinearity (correlated features) exceptionally well.",
        "cons":"Does not zero out coefficients; model remains dense and still linear.",
        "video": {"url": "https://www.youtube.com/embed/Q81RR3yKn30", "title": "Ridge Regression explained by StatQuest"},
        "gfg":[
            {"label":"Ridge Regression","url":"https://www.geeksforgeeks.org/ridge-regression/"},
            {"label":"L1 vs L2 Regularization","url":"https://www.geeksforgeeks.org/difference-between-l1-and-l2-regularization/"},
        ]
    },
    "Lasso Regression": {
        "type":"Regularized","complexity":"Low",
        "desc":"Linear regression with L1 regularization — can shrink some coefficients to exactly zero (feature selection).",
        "easy":"Like Ridge, but harsher — it completely eliminates unimportant features by zeroing their weights.",
        "example":"🧬 Gene expression: predicting disease severity from thousands of genes. Lasso selects only the relevant ones.",
        "pros":"Built-in feature selection; produces simple, interpretable sparse models.",
        "cons":"Can drop useful features if they are highly correlated with other selected features.",
        "video": {"url": "https://www.youtube.com/embed/NGf0jkz1hI8", "title": "Lasso Regression explained by StatQuest"},
        "gfg":[
            {"label":"Lasso Regression","url":"https://www.geeksforgeeks.org/lasso-regression/"},
            {"label":"Lasso vs Ridge","url":"https://www.geeksforgeeks.org/lasso-vs-ridge-vs-elastic-net-ml/"},
        ]
    },
    "Polynomial Regression": {
        "type":"Non-Linear","complexity":"Medium",
        "desc":"Extends linear regression by adding polynomial feature terms (x², x³…) to model curved relationships.",
        "easy":"Instead of a straight line, fit a curve. The degree controls how many bends the curve can have.",
        "example":"📈 Predicting crop yield vs rainfall: too little or too much water hurts crops — a U-shaped (degree 2) curve.",
        "pros":"Captures complex non-linear patterns; conceptually straightforward extension of LR.",
        "cons":"High degree polynomials lead to severe overfitting and extrapolate extremely poorly.",
        "video": {"url": "https://www.youtube.com/embed/PaFPbb66DxQ", "title": "Linear and Polynomial Regression by StatQuest"},
        "gfg":[
            {"label":"Polynomial Regression","url":"https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/"},
            {"label":"Overfitting in Polynomial Regression","url":"https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/"},
        ]
    },
    "SVR": {
        "type":"Kernel Method","complexity":"High",
        "desc":"Support Vector Regression finds a tube around the regression line where errors are ignored (epsilon-insensitive).",
        "easy":"Draw a tube around the predicted line. Points inside the tube are 'close enough'; only outliers matter.",
        "example":"📉 Stock price prediction with noisy data — SVR focuses on the general trend, ignoring small fluctuations.",
        "pros":"Robust to outliers; maps non-linear data into high dimensions using the kernel trick.",
        "cons":"Slow to train on large datasets; requires rigorous feature scaling.",
        "video": {"url": "https://www.youtube.com/embed/efR1C6CvhmE", "title": "Support Vector Regression / Machines by StatQuest"},
        "gfg":[
            {"label":"Support Vector Regression","url":"https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/"},
        ]
    },
    "Random Forest Regressor": {
        "type":"Ensemble","complexity":"Medium-High",
        "desc":"Builds many decision tree regressors and averages their predictions to reduce variance.",
        "easy":"Ask 100 different experts for a price estimate and average their answers — more reliable than one expert.",
        "example":"🚗 Used car price prediction: Forest considers brand, mileage, age, fuel type all at once.",
        "pros":"Handles complex non-linear relationships; very robust; yields feature importances.",
        "cons":"Slow to predict on large trees; predictions are step-like and less interpretable.",
        "video": {"url": "https://www.youtube.com/embed/J4Wdy0Wc_xQ", "title": "Random Forest explained by StatQuest"},
        "gfg":[
            {"label":"Random Forest Regression","url":"https://www.geeksforgeeks.org/random-forest-regression-in-python/"},
        ]
    },
    "Gradient Boosting Regressor": {
        "type":"Ensemble","complexity":"High",
        "desc":"Builds trees sequentially, each correcting the residual errors of the previous ensemble.",
        "easy":"Like editing a document — each round fixes the remaining errors until the prediction is precise.",
        "example":"🏆 Kaggle competitions: GBR (XGBoost/LightGBM) consistently wins on structured data like insurance claims.",
        "pros":"State-of-the-art predictive accuracy on structured / tabular datasets.",
        "cons":"Many hyperparameters to tune; high risk of overfitting if parameters are unconstrained.",
        "video": {"url": "https://www.youtube.com/embed/3CC4N4z3GJc", "title": "Gradient Boosting explained by StatQuest"},
        "gfg":[
            {"label":"Gradient Boosting Regression","url":"https://www.geeksforgeeks.org/gradientboosting-vs-adaboost-vs-xgboost-vs-catboost-vs-lightgbm/"},
            {"label":"XGBoost Tutorial","url":"https://www.geeksforgeeks.org/xgboost/"},
        ]
    },
}

QUIZZES = {
    "Linear Regression": [
        {
            "question": "What does the Ordinary Least Squares (OLS) optimization objective minimize?",
            "options": [
                "The sum of absolute residuals (MAE).",
                "The sum of squared residuals (SSR).",
                "The maximum residual value.",
                "The correlation coefficient between features."
            ],
            "correct": 1,
            "explanation": "OLS fits a linear model by minimizing the sum of squared differences (residuals) between observed targets and the predicted line: $SSR = \\sum (y_i - \\hat{y}_i)^2$."
        },
        {
            "question": "What is the physical meaning of an R-squared value of 0.85?",
            "options": [
                "85% of predictions are perfectly accurate.",
                "85% of the variance in the target variable is explained by the features.",
                "The slope of the regression line is 0.85.",
                "The mean absolute error is 0.85."
            ],
            "correct": 1,
            "explanation": "The Coefficient of Determination $R^2$ represents the proportion of variance in the dependent variable that is predictable from the independent variables."
        },
        {
            "question": "Why is standard Linear Regression highly sensitive to outliers?",
            "options": [
                "Because it ignores extreme values completely.",
                "Because squaring the residuals in the OLS loss function disproportionately penalizes large errors, pulling the line toward outliers.",
                "Because it uses a log scale internally.",
                "Because linear regression requires variables to be integers."
            ],
            "correct": 1,
            "explanation": "Because OLS squares the residuals, a large outlier creates an extremely large squared error. The algorithm will warp the fit line to reduce this massive single penalty."
        }
    ],
    "Ridge Regression": [
        {
            "question": "What type of penalty does Ridge Regression apply to prevent overfitting?",
            "options": [
                "L1 penalty (absolute sum of coefficients).",
                "L2 penalty (squared sum of coefficients).",
                "Elastic Net mixing penalty.",
                "Logarithmic constraint on training iterations."
            ],
            "correct": 1,
            "explanation": "Ridge adds an $L_2$ regularization penalty: $\\alpha \\sum_{j=1}^d w_j^2$ to the OLS cost function, penalizing large weights."
        },
        {
            "question": "How does increasing the regularization parameter alpha affect Ridge Regression?",
            "options": [
                "Decreases bias and increases variance (overfitting).",
                "Increases bias and reduces variance (shrinks weights, preventing overfitting).",
                "Has no effect on weight coefficients.",
                "Forces all coefficients to become exactly zero."
            ],
            "correct": 1,
            "explanation": "Higher alpha places a heavier penalty on coefficient sizes, shrinking weights closer to zero. This increases bias but reduces variance, which generalizes better on noisy data."
        },
        {
            "question": "Does Ridge Regression perform automatic feature selection by zeroing out coefficients?",
            "options": [
                "Yes, it zeros out unimportant features completely.",
                "No, it shrinks coefficients asymptotically toward zero but they remain non-zero.",
                "Yes, but only for degree 1 variables.",
                "Yes, if alpha is set to exactly 10.0."
            ],
            "correct": 1,
            "explanation": "Unlike Lasso, the quadratic $L_2$ penalty does not force coefficients to exactly zero. All features are retained in the model, just with shrunken weights."
        }
    ],
    "Lasso Regression": [
        {
            "question": "What unique property does Lasso regression possess due to its L1 regularization?",
            "options": [
                "It restricts the model to only fit positive slopes.",
                "It shrinks coefficients of unimportant features to exactly zero, performing automatic feature selection.",
                "It increases training speed by a factor of 10.",
                "It makes the regression surface step-like."
            ],
            "correct": 1,
            "explanation": "Lasso adds an $L_1$ penalty: $\\alpha \\sum |w_j|$. Geometrically, the constraint region has sharp corners that intersect the axes, forcing some weight coefficients to be exactly zero."
        },
        {
            "question": "If two features are highly correlated, how does Lasso Regression generally behave?",
            "options": [
                "It retains both features and distributes weights equally.",
                "It selects one feature arbitrarily and drops the other (zeros its coefficient).",
                "It increases both coefficients to infinity.",
                "It converts both features into polynomials."
            ],
            "correct": 1,
            "explanation": "Lasso tends to select only one of the highly correlated variables and shrink the other to exactly zero. Ridge split-shares the weights among correlated features."
        },
        {
            "question": "What is the penalty term minimized in Lasso Regression?",
            "options": [
                "Square root of coefficients.",
                "Sum of squared weights (L2).",
                "Sum of absolute weights (L1).",
                "Log-likelihood ratio."
            ],
            "correct": 2,
            "explanation": "Lasso minimizes the OLS loss plus the sum of absolute values of the coefficients: $\\alpha \\sum_{j=1}^d |w_j|$."
        }
    ],
    "Polynomial Regression": [
        {
            "question": "How does Polynomial Regression model non-linear curved data using standard linear regression algorithms?",
            "options": [
                "By using logarithmic cost functions in gradient descent.",
                "By transforming the original features into higher-order powers (x², x³...) and fitting a linear hyperplane in that expanded feature space.",
                "By linking decision trees sequentially.",
                "By enforcing a soft-margin probability."
            ],
            "correct": 1,
            "explanation": "Polynomial regression transforms input feature $x$ into $[1, x, x^2, \dots, x^d]^T$. A standard linear model fits weights to these new features, creating a curved fit in the original space."
        },
        {
            "question": "What is the primary risk of using a very high polynomial degree (e.g. Degree 6)?",
            "options": [
                "Severe underfitting.",
                "Severe overfitting, where the model wiggles excessively to capture random noise in the training set.",
                "The model will become strictly linear.",
                "It prevents any predictions outside 0 and 1."
            ],
            "correct": 1,
            "explanation": "A high-degree polynomial has excessive flexibility. It will pass perfectly through training points by wiggling wildly, fitting the noise and failing catastrophically on test data."
        },
        {
            "question": "What is a major limitation when extrapolating predictions with Polynomial Regression?",
            "options": [
                "It always yields zero.",
                "High-degree terms diverge rapidly toward infinity outside the training range, making predictions extremely erratic and unreliable.",
                "It only works inside a unit circle.",
                "It scales all values to mean values."
            ],
            "correct": 1,
            "explanation": "Polynomials diverge rapidly outside the range of the training data. Predicting values far outside the training boundaries leads to extreme values due to the dominating high-degree terms."
        }
    ],
    "SVR": [
        {
            "question": "What is the role of the epsilon (ε) parameter in Support Vector Regression?",
            "options": [
                "The maximum iteration count.",
                "The radius of the insensitive tube around the fit where no penalty is applied to errors.",
                "The cost weight of support vectors.",
                "The polynomial degree of kernels."
            ],
            "correct": 1,
            "explanation": "Epsilon $\\epsilon$ defines a tube of insensitivity. Any training point lying within a distance of $\\epsilon$ from the predicted line incurs zero penalty."
        },
        {
            "question": "How does the C parameter in SVR control the fitted boundary?",
            "options": [
                "It changes the kernel type.",
                "It controls the trade-off between model smoothness and penalizing training points that lie outside the epsilon-insensitive tube.",
                "It controls the learning rate of boosting.",
                "It defines the maximum number of support vectors."
            ],
            "correct": 1,
            "explanation": "The parameter $C$ dictates the penalty for predictions that fall outside the $\\epsilon$-insensitive boundary. A higher $C$ forces a tighter, less regularized fit to avoid training errors."
        },
        {
            "question": "How does SVR fit complex non-linear curves?",
            "options": [
                "By using the kernel trick (e.g. RBF kernel) to implicitly fit data in high-dimensional feature spaces.",
                "By running bagging across hundreds of trees.",
                "By performing Lasso feature selection.",
                "By computing prior probabilities using Bayes' Theorem."
            ],
            "correct": 0,
            "explanation": "SVR uses the kernel trick (e.g., Radial Basis Function) to project data into a higher-dimensional space where a linear regression plane is fit, which manifests as a non-linear curve in the original space."
        }
    ],
    "Random Forest Regressor": [
        {
            "question": "How does a Random Forest Regressor aggregate individual tree predictions?",
            "options": [
                "By majority voting.",
                "By taking the mathematical average of the predictions of all individual trees.",
                "By selecting the single tree with the lowest depth.",
                "By taking the product of all predictions."
            ],
            "correct": 1,
            "explanation": "While classification uses majority voting, Random Forest Regression aggregates individual tree predictions by taking their average."
        },
        {
            "question": "Why are Random Forest predictions step-like rather than perfectly smooth curves?",
            "options": [
                "Because it is a linear model.",
                "Because decision trees split the feature space using orthogonal cuts and predict a constant value (mean) within each leaf.",
                "Because of the kernel trick.",
                "Because it uses Lasso regularization."
            ],
            "correct": 1,
            "explanation": "Each decision tree partitions the dataset into rectangular leaf regions and outputs a constant value (mean of target values) in that region. A forest averaging these steps still results in a stair-step prediction boundary."
        },
        {
            "question": "Does increasing the number of trees (estimators) in a Random Forest lead to overfitting?",
            "options": [
                "Yes, the model becomes too complex.",
                "No, bagging averages independent models; adding more trees stabilizes the variance and does not cause overfitting.",
                "Yes, because trees correlate.",
                "No, but it forces the model to be linear."
            ],
            "correct": 1,
            "explanation": "In bagging, trees are trained in parallel. Increasing the tree count reduces ensemble variance without increasing bias, so the model does not overfit as tree count increases."
        }
    ],
    "Gradient Boosting Regressor": [
        {
            "question": "How does Gradient Boosting build trees compared to Random Forest?",
            "options": [
                "Trees are built completely independently in parallel.",
                "Trees are built sequentially, with each tree trained to predict the residual errors of the current ensemble.",
                "Trees are aggregated using majority voting.",
                "It uses bagging with replacement."
            ],
            "correct": 1,
            "explanation": "Gradient Boosting builds trees sequentially. Each tree is trained to fit the residuals (errors) of the aggregate model, moving closer to the true targets step-by-step."
        },
        {
            "question": "What is the role of the learning rate in Gradient Boosting?",
            "options": [
                "It speeds up the execution time.",
                "It scales the contribution of each tree (shrinkage) to prevent overfitting, requiring more trees for optimal convergence.",
                "It selects the best features automatically.",
                "It sets the maximum tree depth."
            ],
            "correct": 1,
            "explanation": "The learning rate scales the step size of gradient updates. Smaller rates shrink each tree's contribution, which prevents overfitting and improves generalization, but requires more trees."
        },
        {
            "question": "What targets do individual decision trees fit during each stage of Gradient Boosting?",
            "options": [
                "The original target labels $y$.",
                "The residuals (negative gradients of the loss function).",
                "The probability distributions.",
                "The covariance matrix values."
            ],
            "correct": 1,
            "explanation": "At each step, Gradient Boosting fits a weak learner (tree) to predict the pseudo-residuals, representing the errors of the current ensemble."
        }
    ]
}

def set_style(fig, axes):
    fig.patch.set_facecolor('#0a0e1a')
    for ax in (axes if hasattr(axes,'__iter__') else [axes]):
        ax.set_facecolor('#0d1522')
        ax.tick_params(colors='#6b7a8d', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#1e3a5f')
        ax.xaxis.label.set_color('#8899aa')
        ax.yaxis.label.set_color('#8899aa')
        ax.title.set_color('#00d4ff')

def card(title, body, color="#00d4ff"):
    return f"""<div class="feature-card" style="border-left-color: {color};">
<p class="feature-title" style="color:{color};">{title}</p>
<p class="feature-desc">{body}</p></div>"""

def make_reg_dataset(n, noise, ds_type):
    np.random.seed(42)
    # Generate 2 input features (X_1, X_2) for standard 3D regression space
    x1 = np.random.uniform(-3, 3, n)
    x2 = np.random.uniform(-3, 3, n)
    X = np.column_stack([x1, x2])
    
    if ds_type == "Linear":
        y = 1.5 * x1 - 1.0 * x2 + np.random.normal(0, noise * 5, n)
    elif ds_type == "Sinusoidal":
        r = np.sqrt(x1**2 + x2**2)
        y = 4.0 * np.sin(r * 1.5) + np.random.normal(0, noise * 1.0, n)
    elif ds_type == "Quadratic":
        y = 0.8 * (x1**2 + x2**2) - 3.0 + np.random.normal(0, noise * 2.0, n)
    else:  # Exponential
        y = np.exp(0.4 * x1) + np.exp(0.2 * x2) + np.random.normal(0, noise * 1.0, n)
        
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

def get_model_instance(algo):
    """Instantiate the regressor with current selected sidebar hyperparameters."""
    if algo == "Linear Regression":
        return LinearRegression()
    elif algo == "Ridge Regression":
        return Ridge(alpha=st.session_state.get("rg_ridge_alpha", 1.0))
    elif algo == "Lasso Regression":
        return Lasso(alpha=st.session_state.get("rg_lasso_alpha", 0.05), max_iter=5000)
    elif algo == "Polynomial Regression":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=st.session_state.get("rg_poly_deg", 2))),
            ("lr", LinearRegression())
        ])
    elif algo == "SVR":
        return Pipeline([
            ("sc", StandardScaler()),
            ("svr", SVR(
                kernel=st.session_state.get("rg_svr_kernel", "rbf"),
                C=st.session_state.get("rg_svr_c", 10.0),
                epsilon=st.session_state.get("rg_svr_eps", 0.1)
            ))
        ])
    elif algo == "Random Forest Regressor":
        return RandomForestRegressor(
            n_estimators=st.session_state.get("rg_rf_est", 100),
            max_depth=st.session_state.get("rg_rf_depth", 6),
            random_state=42
        )
    elif algo == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(
            n_estimators=st.session_state.get("rg_gb_est", 100),
            learning_rate=st.session_state.get("rg_gb_lr", 0.1),
            max_depth=st.session_state.get("rg_gb_depth", 3),
            random_state=42
        )

def render_regression_quiz(algo_name: str):
    """Renders the 3-question conceptual quiz for regression."""
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

    score_key = f"quiz_score_rg_{algo_name}"
    submitted_key = f"quiz_submitted_rg_{algo_name}"

    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False

    user_choices = []

    for idx, q in enumerate(questions):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        choice = st.radio(
            "Select the correct option:",
            q["options"],
            key=f"q_rg_{algo_name}_{idx}",
            index=None,
            label_visibility="collapsed"
        )
        user_choices.append(choice)
        st.markdown("<br>", unsafe_allow_html=True)

    c_btn, c_score = st.columns([1, 2])
    with c_btn:
        submit = st.button("Submit Quiz", key=f"submit_btn_rg_{algo_name}", disabled=st.session_state[submitted_key])

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

        if st.button("Retry Quiz", key=f"retry_btn_rg_{algo_name}"):
            st.session_state[submitted_key] = False
            for idx in range(len(questions)):
                if f"q_rg_{algo_name}_{idx}" in st.session_state:
                    del st.session_state[f"q_rg_{algo_name}_{idx}"]
            st.rerun()

def render_regression_page():
    with st.sidebar:
        st.markdown("---")
        st.markdown("**📈 REGRESSION**")
        algo  = st.selectbox("Algorithm", list(REGR_INFO.keys()), key="rg_algo")
        ds    = st.selectbox("Dataset", ["Linear","Sinusoidal","Quadratic","Exponential"], key="rg_ds")
        n     = st.slider("Samples", 50, 1000, 300, 50, key="rg_n")
        noise = st.slider("Noise", 0.0, 1.0, 0.2, 0.05, key="rg_noise")
        ts    = st.slider("Test Size %", 10, 40, 20, 5, key="rg_ts")

        st.markdown("---")
        st.markdown("**🎛️ HYPERPARAMETERS**")
        if algo == "Linear Regression":
            st.markdown("*No hyperparameters available.*")
        elif algo == "Ridge Regression":
            st.slider("Alpha (Regularization Strength)", 0.001, 50.0, 1.0, 0.1, key="rg_ridge_alpha")
        elif algo == "Lasso Regression":
            st.slider("Alpha (Regularization Strength)", 0.0001, 5.0, 0.05, 0.001, format="%.4f", key="rg_lasso_alpha")
        elif algo == "Polynomial Regression":
            st.slider("Polynomial Degree", 1, 6, 2, key="rg_poly_deg")
        elif algo == "SVR":
            st.slider("Cost (C)", 0.1, 100.0, 10.0, 0.5, key="rg_svr_c")
            st.slider("Epsilon Tube (ε)", 0.01, 0.5, 0.1, 0.01, key="rg_svr_eps")
            st.selectbox("Kernel type", ["rbf", "linear", "poly"], key="rg_svr_kernel")
        elif algo == "Random Forest Regressor":
            st.slider("Estimators (Trees count)", 10, 200, 100, 10, key="rg_rf_est")
            st.slider("Max Tree Depth", 1, 15, 6, key="rg_rf_depth")
        elif algo == "Gradient Boosting Regressor":
            st.slider("Estimators (Boosting rounds)", 10, 150, 100, 10, key="rg_gb_est")
            st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="rg_gb_lr")
            st.slider("Max Tree Depth", 1, 10, 3, key="rg_gb_depth")

    X, y = make_reg_dataset(n, noise, ds)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=ts/100, random_state=42)

    # Instantiate model with dynamic values configured
    model = get_model_instance(algo)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    info = REGR_INFO[algo]

    st.markdown(f"""<div class="hero-banner">
<p class="hero-title">📈 Regression Lab</p>
<p class="hero-subtitle">Supervised learning · Predict continuous values · Evaluate model fit<br>
<span style="color:#00d4ff88;">Algorithm: {algo} &nbsp;|&nbsp; Dataset: {ds}</span></p></div>""",
    unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("R² Score",  f"{r2:.4f}")
    with c2: st.metric("RMSE",      f"{rmse:.3f}")
    with c3: st.metric("MAE",       f"{mae:.3f}")
    with c4: st.metric("Test Size", len(y_te))

    st.markdown("<br>", unsafe_allow_html=True)
    t1,t2,t3,t4,t5 = st.tabs(["📉 Fit Plot", "📊 Residuals", "🧊 3D Surface", "📚 Theory Notes", "🧠 Brain Challenge"])

    # ── Tab 1: Fit Plot ───────────────────────────────────────────────────────
    with t1:
        ca, cb = st.columns([2,1])
        with ca:
            y_tr_pred = model.predict(X_tr)
            y_te_pred = y_pred
            
            fig, ax = plt.subplots(figsize=(8,5))
            set_style(fig,[ax])
            
            ax.scatter(y_tr_pred, y_tr, c='#00d4ff88', edgecolors='#00d4ff', s=30, label='Train Set', linewidths=0.5)
            ax.scatter(y_te_pred, y_te, c='#ffd16688', edgecolors='#ffd166', s=30, label='Test Set', linewidths=0.5)
            
            # Diagonal line representing a perfect fit (y = x)
            min_val = min(y_tr.min(), y_te.min(), y_tr_pred.min(), y_te_pred.min()) - 0.5
            max_val = max(y_tr.max(), y_te.max(), y_tr_pred.max(), y_te_pred.max()) + 0.5
            ax.plot([min_val, max_val], [min_val, max_val], color='#ff6b6b', lw=2, linestyle='--', label='Perfect Fit (ŷ = y)')
            
            ax.set_xlabel("Predicted Target (ŷ)")
            ax.set_ylabel("Actual Target (y)")
            ax.set_title(f"Actual vs. Predicted — {algo}")
            ax.legend(facecolor='#0d1522', edgecolor='#1e3a5f', labelcolor='#c8d8e8', fontsize=9)
            st.pyplot(fig, use_container_width=True)
        with cb:
            st.markdown(card(algo, info["desc"]), unsafe_allow_html=True)
            st.markdown(card("💡 Intuition", info["easy"], "var(--gold)"), unsafe_allow_html=True)
            st.markdown(card("🌍 Example", info["example"], "var(--purple)"), unsafe_allow_html=True)

    # ── Tab 2: Residuals ──────────────────────────────────────────────────────
    with t2:
        residuals = y_te - y_pred
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        set_style(fig, axes)
        axes[0].scatter(y_pred, residuals, c='#00d4ff88', edgecolors='#00d4ff', s=30, linewidths=0.5)
        axes[0].axhline(0, color='#ff6b6b', lw=1.5, linestyle='--')
        axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual"); axes[0].set_title("Residuals vs Predicted")
        axes[1].hist(residuals, bins=25, color='#00d4ff55', edgecolor='#00d4ff', linewidth=0.8)
        axes[1].set_xlabel("Residual"); axes[1].set_ylabel("Frequency"); axes[1].set_title("Residual Distribution")
        plt.tight_layout(pad=2)
        st.pyplot(fig, use_container_width=True)
        st.info("📌 Ideal residuals are **randomly scattered around 0** with no pattern. A pattern suggests the model is missing structure in the data.")

    # ── Tab 3: 3D Surface ─────────────────────────────────────────────────────
    with t3:
        st.markdown('<p style="color: var(--cyan); font-family: \'Space Mono\', monospace; font-size: 0.95rem; font-weight: 700; margin: 0 0 1rem 0;">🛠️ Customize 3D Perspectives</p>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            rg_3d_color = st.selectbox("Marker Color Perspective", ["Actual target (y)", "Predicted target (y_pred)", "Absolute Error (Residuals)"], index=0, key="rg_3d_col")
        with col_c2:
            rg_3d_scale = st.selectbox("Surface Color Scheme", ["Viridis", "Plasma", "Turbo", "Electric", "Thermal"], index=0, key="rg_3d_scale")
            
        feature_options = ["Feature 1", "Feature 2"]
        col_x, col_y, col_z = st.columns([1, 1, 1])
        with col_x:
            rg_3d_x = st.selectbox("X-Axis Coordinate", feature_options, index=0, key="rg_3d_x")
            x_idx = feature_options.index(rg_3d_x)
        with col_y:
            rg_3d_y = st.selectbox("Y-Axis Coordinate", feature_options, index=1, key="rg_3d_y")
            y_idx = feature_options.index(rg_3d_y)
        with col_z:
            rg_3d_z = st.selectbox("Z-Axis Coordinate", ["Target y", "Predicted y_pred", "Absolute Residual"], index=0, key="rg_3d_z")
            
        # Determine continuous marker coloring values
        y_pred_all = model.predict(X)
        if rg_3d_color == "Actual target (y)":
            marker_colors = y
            colorscale_markers = 'Plasma'
        elif rg_3d_color == "Predicted target (y_pred)":
            marker_colors = y_pred_all
            colorscale_markers = 'Cividis'
        else:
            marker_colors = np.abs(y - y_pred_all)
            colorscale_markers = 'Hot'
            
        # Determine actual Z coordinates of the scatter points
        if rg_3d_z == "Target y":
            z_scatter = y
            z_axis_lbl = "Target (y)"
        elif rg_3d_z == "Predicted y_pred":
            z_scatter = y_pred_all
            z_axis_lbl = "Predicted (y_pred)"
        else:
            z_scatter = np.abs(y - y_pred_all)
            z_axis_lbl = "Absolute Error (Residual)"
            
        # Grid generation for the fitted surface using the remapped axes
        grid_res = 40
        x1_range = np.linspace(X[:, x_idx].min(), X[:, x_idx].max(), grid_res)
        x2_range = np.linspace(X[:, y_idx].min(), X[:, y_idx].max(), grid_res)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        
        # Reconstruct grid for model prediction matching feature names order
        if x_idx == 0:
            X_grid = np.column_stack([x1_grid.ravel(), x2_grid.ravel()])
        else:
            X_grid = np.column_stack([x2_grid.ravel(), x1_grid.ravel()])
            
        y_grid_pred = model.predict(X_grid).reshape(grid_res, grid_res)
        
        # Map surface Z values
        if rg_3d_z in ["Target y", "Predicted y_pred"]:
            z_surface = y_grid_pred
        else:
            # If Z axis represents the Error, perfect prediction floor is at 0
            z_surface = np.zeros(y_grid_pred.shape)
            
        # Render beautiful Plotly 3D graph
        fig3 = go.Figure()
        
        # Add actual scattered points
        fig3.add_trace(go.Scatter3d(
            x=X[:, x_idx], y=X[:, y_idx], z=z_scatter,
            mode='markers',
            marker=dict(size=4.5, color=marker_colors, colorscale=colorscale_markers, opacity=0.85, line=dict(color='#030712', width=0.5)),
            name='Actual Points'
        ))
        
        # Add the fitted surface boundary plane
        fig3.add_trace(go.Surface(
            x=x1_range, y=x2_range, z=z_surface,
            opacity=0.65,
            colorscale=rg_3d_scale,
            showscale=False,
            name='Regression Surface'
        ))
        
        fig3.update_layout(
            title=f"3D Fitted Regression Plane — {algo}",
            paper_bgcolor='#0a0e1a',
            font_color='#c8d8e8',
            margin=dict(l=0,r=0,t=40,b=0),
            height=580,
            legend=dict(x=0.02, y=0.98)
        )
        fig3.update_scenes(
            xaxis=dict(title=rg_3d_x, backgroundcolor='#0d1522', gridcolor='#1e3a5f', showbackground=True),
            yaxis=dict(title=rg_3d_y, backgroundcolor='#0d1522', gridcolor='#1e3a5f', showbackground=True),
            zaxis=dict(title=z_axis_lbl, backgroundcolor='#0d1522', gridcolor='#1e3a5f', showbackground=True)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        st.info("💡 **Interactive 3D View:** Click and drag to rotate the 3D space! You can zoom in/out to physically inspect the exact geometric shape (saddle, flat plane, or step cuts) of the fitted model.")

    # ── Tab 4: Theory Notes ───────────────────────────────────────────────────
    with t4:
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

        if algo == "Linear Regression":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Linear Regression?</p>
                <p class="feature-desc">
                Linear Regression is the bedrock of supervised machine learning. It models a continuous target variable $y$ 
                as a linear combination of independent input features $X$.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Mathematical Formulation")
            st.write("For a single point with features $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$, we compute:")
            st.latex(r"\hat{y} = \mathbf{w}^T \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b")

            st.write("During training, the **Ordinary Least Squares (OLS)** method minimizes the **Sum of Squared Residuals (SSR / Residual Sum of Squares)**:")
            st.latex(r"SSR(\mathbf{w}, b) = \sum_{i=1}^N \left( y^{(i)} - \hat{y}^{(i)} \right)^2 = \sum_{i=1}^N \left( y^{(i)} - (\mathbf{w}^T \mathbf{x}^{(i)} + b) \right)^2")

            st.write("### 📊 Metrics of Performance")
            st.write("The fit is measured using the **Coefficient of Determination ($R^2$ Score)**:")
            st.latex(r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}")

        elif algo == "Ridge Regression":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Ridge Regression?</p>
                <p class="feature-desc">
                Ridge Regression is a **regularized** linear regression that applies an **$L_2$ Tikhonov constraint** to the loss function. 
                It penalizes large weight magnitudes to prevent overfitting and handle highly correlated features.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Mathematical Objective")
            st.write("Ridge minimizes the standard Ordinary Least Squares loss plus a squared sum penalty of the weight coefficients:")
            st.latex(r"\mathcal{L}_{\text{Ridge}}(\mathbf{w}, b) = \sum_{i=1}^N \left( y^{(i)} - (\mathbf{w}^T \mathbf{x}^{(i)} + b) \right)^2 + \alpha \|\mathbf{w}\|_2^2")
            st.write("where $\|\mathbf{w}\|_2^2$ is the squared Euclidean norm of weights:")
            st.latex(r"\|\mathbf{w}\|_2^2 = \sum_{j=1}^d w_j^2")
            st.write("Here, $\\alpha \\ge 0$ is the regularization strength. A larger $\\alpha$ forces weights to become smaller, reducing variance and stabilizing the model against noisy parameters.")

        elif algo == "Lasso Regression":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Lasso Regression?</p>
                <p class="feature-desc">
                <b>Lasso</b> (Least Absolute Shrinkage and Selection Operator) is a regularized linear model that applies an **$L_1$ constraint**. 
                Unlike Ridge, Lasso shrinks the coefficients of unimportant features to **exactly zero**, performing automated feature selection.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Mathematical Objective")
            st.write("Lasso minimizes the OLS loss plus an absolute sum penalty of the weights:")
            st.latex(r"\mathcal{L}_{\text{Lasso}}(\mathbf{w}, b) = \sum_{i=1}^N \left( y^{(i)} - (\mathbf{w}^T \mathbf{x}^{(i)} + b) \right)^2 + \alpha \|\mathbf{w}\|_1")
            st.write("where $\|\mathbf{w}\|_1$ is the absolute sum norm of weight coefficients:")
            st.latex(r"\|\mathbf{w}\|_1 = \sum_{j=1}^d |w_j|")
            st.write("Because the L1 norm constraint forms a diamond shape with sharp vertices, optimization solutions tend to land on the axes where secondary feature coefficients are driven to exactly $0$.")

        elif algo == "Polynomial Regression":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Polynomial Regression?</p>
                <p class="feature-desc">
                Polynomial Regression maps input features into higher-degree polynomial dimensions, allowing a standard linear regressor 
                to fit curves and capture non-linear relationships.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Feature Space Transformation")
            st.write("For a single 1D feature $x$, we map it into a $d$-degree polynomial vector:")
            st.latex(r"\mathbf{\phi}(x) = [1, x, x^2, x^3, \dots, x^d]^T")
            st.write("The linear model is then fit to the transformed vector:")
            st.latex(r"\hat{y} = w_0 + w_1 x + w_2 x^2 + \dots + w_d x^d")
            st.write("Although the boundary shape is non-linear relative to $x$, the optimization remains **linear relative to the weights $w_j$**, so it can be solved quickly using standard OLS.")

        elif algo == "SVR":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Support Vector Regression?</p>
                <p class="feature-desc">
                SVR is the regression version of Support Vector Machines. 
                Instead of minimizing squared residuals, it ignores errors that fall within a defined boundary tube of size **Epsilon** ($\\epsilon$), ignoring noise inside this region.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Epsilon-Insensitive Loss & Margin Optimization")
            st.write("The loss ignores errors smaller than $\\epsilon$:")
            st.latex(r"\mathcal{L}_\epsilon(y, \hat{y}) = \max(0, |y - \hat{y}| - \epsilon)")
            st.write("The SVR objective minimizes coefficients while penalizing training points outside the tube using slack variables $\\xi_i, \\xi_i^*$:")
            st.latex(r"\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^N \left( \xi_i + \xi_i^* \right)")
            st.write("subject to:")
            st.latex(r"y^{(i)} - (\mathbf{w}^T \mathbf{x}^{(i)} + b) \le \epsilon + \xi_i \quad , \quad (\mathbf{w}^T \mathbf{x}^{(i)} + b) - y^{(i)} \le \epsilon + \xi_i^*")

        elif algo == "Random Forest Regressor":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Random Forest Regression?</p>
                <p class="feature-desc">
                Random Forest Regressor is an **ensemble bagging** method. It constructs many independent decision tree regressors in parallel 
                using random bootstrap datasets and averages their predicted values to yield a final score.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Ensemble Averaging")
            st.write("For $B$ bootstrapped decision trees, the final continuous prediction is:")
            st.latex(r"\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})")
            st.write("By averaging independent high-variance trees, Random Forest drastically reduces variance, offering a robust model that generalizes well without overfitting.")

        elif algo == "Gradient Boosting Regressor":
            st.markdown("""
            <div class="feature-card" style="border-left-color: var(--cyan);">
                <p class="feature-title">What is Gradient Boosting Regression?</p>
                <p class="feature-desc">
                Gradient Boosting Regressor builds small decision trees **sequentially**. 
                Instead of parallel trees, each new tree is trained to predict the **residuals (errors)** of the current ensemble, gradually refining the overall fit.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("### 🧮 Sequential Boosting Core")
            st.write("The model starts with a simple baseline prediction (usually the target mean):")
            st.latex(r"F_0(\mathbf{x}) = \arg\min_{\gamma} \sum_{i=1}^N \mathcal{L}(y_i, \gamma)")
            st.write("For each sequential iteration $m$, pseudo-residuals (negative loss gradients) are computed:")
            st.latex(r"r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)} \right]")
            st.write("We train a tree $h_m(\\mathbf{x})$ to predict $r_{im}$, and add its shrunken contribution to the ensemble:")
            st.latex(r"F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})")
            st.write("where $\\eta$ is the learning rate.")

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

    # ── Tab 5: Brain Challenge Quiz ───────────────────────────────────────────
    with t5:
        render_regression_quiz(algo)
