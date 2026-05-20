from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

CLASSIFIERS = {
    "Logistic Regression":    LogisticRegression(max_iter=1000),
    "Decision Tree":          DecisionTreeClassifier(),
    "Random Forest":          RandomForestClassifier(n_estimators=100),
    "SVM (RBF Kernel)":       SVC(kernel='rbf', probability=True),
    "K-Nearest Neighbors":    KNeighborsClassifier(),
    "Naive Bayes":            GaussianNB(),
    "Gradient Boosting":      GradientBoostingClassifier(),
}

ALGO_INFO = {
    "Logistic Regression": {
        "type": "Linear", "complexity": "Low",
        "desc": "Models the probability of class membership using a sigmoid function. Best for linearly separable data.",
        "easy": "It draws a straight line (or flat surface) that best separates classes and predicts which side a point falls on.",
        "example": "Like deciding if a student passes or fails based on one exam score and attendance: higher values mean pass.",
        "pros": "Fast, interpretable, probabilistic output",
        "cons": "Cannot model non-linear boundaries",
    },
    "Decision Tree": {
        "type": "Tree", "complexity": "Medium",
        "desc": "Splits data recursively using feature thresholds to create a tree of decisions.",
        "easy": "It asks a series of yes/no questions, like a flowchart, until it reaches a prediction.",
        "example": "Like sorting fruit by asking: Is it round? Is it red? Is it soft? Then decide apple or orange.",
        "pros": "Highly interpretable, handles non-linear data",
        "cons": "Prone to overfitting on training data",
    },
    "Random Forest": {
        "type": "Ensemble", "complexity": "Medium-High",
        "desc": "Builds many decision trees and combines their votes — bagging reduces variance.",
        "easy": "It grows many small trees and then takes the majority vote, so one tree's mistake is less harmful.",
        "example": "Like asking several doctors for a diagnosis and choosing the most common answer.",
        "pros": "Robust, handles high dimensions, feature importance",
        "cons": "Less interpretable, slower than single trees",
    },
    "SVM (RBF Kernel)": {
        "type": "Kernel Method", "complexity": "High",
        "desc": "Finds the maximum-margin hyperplane; RBF kernel maps data into higher dimensions.",
        "easy": "It finds the widest possible gap between classes, even after bending the data into a new shape.",
        "example": "Like drawing a curved line on a map to separate two groups of houses as cleanly as possible.",
        "pros": "Powerful for non-linear data, effective in high dims",
        "cons": "Slow on large datasets, needs feature scaling",
    },
    "K-Nearest Neighbors": {
        "type": "Instance-Based", "complexity": "Low (train) / High (predict)",
        "desc": "Classifies based on the majority vote of the k closest training samples.",
        "easy": "It looks at the nearest neighbors and copies their label, like living in a neighborhood.",
        "example": "If most nearby restaurants are Italian, you guess the new restaurant is probably Italian too.",
        "pros": "No training phase, naturally multi-class",
        "cons": "Slow at prediction, sensitive to irrelevant features",
    },
    "Naive Bayes": {
        "type": "Probabilistic", "complexity": "Very Low",
        "desc": "Applies Bayes' theorem assuming features are conditionally independent.",
        "easy": "It combines simple chances for each feature and picks the label with the highest combined probability.",
        "example": "Like deciding if an email is spam by checking independent clues: sender, words used, and links.",
        "pros": "Extremely fast, works well with small data",
        "cons": "Independence assumption is often violated",
    },
    "Gradient Boosting": {
        "type": "Ensemble", "complexity": "High",
        "desc": "Sequentially builds trees where each one corrects the errors of the previous.",
        "easy": "It builds a series of models, each one learning from the mistakes of the last.",
        "example": "Like improving a recipe by tasting and fixing one problem at a time until it tastes right.",
        "pros": "State-of-the-art accuracy on tabular data",
        "cons": "Many hyperparameters, risk of overfitting",
    },
}

FEATURES = [
    ("🗺️ Decision Boundary Plot",
     "Visualizes how the algorithm divides the feature space into class regions. The colored background shows predicted zones; dots show actual data points. Compare train vs test to spot overfitting."),
    ("📊 Confusion Matrix",
     "A grid showing correct vs incorrect predictions for each class. Diagonal cells = correct; off-diagonal = errors. Essential for understanding where the model makes mistakes."),
    ("📈 ROC / AUC Curve",
     "Plots True Positive Rate vs False Positive Rate at every classification threshold. Area Under Curve (AUC) summarizes overall discriminative ability — 1.0 is perfect."),
    ("📋 Precision / Recall / F1",
     "Per-class breakdown of model quality. Precision = of all predicted positives, how many were real. Recall = of all real positives, how many were found. F1 balances both."),
    ("🎛️ Dataset Selection",
     "Choose from synthetic (Moons, Circles, Blobs) or real-world (Iris, Wine) datasets. Synthetic datasets let you control noise and visualize boundaries in 2D clearly."),
    ("⚙️ Algorithm Selector",
     "7 classifiers from linear to ensemble. Each has unique strengths — try all on the same dataset to compare decision boundaries and accuracy side-by-side."),
    ("🔊 Noise Control",
     "Inject noise into synthetic datasets to simulate messy real-world data. Watch how robust algorithms (like Random Forest) maintain accuracy vs fragile ones (like Decision Tree)."),
    ("📐 Train/Test Split",
     "Control what fraction of data is held out for evaluation. The split prevents data leakage — always evaluate on unseen data to get an honest accuracy estimate."),
]
