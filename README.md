# 🌌 AlgoVerse — A Universe of Machine Learning Algorithms

> **An interactive, visual ML education platform built with Streamlit.**  
> Explore Classification, Clustering & Regression — with theory, 3D visualizations, and GeeksForGeeks-style notes.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📸 Overview

AlgoVerse is a fully interactive Machine Learning dashboard that lets you:

- 🎛️ **Tune hyperparameters** with live sliders
- 📊 **Visualize** decision boundaries, cluster maps, regression fits
- 🧊 **Explore data** in interactive 3D scatter plots
- 📚 **Read theory notes** written in simple English (GFG style)
- 🔗 **Access curated GeeksForGeeks links** for deeper learning

---

## 🧩 Sections

### 📊 Classification (7 Algorithms)
| Algorithm | Type |
|---|---|
| Logistic Regression | Linear |
| Decision Tree | Tree-Based |
| Random Forest | Ensemble |
| SVM (RBF Kernel) | Kernel Method |
| K-Nearest Neighbors | Instance-Based |
| Naive Bayes | Probabilistic |
| Gradient Boosting | Ensemble |

**Features:** Decision boundary plots · Confusion matrix · ROC/AUC curve · Precision/Recall/F1 · Theory notes · GFG links

---

### 🔵 Clustering (4 Algorithms)
| Algorithm | Type |
|---|---|
| K-Means | Centroid-Based |
| DBSCAN | Density-Based |
| Hierarchical | Agglomerative |
| Gaussian Mixture Model | Probabilistic |

**Features:** Cluster scatter plots · 3D visualization · Elbow curve · Silhouette score · Theory notes · GFG links

---

### 📈 Regression (7 Algorithms)
| Algorithm | Type |
|---|---|
| Linear Regression | Linear |
| Ridge Regression | Regularized |
| Lasso Regression | Regularized |
| Polynomial Regression | Non-Linear |
| SVR | Kernel Method |
| Random Forest Regressor | Ensemble |
| Gradient Boosting Regressor | Ensemble |

**Features:** Fit plots · Residual analysis · 3D surface view · R² / RMSE / MAE metrics · Theory notes · GFG links

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AlgoVerse.git
cd AlgoVerse
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run AlgoVerse.py
```

Open your browser at **`http://localhost:8501`**

---

## 📁 Project Structure

```
AlgoVerse/
│
├── AlgoVerse.py          # 🌌 Main app — entry point
├── Shakti.py             # Classification-only standalone app
├── clustering_page.py    # 🔵 Clustering section
├── regression_page.py    # 📈 Regression section
├── algos.py              # Algorithm configs & metadata
├── data_loader.py        # Dataset loader
├── plot_utils.py         # Matplotlib & Plotly utilities
├── style.py              # CSS theme & styles
├── theory.py             # Theory notes + GFG links
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 📦 Dependencies

```
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
plotly>=5.18.0
scipy>=1.11.0
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to **GitHub** (must be public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **"New app"** → Select this repo
5. Set **Main file:** `AlgoVerse.py`
6. Click **Deploy!** 🎉

---

## 🎯 Features at a Glance

| Feature | Description |
|---|---|
| 🎛️ Live Controls | Sliders for noise, sample size, hyperparameters |
| 🗺️ Decision Boundaries | See how each classifier divides feature space |
| 🧊 3D Visualization | Rotate & explore data in 3D with Plotly |
| 📖 Theory Notes | GFG-style explanations with math & real examples |
| 🔗 GFG Links | Curated external reading for every algorithm |
| 📊 Full Metrics | Accuracy, R², RMSE, Silhouette, ROC, F1 and more |

---

## 👨‍💻 Built With

- [Streamlit](https://streamlit.io) — Web framework
- [Scikit-Learn](https://scikit-learn.org) — ML algorithms
- [Plotly](https://plotly.com) — Interactive 3D charts
- [Matplotlib](https://matplotlib.org) — Static visualizations
- [SciPy](https://scipy.org) — Hierarchical clustering

---

## 📄 License

This project is licensed under the **MIT License** — free to use, share, and modify.

---

<div align="center">
  <strong>🌌 AlgoVerse — Explore the Universe of Machine Learning</strong><br>
  Made with ❤️ using Streamlit & Scikit-Learn
</div>
