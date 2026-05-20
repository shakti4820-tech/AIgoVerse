import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def set_plot_style(fig, ax_list):
    fig.patch.set_facecolor('#0a0e1a')
    for ax in ax_list:
        ax.set_facecolor('#0d1522')
        ax.tick_params(colors='#6b7a8d', labelsize=8)
        ax.xaxis.label.set_color('#8899aa')
        ax.yaxis.label.set_color('#8899aa')
        ax.title.set_color('#00d4ff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')


def plot_decision_boundary(clf, X, y, ax, feature_names, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    if X.shape[1] > 2:
        fixed_val = X[:, 2].mean()
        grid = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, fixed_val)]
    else:
        grid = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    n_classes = len(np.unique(y))
    palette = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#a855f7']
    cmap_bg = ListedColormap([c + '33' for c in palette[:n_classes]])
    cmap_pts = ListedColormap(palette[:n_classes])

    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_bg)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts,
                         edgecolors='white', linewidths=0.4, s=30, zorder=3)
    ax.set_title(title, fontsize=9, pad=8)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    return scatter


def create_3d_scatter(X, y, feature_names, title="3D Feature Space", clf=None,
                      x_idx=0, y_idx=1, z_idx=2,
                      color_by="Actual Labels", colorscale_name="Vibrant Space"):
    import plotly.graph_objects as go
    
    # 1. Prepare coordinates
    x_coords = X[:, x_idx]
    y_coords = X[:, y_idx]
    z_coords = X[:, z_idx]
    
    # Define custom discrete colors
    vibrant_sequence = ['#00d4ff', '#ff6b6b', '#ffd166', '#06d6a0', '#a855f7']
    plasma_sequence = ['#0d0887', '#9c179e', '#ed7953', '#f0f921']
    viridis_sequence = ['#440154', '#31688e', '#35b779', '#fde725']
    tealrose_sequence = ['#009B9E', '#C4EC74', '#F1B6DA', '#D01C8B']
    rainbow_sequence = ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#8B00FF']
    
    palettes = {
        "Vibrant Space": vibrant_sequence,
        "Plasma": plasma_sequence,
        "Viridis": viridis_sequence,
        "Tealrose": tealrose_sequence,
        "Rainbow": rainbow_sequence
    }
    
    discrete_seq = palettes.get(colorscale_name, vibrant_sequence)
    
    # 2. Determine coloring perspective
    color_val = []
    hover_labels = []
    is_continuous = False
    
    if color_by == "Actual Labels":
        color_val = [str(val) for val in y]
        hover_labels = [f"Actual: {val}" for val in y]
    elif color_by == "Predicted Labels" and clf is not None:
        y_pred = clf.predict(X)
        color_val = [str(val) for val in y_pred]
        hover_labels = [f"Predicted: {val}" for val in y_pred]
    elif color_by == "Match/Mismatch (Errors)" and clf is not None:
        y_pred = clf.predict(X)
        color_val = ["Correct" if act == pred else "Incorrect" for act, pred in zip(y, y_pred)]
        hover_labels = [f"Actual: {act} | Pred: {pred}" for act, pred in zip(y, y_pred)]
        # Force a green/red palette
        discrete_seq = ['#06d6a0', '#ff6b6b']  # Correct: Green, Incorrect: Red
    elif color_by == "Prediction Confidence" and clf is not None:
        is_continuous = True
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)
            color_val = np.max(probs, axis=1)
        else:
            color_val = np.ones(len(y))
        hover_labels = [f"Conf: {c:.2f}" for c in color_val]
    else:
        # Fallback to actual labels
        color_val = [str(val) for val in y]
        hover_labels = [f"Label: {val}" for val in y]
        
    df = pd.DataFrame({
        'x': x_coords,
        'y': y_coords,
        'z': z_coords,
        'color': color_val,
        'hover': hover_labels
    })
    
    # Build the scatter plot
    if is_continuous:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='color',
            hover_name='hover',
            opacity=0.85,
            title=title,
            width=900,
            height=600,
            color_continuous_scale=colorscale_name if colorscale_name in ["Plasma", "Viridis", "Tealrose"] else "Viridis"
        )
    else:
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='color',
            hover_name='hover',
            symbol='color',
            opacity=0.85,
            title=title,
            width=900,
            height=600,
            color_discrete_sequence=discrete_seq
        )
        
    # 3. Overlay the classification decision slice plane
    if clf is not None:
        try:
            x1_min, x1_max = x_coords.min() - 0.2, x_coords.max() + 0.2
            x2_min, x2_max = y_coords.min() - 0.2, y_coords.max() + 0.2
            
            x1_grid = np.linspace(x1_min, x1_max, 50)
            x2_grid = np.linspace(x2_min, x2_max, 50)
            xx, yy = np.meshgrid(x1_grid, x2_grid)
            
            # Place horizontal slice plane at the mean of the Z-axis coordinate
            z_fixed = z_coords.mean()
            
            # Reconstruct full 3D row grid for classification prediction
            grid_3d = np.zeros((len(xx.ravel()), 3))
            grid_3d[:, x_idx] = xx.ravel()
            grid_3d[:, y_idx] = yy.ravel()
            grid_3d[:, z_idx] = z_fixed
            
            zz_preds = clf.predict(grid_3d).reshape(xx.shape)
            n_classes = len(np.unique(y))
            
            # Discrete colorscale for prediction slice
            colorscale = []
            for i in range(n_classes):
                val_start = i / n_classes
                val_end = (i + 1) / n_classes
                colorscale.append([val_start, discrete_seq[i % len(discrete_seq)]])
                colorscale.append([val_end, discrete_seq[i % len(discrete_seq)]])
                
            fig.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=np.full(xx.shape, z_fixed),
                    surfacecolor=zz_preds,
                    colorscale=colorscale,
                    opacity=0.25,
                    showscale=False,
                    name=f"Decision Boundary at {feature_names[z_idx]} mean",
                    hoverinfo='skip'
                )
            )
        except Exception:
            pass
            
    fig.update_layout(
        paper_bgcolor='#0a0e1a',
        plot_bgcolor='#0d1522',
        font_color='#c8d8e8',
        legend_title_text='Legend',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_scenes(
        xaxis=dict(title=feature_names[x_idx], backgroundcolor='#0d1522', gridcolor='#22354a', showbackground=True, zerolinecolor='#1e3a5f'),
        yaxis=dict(title=feature_names[y_idx], backgroundcolor='#0d1522', gridcolor='#22354a', showbackground=True, zerolinecolor='#1e3a5f'),
        zaxis=dict(title=feature_names[z_idx], backgroundcolor='#0d1522', gridcolor='#22354a', showbackground=True, zerolinecolor='#1e3a5f')
    )
    return fig
