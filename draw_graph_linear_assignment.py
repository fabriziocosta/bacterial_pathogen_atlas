# draw_graph_linear_assignment.py

# ===============================
# Consolidated Import Section
# ===============================
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.drawing.nx_agraph import graphviz_layout  # Requires PyGraphviz
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from collections import defaultdict
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import sample_colorscale
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

# ===============================
# Function Definitions
# ===============================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import re


def darken_color(hex_color, factor=0.7):
    """
    Darkens a given HEX color by multiplying its RGB components by a factor.
    """
    if hex_color.startswith('#'):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f'#{r:02x}{g:02x}{b:02x}'
    return hex_color


def plot_tsne_with_ellipses(df, coordinate_col, color_col, 
                            confidence=1, 
                            tsne_perplexity=30, 
                            random_state=42, 
                            n_std=1, 
                            plot_width=800, 
                            plot_height=600,
                            min_n_instances=3,
                            ellipse_darken_factor=0.7,
                            title="",
                            show_scatter=True):
    """
    Creates a TSNE scatter plot with ellipses for each group. In addition to
    the original features, each cluster is labeled with a progressive number
    at its center. A white circle (with alpha=0.5) is drawn over the underlying
    scatter points at the cluster center so that those points are partially hidden,
    and then the number is drawn on top.
    
    If `show_scatter` is False, the scatter traces are set to "legendonly" so that
    their legend entries remain visible but they are not drawn on the plot.
    
    The plot title can be defined by the user and the figure is automatically
    saved as a PDF with a filename derived from the title.
    """
    # 1. Compute TSNE embedding.
    X = np.vstack(df[coordinate_col].values)  # (n_samples, n_features)
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    # Add TSNE coordinates to a copy of the DataFrame.
    df_plot = df.copy()
    df_plot['tsne_x'] = X_tsne[:, 0]
    df_plot['tsne_y'] = X_tsne[:, 1]
    
    # 2. Filter out groups with fewer than min_n_instances.
    df_plot = df_plot.groupby(color_col).filter(lambda group: len(group) >= min_n_instances)
    
    # 3. Compute the number of instances per group (for tooltip purposes).
    df_plot['group_count'] = df_plot.groupby(color_col)[color_col].transform('count')
    
    # 4. Contract TSNE points for each group (if confidence != 1).
    if confidence != 1:
        for group_name, group_data in df_plot.groupby(color_col):
            group_mean = group_data[['tsne_x', 'tsne_y']].mean()
            df_plot.loc[group_data.index, 'tsne_x'] = group_mean['tsne_x'] + (group_data['tsne_x'] - group_mean['tsne_x']) / confidence
            df_plot.loc[group_data.index, 'tsne_y'] = group_mean['tsne_y'] + (group_data['tsne_y'] - group_mean['tsne_y']) / confidence
    
    # 5. Create a color mapping using the gist_ncar colormap.
    unique_groups = sorted(df_plot[color_col].unique())
    n_colors = len(unique_groups)
    base_colors = get_gist_ncar_colors(n_colors)         # List of HEX strings.
    marker_colors = [hex_to_rgba(c, alpha=0.8) for c in base_colors]  # For markers.
    
    # Mapping for the scatter plot.
    color_map = {group: marker_colors[i] for i, group in enumerate(unique_groups)}
    # Base colors for ellipses.
    base_color_map = {group: base_colors[i] for i, group in enumerate(unique_groups)}
    
    # Create mapping from group to a progressive number.
    group_to_number = {group: idx+1 for idx, group in enumerate(unique_groups)}
    
    # 6. Create the scatter plot using Plotly Express.
    fig = px.scatter(
        df_plot,
        x='tsne_x',
        y='tsne_y',
        color=color_col,
        color_discrete_map=color_map,
        custom_data=['group_count'],
        hover_data={color_col: True, 'tsne_x': False, 'tsne_y': False}
    )
    
    # Update marker trace names to include the progressive cluster number.
    for trace in fig.data:
        if trace.mode == 'markers' and trace.name in group_to_number:
            trace.name = f"{group_to_number[trace.name]}. {trace.name}"
    
    # 7. Remove axes, gridlines, and add the user-defined title.
    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False),
        yaxis=dict(visible=False, showgrid=False),
        title_text=title,
        plot_bgcolor='white',
        width=plot_width,
        height=plot_height
    )
    
    # 7.1. Add a thin black edge to each marker.
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))
    
    # Prepare lists to collect our extra traces.
    white_circle_traces = []
    text_traces = []
    
    # 8. For each group, compute an ellipse and add extra traces for the number.
    for group_name, group_data in df_plot.groupby(color_col):
        if group_data.shape[0] < 2:
            continue  # Not enough points to compute covariance.
        # Compute ellipse from TSNE coordinates.
        points = group_data[['tsne_x', 'tsne_y']].values
        mean = np.mean(points, axis=0)
        cov = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        t = np.linspace(0, 2 * np.pi, 100)
        circle = np.array([np.cos(t), np.sin(t)])
        ellipse_transformation = eigenvectors @ np.diag(n_std * np.sqrt(eigenvalues))
        ellipse = ellipse_transformation @ circle
        ellipse = ellipse + mean.reshape(2, 1)
        ellipse_x = ellipse[0, :]
        ellipse_y = ellipse[1, :]
        
        # Compute the total variance using the original coordinates.
        group_orig = np.vstack(group_data[coordinate_col].values)
        total_variance = np.trace(np.cov(group_orig, rowvar=False))
        variance_text = f"{total_variance:.2f}"
        
        # Retrieve and darken the base color.
        base_color = base_color_map[group_name]
        ellipse_color = darken_color(base_color, factor=ellipse_darken_factor)
        
        # Add the ellipse as an interactive trace.
        fig.add_trace(go.Scatter(
            x=ellipse_x,
            y=ellipse_y,
            mode='lines',
            line=dict(color=ellipse_color, width=2),
            hoverinfo='text',
            hovertext=f"Group: {group_name}<br>Total Variance: {variance_text}",
            showlegend=False
        ))
        
        # Create a white circle trace (as a marker) for the cluster center.
        white_circle_trace = go.Scatter(
            x=[mean[0]],
            y=[mean[1]],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=30,
                color='rgba(255,255,255,0.5)'  # White fill with 50% opacity.
            ),
            showlegend=False
        )
        white_circle_trace.update(meta={"layer": "white_circle"})
        white_circle_traces.append(white_circle_trace)
        
        # Create a text trace for the cluster number.
        text_trace = go.Scatter(
            x=[mean[0]],
            y=[mean[1]],
            mode='text',
            text=[str(group_to_number[group_name])],
            textposition='middle center',
            textfont=dict(color=ellipse_color, size=16),
            showlegend=False
        )
        text_trace.update(meta={"layer": "text"})
        text_traces.append(text_trace)
    
    # Add the extra traces to the figure.
    for trace in white_circle_traces:
        fig.add_trace(trace)
    for trace in text_traces:
        fig.add_trace(trace)
    
    # 9. Update hovertemplate for the original scatter traces.
    for trace in fig.data:
        if trace.mode == 'markers':
            meta = getattr(trace, 'meta', {}) or {}
            if meta.get("layer") != "white_circle":
                trace.hovertemplate = (
                    "Group: " + f"{trace.name}" + "<br>" +
                    "Instances: %{customdata[0]}<extra></extra>"
                )
    
    # 10. Reorder traces so that:
    #     - Scatter (original) traces are at the bottom.
    #     - Then white circle traces,
    #     - Then ellipse (lines) traces,
    #     - Then text traces are on top.
    scatter_traces = []
    ellipse_traces = []
    white_circle_list = []
    text_list = []
    
    for trace in fig.data:
        mode = trace.mode
        if mode == "markers":
            if getattr(trace, 'meta', {}) and trace.meta.get("layer") == "white_circle":
                white_circle_list.append(trace)
            else:
                scatter_traces.append(trace)
        elif mode == "lines":
            ellipse_traces.append(trace)
        elif mode == "text":
            text_list.append(trace)
        elif mode == "markers+text":
            text_list.append(trace)
    
    # If show_scatter is False, mark the scatter traces as "legendonly"
    if not show_scatter:
        for trace in scatter_traces:
            trace.visible = "legendonly"
    
    final_traces = scatter_traces + white_circle_list + ellipse_traces + text_list
    fig.data = tuple(final_traces)
    
    # 11. Automatically save the figure as a PDF.
    safe_title = re.sub(r'\W+', '_', title) if title else "plot"
    pdf_filename = f"{safe_title}.pdf"
    fig.write_image(pdf_filename)
    
    return fig

def plot_radar_from_series(
    data_series,
    axis_labels,
    num_per_row=5,
    selected_names=None,
    selected_axis_names=None,
    comparative_selected_names=None,
    size=4,
    fix_axis_range=False,
    log_scale=False,
    filename=None
):
    """
    Plots radar charts for selected entries in a Series, with optional comparison and scaling.

    Parameters:
    - data_series: pandas Series with index as labels and values as np.ndarray or list.
    - axis_labels: list of all axis labels (same length as data vectors).
    - num_per_row: number of radar plots per figure row.
    - selected_names: list of index keys to include as primary plots.
    - selected_axis_names: optional list of axis labels to restrict display.
    - comparative_selected_names: optional list of index keys for overlay comparison.
    - size: size multiplier for figure dimensions.
    - fix_axis_range: if True, sets radial range to [0, 1] or [0, log(2)] if log_scale is True.
    - log_scale: if True, applies log(value + 1) to all data values.
    """
    
    if selected_names is None:
        selected_names = list(data_series.index)

    if not all(name in data_series.index for name in selected_names):
        missing = [name for name in selected_names if name not in data_series.index]
        raise ValueError(f"Missing selected names: {missing}")

    if comparative_selected_names is not None:
        if len(comparative_selected_names) != len(selected_names):
            raise ValueError("comparative_selected_names must match selected_names in length.")
        if not all(name in data_series.index for name in comparative_selected_names):
            missing = [name for name in comparative_selected_names if name not in data_series.index]
            raise ValueError(f"Missing comparative names: {missing}")
        name_pairs = list(zip(selected_names, comparative_selected_names))
    else:
        name_pairs = [(name, None) for name in selected_names]

    # Handle axis selection
    if selected_axis_names is not None:
        axis_indices = [axis_labels.index(name) for name in selected_axis_names]
        axis_labels_used = selected_axis_names
    else:
        axis_indices = list(range(len(axis_labels)))
        axis_labels_used = axis_labels

    num_axes = len(axis_indices)
    theta = np.linspace(0, 2 * np.pi, num_axes, endpoint=False)
    theta = np.concatenate([theta, [theta[0]]])  # close the circle

    for i in range(0, len(name_pairs), num_per_row):
        batch = name_pairs[i:i+num_per_row]
        fig, axs = plt.subplots(1, len(batch), subplot_kw=dict(polar=True), figsize=(num_per_row * size, size))
        if len(batch) == 1:
            axs = [axs]

        for ax, (main_name, comp_name) in zip(axs, batch):
            # Get and process main values
            main_vals = np.asarray(data_series.loc[main_name])
            if len(main_vals) != len(axis_labels):
                raise ValueError(f"Main data length mismatch for '{main_name}'")

            main_vals = main_vals[axis_indices]
            if log_scale:
                main_vals = np.log1p(main_vals)
            main_vals = np.concatenate([main_vals, [main_vals[0]]])

            # Plot main
            ax.plot(theta, main_vals, label=str(main_name), linewidth=2)
            ax.fill(theta, main_vals, alpha=0.25)

            # Get and process comparative values
            if comp_name is not None:
                comp_vals = np.asarray(data_series.loc[comp_name])
                if len(comp_vals) != len(axis_labels):
                    raise ValueError(f"Comparative data length mismatch for '{comp_name}'")
                comp_vals = comp_vals[axis_indices]
                if log_scale:
                    comp_vals = np.log1p(comp_vals)
                comp_vals = np.concatenate([comp_vals, [comp_vals[0]]])

                ax.plot(theta, comp_vals, linestyle='-', label=f"{comp_name}", linewidth=2)
                ax.fill(theta, comp_vals, alpha=0.15)

            # Set up radar chart appearance
            ax.set_title(str(main_name), size=12)
            ax.set_xticks(theta[:-1])
            ax.set_xticklabels(axis_labels_used)
            ax.set_yticklabels([])

            if fix_axis_range:
                if log_scale:
                    ax.set_ylim(0, np.log1p(1))  # ~0.693
                else:
                    ax.set_ylim(0, 1)

            ax.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.1, 1.1))

        if filename is not None:
            plt.savefig(filename, format="pdf", dpi=300)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_entropy_per_species(data_series, axis_labels, selected_axis_names, selected_names=None, normalize=True):
    """
    Computes Shannon entropy for each species in data_series over specified axes.

    Parameters:
    - data_series: pandas Series, index format "Genus species", values are arrays/lists of counts or weights.
    - axis_labels: list of all axis labels corresponding to the vector positions.
    - selected_axis_names: list of axis labels to include in entropy calculation.
    - selected_names: optional list of species names (index) to restrict computation.
    - normalize: bool, if True normalizes entropy by log of number of non-zero categories.

    Returns:
    - pandas Series indexed by species name, with entropy values.
    """
    # Optionally filter to selected species
    if selected_names is not None:
        missing = [name for name in selected_names if name not in data_series.index]
        if missing:
            raise ValueError(f"Some selected names are not in the data_series: {missing}")
        data_series = data_series.loc[selected_names]

    # Determine indices of interest
    axis_indices = [axis_labels.index(a) for a in selected_axis_names]
    entropies = {}

    for species, values in data_series.items():
        vec = np.asarray(values)[axis_indices]
        total = vec.sum()
        if total == 0:
            entropy = 0.0
        else:
            probs = vec / total
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            if normalize and len(probs) > 1:
                entropy = entropy / np.log(len(probs))
        entropies[species] = entropy

    return pd.Series(entropies)


def compute_entropy_distribution_by_genus(
        data_series,
        axis_labels,
        selected_axis_names,
        selected_names=None,
        exclude_zero_values=True,
        title='Entropy Distribution by Genus',
        combined_plot=False,
        scale=1.0,
        filename=None
    ):
    """
    Computes entropy per species, groups by Genus, and plots either:
      - two-panel layout (boxplot + jittered scatter), or
      - single combined plot (boxplot with scatter overlay) when combined_plot=True.

    Parameters:
    - data_series: pandas Series, index "Genus species", values are numeric arrays/lists.
    - axis_labels: full list of axis labels.
    - selected_axis_names: list of axis labels to compute entropy over.
    - selected_names: optional list of species to include.
    - exclude_zero_values: bool, if True filters out zero-entropies.
    - title: title for the boxplot.
    - combined_plot: bool, if True overlays scatter on boxplot.
    """
    # Compute per-species entropies
    entropy_series = compute_entropy_per_species(
        data_series,
        axis_labels,
        selected_axis_names,
        selected_names,
        normalize=True
    )

    # Group by genus
    genus_entropy_map = defaultdict(list)
    for species, entropy in entropy_series.items():
        if exclude_zero_values and entropy == 0:
            continue
        genus = species.split()[0]
        genus_entropy_map[genus].append(entropy)

    # Prepare labels and data
    genus_list = sorted(genus_entropy_map.keys())
    labels_with_counts = [f"{g} ({len(genus_entropy_map[g])})" for g in genus_list]
    entropy_data = [genus_entropy_map[g] for g in genus_list]
    n_genus = len(genus_list)

    # Jitter settings
    rng = np.random.default_rng()
    jitter_strength = 0.2

    if combined_plot:
        # Single combined plot
        fig, ax = plt.subplots(figsize=(10*scale, max(4, n_genus * 0.8)*scale))

        # Draw boxplot at zorder=1
        box = ax.boxplot(
            entropy_data,
            labels=labels_with_counts,
            patch_artist=True,
            vert=False
        )
        for element in ('boxes', 'medians', 'whiskers', 'caps', 'fliers'):
            for artist in box[element]:
                artist.set_zorder(1)
                if element == 'boxes':
                    artist.set_facecolor('white')
                    artist.set_edgecolor('black')
                    artist.set_linewidth(1.5)
                elif element == 'medians':
                    artist.set_color('black')
                    artist.set_linewidth(2.5)
                elif element in ('whiskers', 'caps'):
                    artist.set_color('black')
                elif element == 'fliers':
                    artist.set_markeredgecolor('gray')

        # Overlay scatter at zorder=2
        for idx, genus in enumerate(genus_list):
            values = genus_entropy_map[genus]
            y = np.full(len(values), idx + 1)
            jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(values))
            ax.scatter(values, y + jitter,
                       alpha=0.7,
                       edgecolors='black',
                       zorder=2)

        # Final styling
        ax.set_title(f"{title} (combined)")
        ax.set_xlabel("Shannon Entropy")
        ax.set_ylabel("Genus (count)")
        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

        if filename is not None:
            plt.savefig(filename, format="pdf", dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    else:
        # Two-panel layout
        fig, axes = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=(12*scale, max(4, n_genus * 0.8)*scale)
        )

        # Left: boxplot
        box = axes[0].boxplot(
            entropy_data,
            labels=labels_with_counts,
            patch_artist=True,
            vert=False
        )
        for patch in box['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        for median in box['medians']:
            median.set_color('black')
            median.set_linewidth(2.5)
        for whisker in box['whiskers']:
            whisker.set_color('black')
        for cap in box['caps']:
            cap.set_color('black')
        for flier in box['fliers']:
            flier.set_markeredgecolor('gray')
        axes[0].set_title(title)
        axes[0].set_xlabel("Shannon Entropy")
        axes[0].set_ylabel("Genus (count)")
        axes[0].grid(True, linestyle='--', alpha=0.4)

        # Right: scatter with jitter
        for idx, genus in enumerate(genus_list):
            values = genus_entropy_map[genus]
            y = np.full(len(values), idx + 1)
            jitter = rng.uniform(-jitter_strength, jitter_strength, size=len(values))
            axes[1].scatter(values, y + jitter, alpha=0.7, edgecolors='black')

        axes[1].set_title('Individual Values with Jitter')
        axes[1].set_xlabel("Shannon Entropy")
        axes[1].set_ylabel("Genus (count)")
        axes[1].set_yticks(np.arange(1, n_genus + 1))
        axes[1].set_yticklabels(labels_with_counts)
        axes[1].yaxis.set_label_position('right')
        axes[1].yaxis.set_ticks_position('right')
        axes[1].tick_params(axis='y', which='both', labelleft=False, labelright=True)
        axes[1].grid(True, linestyle='--', alpha=0.4)
        
        if filename is not None:
            plt.savefig(filename, format="pdf", dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()


def get_gist_ncar_colors(n):
    """
    Generates 'n' distinct colors from Matplotlib's 'gist_ncar' colormap.

    Parameters:
    - n: int, number of unique colors to generate.

    Returns:
    - List of HEX color strings.
    """
    cmap = cm.get_cmap('gist_ncar', n)
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(n)]
    return colors

def hex_to_rgba(hex_color, alpha=0.4):
    """
    Converts a HEX color string to an RGBA string with the specified alpha.

    Parameters:
    - hex_color: str, HEX color code (e.g., '#FF5733').
    - alpha: float, alpha value between 0 and 1.

    Returns:
    - str, RGBA color string (e.g., 'rgba(255,87,51,0.4)').
    """
    rgb = mcolors.to_rgb(hex_color)  # Returns tuple of floats in [0,1]
    return f'rgba({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)}, {alpha})'

import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Helper utilities must still be available in the caller’s scope:  
#   • hex_to_rgba  
#   • get_gist_ncar_colors
# ---------------------------------------------------------------


def plot_emergence_timeline(
    df: pd.DataFrame,
    *,
    top_n: int = 25,
    x_jitter_amt: float = 0,
    y_jitter_amt: float = 5,
    year_per_char: float = 0.6,
    padding: float = 15,
    lane_gap: float = 20,
    dtick: float = 5,
    random_state: int = 42,
    separate_by_cluster: bool = True,
    base_row_height: float = 150,
    filename=None,
    use_score=False,
    cmap: str = "hot",
    export_width: int = 1600,
    export_height: int = 800,
    labels_from_clusters=None,
    avoid_label_overlap: bool = True,
    _max_overlap_iters: int = 8,
) -> go.Figure:
    """Emergence‑timeline scatter‑plot with optional score colouring.

    The function now *guarantees* that label text never spills into the
    neighbouring cluster band.  During the post‑processing collision‑removal
    pass, vertical nudges are limited to the current cluster’s lane extent – so
    labels always remain visually associated with the correct band.
    """

    # ────────────────────────────────────────────────────────────
    # Sanity checks & frame preparation
    # ────────────────────────────────────────────────────────────
    if use_score and "score" not in df.columns:
        raise ValueError("use_score=True but column 'score' not found in DataFrame")

    df = df.copy()
    df["radius"] = df["pubs_in_first_yrs"] / df["pubs_in_first_yrs"].max() * 50
    df = df.sort_values("year_n_pubs").reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    base_x = df["year_n_pubs"].values
    x_circle = base_x + rng.uniform(-x_jitter_amt, x_jitter_amt, len(df))

    # ────────────────────────────────────────────────────────────
    # Label‑selection logic
    # ────────────────────────────────────────────────────────────
    if labels_from_clusters is not None and "cluster" in df.columns:
        label_df = df[df["cluster"].isin(labels_from_clusters)]
    else:
        label_df = df

    if use_score:
        label_idx = label_df[use_score].nlargest(top_n).index.tolist()
    else:
        label_idx = label_df["pubs_in_first_yrs"].nlargest(top_n).index.tolist()
    label_idx.sort(key=lambda i: base_x[i])

    # ────────────────────────────────────────────────────────────
    # Cluster & colour preparation
    # ────────────────────────────────────────────────────────────
    if "cluster" in df.columns:
        cluster_vals = df["cluster"].astype(str).fillna("NA")
    else:
        cluster_vals = pd.Series(["NA"] * len(df))
    unique_clusters = sorted(cluster_vals.unique(), key=lambda x: int(x) if str(x).isdigit() else x)

    cluster_palette = {
        cl: hex_to_rgba(col)
        for cl, col in zip(unique_clusters, get_gist_ncar_colors(len(unique_clusters)))
    }

    # ────────────────────────────────────────────────────────────
    # Per‑cluster geometry helpers
    # ────────────────────────────────────────────────────────────
    def _compute_label_lanes(x_pos_map, indices):
        """Greedy horizontal packing → lane index per label."""
        lanes, lane_of = [], {}
        boxes = {
            i: (
                x_pos_map[i] - len(df.at[i, "CanonicalSpecies"]) * year_per_char / 2,
                x_pos_map[i] + len(df.at[i, "CanonicalSpecies"]) * year_per_char / 2,
            )
            for i in indices
        }
        for i in indices:
            li, ri = boxes[i]
            for lane_no, lane in enumerate(lanes):
                if all(ri <= boxes[j][0] or li >= boxes[j][1] for j in lane):
                    lane.append(i)
                    lane_of[i] = lane_no
                    break
            else:
                lanes.append([i])
                lane_of[i] = len(lanes) - 1
        return lanes, lane_of

    # Initial single‑band y coordinates (later offset by cluster)
    y_circle = base_row_height / 2 + rng.uniform(-y_jitter_amt, y_jitter_amt, len(df))
    lanes, lane_of = _compute_label_lanes({i: x_circle[i] for i in label_idx}, label_idx)
    row_lane_height = 2 * padding + lane_gap * ((len(lanes) + 1) // 2 + 1)

    # Initial label y‑coords (pre‑repulsion)
    y_label = np.zeros_like(base_x, dtype=float)
    for i, lane_no in lane_of.items():
        direction = 1 if lane_no % 2 == 0 else -1
        level = lane_no // 2
        offset = padding + level * lane_gap + df.at[i, "radius"]
        y_label[i] = base_row_height / 2 + direction * offset

    # ── Collision‑removal with band‑aware bounds ───────────────
    if avoid_label_overlap and len(label_idx) > 1:
        lbl_h = 12  # ~font height in y units
        char_w = year_per_char

        # Pre‑compute per‑cluster vertical limits
        if separate_by_cluster and "cluster" in df.columns and len(unique_clusters) > 1:
            # These limits are symmetrical around the cluster centreline
            row_stride = base_row_height + row_lane_height
            band_limits = {}
            for i_cl, cl in enumerate(unique_clusters):
                centre = i_cl * row_stride + base_row_height / 2
                half_span = row_lane_height / 2
                band_limits[cl] = (centre - half_span, centre + half_span)
        else:
            band_limits = {cl: (-np.inf, np.inf) for cl in unique_clusters}

        for _ in range(_max_overlap_iters):
            moved = False
            for i in label_idx:
                cl_i = cluster_vals[i]
                min_y, max_y = band_limits[cl_i]
                for j in label_idx:
                    if j <= i:  # avoid double‑check
                        continue
                    cl_j = cluster_vals[j]
                    if cl_i != cl_j:
                        continue  # collisions across clusters are impossible due to band separation

                    # BBox approximation
                    wi = len(df.at[i, "CanonicalSpecies"]) * char_w / 2
                    wj = len(df.at[j, "CanonicalSpecies"]) * char_w / 2
                    if abs(x_circle[i] - x_circle[j]) >= wi + wj:
                        continue  # no horizontal overlap
                    if abs(y_label[i] - y_label[j]) >= lbl_h:
                        continue  # no vertical overlap

                    # Resolve by moving the lower label down *if* room remains,
                    # otherwise move the upper label up.
                    if y_label[i] <= y_label[j]:
                        new_y = y_label[i] - lbl_h
                        if new_y >= min_y:
                            y_label[i] = new_y
                        else:
                            y_label[j] = min(y_label[j] + lbl_h, max_y)
                    else:
                        new_y = y_label[j] - lbl_h
                        if new_y >= min_y:
                            y_label[j] = new_y
                        else:
                            y_label[i] = min(y_label[i] + lbl_h, max_y)
                    moved = True
            if not moved:
                break

    # ─── Cluster vertical offsets ──────────────────────────────
    if separate_by_cluster and "cluster" in df.columns and len(unique_clusters) > 1:
        row_stride = base_row_height + row_lane_height
        cluster_offsets = {cl: i * row_stride for i, cl in enumerate(unique_clusters)}
        offset_series = cluster_vals.map(cluster_offsets).values
        y_circle += offset_series
        y_label += offset_series
        total_height = len(unique_clusters) * row_stride + 100
    else:
        row_stride = base_row_height + row_lane_height
        total_height = 100 + row_lane_height

    # ────────────────────────────────────────────────────────────
    # Build Plotly figure (rest unchanged)
    # ────────────────────────────────────────────────────────────
    fig = go.Figure()
    if use_score:
        score_min, score_max = df[use_score].min(), df[use_score].max()
        order = np.argsort(df[use_score].values)
        fig.add_trace(
            go.Scatter(
                x=x_circle[order],
                y=y_circle[order],
                mode="markers",
                marker=dict(
                    size=(df["radius"].values * 2)[order],
                    color=df[use_score].values[order],
                    colorscale=cmap,
                    cmin=score_min,
                    cmax=score_max,
                    colorbar=dict(title="Score"),
                    line=dict(width=1, color="black"),
                ),
                text=df["CanonicalSpecies"].values[order],
                hoverinfo="text",
                name="Species",
                showlegend=False,
            )
        )
    else:
        for cl in unique_clusters:
            idxs = df.index[cluster_vals == cl].tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_circle[idxs],
                    y=y_circle[idxs],
                    mode="markers",
                    marker=dict(
                        size=df.loc[idxs, "radius"] * 2,
                        color=cluster_palette[cl],
                        line=dict(width=1, color="black"),
                    ),
                    text=df.loc[idxs, "CanonicalSpecies"],
                    hoverinfo="text",
                    name=f"Cluster {cl}",
                )
            )

    fig.add_trace(
        go.Scatter(
            x=x_circle[label_idx],
            y=y_label[label_idx],
            mode="text",
            text=df.loc[label_idx, "CanonicalSpecies"],
            textfont=dict(size=12),
            showlegend=False,
        )
    )
    for i in label_idx:
        fig.add_shape(
            type="line",
            x0=x_circle[i], y0=y_circle[i], x1=x_circle[i], y1=y_label[i],
            line=dict(color="gray", width=1),
        )

    year_min = int(np.floor(df["year_n_pubs"].min() / 5) * 5)
    year_max = int(np.ceil(df["year_n_pubs"].max() / 5) * 5)
    for yr in range(year_min, year_max + 1, 5):
        fig.add_shape(type="line", x0=yr, y0=-150, x1=yr, y1=total_height,
                      line=dict(color="gray", width=1, dash="dot"), layer="below")

    if separate_by_cluster and "cluster" in df.columns and len(unique_clusters) > 1:
        for i in range(1, len(unique_clusters)):
            y_sep = i * row_stride
            fig.add_shape(type="line", x0=year_min, y0=y_sep, x1=year_max, y1=y_sep,
                          line=dict(color="lightgray", width=1, dash="dot"), layer="below")
    # ────────────────────────────────────────────────────────────
    # Layout & export
    # ────────────────────────────────────────────────────────────
    if separate_by_cluster:
        legend_y_offset = -0.05
    else:
        legend_y_offset = -.45
        
    fig.update_layout(
        template="plotly_white",
        height=total_height,
        xaxis=dict(title="Year", dtick=dtick, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        margin=dict(l=40, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=legend_y_offset, xanchor="center", x=0.5),
        showlegend=not use_score,
    )

    # ------------------------------------------------------------------
    # Export / display
    # ------------------------------------------------------------------
    if filename is not None:
        fig.write_image(filename, width=export_width, height=export_height)
    fig.show()
    return fig


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Optional

def _gaussian_weights(std: float, truncate: float = 4.0) -> np.ndarray:
    """
    Build a 1‑D Gaussian kernel normalised to sum=1.
    Radius = truncate * std (4σ by default).
    """
    radius = int(truncate * std + 0.5)
    x = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (x / std) ** 2)
    return w / w.sum()


def _gaussian_weights(std: float, truncate: float = 4.0) -> np.ndarray:
    """
    Build a discrete, normalised Gaussian kernel with given σ (std),
    truncated at ±truncate·σ.
    """
    radius = int(truncate * std + 0.5)
    x = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (x / std) ** 2)
    return w / w.sum()


def plot_smoothed_series(
    df: pd.DataFrame,
    columns: List[str],
    *,
    x_col: str = "year",
    window: int = 5,                       # only used if kernel_std is None
    kernel_std: Optional[float] = None,    # σ for Gaussian (same units as x_col)
    dtick: float = 5,
    filename: Optional[str] = None,
    export_width: int = 1600,
    export_height: int = 800,
    spline_smoothing: float = 1.0,
    columns_legends: Optional[List[str]] = None,
    show_raw_bars: bool = False,           # draw bars instead of stems
    raw_bar_opacity: float = 0.4
) -> go.Figure:
    """
    Plot smoothed curves plus optional raw‑value bars on a secondary y‑axis.

    Parameters
    ----------
    df : DataFrame containing `x_col` and every column in `columns`.
    columns : list of numeric columns to plot.
    x_col   : column for the x‑axis.
    window  : half‑width for centred moving average (ignored if kernel_std set).
    kernel_std : σ for Gaussian kernel (use None for moving average).
    columns_legends : optional display names for legend/hover (same length as columns).
    show_raw_bars   : if True, draw raw values as semitransparent bars with black edge
                      on a **separate y‑axis** (right side).
    All other kwargs tweak appearance or export.
    """

    # ---- validation ------------------------------------------------------
    req = set(columns + [x_col])
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {sorted(missing)}")

    if columns_legends and len(columns_legends) != len(columns):
        raise ValueError("`columns_legends` length must match `columns` length")

    legend_map = (
        dict(zip(columns, columns_legends))
        if columns_legends else {c: c for c in columns}
    )

    # ---- aggregate duplicate x values ------------------------------------
    aggregated = (
        df.groupby(x_col, sort=True)[columns]
          .mean()
          .sort_index()
    )

    # pad integer gaps (good for missing years)
    try:
        start, end = int(aggregated.index.min()), int(aggregated.index.max())
        full_idx = range(start, end + 1)
        aggregated = (
            aggregated.reindex(full_idx)
                      .interpolate()
                      .ffill()
                      .bfill()
        )
    except (ValueError, TypeError):
        pass  # x_col not integer‑like

    # ---- smoothing -------------------------------------------------------
    if kernel_std is None:
        smoothed = (
            aggregated
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )
    else:
        kernel = _gaussian_weights(kernel_std)
        smoothed = aggregated.copy()
        for c in columns:
            smoothed[c] = np.convolve(aggregated[c].to_numpy(),
                                      kernel, mode="same")

    # ---- colours ---------------------------------------------------------
    base_hex = get_gist_ncar_colors(len(columns))
    curve_col = {c: hex_to_rgba(base_hex[i], 0.8) for i, c in enumerate(columns)}
    bar_col   = {c: hex_to_rgba(base_hex[i], raw_bar_opacity) for i, c in enumerate(columns)}

    # ---- build figure ----------------------------------------------------
    fig = go.Figure()
    x_vals = smoothed.index.to_numpy()

    for col in columns:
        y_smooth = smoothed[col].to_numpy()
        label = legend_map[col]

        # outline
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_smooth,
                mode="lines",
                line=dict(width=6, color="black",
                          shape="spline", smoothing=spline_smoothing),
                hoverinfo="skip",
                showlegend=False
            )
        )

        # coloured smooth curve
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_smooth,
                mode="lines",
                name=label,
                line=dict(width=4, color=curve_col[col],
                          shape="spline", smoothing=spline_smoothing),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"{x_col}: %{{x}}<br>"
                    "Value: %{{y:.3f}}<extra></extra>"
                )
            )
        )

        # raw bars on secondary y‑axis
        if show_raw_bars:
            fig.add_trace(
                go.Bar(
                    x=x_vals,
                    y=aggregated[col].to_numpy(),
                    offsetgroup=f"raw_{col}",
                    yaxis="y2",
                    marker=dict(
                        color=bar_col[col],
                        line=dict(color="black", width=1)
                    ),
                    opacity=raw_bar_opacity,
                    hoverinfo="skip",
                    showlegend=False
                )
            )

    # ---- layout ----------------------------------------------------------
    layout_kwargs = dict(
        title="",
        xaxis=dict(title=x_col, tickmode="linear", dtick=dtick, showgrid=False),
        yaxis=dict(title="Density value", showgrid=False),
        height=600,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=80, b=80),
        legend=dict(orientation="h",
                    x=0.5, y=-0.2, xanchor="center", yanchor="top")
    )

    if show_raw_bars:
        layout_kwargs["barmode"] = "overlay"
        layout_kwargs["yaxis2"] = dict(
            title="Raw value",
            overlaying="y",
            side="right",
            showgrid=False
        )

    fig.update_layout(**layout_kwargs)

    # ---- export ----------------------------------------------------------
    if filename:
        fig.write_image(filename, width=export_width, height=export_height)

    fig.show()
    return fig


def hierarchical_clustering(X, n_clusters, contamination=0.1, random_state=None):
    """
    Perform hierarchical (agglomerative) clustering on inliers after detecting outliers using IsolationForest.

    Parameters:
    - X: ndarray, shape (n_samples, n_features)
        The data matrix where rows represent instances and columns represent features.
    - n_clusters: int
        The desired number of clusters for the hierarchical clustering.
    - contamination: float, optional, default=0.1
        The proportion of outliers in the data set. Used in the IsolationForest model.
    - random_state: int, optional
        Seed used by the random number generator.

    Returns:
    - full_labels: ndarray, shape (n_samples,)
        Cluster labels for each instance in X. Outliers are labeled as -1.
    """

    # Step 1: Use Isolation Forest to detect outliers in the data
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(X)

    # Step 2: Extract the inliers (those labeled as 1 by IsolationForest)
    inliers = X[outlier_labels == 1]

    if len(inliers) == 0:
        # If all points are outliers, return all as outliers
        return np.full(X.shape[0], -1)

    # Step 3: Perform hierarchical clustering on the inliers
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    inlier_labels = clustering_model.fit_predict(inliers)

    # Step 4: Create the full label array and assign -1 to the outliers
    full_labels = np.full(X.shape[0], -1)  # Initialize all labels to -1 (outliers)
    full_labels[outlier_labels == 1] = inlier_labels  # Assign cluster labels to inliers

    return full_labels

def filter_classes_by_min_and_max_instances(
    X, y, min_instances=None, max_instances=None, random_state=None,
    mode='random', n_clusters=5, contamination=0.1
):
    """
    Removes instances from X and y based on class occurrence thresholds and duplicates.
    Supports random sampling or clustering-based selection for limiting instances per class.

    Parameters:
    - X (numpy.ndarray or pandas.DataFrame): Data matrix with features.
    - y (numpy.ndarray or pandas.Series): Target classification vector with string labels.
    - min_instances (int, optional): Minimum number of instances a class must have to be kept.
    - max_instances (int, optional): Maximum number of instances per class.
    - random_state (int, optional): Seed for random sampling.
    - mode (str, optional): Mode of instance selection per class. ['random', 'cluster']
    - n_clusters (int, optional): Number of clusters for 'cluster' mode.
    - contamination (float, optional): Proportion of outliers for 'cluster' mode.

    Returns:
    - X_filtered (numpy.ndarray): Filtered data matrix.
    - y_filtered (numpy.ndarray): Filtered target vector.
    """
    try:
        # Validate mode
        if mode not in ['random', 'cluster']:
            raise ValueError("mode must be either 'random' or 'cluster'")

        # Convert to pandas if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        elif isinstance(y, pd.Series):
            y = y
        else:
            y = pd.Series(y)

        # Combine X and y for duplicate removal
        combined_df = X.copy()
        combined_df['target'] = y

        # Remove duplicate rows
        combined_df.drop_duplicates(inplace=True)

        # Apply min_instances filter
        if min_instances is not None:
            class_counts = combined_df['target'].value_counts()
            valid_classes = class_counts[class_counts >= min_instances].index
            combined_df = combined_df[combined_df['target'].isin(valid_classes)]

        # Apply max_instances filter
        if max_instances is not None:
            if mode == 'random':
                # Random sampling per class
                combined_df = combined_df.groupby('target', group_keys=False).apply(
                    lambda group: group.sample(n=min(len(group), max_instances), 
                                               random_state=random_state)
                )
            elif mode == 'cluster':
                # Cluster-based selection per class
                selected_indices = []

                for cls in combined_df['target'].unique():
                    class_data = combined_df[combined_df['target'] == cls].drop('target', axis=1).to_numpy()
                    num_instances = len(class_data)

                    if num_instances == 0:
                        continue

                    if n_clusters >= num_instances:
                        # Edge case: n_clusters >= number of instances
                        if num_instances <= max_instances:
                            # Keep all instances
                            selected_class_indices = combined_df[combined_df['target'] == cls].index.tolist()
                        else:
                            # Randomly sample max_instances
                            class_subset = combined_df[combined_df['target'] == cls]
                            selected_class_indices = class_subset.sample(
                                n=max_instances, random_state=random_state
                            ).index.tolist()
                        selected_indices.extend(selected_class_indices)
                        continue

                    # Perform hierarchical clustering
                    labels = hierarchical_clustering(
                        X=class_data, 
                        n_clusters=n_clusters, 
                        contamination=contamination, 
                        random_state=random_state
                    )

                    unique_clusters = set(labels)
                    unique_clusters.discard(-1)  # Remove outliers

                    if not unique_clusters:
                        # If all points are outliers, treat all as one cluster
                        unique_clusters = {0}

                    centroids = []

                    for cluster_label in unique_clusters:
                        cluster_indices = np.where(labels == cluster_label)[0]
                        cluster_points = class_data[cluster_indices]

                        if len(cluster_points) == 0:
                            continue

                        # Compute pairwise distances
                        distance_matrix = pairwise_distances(cluster_points, metric='euclidean')
                        avg_distances = distance_matrix.mean(axis=1)

                        # Find the index with minimum average distance
                        centroid_idx_within_cluster = np.argmin(avg_distances)
                        centroid_idx = cluster_indices[centroid_idx_within_cluster]
                        centroids.append(centroid_idx)

                    # If number of centroids exceeds max_instances, sample randomly
                    if len(centroids) > max_instances:
                        np.random.seed(random_state)
                        centroids = np.random.choice(centroids, size=max_instances, replace=False).tolist()

                    # Map back to the original DataFrame indices
                    class_indices = combined_df[combined_df['target'] == cls].index
                    selected_class_indices = class_indices[centroids].tolist()
                    selected_indices.extend(selected_class_indices)

                # Create the filtered DataFrame
                combined_df = combined_df.loc[selected_indices]

        # Extract filtered X and y
        X_filtered = combined_df.drop('target', axis=1).to_numpy()
        y_filtered = combined_df['target'].to_numpy()

        return X_filtered, y_filtered
    except Exception as e:
        print(f"An error occurred: {e}")
        return np.array([]), np.array([])

def compute_k_nearest_neighbors_within_class(X, y, k_within_class):
    """
    For each data point, identify its k nearest neighbors within the same class.

    Parameters:
    - X (numpy.ndarray or pandas.DataFrame): Data matrix of shape `(n_samples, n_features)`.
    - y (numpy.ndarray or pandas.Series): Target vector of shape `(n_samples,)`.
    - k_within_class (int): Number of nearest neighbors within the same class.

    Returns:
    - neighbors_within (list of lists): Indices of nearest neighbors within the same class.
    - distances_within (list of lists): Distances to nearest neighbors within the same class.
    """
    try:
        # Convert to numpy arrays if necessary
        X = X.values if isinstance(X, pd.DataFrame) else X
        y = y.values if isinstance(y, pd.Series) else y
        
        n_samples = X.shape[0]
        neighbors_within = [[] for _ in range(n_samples)]
        distances_within = [[] for _ in range(n_samples)]
        classes = np.unique(y)
        
        for cls in classes:
            indices_c = np.where(y == cls)[0]
            X_c = X[indices_c]
            
            actual_k = min(k_within_class, len(indices_c) - 1)
            if actual_k < 1:
                continue
            
            nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='auto', metric='euclidean')
            nbrs.fit(X_c)
            distances, indices = nbrs.kneighbors(X_c)
            
            for local_idx, orig_idx in enumerate(indices_c):
                # Exclude the point itself by taking indices from 1 to actual_k + 1
                neighbors = indices[local_idx][1:actual_k + 1]
                distances_neigh = distances[local_idx][1:actual_k + 1]
                
                neighbor_orig_indices = indices_c[neighbors]
                neighbors_within[orig_idx] = neighbor_orig_indices.tolist()
                distances_within[orig_idx] = distances_neigh.tolist()
        
        return neighbors_within, distances_within
    except:
        return [[] for _ in range(len(y))], [[] for _ in range(len(y))]

def compute_k_distance(X, inter_class_horizon):
    """
    Compute the distance to the k-th nearest neighbor for each data point across all classes.

    Parameters:
    - X (numpy.ndarray): Data matrix of shape `(n_samples, n_features)`.
    - inter_class_horizon (int): The 'k' corresponding to the k-th nearest neighbor.

    Returns:
    - k_distance (numpy.ndarray): Array of shape `(n_samples,)` with distances to the k-th nearest neighbor.
    """
    try:
        if inter_class_horizon < 1 or inter_class_horizon >= X.shape[0]:
            raise ValueError
        nbrs = NearestNeighbors(n_neighbors=inter_class_horizon + 1, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        # The first distance is zero (distance to itself), so take the k-th distance
        k_distance = distances[:, inter_class_horizon]
        return k_distance
    except:
        return np.array([])

def compute_inter_class_edges_linear_assignment(X, y, k_distance, inter_class_horizon):
    """
    Compute inter-class edges ensuring each node has at most one inter-class edge towards each other class,
    based on distance thresholds derived from k-distance.

    Parameters:
    - X (np.ndarray): 
        Data matrix of shape `(n_samples, n_features)`, where each row represents a data point.
        **Assumption:** 
            - `X` contains multiple classes with sufficient data points for assignment.
            - `n_features` is consistent across all samples.
    
    - y (np.ndarray): 
        Target vector of shape `(n_samples,)` containing discrete class labels.
        **Assumption:** 
            - Labels are integer-encoded starting from `0`.
            - The length of `y` matches the number of samples in `X`.
    
    - k_distance (np.ndarray): 
        Array of shape `(n_samples,)` containing the distance to the k-th nearest neighbor for each data point.
        **Assumption:** 
            - `k_distance` has been correctly computed and corresponds to each data point in `X`.
    
    - inter_class_horizon (int): 
        The 'k' used for determining the distance thresholds.
        **Assumption:** 
            - `inter_class_horizon` matches the value used in computing `k_distance`.
    
    Returns:
    - inter_class_edges (list of tuples): 
        List of tuples `(u, v)` representing edges between nodes of different classes.
        Each tuple denotes an undirected edge from node `u` to node `v`.
        
    - distances_inter (list of floats): 
        List of Euclidean distances corresponding to each inter-class edge in `inter_class_edges`.
        
    Raises:
    - ValueError: If input arrays have mismatched lengths or invalid types.
    """

    inter_class_edges = []
    distances_inter = []
    classes = np.unique(y)
    
    # Iterate over all unique pairs of classes
    for class_a, class_b in combinations(classes, 2):
        # Get indices of points in each class
        indices_a = np.where(y == class_a)[0]
        indices_b = np.where(y == class_b)[0]
        
        if len(indices_a) == 0 or len(indices_b) == 0:
            continue  # Skip if one of the classes has no points
        
        m, n = len(indices_a), len(indices_b)
        
        # Compute pairwise distances between class_a and class_b
        # Efficient computation using broadcasting
        X_a = X[indices_a]
        X_b = X[indices_b]
        distance_matrix = np.linalg.norm(X_a[:, np.newaxis, :] - X_b[np.newaxis, :, :], axis=2)  # Shape: (m, n)
        
        # Compute threshold matrix: min(k_distance[u], k_distance[v])
        threshold_matrix = np.minimum(k_distance[indices_a][:, np.newaxis], k_distance[indices_b][np.newaxis, :])
        
        # Set distances exceeding threshold to a large value to exclude them from assignment
        cost_matrix = np.where(distance_matrix <= threshold_matrix, distance_matrix, 1e6)
        
        # Pad the cost matrix to make it square to prevent infeasibility
        if m < n:
            pad_width = ((0, n - m), (0, 0))
            cost_matrix_padded = np.pad(cost_matrix, pad_width, mode='constant', constant_values=1e6)
        elif m > n:
            pad_width = ((0, 0), (0, m - n))
            cost_matrix_padded = np.pad(cost_matrix, pad_width, mode='constant', constant_values=1e6)
        else:
            cost_matrix_padded = cost_matrix.copy()
        
        # Perform linear sum assignment on the padded cost matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix_padded)
        
        # Iterate over the assignments and add edges if within threshold
        for r, c in zip(row_ind, col_ind):
            if r < m and c < n:
                distance = distance_matrix[r, c]
                if distance <= threshold_matrix[r, c]:
                    a_idx = indices_a[r]
                    b_idx = indices_b[c]
                    # To avoid duplicate edges in undirected graph
                    if not (a_idx, b_idx) in inter_class_edges and not (b_idx, a_idx) in inter_class_edges:
                        inter_class_edges.append((a_idx, b_idx))
                        distances_inter.append(distance)
    
    return inter_class_edges, distances_inter

def create_graph_with_separate_edges(X, y, neighbors_within, distances_within, inter_class_edges, distances_inter):
    """
    Construct a NetworkX graph incorporating both within-class and inter-class edges.

    Parameters:
    - X (numpy.ndarray): Data matrix of shape `(n_samples, n_features)`.
    - y (numpy.ndarray): Target vector of shape `(n_samples,)`.
    - neighbors_within (list of lists): Indices of nearest neighbors within the same class.
    - distances_within (list of lists): Distances to nearest neighbors within the same class.
    - inter_class_edges (list of tuples): Edges between nodes of different classes.
    - distances_inter (list of floats): Distances corresponding to each inter-class edge.

    Returns:
    - G (networkx.Graph): The constructed graph with nodes and edges.
    """
    try:
        G = nx.Graph()
        
        # Add nodes with class labels
        for idx, label in enumerate(y):
            G.add_node(idx, label=str(label))
        
        # Add within-class edges
        for idx, neighbors in enumerate(neighbors_within):
            for neighbor_idx, distance in zip(neighbors, distances_within[idx]):
                if not G.has_edge(idx, neighbor_idx):
                    G.add_edge(idx, neighbor_idx, label='within_class', distance=distance)
        
        # Add inter-class edges
        for (u, v), distance in zip(inter_class_edges, distances_inter):
            if not G.has_edge(u, v):
                G.add_edge(u, v, label='inter_class', distance=distance)
        
        return G
    except:
        return nx.Graph()

def build_graph(X, y, k_within_class, inter_class_horizon, confidence, random_state=None):
    """
    Orchestrate the construction of the k-Nearest Neighbors graph with both within-class and inter-class edges.

    Parameters:
    - X (numpy.ndarray): Data matrix of shape `(n_samples, n_features)`.
    - y (numpy.ndarray): Target vector of shape `(n_samples,)`.
    - k_within_class (int): Number of nearest neighbors within the same class.
    - inter_class_horizon (int): The 'k' used for determining distance thresholds.
    - confidence (float): Parameter to adjust the influence of within-class edge lengths.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - G (networkx.Graph): The fully constructed graph with node positions.
    """
    try:
        # Step 1: Compute intra-class nearest neighbors
        neighbors_within, distances_within = compute_k_nearest_neighbors_within_class(
            X, y, k_within_class
        )
        
        # Step 2: Compute distance thresholds for inter-class edges
        k_distance = compute_k_distance(X, inter_class_horizon)
        if k_distance.size == 0:
            return nx.Graph()
        
        # Step 3: Compute inter-class edges using linear assignment
        inter_class_edges, distances_inter = compute_inter_class_edges_linear_assignment(
            X, y, k_distance, inter_class_horizon
        )
        
        # Step 4: Create the graph with within-class and inter-class edges
        G = create_graph_with_separate_edges(
            X, y, neighbors_within, distances_within, inter_class_edges, distances_inter
        )
        
        if G.number_of_nodes() == 0:
            return G
        
        # Step 5: Modify within_class distances based on confidence and set uniform weights
        for u, v, data in G.edges(data=True):
            if data['label'] == 'within_class':
                original_distance = data['distance']
                # Example scaling: inverse relation with confidence
                modified_distance = original_distance / (1 + confidence)
                G[u][v]['len'] = modified_distance  # Desired length for layout
            else:
                G[u][v]['len'] = data['distance']  # Use original distance as desired length
            G[u][v]['weight'] = 1  # Set weight to 1 for all edges
            # Prevent zero-length edges which can cause layout issues
            if G[u][v]['len'] == 0:
                G[u][v]['len'] = 1e-3
        
        # Step 6: Compute Graphviz layout and assign positions to nodes as 'pos' attribute
        try:
            pos = graphviz_layout(G, prog='neato', args='')
            for node in G.nodes():
                G.nodes[node]['pos'] = pos[node]
        except:
            # Assign random positions as a fallback
            pos = {node: (np.random.rand(), np.random.rand()) for node in G.nodes()}
            nx.set_node_attributes(G, pos, 'pos')
        
        return G
    except:
        return nx.Graph()

def matplotlib_to_plotly_colormap(cmap_name):
    """
    Converts a Matplotlib colormap to a Plotly colorscale.

    Parameters:
    - cmap_name (str): Name of the Matplotlib colormap.

    Returns:
    - list: Plotly colorscale as a list of [normalized_position, color] pairs.
    """
    try:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, 'colors'):
            colors = cmap.colors
        else:
            colors = cmap(np.linspace(0, 1, cmap.N))
        
        colorscale = [
            [i / (len(colors) - 1), f'rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, {color[3]:.2f})']
            for i, color in enumerate(colors)
        ]
        return colorscale
    except:
        return []

def sample_colorscale_func(colorscale, values):
    """
    Sample colors from a Plotly colorscale based on normalized values.

    Parameters:
    - colorscale (list): Plotly colorscale list.
    - values (list of float): Normalized values between 0 and 1.

    Returns:
    - sampled_colors (list of str): Color strings corresponding to the sampled values.
    """
    try:
        sampled_colors = sample_colorscale(colorscale, values)
        return sampled_colors
    except:
        return ["rgba(0,0,0,1)"] * len(values)  # Default to black

def rgb_to_rgba(color, alpha=1.0):
    """
    Convert a hex or rgb color to rgba format with the specified alpha.

    Parameters:
    - color (str): Hex color string (e.g., '#FF5733') or rgb string (e.g., 'rgb(255,87,51)').
    - alpha (float): Alpha value between 0 and 1.

    Returns:
    - rgba_color (str): RGBA color string (e.g., 'rgba(255,87,51,0.5)').
    """
    try:
        if color.startswith('#'):
            rgb = pc.hex_to_rgb(color)
        elif color.startswith('rgb'):
            # Extract the numerical values from 'rgb(r, g, b)' or 'rgba(r, g, b, a)'
            rgb_values = color[color.find('(')+1:color.find(')')].split(',')
            rgb = tuple(map(int, rgb_values[:3]))
        else:
            # Attempt to convert named colors
            rgb = pc.to_rgb(color)
            rgb = tuple(int(255 * c) for c in rgb)
        return f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})'
    except:
        return 'rgba(0,0,0,1)'


def draw_convex_hull_graph_plotly(
    G,
    cmap,
    edge_alpha=0.5,
    edge_width=1,
    within_class_edge_color='red',
    inter_class_edge_color='cornflowerblue',
    draw_convex_hull=True,
    aggregate_edge_thickness=9,
    draw_class_text=True,
    max_num_interclass_edges=None, 
    min_num_interclass_edges=None, 
    size=800,
    save_as_svg=False,
    svg_filename="graph_plot.svg"
):
    """
    Draw the graph with nodes, edges, and optionally convex hulls for each class using Plotly.
    
    Parameters:
    - G (networkx.Graph): The graph containing nodes and edges.
    - cmap (list): Plotly colorscale for node coloring.
    - edge_alpha (float): Transparency level of the edges.
    - edge_width (float): Width of the edges.
    - within_class_edge_color (str): Color for within-class edges.
    - inter_class_edge_color (str): Color for inter-class edges.
    - draw_convex_hull (bool): If True, draws the convex hull for each class.
    - aggregate_edge_thickness (int): Thickness scaling for aggregated inter-class edges.
    - draw_class_text (bool): If True, renders the class label at the centroid.
    - max_num_interclass_edges (int, optional): Maximum number of top inter-class edges to aggregate.
    - size (int): Size of the Plotly figure.
    - save_as_svg (bool): If True, saves the figure as an SVG file.
    - svg_filename (str): Filename for the saved SVG.
    
    Returns:
    - fig (plotly.graph_objects.Figure): The Plotly figure object.
    """
    pos = nx.get_node_attributes(G, 'pos')
    labels = nx.get_node_attributes(G, 'label')

    # Extract inter-class edges
    inter_class_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get('label') == 'inter_class'
    ]

    unique_labels = sorted(set(labels.values()))
    num_classes = len(unique_labels)

    if num_classes > 1:
        label_to_value = {label: i / (num_classes - 1) for i, label in enumerate(unique_labels)}
    else:
        label_to_value = {label: 0.5 for label in unique_labels}

    # Generate class colors using the colorscale
    class_color_values = [label_to_value[label] for label in unique_labels]
    class_colors = sample_colorscale_func(cmap, class_color_values)
    label_to_class_color = dict(zip(unique_labels, class_colors))

    fig = go.Figure()

    # Aggregate inter-class edges
    if inter_class_edges:
        class_centers = {}
        for label in unique_labels:
            nodes_in_class = [node for node in G.nodes() if labels[node] == label]
            points = np.array([pos[node] for node in nodes_in_class])
            if len(points) > 0:
                centroid = points.mean(axis=0)
                class_centers[label] = centroid

        inter_class_counts = defaultdict(int)
        for u, v in inter_class_edges:
            class_u = labels[u]
            class_v = labels[v]
            class_pair = tuple(sorted([class_u, class_v]))
            inter_class_counts[class_pair] += 1
 
        # Sort class pairs by count descending and select top max_num_interclass_edges
        if inter_class_counts:
            sorted_class_pairs = sorted(inter_class_counts.items(), key=lambda item: item[1], reverse=True)
            if min_num_interclass_edges is None and max_num_interclass_edges is not None:
                sorted_class_pairs = sorted_class_pairs[:max_num_interclass_edges]
            if min_num_interclass_edges is not None:
                sorted_class_pairs = [(k,v) for k,v in sorted_class_pairs if v >= min_num_interclass_edges]
            max_count = max(inter_class_counts.values()) if inter_class_counts else 1
            for (class_a, class_b), count in sorted_class_pairs:
                if (
                    class_a in class_centers
                    and class_b in class_centers
                ):
                    start = class_centers[class_a]
                    end = class_centers[class_b]
                    scaled_width = (count / max_count) * aggregate_edge_thickness

                    # Prepare hover text
                    hover_text = f'{class_a} - {class_b}<br>Count: {count}'

                    fig.add_trace(
                        go.Scatter(
                            x=[start[0], end[0]],
                            y=[start[1], end[1]],
                            mode='lines',
                            line=dict(color=inter_class_edge_color, width=scaled_width),
                            opacity=edge_alpha,
                            hoverinfo='none',  # No hover on visible line
                            showlegend=False
                        )
                    )
                    
                    # Transparent hover line (wider for better hover area)
                    fig.add_trace(
                        go.Scatter(
                            x=[start[0], end[0]],
                            y=[start[1], end[1]],
                            mode='lines',
                            line=dict(color='rgba(0,0,0,0)', width=20),  # Transparent and wide
                            opacity=0,  # Fully transparent
                            hoverinfo='text',  # Enable hover info
                            hovertext=hover_text,  # Set hover text
                            showlegend=False
                        )
                    )
                    

    # Draw convex hulls
    if draw_convex_hull:
        for label in unique_labels:
            nodes_in_class = [node for node in G.nodes() if labels[node] == label]
            points = np.array([pos[node] for node in nodes_in_class])
            num_points = len(points)
            hover_text = f'{label}<br>{num_points}' if num_points != 1 else f'Class {label}<br>{num_points}'
            legend_label = f'{label} ({num_points})'

            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close the hull
                    hull_x = hull_points[:, 0]
                    hull_y = hull_points[:, 1]

                    hull_fill_color = rgb_to_rgba(label_to_class_color[label], alpha=0.4)

                    # Fill Trace with legend entry (only one per class)
                    fig.add_trace(
                        go.Scatter(
                            x=hull_x,
                            y=hull_y,
                            mode='lines',
                            fill='toself',
                            fillcolor=hull_fill_color,
                            line=dict(
                                width=1,  # Minimal line width to capture hover
                                color='rgba(0,0,0,0)'  # Transparent line color
                            ),
                            hoverinfo='text',
                            hovertext=hover_text,
                            name=legend_label,  # Assign class label and number of points to legend
                            showlegend=True
                        )
                    )

                    # Outline Trace without hover info and legend
                    fig.add_trace(
                        go.Scatter(
                            x=hull_x,
                            y=hull_y,
                            mode='lines',
                            line=dict(color='black', width=0.5),
                            opacity=1,
                            hoverinfo='none',
                            showlegend=False
                        )
                    )

                except Exception as e:
                    print(f"Convex hull failed for label {label}: {e}")

            elif len(points) == 2:
                # Draw a line connecting the two points
                x0, y0 = points[0]
                x1, y1 = points[1]

                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x0],
                        y=[y0, y1, y0],
                        mode='lines',
                        fill='toself',
                        fillcolor=rgb_to_rgba(label_to_class_color[label], alpha=0.3),
                        line=dict(color='black', width=0.5),
                        hoverinfo='text',
                        hovertext=hover_text,
                        name=legend_label,  # Assign class label and number of points to legend
                        showlegend=True
                    )
                )

            elif len(points) == 1:
                # Represent single points with a marker
                x0, y0 = points[0]
                fig.add_trace(
                    go.Scatter(
                        x=[x0],
                        y=[y0],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=label_to_class_color[label],
                        ),
                        hoverinfo='text',
                        hovertext=hover_text,
                        name=legend_label,
                        showlegend=True
                    )
                )

        # Add class text labels
        if draw_class_text:
            for label in unique_labels:
                nodes_in_class = [node for node in G.nodes() if labels[node] == label]
                points = np.array([pos[node] for node in nodes_in_class])
                num_points = len(points)
                if len(points) > 0:
                    centroid = points.mean(axis=0)
                    fig.add_trace(
                        go.Scatter(
                            x=[centroid[0]],
                            y=[centroid[1]],
                            mode='text',
                            text=[str(label)],
                            textposition='middle center',
                            hoverinfo='skip',
                            showlegend=False
                        )
                    )

    # Update layout with white background and legend at the bottom
    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(l=0, r=0, t=0, b=0),
        width=size,
        height=size,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            x=0.5,
            y=-0.2,  # Adjusted y position to place legend below the plot
            xanchor='center',
            yanchor='top',
            orientation='h',
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='Black',
            borderwidth=1
        )
    )
    
    # Save as SVG if requested
    if save_as_svg:
        try:
            fig.write_image(svg_filename, engine="kaleido")
        except Exception as e:
            print(f"Failed to save SVG: {e}")

    return fig




def build_and_draw_graph_plotly(
    X,
    y,
    min_instances=4,
    max_instances=None,
    mode='random', 
    n_clusters=5, 
    contamination=0.1,
    k_within_class=10,
    inter_class_horizon=20,
    confidence=50,
    max_num_interclass_edges=2,
    min_num_interclass_edges=None,
    draw_class_text=True,
    matplotlib_cmap='gist_ncar',
    size=1000,
    save=True,
    filename='graph_plot.svg',
    random_state=None
):
    """
    Orchestrate the filtering, graph building, and visualization process.

    Parameters:
    - X (numpy.ndarray or pandas.DataFrame): Feature matrix.
    - y (numpy.ndarray or pandas.Series): Target labels.
    - min_instances (int, optional): Minimum instances per class.
    - max_instances (int, optional): Maximum instances per class.
    - k_within_class (int): Number of within-class neighbors.
    - inter_class_horizon (int): 'k' for inter-class distance thresholding.
    - confidence (float): Scaling factor for within-class edge lengths.
    - max_num_interclass_edges (int): Minimum inter-class edges to aggregate.
    - matplotlib_cmap (str): Matplotlib colormap name.
    - size (int): Size of the Plotly figure.
    - save (bool): Whether to save the figure as an SVG file.
    - filename (str): Filename for the saved SVG.
    - random_state (int, optional): Seed for reproducibility.

    Returns:
    - fig (plotly.graph_objects.Figure): The Plotly figure object.
    - filtered_X (numpy.ndarray): Filtered feature matrix.
    - filtered_y (numpy.ndarray): Filtered target labels.
    - graph (networkx.Graph): The constructed graph.
    """
    try:
        # Step 1: Filter classes
        filtered_X, filtered_y = filter_classes_by_min_and_max_instances(
            X, y, min_instances, max_instances, random_state=random_state, mode=mode, n_clusters=n_clusters, contamination=contamination
        )
        
        if filtered_X.size == 0 or filtered_y.size == 0:
            return go.Figure(), filtered_X, filtered_y, nx.Graph()
        
        # Step 2: Build the graph
        graph = build_graph(
            filtered_X, filtered_y, k_within_class, inter_class_horizon, confidence, random_state=random_state
        )
        
        if graph.number_of_nodes() == 0:
            return go.Figure(), filtered_X, filtered_y, graph
        
        # Step 3: Convert Matplotlib colormap to Plotly colorscale
        plotly_cmap = matplotlib_to_plotly_colormap(matplotlib_cmap)
        
        if not plotly_cmap:
            plotly_cmap = matplotlib_to_plotly_colormap('Viridis')
        
        # Step 4: Draw the graph
        fig = draw_convex_hull_graph_plotly(
            graph,
            cmap=plotly_cmap,
            draw_convex_hull=True,
            aggregate_edge_thickness=8,
            draw_class_text=draw_class_text,
            max_num_interclass_edges=max_num_interclass_edges,
            min_num_interclass_edges=min_num_interclass_edges,
            size=size,
            save_as_svg=save,
            svg_filename=filename
        )
        # Display the figure
        fig.show()
        return fig, filtered_X, filtered_y, graph
    except Exception as e:
        print(f"Error encountered: {e}")
        return go.Figure(), np.array([]), np.array([]), nx.Graph()

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import matplotlib.patches as mpatches

def get_timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def plot_circular_dendrogram(X, labels, title='Dendrogram', figsize=10, filename=None,
                             label_colors=None, use_legend=False, legend_mapping=None):
    """
    Creates a circular dendrogram based on hierarchical clustering of the input data.
    Labels are positioned at the termini of the branches and oriented along the branch directions.
    
    Parameters:
    - X: Data matrix where each row represents a data point.
    - labels: List of names corresponding to each data point in X. These labels should be
              already prefixed with a progressive number (e.g. '23. label').
    - title: Title of the plot.
    - figsize: Diameter of the resulting plot in inches.
    - filename: If provided, the plot is saved as a PDF with this filename prefix.
    - label_colors: Optional dict mapping original labels (without the number prefix)
                    to a background color.
    - use_legend: Boolean indicating whether to display a color legend.
    - legend_mapping: Dict mapping legend display names (e.g. '1. SomeValue') to colors.
    """
    # Compute the hierarchical clustering using Ward's method
    Z = linkage(X, method='ward')
    
    # Generate the dendrogram structure without rendering it
    dendro = dendrogram(Z, labels=labels, no_plot=True)
    
    # Total number of leaf nodes in the dendrogram
    num_leaves = len(dendro['ivl'])
    
    # Determine evenly spaced angles around the circle for each leaf
    angles = np.linspace(0, 2 * np.pi, num_leaves, endpoint=False)
    
    # Associate each leaf node with its corresponding angle
    node_angle_map = {}
    for leaf, angle in zip(dendro['leaves'], angles):
        node_angle_map[leaf] = angle
    
    # Set up a polar plot for the circular dendrogram
    fig, ax = plt.subplots(figsize=(figsize, figsize), subplot_kw=dict(polar=True))
    ax.set_axis_off()
    plot_kwds = {'color': 'k', 'lw': 0.5}
    
    def plot_branch(node, parent_radius):
        """
        Recursively draws the branches of the dendrogram and annotates leaf nodes with labels.
    
        Parameters:
        - node: Index of the current node in the clustering hierarchy.
        - parent_radius: Radial distance from the center for the parent node.
    
        Returns:
        - current_angle: The angular position assigned to the current node.
        """
        if node < num_leaves:
            # Handling a leaf node
            angle = node_angle_map[node]
            ax.plot([angle, angle], [parent_radius, parent_radius + 1], **plot_kwds)
            
            label_radius = parent_radius + 1.05
            
            # Determine text alignment based on the angle
            if np.pi/2 <= angle <= 3*np.pi/2:
                horizontalalignment = 'right'
            else:
                horizontalalignment = 'left'
            
            # Calculate rotation to align with branch
            rotation_angle = np.degrees(angle)
            if np.pi/2 <= angle <= 3*np.pi/2:
                rotation_angle += 180
            
            # Extract the original group name by removing the prefix (if any)
            actual_label = labels[node]
            if '. ' in actual_label:
                actual_label = actual_label.split('. ', 1)[1]
            
            # If label_colors is provided, use it to set a background color
            bbox = None
            if label_colors is not None and actual_label in label_colors:
                bbox = dict(facecolor=label_colors[actual_label], edgecolor='none', pad=1)
            
            # Draw the label text (with the progressive number already prefixed)
            ax.text(angle, label_radius, labels[node],
                    rotation=rotation_angle,
                    rotation_mode='anchor',
                    horizontalalignment=horizontalalignment,
                    verticalalignment='center',
                    bbox=bbox)
            return angle
        else:
            # Handling an internal node
            left = int(Z[node - num_leaves, 0])
            right = int(Z[node - num_leaves, 1])
            
            left_angle = plot_branch(left, parent_radius + 1)
            right_angle = plot_branch(right, parent_radius + 1)
            
            current_angle = (left_angle + right_angle) / 2
            node_angle_map[node] = current_angle
            
            ax.plot([current_angle, current_angle], [parent_radius, parent_radius + 1], **plot_kwds)
            
            num_points = 100
            if right_angle < left_angle:
                theta = np.linspace(left_angle, right_angle + 2*np.pi, num_points)
                theta = theta % (2*np.pi)
            else:
                theta = np.linspace(left_angle, right_angle, num_points)
            arc_radius = parent_radius + 1
            r = np.full_like(theta, arc_radius)
            ax.plot(theta, r, **plot_kwds)
            
            return current_angle
    
    # Start drawing from the root node
    root_node = len(Z) + num_leaves - 1
    plot_branch(root_node, 0)
    
    #plt.title(title, fontsize=15)
    
    # If legend is requested, add it using the provided legend_mapping
    if use_legend and legend_mapping is not None:
        handles = [mpatches.Patch(color=color, label=str(label))
                   for label, color in legend_mapping.items()]
        ax.legend(handles=handles, title="Color Legend", bbox_to_anchor=(1.1, 1.1))
    
    if filename is not None:
        timestamp = get_timestamp()
        fname = timestamp + '_' + filename + '.pdf'
        plt.savefig(fname, format="pdf", dpi=300)
    plt.show()

def draw_dendrogram(data_df, column_group_by='CanonicalSpecies', column_embedding='embedding',
                    column_color_by=None, white_fraction=0.1, use_legend=True, min_num_to_show=10,
                    figsize=30, filename=None):
    """
    Draws a dendrogram based on aggregated embeddings.
    
    Parameters:
    - data_df: pandas DataFrame containing the data.
    - column_group_by: Column name to group data by.
    - column_embedding: Column name containing numpy array embeddings.
    - column_color_by: (Optional) Column name to determine label background color.
        For each group in column_group_by, the corresponding value in column_color_by is mapped
        to a color using the gist_ncar colormap.
    - white_fraction: Fraction of white to blend into the original color (0 gives the original color,
                      1 gives pure white).
    - use_legend: Boolean to indicate whether to display a color legend.
    - min_num_to_show: Minimum count of instances per group to include in the dendrogram.
    - figsize: Diameter of the resulting plot in inches.
    - filename: If provided, the plot is saved as a PDF with this filename prefix.
    """
    # Filter out rows with missing embeddings
    embedding_data_df = data_df[data_df[column_embedding].notna()].copy()
    # Aggregate embeddings by group
    species_df = aggregate_embeddings(
        embedding_data_df, 
        column_group_by=column_group_by, 
        column_embedding=column_embedding
    )
    sel_species_df = species_df[species_df['counts'] >= min_num_to_show]
    
    # Prepare the data matrix and corresponding group labels for the dendrogram
    X = np.vstack(sel_species_df[column_embedding].values)
    y = sel_species_df[column_group_by].values
    
    label_colors = None
    legend_mapping_display = None
    
    if column_color_by is not None:
        # For each group, get its representative value from the color column.
        group_color_values = data_df.groupby(column_group_by)[column_color_by].first()
        unique_color_vals = sorted(data_df[column_color_by].dropna().unique())
        cmap = plt.get_cmap('gist_ncar')
        
        def blend_with_white(color, white_fraction):
            r, g, b, a = color
            return (
                (1 - white_fraction) * r + white_fraction,
                (1 - white_fraction) * g + white_fraction,
                (1 - white_fraction) * b + white_fraction,
                a
            )
        
        # Map each unique value to a blended color
        legend_mapping = {
            val: blend_with_white(cmap(i / len(unique_color_vals)), white_fraction)
            for i, val in enumerate(unique_color_vals)
        }
        
        # 0-based numbering for groups
        group_number_mapping = {
            group: unique_color_vals.index(color_val)
            for group, color_val in group_color_values.items()
        }
        
        # Create label colors and prepend 0-based number to each label
        label_colors = {
            group: legend_mapping.get(color_val, 'white')
            for group, color_val in group_color_values.items()
        }
        numbered_labels = [f"{group_number_mapping[label]}. {label}" for label in y]
        y = numbered_labels
        
        # Create the color legend with 0-based indexing
        legend_mapping_display = {
            f"{i}. {val}": legend_mapping[val]
            for i, val in enumerate(unique_color_vals)
        }

    # Call the plotting function
    plot_circular_dendrogram(
        X, y, figsize=figsize,
        title=f'Group by: {column_group_by}   Embedding: {column_embedding}',
        filename=filename,
        label_colors=label_colors,
        use_legend=use_legend,
        legend_mapping=legend_mapping_display if use_legend else None
    )

def aggregate_embeddings(df, column_group_by, column_embedding, mode='avg'):
    """
    Aggregates embeddings or numerical values by averaging (or summing) them for each group 
    and counts the number of instances per group.
    
    Parameters:
    - df: pandas DataFrame.
    - column_group_by: Column name to group by.
    - column_embedding: Column name containing numpy arrays (embeddings) or numerical values.
    - mode: Either 'avg' (default) or 'sum'.
    
    Returns:
    - DataFrame with columns [column_group_by, column_embedding, 'counts'].
    """
    if column_group_by not in df.columns:
        raise ValueError(f"Column '{column_group_by}' not found in DataFrame.")
    if column_embedding not in df.columns:
        raise ValueError(f"Column '{column_embedding}' not found in DataFrame.")
    
    first_valid = df[column_embedding].dropna().iloc[0]
    is_array = isinstance(first_valid, np.ndarray)
    
    if is_array:
        if not df[column_embedding].apply(lambda x: isinstance(x, np.ndarray)).all():
            def to_array(x):
                return x if isinstance(x, np.ndarray) else np.array([x])
            df[column_embedding] = df[column_embedding].apply(to_array)
        
        def aggregate_group(group):
            embeddings = group[column_embedding]
            stacked = np.stack(embeddings.values)
            if mode == 'avg':
                aggregated_embedding = np.mean(stacked, axis=0)
            elif mode == 'sum':
                aggregated_embedding = np.sum(stacked, axis=0)
            else:
                raise ValueError("Invalid mode. Please choose 'avg' or 'sum'.")
            count = len(embeddings)
            return pd.Series({
                column_embedding: aggregated_embedding,
                'counts': count
            })
    else:
        if not pd.api.types.is_numeric_dtype(df[column_embedding]):
            raise TypeError(f"Column '{column_embedding}' must contain numerical values.")
        
        def aggregate_group(group):
            if mode == 'avg':
                aggregated_value = group[column_embedding].mean()
            elif mode == 'sum':
                aggregated_value = group[column_embedding].sum()
            else:
                raise ValueError("Invalid mode. Please choose 'avg' or 'sum'.")
            count = group[column_embedding].count()
            return pd.Series({
                column_embedding: aggregated_value,
                'counts': count
            })
    
    aggregated = df.groupby(column_group_by).apply(aggregate_group).reset_index()
    result_df = aggregated[[column_group_by, column_embedding, 'counts']]
    return result_df


import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def safe_norm_extended(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        arr = np.array(x)
        if arr.size == 0:
            return np.nan  # Handle empty arrays
        return np.linalg.norm(arr)
    else:
        return np.nan  # or any default value you prefer
    
def preprocess_list_column(entry):
    """
    Pre-processes an entry in the 'Summary' column.
    
    - If the entry is a list, returns it as is.
    - If the entry is a string, splits it by commas and strips whitespace.
    - Otherwise (e.g., NaN), returns an empty list.
    
    Parameters:
    - entry: The entry to preprocess.
    
    Returns:
    - list of strings
    """
    if isinstance(entry, list):
        return entry
    elif isinstance(entry, str):
        # Split by commas and strip whitespace
        return [item.strip() for item in entry.split(',') if item.strip()]
    else:
        return []

def add_one_hot_encoding(df, col, onehot_column_name='onehot_embedding'):
    """
    Creates one-hot encoded features using MultiLabelBinarizer for a specified column in the DataFrame
    and assigns it to a new column with a specified name.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - col (str): The column name containing lists of strings.
    - onehot_column_name (str): The name of the new column to store one-hot encoded arrays.

    Returns:
    - df (pd.DataFrame): The original DataFrame with the new one-hot encoded column added.
    - unique_classes (list): List of unique classes used in one-hot encoding.
    """
    # Preprocess the target column
    summary_series = df[col].apply(preprocess_list_column)
    
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Fit and transform the data
    one_hot = mlb.fit_transform(summary_series)
    
    # Retrieve unique classes
    unique_classes = sorted(mlb.classes_)
    
    # Assign the one-hot encoded array as a new column
    df.loc[:, onehot_column_name] = list(one_hot)
    
    return df, unique_classes

#------------------------------------------------

import pandas as pd
import numpy as np
import colorsys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
import plotly.graph_objects as go


def generate_rainbow_colors(n):
    """
    Generates a list of 'n' distinct colors evenly spaced across the HSV hue spectrum and converts them to HEX format.

    Parameters:
    - n: int, number of unique colors to generate.

    Returns:
    - List of HEX color strings.
    """
    colors = []
    for i in range(n):
        hue = i / n  # Evenly spaced hues between 0 and 1
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)  # Full saturation and value for vibrant colors
        colors.append('#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255)))
    return colors

def piecewise_linear_transform(values, x_ref, z_ref):
    """
    Applies a piecewise linear transformation to a list of values.
    
    Parameters:
        values (list or array): Input values to transform.
        x_ref (float): The reference input x value.
        z_ref (float): The corresponding desired output value at x_ref.
    
    Returns:
        list: Transformed values.
    """
    import numpy as np

    values = np.array(values)
    min_val = np.min(values)
    max_val = np.max(values)

    def transform(v):
        if v <= x_ref:
            # Interpolate from (min_val, min_val) to (x_ref, z_ref)
            return min_val + (z_ref - min_val) * (v - min_val) / (x_ref - min_val) if x_ref != min_val else z_ref
        else:
            # Interpolate from (x_ref, z_ref) to (max_val, max_val)
            return z_ref + (max_val - z_ref) * (v - x_ref) / (max_val - x_ref) if max_val != x_ref else z_ref

    return [transform(v) for v in values]


def plot_2d_plotly(X, y, counts=None, colors=None, embedding_target=None, n_instances=None,
                   embedding_transformer=None, 
                   color_scheme='discrete', max_size=50, size=800,
                   perplexity=30, n_iter=1000, random_state=42, show_label=False,
                   filename=None, use_pca=True, use_linear_contrast=False,
                   x_ref=5, z_ref=30, cmap='hot_r', highlight_elements=[],
                   show_legend=True):
    """
    Performs a 2D embedding on the input data X using the provided scikit-learn style
    transformer and plots the result using Plotly Graph Objects.

    Parameters:
      - X: numpy array of shape (n_samples, n_features)
      - y: array-like of labels, length n_samples
      - counts: array-like used for scaling marker sizes
      - colors: array-like for continuous or categorical coloring
      - n_instances: array-like of aggregated instance counts per row (for tooltips)
      - embedding_transformer: a scikit-learn style transformer with fit_transform
      - color_scheme: 'continuous' or 'discrete'
      - max_size: maximum marker size in pixels
      - size: size of the figure in pixels (square)
      - perplexity: used if embedding_transformer is None (for TSNE)
      - n_iter: number of iterations (for TSNE)
      - random_state: random seed (for TSNE)
      - show_label: if True, shows label text on markers
      - filename: if provided, saves the figure as a PDF with this filename
      - show_legend: if False, hides both legend (discrete) and colorbar (continuous)
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # You must define these helper functions elsewhere:
    # - piecewise_linear_transform
    # - get_gist_ncar_colors
    # - hex_to_rgba

    if color_scheme not in ['continuous', 'discrete']:
        raise ValueError("color_scheme must be either 'continuous' or 'discrete'.")
    if len(y) != len(X):
        raise ValueError("Length of y must match the number of samples in X.")
    if counts is not None and len(counts) != len(X):
        raise ValueError("Length of counts must match the number of samples in X.")
    if color_scheme == 'continuous':
        if colors is None:
            raise ValueError("For 'continuous' color_scheme, 'colors' must be provided.")
        if len(colors) != len(X):
            raise ValueError("Length of colors must match X.")
    elif color_scheme == 'discrete':
        if colors is not None and len(colors) != len(X):
            raise ValueError("Length of colors must match X.")

    if embedding_transformer is None:
        embedding_transformer = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(X) // 3),
            n_iter=n_iter,
            random_state=random_state,
            init='random',
            learning_rate='auto',
            verbose=0
        )

    X_embedded = embedding_transformer.fit_transform(X, embedding_target) if embedding_target else embedding_transformer.fit_transform(X)
    if use_pca:
        X_embedded = PCA(n_components=2).fit_transform(X_embedded)

    df_tsne = pd.DataFrame({
        'Dim1': X_embedded[:, 0],
        'Dim2': X_embedded[:, 1],
        'Label': y
    })

    if colors is not None:
        df_tsne['ColorValue'] = colors

    if counts is not None:
        min_marker_size = 7
        norm_counts = (counts - np.min(counts)) / (np.max(counts) - np.min(counts)) if np.max(counts) > np.min(counts) else np.zeros_like(counts)
        df_tsne['MarkerSize'] = norm_counts * (max_size - min_marker_size) + min_marker_size
    else:
        df_tsne['MarkerSize'] = max_size / 2

    if n_instances is not None:
        if len(n_instances) != len(X):
            raise ValueError("Length of n_instances must match X.")
        df_tsne['n_instances'] = n_instances

    def build_hover(row):
        inst = row.get('n_instances', row.get('Counts', 'NA'))
        label = row['Label']
        if color_scheme == 'continuous':
            return f"{label}<br>n_instances: {inst}<br>Color: {row['ColorValue']:.2f}"
        else:
            return f"{label}<br>n_instances: {inst}"

    df_tsne['HoverInfo'] = df_tsne.apply(build_hover, axis=1)
    df_tsne['TextLabel'] = df_tsne['Label'] if show_label else ''

    fig = go.Figure()

    if color_scheme == 'continuous':
        scaled_colors = piecewise_linear_transform(df_tsne['ColorValue'], x_ref, z_ref)
        fig.add_trace(go.Scatter(
            x=df_tsne['Dim1'],
            y=df_tsne['Dim2'],
            mode='markers+text' if show_label else 'markers',
            marker=dict(
                size=df_tsne['MarkerSize'],
                color=scaled_colors if use_linear_contrast else df_tsne['ColorValue'],
                colorscale=cmap,
                showscale=show_legend,
                colorbar=dict(title='Color Scale') if show_legend else None,
                opacity=0.8,
                line=dict(width=0.5, color='black')
            ),
            text=df_tsne['TextLabel'] if show_label else None,
            textposition='middle center' if show_label else None,
            hoverinfo='text',
            hovertext=df_tsne['HoverInfo'],
            name='Data Points'
        ))
    else:
        if colors is not None:
            unique_colors = np.unique(df_tsne['ColorValue'])
            color_map = dict(zip(unique_colors, [hex_to_rgba(c, 0.8) for c in get_gist_ncar_colors(len(unique_colors))]))
            for val in unique_colors:
                df_sub = df_tsne[df_tsne['ColorValue'] == val]
                fig.add_trace(go.Scatter(
                    x=df_sub['Dim1'],
                    y=df_sub['Dim2'],
                    mode='markers+text' if show_label else 'markers',
                    name=str(val),
                    marker=dict(
                        size=df_sub['MarkerSize'],
                        color=color_map[val],
                        opacity=0.8,
                        line=dict(width=0.5, color='black')
                    ),
                    text=df_sub['TextLabel'] if show_label else None,
                    textposition='middle center' if show_label else None,
                    hoverinfo='text',
                    hovertext=df_sub['HoverInfo']
                ))
        else:
            unique_labels = df_tsne['Label'].unique()
            color_map = dict(zip(unique_labels, [hex_to_rgba(c, 0.8) for c in get_gist_ncar_colors(len(unique_labels))]))
            for label in unique_labels:
                df_sub = df_tsne[df_tsne['Label'] == label]
                fig.add_trace(go.Scatter(
                    x=df_sub['Dim1'],
                    y=df_sub['Dim2'],
                    mode='markers+text' if show_label else 'markers',
                    name=label,
                    marker=dict(
                        size=df_sub['MarkerSize'],
                        color=color_map[label],
                        opacity=0.8,
                        line=dict(width=0.5, color='black')
                    ),
                    text=df_sub['TextLabel'] if show_label else None,
                    textposition='middle center' if show_label else None,
                    hoverinfo='text',
                    hovertext=df_sub['HoverInfo']
                ))

    fig.update_layout(
        template='plotly_white',
        width=size + 200 if (color_scheme == 'discrete' and show_legend) else size,
        height=size,
        showlegend=show_legend if color_scheme == 'discrete' else False,
        legend=dict(
            title='Categories',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(0,0,0,0)'
        ) if (color_scheme == 'discrete' and show_legend) else {},
        margin=dict(l=50, r=200 if (color_scheme == 'discrete' and show_legend) else 50, t=50, b=50),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
    )

    if highlight_elements:
        annotations = []
        for label in highlight_elements:
            match = df_tsne[df_tsne['Label'] == label]
            if not match.empty:
                row = match.iloc[0]
                annotations.append(dict(
                    x=row['Dim1'], y=row['Dim2'],
                    xref="x", yref="y",
                    text=label,
                    showarrow=True, arrowhead=2,
                    ax=40, ay=-40,
                    bordercolor="black", borderwidth=1, borderpad=4,
                    bgcolor="lightyellow", opacity=0.9,
                    font=dict(color='black', size=12)
                ))
        fig.update_layout(annotations=annotations)

    fig.show()

    if filename is not None:
        fig.write_image(filename)

    return fig


def fast_visualisation(
        data_df, 
        counts_df, 
        desired_attributes_for_embedding=None, 
        desired_attributes_for_color=None, 
        desired_attributes_for_embedding_axis=None,
        column_group_by='CanonicalSpecies', 
        column_embedding='embedding', 
        n_clusters=None, 
        max_cluster_size=10,
        use_dendrogram_clustering=False,
        contamination=0.1, 
        min_num_to_show=3, 
        max_marker_size=40, 
        figure_size=900, 
        unique_classes=None,
        column_color_by=None, 
        show_label=False,
        highlighted_species=None, 
        hierarchy_df=None,
        filename=None,
        embedding_transformer=None,
        use_pca=True,
        use_multidimensional_attributes_for_embedding_axis=False,
        use_linear_contrast=False, 
        x_ref=5, 
        z_ref=30,
        cmap='hot_r',
        highlight_elements=[],
        show_legend=True
    ):
    """
    Aggregates embeddings and counts, then produces a 2D embedding plot with tooltips showing 
    aggregated instance counts and other info. Supports clustering and coloring options.
    
    Returns:
      - out_df: DataFrame containing [column_group_by, 'cluster'] or color column (if applicable).
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=unique_classes).fit(unique_classes)

    embedding_data_df = data_df[data_df[column_embedding].notna()].copy()
    instances_df = aggregate_embeddings(
        embedding_data_df, 
        column_group_by=column_group_by, 
        column_embedding=column_embedding
    )
    sel_instances_df = instances_df[instances_df['counts'] >= min_num_to_show]

    sel_counts_df = aggregate_embeddings(
        counts_df, 
        column_group_by=column_group_by, 
        column_embedding='pathogenic_index'
    )

    sel_instances_df = pd.merge(
        sel_counts_df,
        sel_instances_df,
        on=column_group_by,
        suffixes=('_counts', '_instances')
    )

    if hierarchy_df is not None:
        sel_instances_df = pd.merge(
            hierarchy_df, 
            sel_instances_df, 
            left_on='species', 
            right_on='CanonicalSpecies', 
            how='right'
        )

    X = np.vstack(sel_instances_df[column_embedding].values)
    y = sel_instances_df[column_group_by].values

    colors = None
    color_scheme = 'discrete'
    out_df = None

    # Color by continuous attributes
    if desired_attributes_for_color is not None:
        positions = np.where(mlb.transform([desired_attributes_for_color])[0])[0]
        onehot_instances_df = aggregate_embeddings(
            embedding_data_df, 
            column_group_by=column_group_by, 
            column_embedding='onehot_embedding', 
            mode='sum'
        )
        sel_onehot_instances_df = onehot_instances_df[onehot_instances_df['counts'] >= min_num_to_show]
        sel_onehot_instances_df = pd.merge(
            sel_counts_df, 
            sel_onehot_instances_df, 
            on=column_group_by, 
            suffixes=('_counts', '_instances')
        )
        colors = np.sum(np.vstack(sel_onehot_instances_df['onehot_embedding'].values)[:, positions], axis=1)
        color_scheme = 'continuous'

    # Set embedding_target based on desired attributes for embedding axis
    embedding_target = None
    if desired_attributes_for_embedding_axis is not None:
        positions = np.where(mlb.transform([desired_attributes_for_embedding_axis])[0])[0]
        onehot_instances_df = aggregate_embeddings(
            embedding_data_df, 
            column_group_by=column_group_by, 
            column_embedding='onehot_embedding', 
            mode='sum'
        )
        sel_onehot_instances_df = onehot_instances_df[onehot_instances_df['counts'] >= min_num_to_show]
        sel_onehot_instances_df = pd.merge(
            sel_counts_df, 
            sel_onehot_instances_df, 
            on=column_group_by, 
            suffixes=('_counts', '_instances')
        )
        if use_multidimensional_attributes_for_embedding_axis:
            embedding_target = np.vstack(sel_onehot_instances_df['onehot_embedding'].values)[:, positions]
        else:
            embedding_target = np.sum(np.vstack(sel_onehot_instances_df['onehot_embedding'].values)[:, positions], axis=1)

    elif column_color_by is not None:
        if column_color_by in sel_instances_df.columns:
            color_labels = sel_instances_df[column_color_by].fillna("Missing").values
        else:
            raise ValueError(f"Column '{column_color_by}' not found in the aggregated dataframe.")
        colors = color_labels
        color_scheme = 'discrete'
        out_df = pd.DataFrame({column_group_by: y, column_color_by: color_labels})

    elif n_clusters is not None:
        clustering_method = 'dendrogram' if use_dendrogram_clustering else 'hybrid'
        cluster_df = assign_clusters(
            X, y, method=clustering_method,
            n_clusters=n_clusters,
            max_cluster_size=max_cluster_size,
            contamination=contamination,
            return_df=True,
            column_group_by=column_group_by
        )
        colors = cluster_df['cluster'].values
        color_scheme = 'discrete'
        out_df = cluster_df

    if desired_attributes_for_embedding is not None:
        assert column_embedding == 'onehot_embedding', (
            'ERROR: desired_attributes_for_embedding can only be used if column_embedding is onehot_embedding'
        )
        positions = np.where(mlb.transform([desired_attributes_for_embedding])[0])[0]
        X = X[:, positions]    

    if highlighted_species is not None:
        colors = np.array([
            '0.Selected' if species in highlighted_species else '1.Other' 
            for species in y
        ])
        color_scheme = 'discrete'

    marker_sizes = sel_instances_df['pathogenic_index'].round(2)
    n_instances = sel_instances_df['counts_instances']

    if embedding_transformer is None:
        embedding_transformer = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            random_state=42,
            init='random',
            learning_rate='auto',
            verbose=0
        )

    plot_2d_plotly(
        X, y, counts=marker_sizes, colors=colors, embedding_target=embedding_target, 
        n_instances=n_instances,
        embedding_transformer=embedding_transformer,
        max_size=max_marker_size, size=figure_size, color_scheme=color_scheme, 
        show_label=show_label, filename=filename, use_pca=use_pca, 
        use_linear_contrast=use_linear_contrast, x_ref=x_ref, z_ref=z_ref, cmap=cmap, 
        highlight_elements=highlight_elements,
        show_legend=show_legend
    )

    return out_df


from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

def assign_clusters(
    X, y, method='dendrogram', n_clusters=5, contamination=0.1, 
    return_df=True, column_group_by='CanonicalSpecies',
    max_cluster_size=10
):
    """
    Assigns cluster labels with optional max size per cluster (enforced recursively).

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : array-like (n_samples,)
    method : str
        'dendrogram' or 'hybrid'
    n_clusters : int
    contamination : float
        Used in hybrid mode
    return_df : bool
    column_group_by : str
    max_cluster_size : int or None

    Returns
    -------
    DataFrame with [column_group_by, 'cluster'] or np.ndarray
    """

    if method == 'dendrogram':
        Z = linkage(X, method='ward')
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    elif method == 'hybrid':
        iso = IsolationForest(contamination=contamination, random_state=42)
        inliers = iso.fit_predict(X) == 1
        cluster_labels = np.full(len(X), -1)
        if np.sum(inliers) > 0:
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels[inliers] = clustering.fit_predict(X[inliers])
    else:
        raise ValueError("Invalid method. Choose 'dendrogram' or 'hybrid'.")

    cluster_labels = np.array(cluster_labels)

    if max_cluster_size is not None:
        new_labels = np.full_like(cluster_labels, fill_value=-1)
        current_label = 0
        queue = [(np.where(cluster_labels == cid)[0]) for cid in np.unique(cluster_labels)]

        while queue:
            indices = queue.pop(0)
            if len(indices) <= max_cluster_size:
                new_labels[indices] = current_label
                current_label += 1
            else:
                n_sub = int(np.ceil(len(indices) / max_cluster_size))
                sub_clustering = AgglomerativeClustering(n_clusters=n_sub)
                sub_labels = sub_clustering.fit_predict(X[indices])
                for sub_id in np.unique(sub_labels):
                    sub_indices = indices[sub_labels == sub_id]
                    queue.append(sub_indices)

        cluster_labels = new_labels

    if return_df:
        return pd.DataFrame({column_group_by: y, 'cluster': cluster_labels})
    else:
        return cluster_labels
