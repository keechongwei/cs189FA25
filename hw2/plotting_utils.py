"""
Plotting utilities for Arena Style Control analysis.

This module contains standalone plotting functions extracted from the main notebook
to reduce clutter and improve code organization.
"""

import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List


def plot_rank_heatmap(df: pd.DataFrame, title: str = "Rank Heatmap", top_n: int = 20, 
                     categories: Optional[List[str]] = None, selected_models: Optional[List[str]] = None) -> go.Figure:
    """
    Create a heatmap showing ranks across categories using tidy data format.
    
    Args:
        df: pd.DataFrame
            Tidy DataFrame with columns ['Model', 'Rank', 'Category']
        title: str
            Title of the heatmap
        top_n: int
            Number of top models to show (default 20)
        categories: list[str], optional
            List of categories to include. If None, uses all categories in df
    
    Returns:
        plotly.graph_objects.Figure
            Interactive heatmap showing model ranks across categories
    """
    # Validate input format
    required_cols = ['Model', 'Rank', 'Category']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Filter categories if specified
    if categories is not None:
        df = df[df['Category'].isin(categories)].copy()
        
    if selected_models is not None:
        df = df[df['Model'].isin(selected_models)].copy()
    
    # Get the categories in the data (sorted for consistency)
    all_categories = sorted(df['Category'].unique())
    
    # Find top models based on overall ranking (or first category if no "Overall")
    if 'Overall' in df['Category'].values:
        top_models_df = df[df['Category'] == 'Overall'].nsmallest(top_n, 'Rank')
    else:
        # Use the first category alphabetically
        first_category = all_categories[0]
        top_models_df = df[df['Category'] == first_category].nsmallest(top_n, 'Rank')
    
    top_models = top_models_df['Model'].tolist()
    
    # Filter to top models and pivot to wide format for heatmap
    filtered_df = df[df['Model'].isin(top_models)].copy()
    
    # Pivot tidy data to wide format for heatmap
    pivot_df = filtered_df.pivot(index='Model', columns='Category', values='Rank')
    
    # Reorder models by their overall rank (or first category rank)
    if 'Overall' in pivot_df.columns:
        model_order = pivot_df.sort_values('Overall')['Overall'].index.tolist()
    else:
        first_category = all_categories[0]
        model_order = pivot_df.sort_values(first_category)[first_category].index.tolist()
    
    # Reorder rows and columns
    pivot_df = pivot_df.loc[model_order, all_categories]
    
    # Prepare data for heatmap
    rank_data = pivot_df.values
    models = pivot_df.index.tolist()
    category_names = pivot_df.columns.tolist()
    
    # Reverse model order so best models appear at top
    models.reverse()
    rank_data = rank_data[::-1]
    
    fig = go.Figure(data=go.Heatmap(
        z=rank_data,
        x=category_names,
        y=models,
        colorscale='Viridis_r',  # darker = better rank
        text=rank_data,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='Model: %{y}<br>Category: %{x}<br>Rank: %{z}<extra></extra>',
        showscale=False
    ))

    fig.update_layout(
        title=f'Rank Heatmap for Top {top_n} Models {title}',
        xaxis_title='Category',
        yaxis_title='Model',
        height=max(400, len(models) * 32) 
    )
    
    return fig


def plot_style_features(results_df: pd.DataFrame, selected_models: List[str]) -> go.Figure:
    """
    Plot style feature scores with confidence intervals.
    
    Args:
        results_df: DataFrame containing model results with columns 'Model', 'Average Score', 
                   'Lower Bound', 'Upper Bound', 'Rank'
        selected_models: List of model names to exclude (we want only style features)
    
    Returns:
        Plotly Figure object showing style feature scores with error bars
    """
    # Filter for style features only (not in selected_models)
    style_results = results_df[~results_df.Model.isin(selected_models)]

    # Sort by Rank for nice plotting
    style_results = style_results.sort_values("Rank")

    fig = go.Figure()

    for idx, row in style_results.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["Model"]],
            y=[row["Average Score"]],
            mode="markers",
            marker=dict(size=12),
            name=row["Model"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["Upper Bound"] - row["Average Score"]],
                arrayminus=[row["Average Score"] - row["Lower Bound"]],
                thickness=2,
                width=8,
                color="rgba(0,0,0,0.7)"
            ),
            showlegend=False,
            hovertemplate=(
                f"<b>{row['Model']}</b><br>"
                f"Score: {row['Average Score']:.3f}<br>"
                f"95% CI: [{row['Lower Bound']:.3f}, {row['Upper Bound']:.3f}]<br>"
                f"Rank: {int(row['Rank'])}"
            )
        ))

    fig.update_layout(
        title="Style Feature Scores with Confidence Intervals",
        xaxis_title="Style Feature",
        yaxis_title="Average Score",
        xaxis=dict(tickmode='array', tickvals=style_results["Model"], ticktext=style_results["Model"]),
        yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='gray'),
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=80)
    )

    return fig


def plot_rank_delta_heatmap(style_control_df: pd.DataFrame, baseline_df: pd.DataFrame, 
                           selected_models: List[str]) -> go.Figure:
    """
    Create a heatmap showing the delta in model rankings between baseline and style control results.
    
    Args:
        style_control_df: DataFrame with style control results in tidy format 
                         (columns: Model, Category, Rank)
        baseline_df: DataFrame with baseline results in tidy format
                    (columns: Model, Category, Rank)  
        selected_models: List of model names to include in comparison
        
    Returns:
        plotly.graph_objects.Figure
            Heatmap showing rank deltas (positive = worse with style control)
    """
    # Get style control results for models only (exclude style features)
    style_control_models = style_control_df[
        style_control_df['Model'].isin(selected_models)
    ].copy()

    # Get baseline results for models only
    baseline_models = baseline_df[
        baseline_df['Model'].isin(selected_models)
    ].copy()

    # Pivot both dataframes to wide format for comparison
    style_pivot = style_control_models.pivot(index='Model', columns='Category', values='Rank')
    baseline_pivot = baseline_models.pivot(index='Model', columns='Category', values='Rank')

    # Create mapping between style control and baseline categories
    # Remove " w/ Style Control" suffix to match with baseline categories
    category_mapping = {}
    for style_cat in style_pivot.columns:
        baseline_cat = style_cat.replace(" w/ Style Control", "")
        if baseline_cat in baseline_pivot.columns:
            category_mapping[style_cat] = baseline_cat

    if not category_mapping:
        raise ValueError("No matching categories found between style control and baseline results")
    
    # Get common models
    common_models = list(set(style_pivot.index) & set(baseline_pivot.index))
    
    if not common_models:
        raise ValueError("No common models found between datasets")
    
    # Create aligned dataframes for comparison
    style_aligned = pd.DataFrame(index=common_models)
    baseline_aligned = pd.DataFrame(index=common_models)
    
    # Align categories using the mapping
    for style_cat, baseline_cat in category_mapping.items():
        style_aligned[baseline_cat] = style_pivot.loc[common_models, style_cat]
        baseline_aligned[baseline_cat] = baseline_pivot.loc[common_models, baseline_cat]
    
    # Compute rank deltas (baseline - style_control)
    delta_data = baseline_aligned - style_aligned
    
    # Remove Overall column if it exists (often not meaningful for deltas)
    if 'Overall' in delta_data.columns:
        delta_data = delta_data.drop(columns=['Overall'])
    
    # Prepare data for heatmap
    heatmap_x = delta_data.columns.tolist()  # Categories
    
    # Sort models by their average delta across all categories
    avg_delta = delta_data.mean(axis=1).sort_values(ascending=False)
    delta_data_sorted = delta_data.loc[avg_delta.index]
    heatmap_z = delta_data_sorted.values
    heatmap_y = delta_data_sorted.index.tolist()
    
    # Add delta numbers as annotations
    annotations = []
    for i, model in enumerate(heatmap_y):
        for j, category in enumerate(heatmap_x):
            value = heatmap_z[i][j]
            if not pd.isna(value):  # Only annotate if value exists
                annotations.append(
                    dict(
                        x=category,
                        y=model,
                        text=f"{int(value)}",
                        showarrow=False,
                        font=dict(color="black" if abs(int(value)) < 2 else "white", size=12)
                    )
                )
    
    # Calculate height to ensure all y labels are visible
    n_models = len(heatmap_y)
    height = max(400, n_models * 30)
    
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_z,
            x=heatmap_x,
            y=heatmap_y,
            colorscale="RdBu",
            colorbar=dict(title="Rank Delta (Baseline - Style Control)"),
            zmid=0
        )
    )
    fig.update_layout(
        title="Delta in Model Rankings With Style Control (Category-Specific)",
        xaxis_title="Category",
        yaxis_title="Model",
        yaxis_autorange="reversed",
        annotations=annotations,
        height=height
    )
    
    return fig
