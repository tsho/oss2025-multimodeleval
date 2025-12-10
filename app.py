"""
Streamlit App for Embedding Model Evaluation Results Visualization

Usage:
    streamlit run app.py
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Page config
st.set_page_config(
    page_title="Embedding Model Evaluation",
    page_icon="📊",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
</style>
""", unsafe_allow_html=True)


def find_result_dirs() -> list[Path]:
    """
    
    Find all result directories

    Args:
        None
    
    Returns:
        results_dirs: List of result directories
    """
    results_dirs = []
    
    # Check standard results directory
    for base_dir in [Path("./results"), Path("./results_custom")]:
        if base_dir.exists():
            for subdir in sorted(base_dir.iterdir(), reverse=True):
                if subdir.is_dir():
                    # Check if it contains results
                    if (subdir / "comparison_results.csv").exists() or \
                       (subdir / "custom_results.csv").exists():
                        results_dirs.append(subdir)
    
    return results_dirs


def load_results(result_dir: Path) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Load results from a directory

    Args:
        result_dir: Path to result directory

    Returns:
        df: DataFrame of results
        config: Dictionary of configuration
    """
    df = None
    config = None
    
    # Try loading comparison_results.csv (MTEB results)
    if (result_dir / "comparison_results.csv").exists():
        df = pd.read_csv(result_dir / "comparison_results.csv")
        df["source"] = "MTEB"
    
    # Try loading custom_results.csv
    elif (result_dir / "custom_results.csv").exists():
        df = pd.read_csv(result_dir / "custom_results.csv")
        df["source"] = "Custom"
    
    # Load config if exists
    if (result_dir / "config.json").exists():
        with open(result_dir / "config.json") as f:
            config = json.load(f)
    
    return df, config


def main():
    st.markdown('<p class="main-header">📊 Embedding Model Evaluation Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Find result directories
    result_dirs = find_result_dirs()
    
    if not result_dirs:
        st.warning("No result directories found. Run evaluation first:")
        st.code("uv run eval_retrieval.py --models minilm bge-small --task-set quick")
        return
    
    # Select result directory
    selected_dir = st.sidebar.selectbox(
        "Select Results",
        result_dirs,
        format_func=lambda x: f"{x.parent.name}/{x.name}",
    )
    
    # Load results
    df, config = load_results(selected_dir)
    
    if df is None:
        st.error(f"Could not load results from {selected_dir}")
        return
    
    # Display config info
    if config:
        with st.sidebar.expander("Configuration", expanded=False):
            st.json(config)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Results Overview")
        
        # Show data source
        source = df["source"].iloc[0] if "source" in df.columns else "Unknown"
        st.info(f"Data Source: **{source}** | Directory: `{selected_dir.name}`")
    
    with col2:
        st.subheader("📋 Summary Stats")
        if "model_name" in df.columns:
            st.metric("Models Evaluated", df["model_name"].nunique())
        if "task" in df.columns:
            st.metric("Tasks", df["task"].nunique())
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📋 Data Table", "🔥 Heatmap", "📈 Comparison"])
    
    with tab1:
        render_charts(df)
    
    with tab2:
        render_data_table(df)
    
    with tab3:
        render_heatmap(df)
    
    with tab4:
        render_comparison(df, result_dirs)


def render_charts(df: pd.DataFrame) -> None:
    """
    Render interactive charts

    Args:
        df: DataFrame of results

    Returns:
        None
    """
    
    st.subheader("NDCG@10 by Model")
    
    # Determine metric column
    metric_col = "ndcg_at_10" if "ndcg_at_10" in df.columns else None
    
    if metric_col and df[metric_col].notna().any():
        # Group by model
        if "model_name" in df.columns:
            if "task" in df.columns:
                # MTEB results - show by task
                fig = px.bar(
                    df,
                    x="task",
                    y=metric_col,
                    color="model_name",
                    barmode="group",
                    title="NDCG@10 by Task and Model",
                    labels={metric_col: "NDCG@10", "task": "Task", "model_name": "Model"},
                )
            else:
                # Custom results - simple bar
                avg_scores = df.groupby("model_name")[metric_col].mean().reset_index()
                fig = px.bar(
                    avg_scores,
                    x="model_name",
                    y=metric_col,
                    title="Average NDCG@10 by Model",
                    labels={metric_col: "NDCG@10", "model_name": "Model"},
                    color=metric_col,
                    color_continuous_scale="Blues",
                )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            if "recall_at_10" in df.columns and df["recall_at_10"].notna().any():
                st.subheader("Recall@10")
                if "model_name" in df.columns:
                    avg_recall = df.groupby("model_name")["recall_at_10"].mean().reset_index()
                    fig = px.bar(
                        avg_recall,
                        x="model_name",
                        y="recall_at_10",
                        title="Average Recall@10 by Model",
                        color="recall_at_10",
                        color_continuous_scale="Greens",
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if "mrr_at_10" in df.columns and df["mrr_at_10"].notna().any():
                mrr_col = "mrr_at_10"
            elif "mrr" in df.columns and df["mrr"].notna().any():
                mrr_col = "mrr"
            else:
                mrr_col = None
            
            if mrr_col:
                st.subheader("MRR")
                if "model_name" in df.columns:
                    avg_mrr = df.groupby("model_name")[mrr_col].mean().reset_index()
                    fig = px.bar(
                        avg_mrr,
                        x="model_name",
                        y=mrr_col,
                        title="Average MRR by Model",
                        color=mrr_col,
                        color_continuous_scale="Oranges",
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No NDCG@10 data available")


def render_data_table(df: pd.DataFrame):
    """
    Render data table

    Args:
        df: DataFrame of results

    Returns:
        None
    """
    st.subheader("Raw Data")
    
    # Column selector
    all_cols = df.columns.tolist()
    default_cols = [c for c in ["model_name", "task", "ndcg_at_10", "recall_at_10", "mrr_at_10", "mrr"] if c in all_cols]
    
    selected_cols = st.multiselect(
        "Select columns to display",
        all_cols,
        default=default_cols if default_cols else all_cols[:5],
    )
    
    if selected_cols:
        st.dataframe(
            df[selected_cols].style.format({
                col: "{:.4f}" for col in selected_cols if df[col].dtype == float
            }),
            use_container_width=True,
            height=400,
        )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="results.csv",
        mime="text/csv",
    )


def render_heatmap(df: pd.DataFrame) -> None:
    """
    Render heatmap

    Args:
        df: DataFrame of results

    Returns:
        None
    """
    st.subheader("NDCG@10 Heatmap")
    
    if "model_name" not in df.columns or "ndcg_at_10" not in df.columns:
        st.warning("Required columns not found")
        return
    
    if "task" in df.columns:
        # Create pivot table
        pivot = df.pivot_table(
            index="model_name",
            columns="task",
            values="ndcg_at_10",
            aggfunc="mean",
        )
        
        fig = px.imshow(
            pivot,
            labels=dict(x="Task", y="Model", color="NDCG@10"),
            aspect="auto",
            color_continuous_scale="Blues",
            title="NDCG@10 Heatmap",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Heatmap requires task column (available in MTEB results)")


def render_comparison(df: pd.DataFrame, result_dirs: list[Path]) -> None:
    """
    Render comparison between different results

    Args:
        df: DataFrame of results
        result_dirs: List of result directories

    Returns:
        None
    """
    st.subheader("Compare Results")
    
    # Allow selecting multiple result directories
    selected_dirs = st.multiselect(
        "Select results to compare",
        result_dirs,
        format_func=lambda x: f"{x.parent.name}/{x.name}",
        default=[result_dirs[0]] if result_dirs else [],
    )
    
    if len(selected_dirs) < 2:
        st.info("Select at least 2 result sets to compare")
        return
    
    # Load all selected results
    all_dfs = []
    for dir_path in selected_dirs:
        df_temp, _ = load_results(dir_path)
        if df_temp is not None:
            df_temp["result_set"] = dir_path.name
            all_dfs.append(df_temp)
    
    if not all_dfs:
        st.warning("Could not load selected results")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Compare metrics
    if "model_name" in combined_df.columns and "ndcg_at_10" in combined_df.columns:
        avg_scores = combined_df.groupby(["result_set", "model_name"])["ndcg_at_10"].mean().reset_index()
        
        fig = px.bar(
            avg_scores,
            x="model_name",
            y="ndcg_at_10",
            color="result_set",
            barmode="group",
            title="NDCG@10 Comparison Across Result Sets",
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

