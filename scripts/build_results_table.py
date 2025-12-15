"""
Build Results Table

Script to aggregate CV results and generate summary tables and markdown reports.
"""

import argparse
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(results_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(df)} result rows from {results_path}")
    
    return df


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results to compute mean and std per embedding+model combination.
    
    Args:
        results_df: DataFrame with individual fold results
    
    Returns:
        DataFrame with aggregated statistics
    """
    metrics = ["accuracy", "f1", "precision", "recall", "train_seconds", "infer_seconds"]
    
    # Check if embed_seconds column exists
    if "embed_seconds" in results_df.columns:
        metrics.append("embed_seconds")
    
    agg_funcs = {metric: ["mean", "std", "min", "max"] for metric in metrics}
    
    summary = results_df.groupby(["embedding", "model"]).agg(agg_funcs)
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    summary = summary.reset_index()
    
    # Add count of evaluations
    counts = results_df.groupby(["embedding", "model"]).size().reset_index(name="n_evaluations")
    summary = summary.merge(counts, on=["embedding", "model"])
    
    # Sort by mean F1 score (descending)
    summary = summary.sort_values("f1_mean", ascending=False)
    
    return summary


def generate_markdown_table(summary_df: pd.DataFrame) -> str:
    """
    Generate a markdown table from summary statistics.
    
    Args:
        summary_df: DataFrame with aggregated results
    
    Returns:
        Markdown-formatted table string
    """
    lines = [
        "# Results Summary\n",
        "## Cross-Validation Results (5-fold × 4 seeds = 20 evaluations)\n",
        "| Embedding | Model | Accuracy (mean ± std) | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Train Time (s) | Infer Time (s) |",
        "|-----------|-------|----------------------|-----------------|----------------------|-------------------|----------------|----------------|",
    ]
    
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['embedding']} | {row['model']} | "
            f"{row['accuracy_mean']:.4f} ± {row['accuracy_std']:.4f} | "
            f"{row['f1_mean']:.4f} ± {row['f1_std']:.4f} | "
            f"{row['precision_mean']:.4f} ± {row['precision_std']:.4f} | "
            f"{row['recall_mean']:.4f} ± {row['recall_std']:.4f} | "
            f"{row['train_seconds_mean']:.2f} ± {row['train_seconds_std']:.2f} | "
            f"{row['infer_seconds_mean']:.2f} ± {row['infer_seconds_std']:.2f} |"
        )
    
    lines.append("\n## Best Results by F1 Score\n")
    
    # Top 5 by F1
    top_5 = summary_df.head(5)
    lines.append("| Rank | Embedding | Model | F1 Score | Accuracy |")
    lines.append("|------|-----------|-------|----------|----------|")
    
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        lines.append(
            f"| {idx} | {row['embedding']} | {row['model']} | "
            f"{row['f1_mean']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build results table from CV results")
    parser.add_argument(
        "--input",
        type=str,
        default="reports/results/results_long.csv",
        help="Path to input CSV file with detailed results",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default="reports/results/results_summary.csv",
        help="Path to output summary CSV file",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default="reports/results/results_table.md",
        help="Path to output markdown table file",
    )
    
    args = parser.parse_args()
    
    # Load results
    results_df = load_results(args.input)
    
    # Aggregate results
    summary_df = aggregate_results(results_df)
    
    # Save summary CSV
    output_summary_path = Path(args.output_summary)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_summary_path, index=False)
    logger.info(f"Saved summary to {output_summary_path}")
    
    # Generate and save markdown table
    markdown_table = generate_markdown_table(summary_df)
    output_md_path = Path(args.output_markdown)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown_table)
    logger.info(f"Saved markdown table to {output_md_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal experiments: {len(results_df)}")
    print(f"Unique combinations: {len(summary_df)}")
    print(f"\nTop 3 by F1 score:")
    print(summary_df[["embedding", "model", "f1_mean", "accuracy_mean"]].head(3).to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
