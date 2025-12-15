#!/usr/bin/env python3
"""
CLI script to aggregate experiment results and generate summary tables.

Usage:
    python scripts/build_results_table.py
    python scripts/build_results_table.py --results-dir reports/results --output results_summary
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import format_results_table, ensure_dir


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build results summary tables")
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="reports/results",
        help="Directory containing result CSV files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output filename prefix (without extension)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_summary.csv",
        help="Pattern for summary CSV files"
    )
    
    return parser.parse_args()


def aggregate_results(results_dir: str, pattern: str = "*_summary.csv") -> pd.DataFrame:
    """
    Aggregate all summary CSV files in directory.
    
    Args:
        results_dir: Directory containing result files
        pattern: File pattern to match
    
    Returns:
        Aggregated DataFrame
    """
    results_dir = Path(results_dir)
    
    # Find all summary files
    summary_files = list(results_dir.glob(pattern))
    
    if not summary_files:
        print(f"Warning: No files matching '{pattern}' found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(summary_files)} summary files")
    
    # Load and concatenate
    dfs = []
    for file in summary_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    return combined


def create_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated summary table grouped by embedding+variant+model.
    
    Args:
        df: DataFrame with individual experiment results
    
    Returns:
        Summary DataFrame with mean±std statistics
    """
    # Group by embedding, variant, and model
    group_cols = ['embedding_type', 'embedding_variant', 'model_type']
    
    # Metrics to aggregate
    metric_cols = [col for col in df.columns if any(m in col for m in ['accuracy', 'f1', 'precision', 'recall', 'runtime'])]
    
    # If already aggregated (has _mean, _std), just return relevant columns
    if any('_mean' in col for col in df.columns):
        cols_to_keep = group_cols + metric_cols
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        
        # Remove duplicates
        summary = df[cols_to_keep].drop_duplicates().reset_index(drop=True)
    else:
        # Aggregate from long format
        summary_data = []
        
        for name, group in df.groupby(group_cols):
            row = {
                'embedding_type': name[0],
                'embedding_variant': name[1],
                'model_type': name[2]
            }
            
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'runtime']:
                if metric in group.columns:
                    row[f'{metric}_mean'] = group[metric].mean()
                    row[f'{metric}_std'] = group[metric].std()
            
            summary_data.append(row)
        
        summary = pd.DataFrame(summary_data)
    
    # Sort by accuracy
    if 'accuracy_mean' in summary.columns:
        summary = summary.sort_values('accuracy_mean', ascending=False)
    
    return summary


def main():
    """Main function."""
    args = parse_args()
    
    print("="*80)
    print("Building Results Summary Tables")
    print("="*80)
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return
    
    # Aggregate summary files
    print(f"\nAggregating results from {results_dir}...")
    combined_df = aggregate_results(results_dir, args.pattern)
    
    if combined_df.empty:
        print("No results to process. Exiting.")
        return
    
    print(f"Total experiments: {len(combined_df)}")
    
    # Create summary table
    print("\nCreating summary table...")
    summary_df = create_summary_table(combined_df)
    
    # Save summary CSV
    summary_csv = results_dir / f"{args.output}_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary table saved to: {summary_csv}")
    
    # Also aggregate all CV long results if they exist
    print("\nLooking for CV long-format results...")
    long_files = list(results_dir.glob("*_cv_long.csv"))
    
    if long_files:
        print(f"Found {len(long_files)} CV long-format files")
        long_dfs = []
        for file in long_files:
            df = pd.read_csv(file)
            long_dfs.append(df)
        
        combined_long = pd.concat(long_dfs, ignore_index=True)
        long_csv = results_dir / f"{args.output}_long.csv"
        combined_long.to_csv(long_csv, index=False)
        print(f"Long-format results saved to: {long_csv}")
    
    # Generate markdown table
    print("\nGenerating markdown table...")
    md_table = format_results_table(summary_df)
    
    md_file = results_dir / f"{args.output}_table.md"
    with open(md_file, 'w') as f:
        f.write("# IMDB Sentiment Classification Results\n\n")
        f.write(md_table)
        f.write("\n")
    
    print(f"Markdown table saved to: {md_file}")
    
    # Print table to console
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("Markdown Table Preview")
    print("="*80)
    print(md_table)
    
    print("\n" + "="*80)
    print("Table generation completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
