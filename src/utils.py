"""
Utility functions for configuration, logging, and results management.
"""
import os
import json
import yaml
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, loads default config.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / "configs" / "default_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def save_results_csv(results: pd.DataFrame, filename: str, results_dir: Optional[str] = None) -> str:
    """
    Save results DataFrame to CSV.
    
    Args:
        results: Results DataFrame
        filename: Name of the CSV file
        results_dir: Directory to save results. If None, uses default reports/results/
    
    Returns:
        Path to saved file
    """
    if results_dir is None:
        results_dir = str(get_project_root() / "reports" / "results")
    
    ensure_dir(results_dir)
    filepath = os.path.join(results_dir, filename)
    results.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    return filepath


def load_results_csv(filename: str, results_dir: Optional[str] = None) -> pd.DataFrame:
    """Load results DataFrame from CSV."""
    if results_dir is None:
        results_dir = str(get_project_root() / "reports" / "results")
    
    filepath = os.path.join(results_dir, filename)
    return pd.read_csv(filepath)


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_environment_info() -> Dict[str, str]:
    """
    Get information about the current environment.
    
    Returns:
        Dictionary with environment information
    """
    import platform
    import sys
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    except ImportError:
        cuda_available = False
        cuda_version = "N/A"
        gpu_name = "N/A"
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": str(cuda_available),
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "timestamp": get_timestamp()
    }


def format_results_table(df: pd.DataFrame, metric_cols: list = None) -> str:
    """
    Format results DataFrame as markdown table.
    
    Args:
        df: Results DataFrame
        metric_cols: List of metric columns to include
    
    Returns:
        Markdown formatted table string
    """
    if metric_cols is None:
        metric_cols = ['accuracy', 'f1', 'precision', 'recall']
    
    # Create a formatted version for markdown
    md_lines = []
    md_lines.append("| Embedding | Variant | Model | " + " | ".join([f"{m.title()}" for m in metric_cols]) + " | Runtime (s) |")
    md_lines.append("|" + "---|" * (3 + len(metric_cols) + 1))
    
    for _, row in df.iterrows():
        values = [
            row.get('embedding_type', 'N/A'),
            row.get('embedding_variant', 'N/A'),
            row.get('model_type', 'N/A')
        ]
        
        for metric in metric_cols:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            if mean_col in row and std_col in row:
                values.append(f"{row[mean_col]:.4f} ± {row[std_col]:.4f}")
            elif metric in row:
                values.append(f"{row[metric]:.4f}")
            else:
                values.append("N/A")
        
        # Runtime
        if 'runtime_mean' in row and 'runtime_std' in row:
            values.append(f"{row['runtime_mean']:.2f} ± {row['runtime_std']:.2f}")
        elif 'runtime' in row:
            values.append(f"{row['runtime']:.2f}")
        else:
            values.append("N/A")
        
        md_lines.append("| " + " | ".join(str(v) for v in values) + " |")
    
    return "\n".join(md_lines)
