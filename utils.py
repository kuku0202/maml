import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import time
from contextlib import contextmanager

def check_system_requirements():
    """Check if system has required dependencies and hardware"""
    print("System Requirements Check:")
    print("="*30)
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name()}")
            print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("✗ PyTorch not installed")
    
    # Check Transformers
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed")
    
    # Check other dependencies
    dependencies = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn', 'tqdm']
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✓ {dep} available")
        except ImportError:
            print(f"✗ {dep} not installed")


def estimate_memory_usage(dataset_size: int, max_length: int = 512, batch_size: int = 8) -> Dict[str, float]:
    """
    Estimate memory usage for a given dataset configuration
    
    Args:
        dataset_size: Number of samples in dataset
        max_length: Maximum sequence length
        batch_size: Batch size for training
        
    Returns:
        dict: Memory usage estimates in GB
    """
    # Rough estimates based on BERT-base
    token_memory_per_sample = max_length * 4  # bytes (int32)
    model_params = 110e6  # BERT-base parameters
    param_memory = model_params * 4  # bytes (float32)
    
    # Forward pass memory (activations)
    activation_memory_per_sample = max_length * 768 * 4  # hidden_size * bytes
    
    # Batch memory
    batch_memory = batch_size * (token_memory_per_sample + activation_memory_per_sample)
    
    # Total memory estimate
    total_memory = param_memory + batch_memory
    
    return {
        'model_parameters_gb': param_memory / 1e9,
        'batch_memory_gb': batch_memory / 1e9,
        'total_memory_gb': total_memory / 1e9,
        'recommended_gpu_memory_gb': total_memory * 2 / 1e9  # with safety margin
    }


def create_visualization_summary(datasets_info: List[Dict], save_path: str = 'dataset_summary.png'):
    """
    Create visualization summary of multiple datasets
    
    Args:
        datasets_info: List of dictionaries with dataset information
        save_path: Path to save the plot
    """
    if not datasets_info:
        print("No dataset info provided for visualization")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract information
    names = [info['name'] for info in datasets_info]
    sizes = [info['size'] for info in datasets_info]
    means = [info['target_stats']['mean'] for info in datasets_info]
    stds = [info['target_stats']['std'] for info in datasets_info]
    
    # Plot 1: Dataset sizes
    axes[0, 0].bar(range(len(names)), sizes, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Dataset Sizes')
    axes[0, 0].set_xlabel('Datasets')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in names], rotation=45)
    
    # Plot 2: Target means
    axes[0, 1].bar(range(len(names)), means, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Target Mean Values')
    axes[0, 1].set_xlabel('Datasets')
    axes[0, 1].set_ylabel('Mean Target Value')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in names], rotation=45)
    
    # Plot 3: Target standard deviations
    axes[1, 0].bar(range(len(names)), stds, color='salmon', alpha=0.7)
    axes[1, 0].set_title('Target Standard Deviations')
    axes[1, 0].set_xlabel('Datasets')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in names], rotation=45)
    
    # Plot 4: Mean vs Std scatter
    axes[1, 1].scatter(means, stds, s=sizes, alpha=0.6, c=range(len(names)), cmap='viridis')
    axes[1, 1].set_xlabel('Mean Target Value')
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Mean vs Std (bubble size = dataset size)')
    
    # Add text annotations for scatter plot
    for i, name in enumerate(names):
        axes[1, 1].annotate(name[:8], (means[i], stds[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dataset summary visualization saved to {save_path}")


def explore_dataset(csv_path: str, target_col: str = None):
    """
    Explore and visualize a protein dataset
    
    Args:
        csv_path: Path to CSV file
        target_col: Target column name (auto-detected if None)
    """
    df = pd.read_csv(csv_path)
    
    print(f"\nDataset Exploration: {os.path.basename(csv_path)}")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Auto-detect target column if not provided
    if target_col is None:
        from data import _auto_detect_target_helper
        target_col = _auto_detect_target_helper(df)
        if target_col:
            print(f"Auto-detected target column: '{target_col}'")
    
    # Basic statistics
    if target_col and target_col in df.columns:
        print(f"\nTarget Column ({target_col}) Statistics:")
        print(f"Mean: {df[target_col].mean():.4f}")
        print(f"Std: {df[target_col].std():.4f}")
        print(f"Min: {df[target_col].min():.4f}")
        print(f"Max: {df[target_col].max():.4f}")
        print(f"Missing values: {df[target_col].isna().sum()}")
        print(f"Data type: {df[target_col].dtype}")
    
    # Sequence length statistics
    if 'sequence' in df.columns:
        seq_lengths = df['sequence'].str.len()
        print(f"\nSequence Length Statistics:")
        print(f"Mean length: {seq_lengths.mean():.1f}")
        print(f"Min length: {seq_lengths.min()}")
        print(f"Max length: {seq_lengths.max()}")
    
    # Mutation analysis
    mutation_cols = [col for col in df.columns if 'mutation' in col.lower()]
    if mutation_cols:
        print(f"\nMutation Columns: {mutation_cols}")
        for col in mutation_cols:
            non_null = df[col].notna().sum()
            print(f"{col}: {non_null} non-null values ({non_null/len(df)*100:.1f}%)")
    
    return df


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration file for the pipeline"""
    config = {
        "initial_task": "data/initial_task.csv",
        "meta_learning_tasks": [
            "data/meta_task_1.csv",
            "data/meta_task_2.csv",
            "data/meta_task_3.csv"
        ],
        "test_tasks": [
            "data/test_task_1.csv",
            "data/test_task_2.csv"
        ],
        "initial_target_col": None,  # Auto-detect
        "pretrain_epochs": 10,
        "maml_epochs": 50,
        "batch_size": 16,
        "inner_lr": 0.01,
        "meta_lr": 0.001,
        "adaptation_steps": 5,
        "support_size": 16,
        "query_size": 16,
        "max_length": 512,
        "seed": 42,
        "save_dir": "./results",
        "freeze_bert": True,  # For faster training
        "debug": False
    }
    
    with open('example_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Example configuration saved to example_config.json")
    return config


@contextmanager
def timer(description: str):
    """Context manager for timing code blocks"""
    start = time.time()
    print(f"Starting: {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Completed: {description} in {elapsed:.2f}s")


def save_training_metrics(metrics: Dict, save_path: str):
    """Save training metrics to JSON file"""
    # Convert any numpy types to native Python types
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            serializable_metrics[key] = [float(x) for x in value]
        elif isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"Training metrics saved to {save_path}")


def print_model_info(model):
    """Print model information including parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")
    print(f"  Memory estimate: {total_params * 4 / 1e9:.2f} GB (float32)")


def get_gpu_memory_info():
    """Get GPU memory information if CUDA is available"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        
        return {
            'total_gb': total_memory,
            'allocated_gb': allocated,
            'cached_gb': cached,
            'free_gb': total_memory - cached
        }
    except Exception as e:
        return f"Error getting GPU info: {e}"


def optimize_dataloader_settings(dataset_size: int, batch_size: int) -> Dict[str, int]:
    """
    Suggest optimal DataLoader settings based on dataset size and hardware
    
    Args:
        dataset_size: Number of samples in dataset
        batch_size: Batch size for training
        
    Returns:
        dict: Suggested DataLoader settings
    """
    # Suggest number of workers based on CPU count and dataset size
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    
    if dataset_size < 1000:
        num_workers = min(2, cpu_count)
    elif dataset_size < 10000:
        num_workers = min(4, cpu_count)
    else:
        num_workers = min(8, cpu_count)
    
    # Adjust for batch size
    if batch_size < 8:
        num_workers = max(1, num_workers // 2)
    
    settings = {
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': num_workers > 0 and dataset_size > 1000,
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    
    return settings


if __name__ == "__main__":
    print("Protein MAML Utils")
    print("="*20)
    
    check_system_requirements()
    
    # Example memory estimation
    estimates = estimate_memory_usage(dataset_size=1000, max_length=512, batch_size=16)
    print(f"\nMemory Estimates (1000 samples, batch_size=16):")
    for key, value in estimates.items():
        print(f"{key}: {value:.2f} GB")
    
    # Create example config
    create_example_config()
    
    # GPU memory info
    gpu_info = get_gpu_memory_info()
    if isinstance(gpu_info, dict):
        print(f"\nGPU Memory Info:")
        for key, value in gpu_info.items():
            print(f"{key}: {value:.2f} GB")
    else:
        print(f"\nGPU Info: {gpu_info}")