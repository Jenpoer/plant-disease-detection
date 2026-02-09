"""
Plot training metrics from training_log.csv

Usage:
    python src/utils/plot_training.py --log-file outputs/training_log.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_metrics(log_file, output_dir="outputs/plots", output_name="training_metrics.png"):
    """
    Generate training and validation loss/accuracy plots.
    
    Args:
        log_file: Path to training_log.csv
        output_dir: Directory to save plots
    """
    # Read training log
    df = pd.read_csv(log_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss
    ax1.plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(df['epoch'], df['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    ax2.plot(df['epoch'], df['train_acc'], 'b-o', label='Train Acc', linewidth=2, markersize=6)
    ax2.plot(df['epoch'], df['val_acc'], 'r-s', label='Val Acc', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])  # Accuracy is 0-1
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plot
    plot_path = output_path / output_name
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved training plot to {plot_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total Epochs: {len(df)}")
    print(f"Best Val Accuracy: {df['val_acc'].max():.4f} (Epoch {df['val_acc'].idxmax() + 1})")
    print(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"Final Val Loss: {df['val_loss'].iloc[-1]:.4f}")
    print(f"Final Train Acc: {df['train_acc'].iloc[-1]:.4f}")
    print(f"Final Val Acc: {df['val_acc'].iloc[-1]:.4f}")
    
    # Check for overfitting
    final_gap = df['train_acc'].iloc[-1] - df['val_acc'].iloc[-1]
    if final_gap > 0.1:
        print(f"\nWarning: Possible overfitting detected (train-val gap: {final_gap:.4f})")
    else:
        print(f"\nTrain-val gap is healthy: {final_gap:.4f}")
    
    print("="*60)
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--log-file", type=str, default="outputs/training_log.csv",
                        help="Path to training_log.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/plots",
                        help="Directory to save plots")
    parser.add_argument("--output-name", type=str, default="training_metrics.png",
                        help="Filename for the output plot image")
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}")
        print("Please run training first to generate the log file.")
        return
    
    plot_training_metrics(args.log_file, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()
