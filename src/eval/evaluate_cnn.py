"""
CNN Model Evaluation Script

This module evaluates trained CNN models on test datasets and computes comprehensive metrics.
It handles:
- Loading trained model checkpoints
- Evaluating on multiple test sets (PlantVillage in-domain, PlantDoc cross-domain)
- Computing detailed metrics: Accuracy, Precision, Recall, F1 (Macro/Micro/Weighted)
- Generating per-class classification reports
- Saving results to CSV for comparison

Usage:
    python src/eval/evaluate_cnn.py --model-path checkpoints/cnn_baseline_mobilenet_v3_small.pt --model-name mobilenet_v3_small
    python src/eval/evaluate_cnn.py --model-name all  # Evaluate all available models
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
import sys
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from tqdm import tqdm

# Add project root to path BEFORE importing local modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Load helpers for data and model loading
from src.utils.transformations import get_default_transforms
from src.utils.loader_cnn import get_test_dataloader
from src.utils.baseline_models_cnn import get_model


def evaluate(model, loader, device):
    """
    Runs inference on a dataset and collects predictions for metric computation.

    This function:
    - Sets the model to evaluation mode (disables dropout, freezes batch norm)
    - Disables gradient computation for memory efficiency
    - Iterates through the test dataset with a progress bar
    - Collects all predictions and ground truth labels

    Args:
        model: Trained PyTorch model to evaluate
        loader: DataLoader containing test data
        device: Device to run inference on (cpu/cuda/mps)

    Returns:
        Tuple of (all_labels, all_preds) - lists of ground truth labels and predictions
    """
    # Freeze model (no dropout, fixed weights)
    model.eval()

    # Initialize lists for predictions and labels
    all_preds = []
    all_labels = []

    # Disable gradient calculation for efficiency
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Append to lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Evaluate CNN")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to .pt checkpoint"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["mobilenet_v3_small", "efficientnet_b0", "vit_base_patch16_224"],
    )
    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-file", type=str, default="outputs/evaluation_results.csv"
    )

    # Parse arguments
    args = parser.parse_args()

    # Device switching to utilize mps/gpu if available, otherwise use CPU
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = get_model(args.model_name, num_classes=26, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Define test sets
    test_sets = {
        "PV_Test": Path(args.splits_dir) / "pv_test.csv",
        "PlantDoc_Test": Path(args.splits_dir) / "plantdoc_test_mapped.csv",
    }

    results = []

    # Evaluate on each test set
    for name, csv_path in test_sets.items():
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        print(f"\nEvaluating on {name}...")
        # Get appropriate transforms
        _, _, transform_test = get_default_transforms(
            model_name=args.model_name, image_size=224
        )

        # Create DataLoader
        loader = get_test_dataloader(
            csv_path, root_dir=args.data_dir, batch_size=args.batch_size, transforms=transform_test
        )

        # Evaluate model
        y_true, y_pred = evaluate(model, loader, device)

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class report
        report = classification_report(y_true, y_pred, zero_division=0)

        # Save per-class report
        class_report_path = (
            Path(args.output_file).parent / f"report_{args.model_name}_{name}.txt"
        )
        with open(class_report_path, "w") as f:
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Test Set: {name}\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        print(f"  --> Saved per-class report to {class_report_path}")

        # Print results
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  Precision:   {precision:.4f} (weighted)")
        print(f"  Recall:      {recall:.4f} (weighted)")
        print(f"  F1 (Macro):  {f1_macro:.4f}")
        print(f"  F1 (Micro):  {f1_micro:.4f}")
        print(f"  F1 (Weighted):   {f1_weighted:.4f}")

        results.append(
            {
                "model": args.model_name,
                "checkpoint": Path(args.model_path).name,
                "test_set": name,
                "accuracy": acc,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
            }
        )

    # Append to results file
    out_path = Path(args.output_file)
    df = pd.DataFrame(results)

    # Append to existing file or create new
    if out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
