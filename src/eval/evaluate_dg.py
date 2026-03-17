"""
Domain Generalization Model Evaluation Script

Evaluates DG models trained with contrastive learning and class/style disentanglement.

Features:
- Loads SubspaceDGModel checkpoints
- Evaluates on multiple datasets (PlantVillage, PlantDoc)
- Computes Accuracy, Precision, Recall, F1
- Saves per-class classification reports
- Appends results to CSV for experiment comparison

Usage:
python src/eval/evaluate_dg.py \
    --model-path checkpoints/vit_dg_best.pt \
    --model-name vit_base_patch16_224
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm

import timm
from timm.data import resolve_model_data_config, create_transform

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.dataloaders import get_test_dataloader
from src.utils.subspace_factorization import SubspaceDGModel


def evaluate(model, loader, device):
    """
    Run inference and collect predictions.
    """

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):

            images = images.to(device)

            # Forward pass
            _, _, _, class_logits, _ = model(images)

            _, preds = torch.max(class_logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds


def main():

    parser = argparse.ArgumentParser(description="Evaluate DG model")

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        choices=["mobilenet_v3_small", "efficientnet_b0", "vit_base_patch16_224", "cct_14_7x2_224", "swin_base_patch4_window7_224", "maxvit_base_tf_224"],
    )

    parser.add_argument("--splits-dir", type=str, default="data/splits")
    parser.add_argument("--data-dir", type=str, default=".")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/evaluation_results_dg.csv",
    )

    args = parser.parse_args()

    # Device selection
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -----------------------
    # Load model
    # -----------------------

    print(f"Loading model from {args.model_path}")

    checkpoint = torch.load(args.model_path, map_location=device)

    model = SubspaceDGModel(
        backbone_name=args.model_name,
        num_classes=26,
        num_styles=5,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # -----------------------
    # Setup transforms
    # -----------------------

    timm_model_name = "mobilenetv3_small_100" if args.model_name == "mobilenet_v3_small" else args.model_name
    backbone = timm.create_model(timm_model_name, pretrained=False)

    data_config = resolve_model_data_config(backbone)

    transform_test = create_transform(
        **data_config,
        is_training=False,
    )

    # -----------------------
    # Define test datasets
    # -----------------------

    test_sets = {
        "PV_Test": Path(args.splits_dir) / "pv_test.csv",
        "PlantDoc_Test": Path(args.splits_dir) / "plantdoc_test_mapped.csv",
    }

    results = []

    # -----------------------
    # Evaluate
    # -----------------------

    for name, csv_path in test_sets.items():

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Skipping.")
            continue

        print(f"\nEvaluating on {name}")

        loader = get_test_dataloader(
            csv_path,
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            transforms=transform_test,
        )

        y_true, y_pred = evaluate(model, loader, device)

        # -----------------------
        # Metrics
        # -----------------------

        acc = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        recall = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        f1_macro = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )

        f1_micro = f1_score(
            y_true, y_pred, average="micro", zero_division=0
        )

        f1_weighted = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )

        # Save report
        report_path = (
            Path(args.output_file).parent
            / f"report_dg_{args.model_name}_{name}.txt"
        )

        with open(report_path, "w") as f:

            f.write(f"Model: {args.model_name}\n")
            f.write(f"Test Set: {name}\n")
            f.write("=" * 60 + "\n")
            f.write(report)

        print(f"Saved report → {report_path}")

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Micro: {f1_micro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")

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

    # -----------------------
    # Save results
    # -----------------------

    out_path = Path(args.output_file)

    df = pd.DataFrame(results)

    if out_path.exists():
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()