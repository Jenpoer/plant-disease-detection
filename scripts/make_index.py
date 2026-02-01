# scripts/make_index.py
"""
M1: Index datasets for ViT plant disease project.

Creates CSV indexes for PlantVillage and PlantDoc datasets, plus a dataset summary.
"""

from pathlib import Path
import pandas as pd


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def index_plantvillage(base_path: Path, project_root: Path) -> pd.DataFrame:
    records = []
    if not base_path.exists():
        print(f"Warning: PlantVillage path not found: {base_path}")
        return pd.DataFrame(columns=["dataset", "split", "raw_label", "filepath_rel"])

    for class_folder in base_path.iterdir():
        if not class_folder.is_dir():
            continue

        raw_label = class_folder.name
        for img_file in class_folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTENSIONS:
                rel_path = img_file.relative_to(project_root).as_posix()
                records.append(
                    {
                        "dataset": "plantvillage",
                        "split": "all",
                        "raw_label": raw_label,
                        "filepath_rel": rel_path,
                    }
                )

    return pd.DataFrame(records)


def index_plantdoc(base_path: Path, project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def index_split(split_name: str) -> pd.DataFrame:
        records = []
        split_path = base_path / split_name

        if not split_path.exists():
            print(f"Warning: PlantDoc {split_name} path not found: {split_path}")
            return pd.DataFrame(columns=["dataset", "split", "raw_label", "filepath_rel"])

        for class_folder in split_path.iterdir():
            if not class_folder.is_dir():
                continue

            raw_label = class_folder.name
            for img_file in class_folder.iterdir():
                if img_file.is_file() and img_file.suffix.lower() in VALID_EXTENSIONS:
                    rel_path = img_file.relative_to(project_root).as_posix()
                    records.append(
                        {
                            "dataset": "plantdoc",
                            "split": split_name,
                            "raw_label": raw_label,
                            "filepath_rel": rel_path,
                        }
                    )

        return pd.DataFrame(records)

    train_df = index_split("train")
    test_df = index_split("test")
    return train_df, test_df


def generate_summary(pv: pd.DataFrame, pd_train: pd.DataFrame, pd_test: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append("=" * 80)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 80)
    lines.append("")

    def block(title: str, df: pd.DataFrame) -> None:
        lines.append(title)
        lines.append("-" * 40)
        if df.empty:
            lines.append("No data found")
            lines.append("")
            return

        lines.append(f"Total files: {len(df):,}")
        lines.append(f"Unique labels: {df['raw_label'].nunique()}")
        lines.append("")
        lines.append("Top 10 labels by file count:")
        for label, count in df["raw_label"].value_counts().head(10).items():
            lines.append(f"  {label}: {count:,}")
        lines.append("")

    block("PLANTVILLAGE", pv)
    block("PLANTDOC - TRAIN", pd_train)
    block("PLANTDOC - TEST", pd_test)

    lines.append("=" * 80)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Summary written to {output_path}")


def main() -> None:
    project_root = Path.cwd()
    data_raw = project_root / "data" / "raw"
    data_interim = project_root / "data" / "interim"
    outputs_dir = project_root / "outputs"

    data_interim.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Indexing datasets...")

    pv_path = data_raw / "plantvillage"
    pd_path = data_raw / "plantdoc"

    print("  → PlantVillage...")
    pv_df = index_plantvillage(pv_path, project_root)

    print("  → PlantDoc train/test...")
    pd_train_df, pd_test_df = index_plantdoc(pd_path, project_root)

    print("\nWriting index files...")
    pv_df.to_csv(data_interim / "plantvillage_index.csv", index=False)
    print(f"  ✓ plantvillage_index.csv ({len(pv_df):,} rows)")

    pd_train_df.to_csv(data_interim / "plantdoc_train_index.csv", index=False)
    print(f"  ✓ plantdoc_train_index.csv ({len(pd_train_df):,} rows)")

    pd_test_df.to_csv(data_interim / "plantdoc_test_index.csv", index=False)
    print(f"  ✓ plantdoc_test_index.csv ({len(pd_test_df):,} rows)")

    print("\nGenerating summary report...")
    generate_summary(pv_df, pd_train_df, pd_test_df, outputs_dir / "dataset_summary.txt")

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
