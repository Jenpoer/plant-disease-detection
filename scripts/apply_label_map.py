# scripts/apply_label_map.py
"""
M1: Apply label mapping for ViT plant disease project.

Creates a unified label mapping table enforcing intersection-only policy:
- Only confirmed 26 PV↔PD mappings are included
- Everything else is excluded with clear notes
"""

from pathlib import Path
import pandas as pd
import re


# Confirmed 26 mappings: PV -> PD
CONFIRMED_MAPPINGS = {
    "Apple___Apple_scab": "Apple Scab Leaf",
    "Apple___Cedar_apple_rust": "Apple rust leaf",
    "Apple___healthy": "Apple leaf",
    "Blueberry___healthy": "Blueberry leaf",
    "Cherry___healthy": "Cherry leaf",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray leaf spot",
    "Corn___Common_rust": "Corn rust leaf",
    "Corn___Northern_Leaf_Blight": "Corn leaf blight",
    "Grape___Black_rot": "grape leaf black rot",
    "Grape___healthy": "grape leaf",
    "Peach___healthy": "Peach leaf",
    "Potato___Early_blight": "Potato leaf early blight",
    "Potato___Late_blight": "Potato leaf late blight",
    "Raspberry___healthy": "Raspberry leaf",
    "Soybean___healthy": "Soyabean leaf",
    "Squash___Powdery_mildew": "Squash Powdery mildew leaf",
    "Strawberry___healthy": "Strawberry leaf",
    "Tomato___Bacterial_spot": "Tomato leaf bacterial spot",
    "Tomato___Early_blight": "Tomato Early blight leaf",
    "Tomato___Late_blight": "Tomato leaf late blight",
    "Tomato___Leaf_Mold": "Tomato mold leaf",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria leaf spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato two spotted spider mites leaf",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf yellow virus",
    "Tomato___Tomato_mosaic_virus": "Tomato leaf mosaic virus",
    "Tomato___healthy": "Tomato leaf",
}

# Optional canonical overrides for cleaner naming
CANONICAL_OVERRIDES = {
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato__yellow_leaf_curl_virus",
}

# Specific exclusions with reasons
PV_EXCLUSIONS = {
    "Background_without_leaves": "exclude: background-only (no leaf); out of scope",
    "Cherry___Powdery_mildew": "exclude: not in confirmed PV↔PD mapping",
    "Corn___healthy": "exclude: not in confirmed PV↔PD mapping",
    "Grape___Esca_(Black_Measles)": "exclude: not in confirmed PV↔PD mapping",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "exclude: not in confirmed PV↔PD mapping",
    "Orange___Haunglongbing_(Citrus_greening)": "exclude: not in confirmed PV↔PD mapping",
    "Peach___Bacterial_spot": "exclude: not in confirmed PV↔PD mapping",
    "Pepper,_bell___Bacterial_spot": "exclude: not in confirmed PV↔PD mapping",
    "Pepper,_bell___healthy": "exclude: not in confirmed PV↔PD mapping",
    "Potato___healthy": "exclude: not in confirmed PV↔PD mapping",
    "Strawberry___Leaf_scorch": "exclude: not in confirmed PV↔PD mapping",
    "Tomato___Target_Spot": "exclude: not in confirmed PV↔PD mapping",
    "Apple___Black_rot": "exclude: not in confirmed PV↔PD mapping",
}

PD_EXCLUSIONS = {
    "Bell_pepper leaf": "exclude: not in confirmed PV↔PD mapping",
    "Bell_pepper leaf spot": "exclude: not in confirmed PV↔PD mapping",
}


def _norm_token(s: str) -> str:
    """
    Normalize one token:
    - lowercase
    - remove commas
    - spaces/hyphens -> underscores
    - collapse underscores
    """
    s = s.lower().replace(",", "")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalize_canonical(pv_label: str) -> str:
    """
    Convert PlantVillage label to canonical format: crop__condition
    PV folders are typically: Crop___Condition
    """
    if "___" in pv_label:
        crop, cond = pv_label.split("___", 1)
        return f"{_norm_token(crop)}__{_norm_token(cond)}"
    # Fallback (shouldn't happen)
    return _norm_token(pv_label)


def build_label_map(pv_labels: set, pd_labels: set) -> pd.DataFrame:
    """
    Build label mapping table enforcing intersection-only policy.
    """
    records = []
    
    # Reverse mapping: PD -> canonical (via PV)
    pd_to_canonical = {}
    for pv_raw, pd_raw in CONFIRMED_MAPPINGS.items():
        canonical = CANONICAL_OVERRIDES.get(pv_raw, normalize_canonical(pv_raw))
        pd_to_canonical[pd_raw] = canonical
    
    # Process PlantVillage labels
    for raw_label in sorted(pv_labels):
        if raw_label in CONFIRMED_MAPPINGS:
            # Included mapping
            canonical = CANONICAL_OVERRIDES.get(raw_label, normalize_canonical(raw_label))
            records.append({
                "dataset": "plantvillage",
                "raw_label": raw_label,
                "canonical_label": str(canonical),
                "include": 1,
                "notes": str(""),
            })
        elif raw_label in PV_EXCLUSIONS:
            # Explicit exclusion
            records.append({
                "dataset": "plantvillage",
                "raw_label": raw_label,
                "canonical_label": str(""),
                "include": 0,
                "notes": str(PV_EXCLUSIONS[raw_label]),
            })
        else:
            # Default exclusion
            records.append({
                "dataset": "plantvillage",
                "raw_label": raw_label,
                "canonical_label": str(""),
                "include": 0,
                "notes": str("exclude: not in confirmed PV↔PD mapping"),
            })
    
    # Process PlantDoc labels
    for raw_label in sorted(pd_labels):
        if raw_label in pd_to_canonical:
            # Included mapping
            canonical = pd_to_canonical[raw_label]
            records.append({
                "dataset": "plantdoc",
                "raw_label": raw_label,
                "canonical_label": str(canonical),
                "include": 1,
                "notes": str(""),
            })
        elif raw_label in PD_EXCLUSIONS:
            # Explicit exclusion
            records.append({
                "dataset": "plantdoc",
                "raw_label": raw_label,
                "canonical_label": str(""),
                "include": 0,
                "notes": str(PD_EXCLUSIONS[raw_label]),
            })
        else:
            # Default exclusion
            records.append({
                "dataset": "plantdoc",
                "raw_label": raw_label,
                "canonical_label": str(""),
                "include": 0,
                "notes": str("exclude: not in confirmed PV↔PD mapping"),
            })
    
    df = pd.DataFrame(records)
    
    # Ensure correct dtypes (avoid FutureWarning)
    df["dataset"] = df["dataset"].astype(str)
    df["raw_label"] = df["raw_label"].astype(str)
    df["canonical_label"] = df["canonical_label"].astype(str)
    df["include"] = df["include"].astype(int)
    df["notes"] = df["notes"].astype(str)
    
    return df


def validate_label_map(df: pd.DataFrame) -> None:
    """
    Run validations and print summary statistics.
    """
    print("\n" + "=" * 60)
    print("LABEL MAP VALIDATION")
    print("=" * 60)
    
    total_rows = len(df)
    pv_rows = len(df[df["dataset"] == "plantvillage"])
    pd_rows = len(df[df["dataset"] == "plantdoc"])
    
    included = df[df["include"] == 1]
    pv_included = len(included[included["dataset"] == "plantvillage"])
    pd_included = len(included[included["dataset"] == "plantdoc"])
    
    unique_canonical = included["canonical_label"].nunique()
    
    print(f"\nTotal rows: {total_rows}")
    print(f"  PlantVillage: {pv_rows}")
    print(f"  PlantDoc: {pd_rows}")
    
    print(f"\nIncluded rows (include=1): {len(included)}")
    print(f"  PlantVillage: {pv_included}")
    print(f"  PlantDoc: {pd_included}")
    
    print(f"\nUnique canonical labels (included): {unique_canonical}")
    
    # Assertions/warnings
    print("\n" + "-" * 60)
    print("VALIDATION CHECKS")
    print("-" * 60)
    
    checks_passed = True
    
    if unique_canonical != 26:
        print(f"WARNING: Expected 26 unique canonical labels, got {unique_canonical}")
        checks_passed = False
    else:
        print(f"✓ Unique canonical labels = 26")
    
    if pv_included != 26:
        print(f"WARNING: Expected 26 included PV labels, got {pv_included}")
        checks_passed = False
    else:
        print(f"✓ Included PlantVillage labels = 26")
    
    if pd_included != 26:
        print(f"WARNING: Expected 26 included PD labels, got {pd_included}")
        checks_passed = False
    else:
        print(f"✓ Included PlantDoc labels = 26")
    
    if len(included) != 52:
        print(f"WARNING: Expected 52 total included rows (26+26), got {len(included)}")
        checks_passed = False
    else:
        print(f"✓ Total included rows = 52")
    
    if checks_passed:
        print("\n✓ All validations passed!")
    else:
        print("\n⚠ Some validation checks failed - review above")
    
    print("=" * 60)


def main() -> None:
    project_root = Path.cwd()
    data_interim = project_root / "data" / "interim"
    src_data = project_root / "src" / "data"
    
    # Create output directory
    src_data.mkdir(parents=True, exist_ok=True)
    
    print("Reading index files...")
    
    # Read PlantVillage index
    pv_index = pd.read_csv(data_interim / "plantvillage_index.csv")
    pv_labels = set(pv_index["raw_label"].unique())
    print(f"  → PlantVillage: {len(pv_labels)} unique labels")
    
    # Read PlantDoc indexes (train + test, deduplicate)
    pd_train_index = pd.read_csv(data_interim / "plantdoc_train_index.csv")
    pd_test_index = pd.read_csv(data_interim / "plantdoc_test_index.csv")
    pd_labels = set(pd_train_index["raw_label"].unique()) | set(pd_test_index["raw_label"].unique())
    print(f"  → PlantDoc: {len(pd_labels)} unique labels (train+test deduplicated)")
    
    # ---- sanity checks: confirmed mapping labels must exist in indexes ----
    missing_pv = [k for k in CONFIRMED_MAPPINGS.keys() if k not in pv_labels]
    missing_pd = [v for v in CONFIRMED_MAPPINGS.values() if v not in pd_labels]
    
    if missing_pv:
        print("\n[WARN] The following CONFIRMED PV labels were NOT found in plantvillage_index.csv:")
        for x in missing_pv:
            print("  -", x)
    
    if missing_pd:
        print("\n[WARN] The following CONFIRMED PD labels were NOT found in plantdoc_(train/test)_index.csv:")
        for x in missing_pd:
            print("  -", x)
    
    # Check if exclusions exist
    missing_pv_excl = [k for k in PV_EXCLUSIONS.keys() if k not in pv_labels]
    missing_pd_excl = [k for k in PD_EXCLUSIONS.keys() if k not in pd_labels]
    
    if missing_pv_excl:
        print("\n[WARN] The following PV_EXCLUSIONS labels were not found (possible typos):")
        for x in missing_pv_excl:
            print("  -", x)
    
    if missing_pd_excl:
        print("\n[WARN] The following PD_EXCLUSIONS labels were not found (possible typos):")
        for x in missing_pd_excl:
            print("  -", x)
    
    print("\nBuilding label mapping table...")
    label_map_df = build_label_map(pv_labels, pd_labels)
    
    # Write to CSV
    output_path = src_data / "label_map.csv"
    label_map_df.to_csv(output_path, index=False)
    print(f"\n✓ Label map written to {output_path}")
    print(f"  Total rows: {len(label_map_df)}")
    
    # Validate
    validate_label_map(label_map_df)
    
    print("\n" + "=" * 60)
    print("LABEL MAPPING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()