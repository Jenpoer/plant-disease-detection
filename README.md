# Plant Disease Classification (ViT) — Milestone 1: Data Indexing & Label Harmonization

Milestone 1 (M1) prepares **reproducible, training-ready datasets** for a plant disease classification project. It standardizes two datasets with different label conventions (**PlantVillage** and **PlantDoc**).

## 1. Baseline Data Protocol (Source + Target)

This project evaluates robustness under dataset domain shift:

- **Source domain (PlantVillage, PV):** used for **train / validation / test**
- **Target domain (PlantDoc, PD):** used as a **held-out target test set** for cross-domain evaluation

> Baseline rule: **PlantDoc test is never used for tuning or model selection.**
> 
> (Optional) PD fine-tuning experiments belong to later milestones and do not belong to PD test untouched.

---

## 2. Repository Expectations

### 2.1 Dataset placement

Place raw datasets under `data/raw/` exactly as follows:
```
data/raw/
├── plantvillage/
│   ├── Apple___Apple_scab/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── Apple___Cedar_apple_rust/
│   └── ...
└── plantdoc/
    ├── train/
    │   ├── Apple Scab Leaf/
    │   │   ├── image001.jpg
    │   │   └── ...
    │   ├── Apple rust leaf/
    │   └── ...
    └── test/
        ├── Apple Scab Leaf/
        └── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png` (case-insensitive)

### 2.2 Environment setup

From the **repository root**:
```bash
pip install -r requirements.txt
```

> Note: Your terminal prompt may display the active git branch name (e.g., `m1-data-label-map`). That is not a folder path.

---

## 3. Milestone 1 Outputs (What you get)

M1 produces three categories of artifacts:

1. **Image inventories** (one row per image; disk scan results)
2. **Label harmonization map** (raw label → canonical label)
3. **Training-ready split CSVs** (mapped + filtered + seeded splits)

Downstream training (M2–M5) reads only the split CSVs.

---

## 4. How to Run M1 (Reproducible Pipeline)

Run the following steps in order from the repository root.

### Step 1 — Index datasets (image inventory)
```bash
python scripts/make_index.py
```

**Writes:**

- `data/interim/plantvillage_index.csv`
- `data/interim/plantdoc_train_index.csv`
- `data/interim/plantdoc_test_index.csv`
- `outputs/dataset_summary.txt`

**Schema:** `dataset, split, raw_label, filepath_rel`

---

### Step 2 — Build label mapping (intersection-only 26 classes)
```bash
python scripts/apply_label_map.py
```

**Reads:**

- `data/interim/*.csv`

**Writes:**

- `src/data/label_map.csv`

**Policy:**

- Only the **26 confirmed PV↔PD correspondences** are included (`include=1`)
- All other labels are excluded (`include=0`) with notes

**Schema:** `dataset, raw_label, canonical_label, include, notes`

**Expected console validation:**

- 26 unique canonical labels
- 26 included PV labels
- 26 included PD labels
- 52 included rows total (26 + 26)

---

### Step 3 — Generate mapped split files (training-ready CSVs)
```bash
python scripts/build_mapped_splits.py
```

**Reads:**

- `data/interim/*.csv` (image-level inventory)
- `src/data/label_map.csv` (label-level mapping)

**Produces:**

- frozen label space (`canonical_id` 0–25)
- mapped PV inventory (filtered to shared classes)
- mapped PD target test set (filtered to shared classes)
- seeded, stratified PV splits: train/val/test

Writes to the splits directory configured in the script (commonly `data/splits/`).

---

## 5. Output Locations (Files to expect)

After Steps 1–3, you should have:
```bash
data/interim/
├── plantvillage_index.csv
├── plantdoc_train_index.csv
└── plantdoc_test_index.csv

src/data/
└── label_map.csv

<data_splits_dir>/                # e.g., data/splits/
├── label_space.csv               # canonical_id (0–25) ↔ canonical_label
├── plantvillage_mapped.csv       # PV filtered to 26 shared classes (pre-split)
├── plantdoc_test_mapped.csv      # PD test filtered to shared classes
├── pv_train.csv                  # PV train split (seeded, stratified)
├── pv_val.csv                    # PV val split (seeded, stratified)
└── pv_test.csv                   # PV test split (seeded, stratified)

outputs/
└── dataset_summary.txt
```

If you are unsure where Step 3 wrote the outputs, locate them from repo root:
```bash
find . -maxdepth 4 -name "pv_train.csv" -o -name "plantdoc_test_mapped.csv"
```

---

## 6. Quick Sanity Checks (Optional)

From the folder containing the split files:
```bash
python -c "import pandas as pd; \
print('PV train', len(pd.read_csv('pv_train.csv'))); \
print('PV val', len(pd.read_csv('pv_val.csv'))); \
print('PV test', len(pd.read_csv('pv_test.csv'))); \
print('PD test', len(pd.read_csv('plantdoc_test_mapped.csv')))"
```

---

## 7. Label Harmonization Notes

- **Canonical label format:** `crop__condition`
- **Normalization:** lowercase; spaces/hyphens → underscores
- **Mapping policy:** intersection-only (26 confirmed classes)

(Include your 26-class mapping table here if required by your rubric)

---

## 8. Next Milestones (Context)

- **M2:** CNN baseline training on PV (train/val), evaluate on PV test + PD test
- **M3:** ViT training + evaluation
- **M4:** Robustness improvements (ROI and/or augmentations) + ablations
- **M5:** Reliability layer (calibration + abstain option)