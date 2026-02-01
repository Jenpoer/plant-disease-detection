# scripts/build_mapped_splits.py
from pathlib import Path
import pandas as pd

SEED = 42
PV_TRAIN = 0.80
PV_VAL   = 0.10
PV_TEST  = 0.10

def stratified_split(df: pd.DataFrame, label_col: str, seed: int):
    """
    Split df into train/val/test stratified by label_col without sklearn dependency.
    Works by sampling within each class.
    """
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).index  # shuffle index
    df = df.loc[rng].reset_index(drop=True)

    parts = []
    for lab, g in df.groupby(label_col, sort=False):
        n = len(g)
        n_train = int(round(n * PV_TRAIN))
        n_val   = int(round(n * PV_VAL))
        # ensure sum doesn't exceed n
        n_train = min(n_train, n)
        n_val   = min(n_val, n - n_train)
        n_test  = n - n_train - n_val

        g_train = g.iloc[:n_train].copy()
        g_val   = g.iloc[n_train:n_train+n_val].copy()
        g_test  = g.iloc[n_train+n_val:].copy()

        g_train["split_final"] = "train"
        g_val["split_final"]   = "val"
        g_test["split_final"]  = "test"

        parts.append(pd.concat([g_train, g_val, g_test], axis=0))

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out

def main():
    project_root = Path.cwd()
    data_interim = project_root / "data" / "interim"
    data_splits  = project_root / "data" / "splits"
    data_splits.mkdir(parents=True, exist_ok=True)

    # Inputs (your existing outputs)
    pv_idx = pd.read_csv(data_interim / "plantvillage_index.csv")
    pd_tr  = pd.read_csv(data_interim / "plantdoc_train_index.csv")
    pd_te  = pd.read_csv(data_interim / "plantdoc_test_index.csv")

    # label_map (your generated mapping)
    label_map = pd.read_csv(project_root / "src" / "data" / "label_map.csv")

    # Keep only included mappings
    inc = label_map[label_map["include"] == 1].copy()

    # Build canonical_id (stable order by canonical_label)
    canonical_space = (
        inc[["canonical_label"]]
        .drop_duplicates()
        .sort_values("canonical_label")
        .reset_index(drop=True)
    )
    canonical_space["canonical_id"] = range(len(canonical_space))
    canonical_space = canonical_space[["canonical_id", "canonical_label"]]

    # Helper: attach mapping to an index
    def apply_map(index_df: pd.DataFrame, dataset_name: str):
        m = inc[inc["dataset"] == dataset_name][["raw_label", "canonical_label"]].copy()
        out = index_df.merge(m, on="raw_label", how="inner")  # inner = drop excluded/unmapped
        out = out.merge(canonical_space, on="canonical_label", how="left")
        return out

    pv_mapped = apply_map(pv_idx, "plantvillage")
    pd_train_mapped = apply_map(pd_tr, "plantdoc")
    pd_test_mapped  = apply_map(pd_te, "plantdoc")

    # PV: make stratified train/val/test
    pv_split = stratified_split(pv_mapped, label_col="canonical_id", seed=SEED)

    # Write outputs
    canonical_space.to_csv(data_splits / "label_space.csv", index=False)
    pv_mapped.to_csv(data_splits / "plantvillage_mapped.csv", index=False)
    pd_test_mapped.to_csv(data_splits / "plantdoc_test_mapped.csv", index=False)

    pv_split[pv_split["split_final"] == "train"].to_csv(data_splits / "pv_train.csv", index=False)
    pv_split[pv_split["split_final"] == "val"].to_csv(data_splits / "pv_val.csv", index=False)
    pv_split[pv_split["split_final"] == "test"].to_csv(data_splits / "pv_test.csv", index=False)

    # --- Sanity checks ---
    print("=== SANITY ===")
    # 1) Label space size
    print("label_space:", len(canonical_space))
    # 2) PV class coverage per split
    for s in ["train", "val", "test"]:
        sub = pv_split[pv_split["split_final"] == s]
        print(f"PV {s}: n={len(sub)} classes={sub['canonical_id'].nunique()}")
    # 3) Leakage check
    train_paths = set(pv_split[pv_split["split_final"] == "train"]["filepath_rel"])
    val_paths   = set(pv_split[pv_split["split_final"] == "val"]["filepath_rel"])
    test_paths  = set(pv_split[pv_split["split_final"] == "test"]["filepath_rel"])
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    print("No PV leakage across splits ✓")
    # 4) PD only contains known canonical ids
    assert pd_test_mapped["canonical_id"].notna().all()
    print("PD test fully mapped ✓")

    print("\n✓ M1 mapped splits ready in:", data_splits)

if __name__ == "__main__":
    main()
