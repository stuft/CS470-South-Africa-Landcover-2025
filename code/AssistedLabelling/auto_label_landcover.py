import os
import pandas as pd
from fastai.vision.all import *

# ==== CONFIGURATION SECTION ====
CSV_PATH = "labeled_with_model.csv"   # your cleaned (# of rows) CSV
IMAGE_DIR = "og_shp_imagesCopy"                # <-- change if your PNG folder has a different name
LABEL_COL = "FinalClass"          # column with your human labels

# We don't have a direct filename column; we build filenames from Sample_num
IMAGE_COL = None
SAMPLE_COL = "Sample_num"         # exact column name in your CSV

def row_to_filename(row):
    # Sample_num = 1    -> "1.png"
    # Sample_num = 2039 -> "2039.png"
    return f"{int(row[SAMPLE_COL])}.png"

# Training settings
VALID_PCT = 0.2        # 20% of labeled data used for validation
EPOCHS = 5             # you can bump this later if you want
MODEL_NAME = "resnet18"  # backbone model: "resnet18" or "resnet34"

# Autolabel settings
CONFIDENCE_THRESHOLD = 0.8  # only auto-fill labels when model confidence >= this

OUTPUT_CSV = "labeled_with_model.csv"
# ================================


def build_df_with_filenames(df):
    """Ensure we have a 'fname' column with full image paths."""
    if IMAGE_COL is not None:
        if IMAGE_COL not in df.columns:
            raise ValueError(f"IMAGE_COL '{IMAGE_COL}' not found in CSV columns: {list(df.columns)}")
        df["fname"] = df[IMAGE_COL].astype(str)
    else:
        if SAMPLE_COL not in df.columns:
            raise ValueError(
                f"SAMPLE_COL '{SAMPLE_COL}' not found in CSV columns: {list(df.columns)}. "
                f"Edit SAMPLE_COL or add such a column."
            )
        df["fname"] = df.apply(row_to_filename, axis=1)

    # Prepend IMAGE_DIR to get full paths
    df["fname"] = df["fname"].apply(lambda x: os.path.join(IMAGE_DIR, str(x)))
    return df


def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Ensure label column exists
    if LABEL_COL not in df.columns:
        raise ValueError(f"LABEL_COL '{LABEL_COL}' not found in CSV columns: {list(df.columns)}")

    # Build filename paths
    df = build_df_with_filenames(df)

    # Split labeled vs unlabeled
    labeled_mask = df[LABEL_COL].notna() & (df[LABEL_COL].astype(str).str.strip() != "")
    df_labeled = df[labeled_mask].copy()
    df_unlabeled = df[~labeled_mask].copy()

    print(f"Total rows: {len(df)}")
    print(f"Labeled rows: {len(df_labeled)}")
    print(f"Unlabeled rows: {len(df_unlabeled)}")

    if len(df_labeled) == 0:
        raise ValueError("No labeled rows found in CSV. Check LABEL_COL and cleaned CSV.")

    # Check that files exist for labeled data; drop any missing
    missing_files = [f for f in df_labeled["fname"] if not os.path.isfile(f)]
    if missing_files:
        print(f"WARNING: {len(missing_files)} labeled image files were not found. Dropping those rows.")
        print("First few missing examples:")
        for f in missing_files[:10]:
            print("  MISSING:", f)
        df_labeled = df_labeled[~df_labeled["fname"].isin(missing_files)]

    print(f"Labeled rows after dropping missing images: {len(df_labeled)}")

    if len(df_labeled) == 0:
        raise ValueError("After dropping missing-image rows, no labeled data remains. Cannot train model.")

    # Create DataLoaders from labeled set
    print("Building DataLoaders for training...")

    dls = ImageDataLoaders.from_df(
        df_labeled,
        fn_col="fname",
        label_col=LABEL_COL,
        valid_pct=VALID_PCT,
        seed=42,
        item_tfms=Resize(224),
        bs=32
    )

    print("Classes:", dls.vocab)

    # Build and train model
    print(f"Training model ({MODEL_NAME}) for {EPOCHS} epochs...")
    if MODEL_NAME == "resnet18":
        arch = resnet18
    elif MODEL_NAME == "resnet34":
        arch = resnet34
    else:
        raise ValueError("Unsupported MODEL_NAME. Use 'resnet18' or 'resnet34'.")

    learn = vision_learner(dls, arch, metrics=accuracy)
    learn.fine_tune(EPOCHS)

    # Predict on unlabeled rows (if any)
    if len(df_unlabeled) > 0:
        print("Predicting labels for unlabeled rows...")
        preds = []
        probs = []

        for idx, row in df_unlabeled.iterrows():
            img_path = row["fname"]
            if not os.path.isfile(img_path):
                print(f"WARNING: Unlabeled image file missing: {img_path}. Skipping.")
                preds.append(None)
                probs.append(None)
                continue

            try:
                pred_class, pred_idx, pred_probs = learn.predict(PILImage.create(img_path))
            except Exception as e:
                print(f"WARNING: Error predicting for {img_path}: {e}")
                preds.append(None)
                probs.append(None)
                continue

            preds.append(str(pred_class))
            probs.append(float(pred_probs[pred_idx]))

        df_unlabeled["ModelClass"] = preds
        df_unlabeled["ModelConf"] = probs

    else:
        print("No unlabeled rows found. Nothing to predict.")
        df_unlabeled["ModelClass"] = []
        df_unlabeled["ModelConf"] = []

    # Get model predictions on labeled rows (for analysis)
    print("Getting model predictions on labeled rows (for analysis)...")
    labeled_preds = []
    labeled_probs = []

    for idx, row in df_labeled.iterrows():
        img_path = row["fname"]
        pred_class, pred_idx, pred_probs = learn.predict(PILImage.create(img_path))
        labeled_preds.append(str(pred_class))
        labeled_probs.append(float(pred_probs[pred_idx]))

    df_labeled["ModelClass"] = labeled_preds
    df_labeled["ModelConf"] = labeled_probs

    # Combine back
    df_combined = pd.concat([df_labeled, df_unlabeled], ignore_index=True)

    # Create a "FinalClass_filled" column:
    # - if human label exists -> use it
    # - else if ModelConf >= threshold -> use ModelClass
    # - else leave blank
    def fill_label(row):
        human = str(row[LABEL_COL]).strip()
        if human and human.lower() != "nan":
            return human
        mc = row.get("ModelClass", None)
        conf = row.get("ModelConf", None)
        if mc is not None and conf is not None and conf >= CONFIDENCE_THRESHOLD:
            return mc
        return ""

    df_combined["FinalClass_filled"] = df_combined.apply(fill_label, axis=1)

    # Save to CSV
    print(f"Saving results to {OUTPUT_CSV}...")
    df_combined.to_csv(OUTPUT_CSV, index=False)
    print("Done.")
    print("Columns added:")
    print("  - fname (image path)")
    print("  - ModelClass (model prediction)")
    print("  - ModelConf (model confidence for that prediction)")
    print("  - FinalClass_filled (human OR confident model label)")


if __name__ == "__main__":
    main()

