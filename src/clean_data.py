
from pathlib import Path
import pandas as pd
import numpy as np
import random

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "raw_data" / "MachineLearningCVE"
OUTPUT_DIR = BASE_DIR / "dataset" / "cleaned_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the dataset balanced and manageable
MAX_PER_CLASS = 50000
CHUNK_SIZE = 50000

samples = {
    "BENIGN": [],
    "ATTACK": []
}

counts = {
    "BENIGN": 0,
    "ATTACK": 0
}

files = list(DATA_DIR.glob("*.csv"))
random.seed(42)
random.shuffle(files)

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

print("Starting NetInsight data cleaning...")
print(f"Found {len(files)} CSV files.")

for file in files:
    print(f"\nProcessing: {file.name}")

    for chunk in pd.read_csv(
        file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace"
    ):
        # Remove unwanted spaces from column names
        chunk.columns = chunk.columns.str.strip()

        if "Label" not in chunk.columns:
            continue

        # Clean the original labels
        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        # Separate normal and attack traffic
        chunk["Target"] = np.where(
            chunk["Label"] == "BENIGN", "BENIGN", "ATTACK"
        )

        for category in ["BENIGN", "ATTACK"]:
            remaining = MAX_PER_CLASS - counts[category]

            if remaining <= 0:
                continue

            selected = chunk[chunk["Target"] == category]

            if len(selected) > remaining:
                selected = selected.sample(
                    n=remaining,
                    random_state=42
                )

            if not selected.empty:
                samples[category].append(selected.copy())
                counts[category] += len(selected)

    print("Collected:", counts)

# Combine selected records
if not samples["BENIGN"] or not samples["ATTACK"]:
    raise ValueError("Could not collect both normal and attack records.")

df = pd.concat(
    samples["BENIGN"] + samples["ATTACK"],
    ignore_index=True
)

# Replace infinite values and clean missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert feature columns to numeric values
feature_columns = [
    col for col in df.columns
    if col not in ["Label", "Target"]
]

for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing feature values
df.dropna(subset=feature_columns, inplace=True)

# Shuffle the final dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save cleaned dataset
output_file = OUTPUT_DIR / "netinsight_cleaned.csv"
df.to_csv(output_file, index=False)

print("\nDATA CLEANING COMPLETED!")
print("Final records:", len(df))
print("\nTraffic distribution:")
print(df["Target"].value_counts())
print("\nSaved to:", output_file)

