import pandas as pd
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def create_balanced_dataset(data, infection_label_col, organ_dysfunction_label_col, min_samples):
    """
    Create a balanced dataset by downsampling all classes to the same number of samples.
    """
    balanced_dfs = []

    # Downsample InfectionLabel classes
    print("Balancing InfectionLabel...")
    for label in data[infection_label_col].unique():
        label_data = data[data[infection_label_col] == label]
        downsampled = label_data.sample(n=min_samples, random_state=42)
        balanced_dfs.append(downsampled)

    # Downsample OrganDysfunctionLabel classes
    print("Balancing OrganDysfunctionLabel...")
    for label in data[organ_dysfunction_label_col].unique():
        label_data = data[data[organ_dysfunction_label_col] == label]
        downsampled = label_data.sample(n=min_samples, random_state=42)
        balanced_dfs.append(downsampled)

    # Combine all balanced subsets
    balanced_data = pd.concat(balanced_dfs).drop_duplicates().reset_index(drop=True)

    print("Balanced dataset created:")
    print(balanced_data[infection_label_col].value_counts())
    print(balanced_data[organ_dysfunction_label_col].value_counts())

    return balanced_data

def balance_classes(df, label_col, target_count=50000):
    """
    Balance classes in the dataset by sampling to match the least frequent class.
    """
    balanced_dfs = []
    for label in df[label_col].unique():
        class_data = df[df[label_col] == label]
        if len(class_data) > target_count:
            class_data = class_data.sample(n=target_count, random_state=42)
        balanced_dfs.append(class_data)
    return pd.concat(balanced_dfs, ignore_index=True)

def stratified_split(df, stratify_cols, test_size=0.2):
    """
    Perform stratified train-test split based on specified columns.
    """
    train, temp = train_test_split(df, test_size=test_size, stratify=df[stratify_cols], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp[stratify_cols], random_state=42)
    return train, val, test

def compute_class_weights(df, label_col):
    """
    Compute class weights for imbalanced datasets.
    """
    classes = np.unique(df[label_col])
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=df[label_col])
    return dict(zip(classes, weights))

def save_data(train, val, test, output_dir):
    """
    Save train, validation, and test sets to CSV files.
    """
    print(f"Saving datasets to {output_dir}...")
    train.to_csv(f"{output_dir}/train.csv", index=False)
    val.to_csv(f"{output_dir}/val.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    print("Datasets saved.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset.csv for sepsis prediction")
    parser.add_argument("--input_file", required=True, help="Path to the raw dataset CSV file")
    parser.add_argument("--output_folder", required=True, help="Path to the raw dataset CSV file")
    args = parser.parse_args()
    # Load and preprocess
    data = pd.read_csv(args.input_file)
    shuffled_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Balance classes
    infection_label_col = 'InfectionLabel'
    organ_dysfunction_label_col = 'OrganDysfunctionLabel'
    min_samples = 54586

    # Create balanced dataset
    balanced_data = create_balanced_dataset(shuffled_data, infection_label_col, organ_dysfunction_label_col, min_samples)

    # Stratified split
    train_df, val_df, test_df = stratified_split(balanced_data, stratify_cols=['InfectionLabel', 'OrganDysfunctionLabel', 'SepsisLabel'])
    print(len(train_df), len(val_df), len(test_df))
    # Save processed datasets
    save_data(train_df, val_df, test_df, args.output_folder)
