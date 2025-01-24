import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from utils import validate_preprocessed_data


def preprocess_dataset(input_file, temp_output_file, final_output_file, chunk_size=10000):
    """
    Preprocess the dataset for sepsis prediction and multi-task learning in chunks.
    Handles missing values, label engineering, and outlier management in chunks.
    Ensures septic cases are preserved, and outliers are handled without removing critical labels.
    """
    # Validate chunk_size
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("'chunk_size' must be an integer >= 1")

    if os.path.exists(final_output_file):
        print(f"Final output file {final_output_file} already exists. Loading preprocessed data...")
        return pd.read_csv(final_output_file)

    print("Processing raw data in chunks...")

    # Remove temp output file if it exists
    if os.path.exists(temp_output_file):
        os.remove(temp_output_file)

    # Calculate total number of rows for progress estimation
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract 1 for header
    num_chunks = (total_rows // chunk_size) + 1

    # Process chunks
    for i, chunk in enumerate(
            tqdm(pd.read_csv(input_file, chunksize=chunk_size), total=num_chunks, desc="Processing chunks")):

        # Handle missing values
        chunk.sort_values(by=["Patient_ID", "ICULOS"], inplace=True)  # Ensure time ordering
        chunk.ffill(inplace=True)  # Forward fill for time-series continuity
        chunk.bfill(inplace=True)  # Backward fill for edge cases
        chunk.fillna(chunk.mean(), inplace=True)

        # Engineer labels
        chunk['InfectionLabel'] = np.where(
            ((chunk['WBC'] > 15) | (chunk['WBC'] < 3)) |
            (chunk['Glucose'] > 250) |
            (chunk['HR'] > 100) |
            (chunk['Temp'] > 39.0) | (chunk['Temp'] < 35.0) |
            (chunk['Resp'] > 25) |
            (chunk['Lactate'] > 3.0),
            1, 0
        )

        # Kidney Dysfunction
        chunk['KidneyDysfunction'] = (chunk['Creatinine'] > 1.5) | (chunk['BUN'] > 25)

        # Liver Dysfunction
        chunk['LiverDysfunction'] = (chunk['Bilirubin_total'] > 2.0) | (chunk['AST'] > 50)

        # Cardiovascular Dysfunction
        chunk['CardioDysfunction'] = (chunk['SBP'] < 80) | (chunk['MAP'] < 60)

        # Respiratory Dysfunction
        # FiO2 is present but PaO2 is not, so only FiO2-based condition can be used
        chunk['RespDysfunction'] = (chunk['FiO2'] > 0.8)

        # Combine all organ dysfunctions
        chunk['OrganDysfunctionLabel'] = (
                chunk['KidneyDysfunction'] |
                chunk['LiverDysfunction'] |
                chunk['CardioDysfunction'] |
                chunk['RespDysfunction']
        ).astype(int)

        # Exclude septic cases from implausible condition filtering
        septic_mask = chunk['SepsisLabel'] == 1
        implausible_conditions = (
                (chunk['Temp'] < 20) |  # Temperature too low
                (chunk['SBP'] < 10)  # Systolic blood pressure too low
        )
        # Retain septic rows while removing implausible non-septic rows
        chunk = chunk[~(implausible_conditions & ~septic_mask)]
        # Save the processed chunk to a temporary file
        if i == 0:
            chunk.to_csv(temp_output_file, mode='w', index=False, header=True)
        else:
            chunk.to_csv(temp_output_file, mode='a', index=False, header=False)

    print("Chunk processing completed. Combining chunks...")

    # Load combined data
    df = pd.read_csv(temp_output_file)
    print(f"Septic cases: {len(df[df['SepsisLabel'] == 1])}")

    # Handle outliers globally for non-septic rows
    print("Handling outliers globally...")
    numeric_cols = df.select_dtypes(include='number').columns
    critical_features = ['Temp', 'HR', 'WBC']
    non_critical_cols = [col for col in numeric_cols if col not in critical_features]

    non_septic_df = df[df['SepsisLabel'] == 0]
    septic_df = df[df['SepsisLabel'] == 1]

    # Apply global outlier handling to non-septic data
    for col in non_critical_cols:
        Q1 = non_septic_df[col].quantile(0.25)
        Q3 = non_septic_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        non_septic_df.loc[:, col] = np.clip(non_septic_df[col], lower_bound, upper_bound)
    # Concatenate septic and non-septic data
    df = pd.concat([septic_df, non_septic_df], ignore_index=True)
    print(f"Septic cases retained: {len(df[df['SepsisLabel'] == 1])}")

    print("Checking for columns with excessive missing values...")
    missing_threshold = 0.5
    cols_to_drop = df.columns[df.isnull().mean() > missing_threshold]
    if not cols_to_drop.empty:
        print(f"Dropping columns: {list(cols_to_drop)}")
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        print("No columns to drop.")
    print(df)
    print(len(df))
    if 'Unnamed: 0' or 'Patient_ID' in df.columns:
        df.drop(columns=['Unnamed: 0', 'Patient_ID'], inplace=True)
    # Save final preprocessed data
    print("Saving final preprocessed data...")
    df.to_csv(final_output_file, index=False)
    print(f"Preprocessed data saved to {final_output_file}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess dataset.csv for sepsis prediction")
    parser.add_argument("--input_file", required=True, help="Path to the raw dataset CSV file")
    parser.add_argument("--temp_output_file", required=True, help="Path to save the temporarily processed dataset")
    parser.add_argument("--final_output_file", required=True, help="Path to save the final preprocessed dataset")
    args = parser.parse_args()

    res = preprocess_dataset(args.input_file, args.temp_output_file, args.final_output_file, chunk_size=10000)
    print(res.columns)
    validate_preprocessed_data(res)
