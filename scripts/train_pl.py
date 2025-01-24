import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # Optional callbacks
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np  # For handling numerical operations
import os  # For handling paths
import sys  # For adding custom paths

# Custom imports from your project
sys.path.insert(0, "D:\\Clinical_data_prediction")  # Adjust the path to your project directory
from models.multitask_model import MultiTaskModel
from utils import calculate_class_weights  # Ensure you have this utility script


def load_datasets(train_file, val_file, test_file):
    """
    Load preprocessed train, validation, and test datasets.
    """
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    return train, val, test



# Example usage
if __name__ == "__main__":
    # Prepare data
    train_file = "./data/processed/train.csv"
    val_file = "./data/processed/val.csv"
    test_file = "./data/processed/test.csv"
    output_dir = "./models"


    # Load datasets
    train, val, test = load_datasets(train_file, val_file, test_file)
    train = train.dropna()
    val = val.dropna()
    test = test.dropna()
    # Extract features and labels
    feature_cols = [col for col in train.columns if
                    col not in ['InfectionLabel', 'OrganDysfunctionLabel', 'SepsisLabel']]
    X_train, y_train_infection, y_train_organ, y_train_sepsis = train[feature_cols], train['InfectionLabel'], train[
        'OrganDysfunctionLabel'], train['SepsisLabel']
    X_val, y_val_infection, y_val_organ, y_val_sepsis = val[feature_cols], val['InfectionLabel'], val[
        'OrganDysfunctionLabel'], val['SepsisLabel']
    X_train = X_train.astype(
        {col: 'int32' for col in ['KidneyDysfunction', 'LiverDysfunction', 'CardioDysfunction', 'RespDysfunction']})
    X_val = X_val.astype(
        {col: 'int32' for col in ['KidneyDysfunction', 'LiverDysfunction', 'CardioDysfunction', 'RespDysfunction']})

    infection_weights = calculate_class_weights(y_train_infection)
    organ_weights = calculate_class_weights(y_train_organ)

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(y_train_infection.values, dtype=torch.float32),
        torch.tensor(y_train_organ.values, dtype=torch.float32),
        torch.tensor(y_train_sepsis.values, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(y_val_infection.values, dtype=torch.float32),
        torch.tensor(y_val_organ.values, dtype=torch.float32),
        torch.tensor(y_val_sepsis.values, dtype=torch.float32)

    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        dirpath=output_dir,  # Directory to save models
        filename="best_model-{epoch:02d}-{val_loss:.2f}",  # Naming convention
        save_top_k=1,  # Save only the best model
        mode="min"  # Minimize validation loss
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=5,  # Number of epochs to wait
        mode="min"  # Minimize validation loss
    )
    model = MultiTaskModel(
        input_dim=X_train.shape[1],
        hidden_dim=64,
        infection_class_weights=infection_weights.to(device),
        organ_class_weights=organ_weights.to(device),
        lr=0.001
    )

    # Logger
    logger = TensorBoardLogger("tb_logs", name="multi_task_model")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        log_every_n_steps=5,
        accelerator="cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # Train
    trainer.fit(model, train_loader, val_loader)
    trainer.logger.experiment.add_graph(model, next(iter(train_loader))[0])
