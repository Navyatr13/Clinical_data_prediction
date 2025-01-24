import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def calculate_class_weights(y_train):
    """
    Calculate class weights for imbalanced datasets.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))


def load_datasets(train_file, val_file, test_file):
    """
    Load preprocessed train, validation, and test datasets.
    """
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)
    test = pd.read_csv(test_file)
    return train, val, test


def train_model(X_train, y_train, model_type='binary', class_weights=None):
    """
    Train a baseline model using Random Forest for binary or multiclass classification.
    """
    if model_type == 'binary':
        model = RandomForestClassifier(class_weight=class_weights, random_state=42)
    elif model_type == 'multiclass':
        model = RandomForestClassifier(class_weight=class_weights, random_state=42)
    else:
        raise ValueError("Invalid model_type. Use 'binary' or 'multiclass'.")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y_true, model_type='binary'):
    """
    Evaluate the model on validation or test data.
    """
    y_pred = model.predict(X)
    if model_type == 'binary':
        y_prob = model.predict_proba(X)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1-Score": f1_score(y_true, y_pred),
            "AUROC": roc_auc_score(y_true, y_prob)
        }
    elif model_type == 'multiclass':
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1-Score (weighted)": f1_score(y_true, y_pred, average='weighted'),
            "Classification Report": classification_report(y_true, y_pred)
        }
    return metrics, y_pred


def save_model(model, output_path):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")


def main():
    # Configuration
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
    X_test, y_test_infection, y_test_organ, y_test_sepsis = test[feature_cols], test['InfectionLabel'], test[
        'OrganDysfunctionLabel'], test['SepsisLabel']

    # Example for InfectionLabel
    infection_class_weights = calculate_class_weights(y_train_infection)
    print("InfectionLabel Class Weights:", infection_class_weights)

    # Example for OrganDysfunctionLabel
    organ_class_weights = calculate_class_weights(y_train_organ)
    print("OrganDysfunctionLabel Class Weights:", organ_class_weights)

    # Disjoint feature sets for Infection and Organ Dysfunction Models
    infection_features = ['WBC', 'Glucose', 'Temp', 'HR', 'Resp']
    organ_features = ['Creatinine', 'Bilirubin_total', 'BUN', 'FiO2', 'SBP', 'MAP']

    # Train models
    print("Training Infection Model...")
    infection_model = train_model(X_train[infection_features], y_train_infection, model_type='binary',
                                  class_weights=infection_class_weights)

    print("Training Organ Dysfunction Model...")
    organ_model = train_model(X_train[organ_features], y_train_organ, model_type='binary',
                              class_weights=organ_class_weights)

    # Evaluate models
    print("Evaluating Infection Model...")
    infection_metrics, y_pred = evaluate_model(infection_model, X_val[infection_features], y_val_infection,
                                               model_type='binary')
    print("Infection Model Metrics:", infection_metrics)

    print("Evaluating Organ Dysfunction Model...")
    organ_metrics, y_pred = evaluate_model(organ_model, X_val[organ_features], y_val_organ, model_type='binary')
    print("Organ Dysfunction Model Metrics:", organ_metrics)

    # Combine predictions for Sepsis
    print("Predicting Sepsis Probability...")
    infection_prob = infection_model.predict_proba(X_val[infection_features])[:, 1]
    organ_prob = organ_model.predict_proba(X_val[organ_features]).max(axis=1)
    sepsis_prob = infection_prob * organ_prob

    # Evaluate Sepsis prediction
    sepsis_label = (sepsis_prob > 0.5).astype(int)
    print("Sepsis AUROC:", roc_auc_score(y_val_sepsis, sepsis_prob))

    # Save models
    save_model(infection_model, f"{output_dir}/infection_model.pkl")
    save_model(organ_model, f"{output_dir}/organ_model.pkl")
    np.save(f"{output_dir}/sepsis_prob.npy", sepsis_prob)

    print("All models and outputs saved.")


if __name__ == "__main__":
    main()
