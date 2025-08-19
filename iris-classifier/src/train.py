import os
from typing import Dict, List
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


"""This script trains various classification models on the Iris dataset and evaluates their performance.
It includes Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest classifiers."""

def print_classification_report_for_model(model_name:str, y_test:np.ndarray, y_pred:np.ndarray, labels:List[str]):
    """
    Print the classification report for a given model.
    """
    logging.info(f"----{model_name}------- ")
    logging.info(f"Predictions = {y_pred[:5]}")    
    logging.info(classification_report(y_test, y_pred, target_names = labels))

def accuracy(y_test:np.ndarray, y_pred:np.ndarray) -> float:
    return accuracy_score(y_test, y_pred)

def display_confusion_matrix(model_name:str, y_test:np.ndarray, y_pred:np.ndarray, labels:List[str], show_plot:bool = True) -> None:
    """
    Display and save the confusion matrix for a given model.
    """
    # Create output folder if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    logging.info(f"\nConfusion Matrix for {model_name}:\n{cm_df}")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
    disp.plot()

    # Save the plot
    output_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Save before plt.show()

    # Optionally display it
    if show_plot:
        plt.show()
    plt.close()

def save_model(model, model_name:str) -> None:
    """
    Save the trained model to a file.
    """
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

def load_model(model_name:str):
    """
    Load a trained model from a file.
    """
    model_path = os.path.join("output", f"{model_name}_model.pkl")
    return joblib.load(model_path)


def train_model(labels:np.ndarray, X:np.ndarray, y:np.ndarray, test_size: float, random_state: int, show_plot: bool) -> pd.DataFrame:
    """
    Train multiple classifiers and evaluate their performance.
    Returns a list of dicts with model names and accuracies.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model_constructors = {
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=random_state),
        'svm': svm.SVC(),
        'RandomForestClassifier': RandomForestClassifier(max_depth=2, random_state=random_state),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3)
    }
    
    results = []
    print("True labels:", y_test[:5])
    
    for model_name, model in model_constructors.items():
        try:
            logging.info(f"Training model: {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy(y_test, y_pred)
            print_classification_report_for_model(model_name, y_test, y_pred, labels)
            display_confusion_matrix(model_name, y_test, y_pred, labels, show_plot=show_plot)
            save_model(model, model_name)
            results.append({'model_name': model_name, 'accuracy': acc})
        except Exception as e:
            logging.error(f"Error training model {model_name}: {e}")

    return pd.DataFrame(results)

def main(test_size: float, random_state: int, show_plot: bool = False) -> pd.DataFrame:

    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logging.info("First 5 samples:")
  
    for i in range(5):
        logging.info(f"Sample {i+1}: Features: {X[i]}, Target: {y[i]}")

    labels = iris.target_names

    return train_model(labels, X, y, test_size, random_state, show_plot=show_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification models on the Iris dataset.")
    parser.add_argument('--test-size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--random-state', type=int, default=42, help="Random state for reproducibility.")
    parser.add_argument('--show-plot', action='store_true', help="Whether to show the confusion matrix plot.")
    args = parser.parse_args()
    main(args.test_size, args.random_state, show_plot=args.show_plot)