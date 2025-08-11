import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse

from typing import Dict, List

"""This script trains various classification models on the Iris dataset and evaluates their performance.
It includes Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest classifiers."""

def print_classification_report_for_model(model_name:str, y_test, y_pred, labels):
    
    print(f"----{model_name}------- ")
    print(f"Predictions = {y_pred[:5]}")    
    print(classification_report(y_test, y_pred, target_names = labels))

def accuracy(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def display_confusion_matrix(model_name:str, y_test, y_pred, labels):
    # Create output folder if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=labels)
    disp.plot()

    # Save the plot
    output_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")  # Save before plt.show()

    # Optionally display it
    plt.show()

def train_model(labels, X, y, test_size, random_state) -> List[Dict[str, float]]:

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model_type = [{'model_name':'DecisionTreeClassifier', 'accuracy': 0.0},{'model_name':'svm', 'accuracy': 0.0},
              {'model_name':'RandomForestClassifier', 'accuracy': 0.0},{'model_name':'KNeighboursClassifier', 'accuracy': 0.0}]

    print("True labels:", y_test[:5])

    for m in model_type:  
        model_name = m['model_name']
        print(f"Training model: {model_name}") 
        if model_name=='DecisionTreeClassifier':
            model_dtc = DecisionTreeClassifier(random_state = random_state)
            model_dtc.fit(X_train,y_train)
            y_pred_dtc = model_dtc.predict(X_test)
            m['accuracy'] = accuracy(y_test, y_pred_dtc)
            print_classification_report_for_model(model_name, y_test, y_pred_dtc, labels)
            display_confusion_matrix(model_name, y_test, y_pred_dtc, labels)
        elif model_name=='svm':
            model_svc = svm.SVC()
            model_svc.fit(X_train,y_train)
            y_pred_svc = model_svc.predict(X_test)
            m['accuracy'] = accuracy(y_test, y_pred_svc)
            print_classification_report_for_model(model_name, y_test, y_pred_svc, labels)
            display_confusion_matrix(model_name, y_test, y_pred_svc, labels)
        elif model_name=='KNeighboursClassifier':
            model_knc = KNeighborsClassifier(n_neighbors = 3)
            model_knc.fit(X_train,y_train)
            y_pred_knc = model_knc.predict(X_test)
            m['accuracy'] = accuracy(y_test, y_pred_knc)
            print_classification_report_for_model(model_name, y_test, y_pred_knc, labels)
            display_confusion_matrix(model_name, y_test, y_pred_knc, labels)
        elif model_name=='RandomForestClassifier':
            model_rf = RandomForestClassifier(max_depth=2, random_state = random_state)
            model_rf.fit(X_train,y_train)
            y_pred_rf = model_rf.predict(X_test)
            m['accuracy'] = accuracy(y_test, y_pred_rf)
            print_classification_report_for_model(model_name, y_test, y_pred_rf, labels)
            display_confusion_matrix(model_name, y_test, y_pred_rf, labels)
        else:
            print(f"Model {model_name} is not implemented.")
    return model_type

def main(test_size, random_state):

    test_size = float(test_size)
    random_state = int(random_state)
    # Load the iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Print the shape of the dataset
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Print the first 5 samples
    print("First 5 samples:")   
    for i in range(5):
        print(f"Sample {i+1}: Features: {X[i]}, Target: {y[i]}")

    labels = iris.target_names

    return train_model(labels, X, y, test_size, random_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classification models on the Iris dataset.")
    parser.add_argument('--test-size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--random-state', type=int, default=42, help="Random state for reproducibility.")
    args = parser.parse_args()
    main(args.test_size, args.random_state)