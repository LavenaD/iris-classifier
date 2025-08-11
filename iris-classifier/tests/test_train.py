import os
import pytest
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from train import main

from io import StringIO
import re

def test_iris_accuracy():


    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    # Call the main function with test parameters
    model_result = main(test_size=0.2, random_state=42)

    # Get the output and reset stdout
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout

    print(model_result)  # For debugging purposes, you can see the output

    # Check if the output contains the expected accuracy
    for model in model_result:
        assert model['accuracy'] > 0.9
        
    # Check if the output contains the expected model names
    expected_models = ['DecisionTreeClassifier', 'svm', 'RandomForestClassifier', 'KNeighboursClassifier']
    assert len(model_result) == len(expected_models), "Model count mismatch"
    for model in expected_models:
        assert any(m['model_name'] == model for m in model_result), f"Model {model} not found in results"