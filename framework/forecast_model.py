import pandas as pd
import numpy as np
import pickle
import boto3
import os
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from helper import *
from train_model import *
from scoring1 import *

def regression_predict(df, X):
    # Load all models
    best_models_forecast = {}
    # Include the simple Linear Regression model here as well
    with open(os.path.join(model_folder_path_reg, 'regression_model.pkl'), 'rb') as f:
        linear_regressor = pickle.load(f)
    print("Model loaded successfully!")

    best_models_forecast['linearregression'] = linear_regressor
    
    # Dictionary to store loaded models
    best_models_forecast = {}
    
    # List of model names
    model_names = ['decisiontree', 'randomforest', 'gradientboosting', 'xgboost', 'lightgbm']
    
    # Loop through models
    for model_name in model_names:
        local_path = os.path.join(model_folder_path_reg, f'{model_name}_model.pkl')
    
        # Load model using pickle
        with open(local_path, 'rb') as f:
            best_models_forecast[model_name] = pickle.load(f)
    print("All models loaded successfully!")

    # Predict overall score using the best models
    regression_predictions = {}
    for model_name, model in best_models_forecast.items():
        Y_pred = model.predict(X)
        regression_predictions[model_name] = Y_pred
        df[f'Predicted Overall Score ({model_name})'] = Y_pred

    return df

def classification_predict(df, X):
    # Load classification models
    local_model_path = os.path.join(model_folder_path_clf, 'classification_model.pkl')
    local_decile_model_path = os.path.join(model_folder_path_clf, 'classification_decile_model.pkl')
    
    # Load classification models
    with open(local_model_path, 'rb') as f:
        classifier = pickle.load(f)
    
    with open(local_decile_model_path, 'rb') as f:
        classifier_decile = pickle.load(f)
    
    print("Models successfully loaded!")

    # Predict classification categories
    y_class_pred = classifier.predict(X)
    df['Predicted Five Score Category'] = pd.Series(y_class_pred).map(FIVE_CATEGORY_MAPPING)

    # Predict decile-based classification
    y_decile_pred = classifier_decile.predict(X)
    df['Predicted Ten Score Category'] = pd.Series(y_decile_pred).map(CATEGORY_MAPPING)

    return df