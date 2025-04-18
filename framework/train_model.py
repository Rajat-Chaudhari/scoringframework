import pandas as pd
import numpy as np
import pickle
import boto3
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from scoring1 import *
from helper import *
from constant import *

# Split the data into train & test
def split_data(X, Y, test_size=0.2, random_state=42):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test

def model_regression_training(X_train, X_test, Y_train, Y_test):
    # Train regression model
    regressor = LinearRegression()
    regressor.fit(X_train, Y_train)

    # Save the regression model locally
    with open(os.path.join(model_folder_path_reg, 'regression_model.pkl'), 'wb') as f:
        pickle.dump(regressor, f)

    # Evaluate Linear Regression
    Y_pred_lr = regressor.predict(X_test)
    lr_mse = mean_squared_error(Y_test, Y_pred_lr)
    lr_r2 = r2_score(Y_test, Y_pred_lr)

    # GridSearchCV for all models
    best_models = {}
    model_stats = [("LinearRegression", {}, lr_r2)]  # Add LR to model stats

    for model_name, (model, params) in models.items():
        grid = GridSearchCV(estimator=model, param_grid=params, scoring='r2', cv=5, n_jobs=-1)
        grid.fit(X_train, Y_train)
        best_models[model_name] = grid.best_estimator_
        model_stats.append((model_name, grid.best_params_, grid.best_score_))
    
    model_stats_df = pd.DataFrame(model_stats, columns=['model_name', 'params', 'score'])
    model_stats_df.to_csv(model_stats_path, index=False)

    # Evaluate models on the test set
    model_score = [("LinearRegression", lr_mse, lr_r2)]  # Include LR first
    
    for model_name, model in best_models.items():
        Y_pred_model = model.predict(X_test)
        model_score.append((model_name, mean_squared_error(Y_test, Y_pred_model), r2_score(Y_test, Y_pred_model)))
        
        # Save each trained model locally
        with open(os.path.join(model_folder_path_reg, f'{model_name.lower()}_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
    
    model_score_df = pd.DataFrame(model_score, columns=['model_name', 'mean_squared_error', 'r2_score'])
    model_score_df.to_csv(model_score_path, index=False)

def model_classification_training(X_train_clf, y_train_clf, y_train_decile_clf):
    # Train classification model for 5 categories
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_clf, y_train_clf)
    
    # Train classification model for 10 categories
    classifier_decile = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier_decile.fit(X_train_clf, y_train_decile_clf)

    # Save the classification models locally
    with open(os.path.join(model_folder_path_clf, 'classification_model.pkl'), 'wb') as f:
        pickle.dump(classifier, f)
    
    with open(os.path.join(model_folder_path_clf, 'classification_decile_model.pkl'), 'wb') as f:
        pickle.dump(classifier_decile, f)
    
    print("Models saved locally")