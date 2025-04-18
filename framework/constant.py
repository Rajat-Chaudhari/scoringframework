import pandas as pd
import numpy as np
import pickle
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
from helper import *

# Constant values
CATEGORICAL_COLUMNS = ['Diagnostic code detected - Priority', 'Collision Detection(Yes/No)']
NUMERICAL_COLUMNS = ['Odometer exchange/rewind possibility detection - Distance', 'Odometer Value',
                     'Fuel efficiency (Km/l)', 'Average Trips per Week',
                     'Average Mileage (Km/Trip)', 'Average Speed', 'Number of Current Warnings',
                     'Number of Historical Warnings', 'DTC Count', 'car_age', "Percentage Speed WRT Mileage < 20km/h",
                     "Percentage Speed WRT Mileage 21 - 60km/h", "Percentage Speed WRT Mileage 61 - 80km/h",
                     "Percentage Speed WRT Mileage 81 - 100 km/h", "Percentage Speed WRT Mileage > 100km/h",
                     "Percentage Speed WRT Trips < 20km/h",
                     "Percentage Speed WRT Trips 21 - 60km/h",
                     "Percentage Speed WRT Trips 61 - 80km/h",
                     "Percentage Speed WRT Trips 81 - 100km/h",
                     "Percentage Speed WRT Trips >100 km/h",
                     "Percentage Road surface category 1 (km)",
                     "Percentage Road surface category 2 (km)",
                     "Percentage Road surface category 3 (km)",
                     "Percentage Road surface category 4 (km)",
                     "Percentage Road surface category 5 (km)",
                     "Percentage Road surface category 6 (km)",
                     "Percentage Trips with DTC",
                     "collision count"
                     ]

#Outlier Columns
outlier_cols = ['Number of Historical Warnings']

# Category mapping
FIVE_CATEGORY_MAPPING = {0: "Poor", 1: "Below Average", 2: "Average", 3: "Good", 4: "Excellent"}
CATEGORY_MAPPING = {0: 'F (Critical)', 1: 'E (Very Poor)', 2: 'D (Poor)', 3: 'D+ (Below Average)',
                           4: 'C (Fair)', 5: 'C+ (Above Average)', 6: 'B (Good)', 7: 'B+ (Very Good)',
                           8: 'A (Excellent)', 9: 'A+ (Outstanding)'}

# Define models and parameter grids for GridSearchCV
models = {
    'DecisionTree': (DecisionTreeRegressor(), {'max_depth': [3, 5, 10, None]}),
    'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}),
    'GradientBoosting': (GradientBoostingRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    'XGBoost': (XGBRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]}),
    'LightGBM': (LGBMRegressor(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]})
}

json_path_logic = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Input\config_nmp.json"

json_path_list = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Input\config_sources.json"

json_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Input\config.json"
 
data_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Curated Input\InputModelData.xlsx"

output_pred_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Output\OutputReportData.xlsx"

dtc_data = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Input\final_dtc_data.csv"
 
model_stats_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Output\model_stats.csv"
 
model_score_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Output\model_score.csv"
 
model_folder_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Model"
 
model_folder_path_reg = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Model\regression"
 
model_folder_path_clf = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Model\classification"
 
final_pred_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Output\LDCM_Predictions.xlsx"

summary_path = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Output\summary.csv"

dashboard_data = r"C:\Users\arunanand@deloitte.com\Downloads\AWS\Persist Output\grouped_data.csv"

# model_run_timestamp = "s3://s3-dx-tdem-ie-nonprod-curated/curated-nmp/curated-nmp-dummy/dummy.csv"

# dummy_bucket = "s3-dx-tdem-ie-nonprod-curated"

# dummy_folder = "curated-nmp/curated-nmp-dummy/"

# dummy_file = "dummy.csv"

# connection_str = "postgresql+psycopg2://awsuser@redshift-dx-tdem-ie-nonprod.ckt2igzwhlzp.ap-southeast-1.redshift.amazonaws.com:5439/tdem_db"