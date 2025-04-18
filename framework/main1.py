import pandas as pd
import numpy as np
import s3fs
import json
import boto3
from datetime import datetime
from scoring1 import *
from forecast_model import *
from train_model import *
from helper import *
from constant import *

if __name__ == "__main__":
    ##################################### Data Read and Preprocess #####################################
    fs = s3fs.S3FileSystem()
    
    # Load configuration files
    with open(json_path_logic, 'r') as f1, \
         open(json_path_list, 'r') as f2:
         config_nmp = json.load(f1)
         config_sources = json.load(f2)

    # Read data
    df = pd.read_excel(data_path)

    # # Remove VINs with missing mileage
    # car_remove = df[df['Average Mileage (Km/Trip)'].isna()]['VIN'].unique().tolist()
    # df = df[~df['VIN'].isin(car_remove)].reset_index(drop=True)

    # Merge DTC data
    df_dtc = pd.read_csv(dtc_data).drop_duplicates(subset=['VIN'])
    car_df = df.merge(df_dtc[['VIN', 'Percentage Trips with DTC']], on='VIN', how='left')
    print(car_df.shape, df_dtc.shape, df.shape)
    print(car_df['VIN'].nunique(), df_dtc['VIN'].nunique(), df['VIN'].nunique())
    
    # Add Speed & Road Surface Columns
    car_df = add_speed_road_columns(car_df)
    car_data = car_df[:]

    # Outlier removal for historical warnings
    car_data = replace_outliers(car_data, outlier_cols)
    
    # Check all columns
    print(car_data.columns.tolist())

    # Filter columns based on config
    filtered_columns, filtered_weighted_scores = filter_columns_by_source(config_nmp, config_sources)

    # Update configuration for scoring
    config = {
        "columns": filtered_columns,
        "weighted_overall_score_columns": filtered_weighted_scores,
        "ranking_categories": config_nmp["ranking_categories"]
    }
    
    # Save updated config to S3
    save_config(config, json_path)

    # Apply scoring logic
    car_data_scoring = scoring_logic_dynamic(car_data, config)

    # Generate summary table
    summary_df = create_summary_table(car_data, config)
    summary_df.to_csv(summary_path, index=False)

    # Replace the modified column with the original in the final DataFrame
    for column in outlier_cols:
        car_data_scoring[column] = car_df[column]

    # Preprocess data
    car_data_encoded = label_encode_columns(car_data_scoring)
    
    ##################################### Data Modeling #####################################
    X = prepare_features_for_modeling(car_data_encoded, NUMERICAL_COLUMNS, CATEGORICAL_COLUMNS)
    X = rename_columns_for_modeling(X)
    Y = car_data_encoded['Normalized Overall Score (Out of 10)']
    
    ##################################### Regression #####################################
    X_train, X_test, Y_train, Y_test = split_data(X, Y)
    print(Y_train.shape, Y_test.shape)
    print(Y_train.dtype)
    model_regression_training(X_train, X_test, Y_train, Y_test)
    car_data_predicted_reg = regression_predict(car_data_encoded, X)
    
    ##################################### Classification #####################################
    car_data_clf_features = classification_features(car_data_predicted_reg)
    X_class = X[:]
    y_class = car_data_clf_features['Five Score Category']
    y_decile_class = car_data_clf_features['Ten Score Category']
    
    X_train_class, X_test_class, y_train_class, y_test_class = split_data(X_class, y_class)
    X_decile_train_class, X_decile_test_class, y_decile_train_class, y_decile_test_class = split_data(X_class, y_decile_class)
    
    model_classification_training(X_train_class, y_train_class, y_decile_train_class)
    final_classification = classification_predict(car_data_clf_features, X)
    
    ##################################### Saving the Data #####################################
    final_classification[CATEGORICAL_COLUMNS] = final_classification[CATEGORICAL_COLUMNS].replace({1: 'Yes', 0: 'No'})
    
    # Convert dates
    date_cols = [
        'Sales Date', 'Connected Data Acquisition Period - Start Date', 'Manufacture Date',
        'Connected Data Acquisition Period - Cut-Off Date', 'Maintenance Data Acquisition Period - Start Date'
    ]
    for col in date_cols:
        final_classification[col] = pd.to_datetime(final_classification[col], errors='coerce').astype('datetime64[us]')
    
    # Save LDCM Prediction data
    ldcm_preds_df = select_valid_columns(final_classification)
    ldcm_preds_df.to_excel(final_pred_path, index=False)
    
    # Save overwritten parquet file
    final_df = final_data(final_classification)
    final_df.to_excel(output_pred_path, index=False)
    print(final_df.shape)
    
    # Save the grouped data
    processed_df = process_grouped_data(final_classification)
    
    ##################################### Dummy Files #####################################    
    # current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # df = pd.DataFrame({'timestamp': [current_timestamp]})
    # file_check = file_exists_in_s3(dummy_bucket, dummy_folder, dummy_file)
    
    # if not file_check:
    #     df.to_csv(model_run_timestamp, index=False)
    # else:
    #     old_df = pd.read_csv(f"s3://{dummy_bucket}/{dummy_folder}{dummy_file}")
    #     df = pd.concat([df, old_df], ignore_index=True)
    #     df.to_csv(model_run_timestamp, index=False)
    
    print("All done till the end!")
    
    # Writing to RedShift
    # write_to_redshift(processed_df, connection_str)n_str)