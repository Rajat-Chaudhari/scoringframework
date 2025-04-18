import pandas as pd
import s3fs
import json
import fsspec
import boto3
from constant import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Helper functions

# Filter source columns from parameter list config file
def filter_columns_by_source(config_nmp, config_sources):
    # Filter columns in config_nmp based on enabled sources in config_sources
    enabled_sources = {key for key, value in config_sources["enabled_sources"].items() if value}
    filtered_columns = {
        key: value for key, value in config_nmp["columns"].items()
        if value["source"] in enabled_sources
    }  
    # Filter weighted overall score columns based on remaining sources
    filtered_weighted_scores = [
        col for col in config_nmp["weighted_overall_score_columns"]
        if col in filtered_columns
    ]
    return filtered_columns, filtered_weighted_scores

# Saving updated config file to S3
def save_config(config, json_path):
    with open(json_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config file saved locally at: {json_path}")

# Read parquet df
def read_all_parquet(s3_folder_path: str) -> pd.DataFrame:
    fs = fsspec.filesystem('s3', anon=False)
    # List all parquet files in the folder
    files = fs.glob(f"{s3_folder_path}*.parquet")
    if not files:
        raise FileNotFoundError(f"No parquet files found in {s3_folder_path}")
    # Read all files and concatenate
    df = pd.concat(
        [pd.read_parquet(f"s3://{file}", storage_options={"anon": False}) for file in files],
        ignore_index=True
    )
    file_name = files[0].split('/')[-1]
    return df, file_name

#Speed & Road Surface Columns Addition
def add_speed_road_columns(df):
    # Define the speed bucket columns for distance, trips, and RSC
    speed_dist_columns = [
        "Speed WRT Mileage < 20km/h",
        "Speed WRT Mileage 21 - 60km/h",
        "Speed WRT Mileage 61 - 80km/h",
        "Speed WRT Mileage 81 - 100 km/h",
        "Speed WRT Mileage > 100km/h"
    ]
    speed_trips_columns = [
        "Speed WRT Trips < 20km/h",
        "Speed WRT Trips 21 - 60km/h",
        "Speed WRT Trips 61 - 80km/h",
        "Speed WRT Trips 81 - 100km/h",
        "Speed WRT Trips >100 km/h"
    ]
    road_surface_category_columns = [
        "Road surface category 1 (km)",
        "Road surface category 2 (km)",
        "Road surface category 3 (km)",
        "Road surface category 4 (km)",
        "Road surface category 5 (km)",
        "Road surface category 6 (km)"
    ]
    
     # Convert to numeric to avoid TypeError
    for col in speed_dist_columns + speed_trips_columns + road_surface_category_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Calculate total distance & trips traveled
    df["Speed WRT Mileage Total Distance"] = df[speed_dist_columns].sum(axis=1)
    df["Speed WRT Mileage Total Trips"] = df[speed_trips_columns].sum(axis=1)
    df["Road Surface Category Total Distance"] = df[road_surface_category_columns].sum(axis=1)
    
    # Avoid division by zero: Replace zero totals with NaN
    # df["Speed WRT Mileage Total Distance"] = df["Speed WRT Mileage Total Distance"].replace(0, None)
    # df["Speed WRT Mileage Total Trips"] = df["Speed WRT Mileage Total Trips"].replace(0, None)
    # df["Road Surface Category Total Distance"] = df["Road Surface Category Total Distance"].replace(0, None)
    
    # Create new columns for percentage of each speed bucket & road surface category (Distance-based)
    for col in speed_dist_columns:
        percentage_col = f"Percentage {col}"
        df[percentage_col] = (df[col] / df["Speed WRT Mileage Total Distance"]) * 100
        
    for col in speed_trips_columns:
        percentage_col = f"Percentage {col}"
        df[percentage_col] = (df[col] / df["Speed WRT Mileage Total Trips"]) * 100
    
    for col in road_surface_category_columns:
        percentage_col = f"Percentage {col}"
        df[percentage_col] = (df[col] / df["Road Surface Category Total Distance"]) * 100
    
    return df

#Outlier Treatment
def replace_outliers(df, columns):
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        IQR = q3 - q1
        lower_bound = q1 - 2 * IQR
        upper_bound = q3 + 2 * IQR
        # Replace outliers
        df[column] = df[column].apply(
            lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df

#Label Encoding
def label_encode_columns(df):
    #Label encode specified categorical columns
    label_encoder = LabelEncoder()
    for col in CATEGORICAL_COLUMNS:
        df[col] = label_encoder.fit_transform(df[col])
    return df

#Missing Value Hnadling for Modeling
def prepare_features_for_modeling(df, numerical_cols, categorical_cols):
    X = df[numerical_cols + categorical_cols].copy()
    # Fill missing for modeling
    columns_with_missing = ['Odometer exchange/rewind possibility detection - Distance',
                            'Odometer Value',
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
                            "collision count"]
    X[columns_with_missing] = X[columns_with_missing].fillna(0)
    return X

#Renaming Columns for Modeling
def rename_columns_for_modeling(X):
    rename_mapping = {
        'Diagnostic code detected - Priority': 'Diagnostic_Code_Priority',
        'Collision Detection(Yes/No)': 'Collision_Detection',
        'Odometer exchange/rewind possibility detection - Distance': 'Odometer_Rewind_Distance',
        'Odometer Value': 'Odometer_Value',
        'Fuel efficiency (Km/l)': 'Fuel_Efficiency',
        'Average Trips per Week': 'Avg_Trips_per_Week',
        'Average Mileage (Km/Trip)': 'Avg_Mileage',
        'Average Speed': 'Avg_Speed',
        'Number of Current Warnings': 'Current_Warnings',
        'Number of Historical Warnings': 'Historical_Warnings',
        'DTC Count': 'DTC_Count',
        'car_age': 'Car_Age',
        "Percentage Speed WRT Mileage < 20km/h": "Speed_less_than_20kmh",
        "Percentage Speed WRT Mileage 21 - 60km/h": "Speed_21_60kmh",
        "Percentage Speed WRT Mileage 61 - 80km/h": "Speed_61_80kmh",
        "Percentage Speed WRT Mileage 81 - 100 km/h": "Speed_81_100kmh",
        "Percentage Speed WRT Mileage > 100km/h": "Speed_greater_than_100kmh",
        "Percentage Speed WRT Trips < 20km/h": "Trips_less_than_20kmh",
        "Percentage Speed WRT Trips 21 - 60km/h": "Trips_21_60kmh",
        "Percentage Speed WRT Trips 61 - 80km/h": "Trips_61_80kmh",
        "Percentage Speed WRT Trips 81 - 100km/h": "Trips_81_100kmh",
        "Percentage Speed WRT Trips >100 km/h": "Trips_greater_than_100kmh",
        "Percentage Road surface category 1 (km)": "Road_Surface_Category_1",
        "Percentage Road surface category 2 (km)": "Road_Surface_Category_2",
        "Percentage Road surface category 3 (km)": "Road_Surface_Category_3",
        "Percentage Road surface category 4 (km)": "Road_Surface_Category_4",
        "Percentage Road surface category 5 (km)": "Road_Surface_Category_5",
        "Percentage Road surface category 6 (km)": "Road_Surface_Category_6",
        "Percentage Trips with DTC": "Percentage_Trips_with_DTC",
        "collison count": "collision_count"
    }
    return X.rename(columns=rename_mapping)

#Classification
def classification_features(df):
    # Define classification model features for 5 categories
    binning = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    df['Five Score Category'] = binning.fit_transform(df[['Normalized Overall Score (Out of 10)']])
    # Define classification model features for 10 categories
    binning_decile = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    df['Ten Score Category'] = binning_decile.fit_transform(df[['Normalized Overall Score (Out of 10)']])
    return df

#Cleaning dataset 
def select_valid_columns(df):
    column_list = [
        'VIN', 'Model Code', 'Series Name',
        'Odometer exchange/rewind possibility detection',
        'Odometer exchange/rewind possibility detection - Distance',
        'Odometer Value', 'Warning Light Illuminates',
        'Number of Current Warnings', 'Number of Historical Warnings',
        'Collision Detection(Yes/No)', 'Average Trips per Week',
        'Fuel efficiency (Km/l)', 'Average Mileage (Km/Trip)',
        'Average Speed', 'Diagnostic code detected - Priority',
        'DTC Count', 'car_age', 'Percentage Speed WRT Mileage < 20km/h',
        'Percentage Speed WRT Mileage 21 - 60km/h',
        'Percentage Speed WRT Mileage 61 - 80km/h',
        'Percentage Speed WRT Mileage 81 - 100 km/h',
        'Percentage Speed WRT Mileage > 100km/h',
        "Percentage Speed WRT Trips < 20km/h",
        "Percentage Speed WRT Trips 21 - 60km/h",
        "Percentage Speed WRT Trips 61 - 80km/h",
        "Percentage Speed WRT Trips 81 - 100km/h",
        "Percentage Speed WRT Trips >100 km/h",
        'Percentage Road surface category 1 (km)',
        'Percentage Road surface category 2 (km)',
        'Percentage Road surface category 3 (km)',
        'Percentage Road surface category 4 (km)',
        'Percentage Road surface category 5 (km)',
        'Percentage Road surface category 6 (km)',
        'Percentage Trips with DTC',
        'collision count',
        'Odometer Rewind Yes/No Score Normalized & Weighted',
        'Odometer Rewind Score Normalized & Weighted',
        'Odometer Value Score Normalized & Weighted',
        'Warning Score Normalized & Weighted',
        'Current Warning Score Normalized & Weighted',
        'Historical Warning Score Normalized & Weighted',
        'Accident Score Normalized & Weighted',
        'Trip Frequency Score Normalized & Weighted',
        'Fuel Efficiency Score Normalized & Weighted',
        'Usage Distance Score Normalized & Weighted',
        'Speed Score Normalized & Weighted',
        'Health Score Normalized & Weighted',
        'DTC Occurrence Score Normalized & Weighted',
        'Age Score Normalized & Weighted',
        'Percentage Speed WRT Mileage < 20km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage 21 - 60km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage 61 - 80km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage 81 - 100 km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage > 100km/h Score Normalized & Weighted',
        "Road Surface Category 1 Score Normalized & Weighted",
        "Road Surface Category 2 Score Normalized & Weighted",
        "Road Surface Category 3 Score Normalized & Weighted",
        "Road Surface Category 4 Score Normalized & Weighted",
        "Road Surface Category 5 Score Normalized & Weighted",
        "Road Surface Category 6 Score Normalized & Weighted",
        "Share of Trips with DTC Score Normalized & Weighted",
        "Count of Collision Score Normalized & Weighted",
        'Weighted Overall Score', 'Normalized Overall Score (Out of 10)',
        'Score Category']
    # Select only columns that exist in df
    existing_columns = [col for col in column_list if col in df.columns]
    return df[existing_columns]

#Final Data
#Processing Data
def final_data(df):
    # Define column groups
    groups = {
        'Group 1 Normalized & Weighted Score': [
            'Odometer Value Score Normalized & Weighted',
            'Usage Distance Score Normalized & Weighted',
            'Trip Frequency Score Normalized & Weighted',
            'Fuel Efficiency Score Normalized & Weighted'
        ],
        'Group 2 Normalized & Weighted Score': [
            'Odometer Rewind Yes/No Score Normalized & Weighted',
            'Odometer Rewind Score Normalized & Weighted',
            'DTC Occurrence Score Normalized & Weighted',
            'Health Score Normalized & Weighted',
            'Accident Score Normalized & Weighted',
            'Current Warning Score Normalized & Weighted',
            'Historical Warning Score Normalized & Weighted'
        ],
        'Group 3 Normalized & Weighted Score': [
            'Percentage Speed WRT Mileage < 20km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage 21 - 60km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage 61 - 80km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage 81 - 100 km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage > 100km/h Score Normalized & Weighted'
        ],
        'Group 4 Normalized & Weighted Score': ['Age Score Normalized & Weighted']
    }

    # Compute group scores
    for group_name, cols in groups.items():
        df[group_name] = df[cols].sum(axis=1)

    # Compute average scores per Series Name (EXCLUDING Fuel Efficiency)
    group_1_columns_excluding_fuel = [
        'Odometer Value Score Normalized & Weighted',
        'Usage Distance Score Normalized & Weighted',
        'Trip Frequency Score Normalized & Weighted'
    ]
    group_1_avg_excluding_fuel = df.groupby("Series Name")[group_1_columns_excluding_fuel].mean().sum(axis=1)

    # Compute Fuel Efficiency average at Model Code level
    fuel_eff_avg_by_model = df.groupby("Model Code")['Fuel Efficiency Score Normalized & Weighted'].mean()

    # Assign Fuel Efficiency back to VINs based on Model Code
    df['Fuel Efficiency Score Adjusted'] = df['Model Code'].map(fuel_eff_avg_by_model)

    # Compute final Group 1 Average per VIN (Series Name avg + Model Code Fuel Efficiency avg)
    df['Group 1 Normalized & Weighted Average Score'] = (
        df['Series Name'].map(group_1_avg_excluding_fuel) + df['Fuel Efficiency Score Adjusted']
    )

    # Compute other group averages per Series Name
    group_averages = {
        key.replace("Score", "Average Score"): df.groupby("Series Name")[cols].mean().sum(axis=1)
        for key, cols in groups.items() if key != 'Group 1 Normalized & Weighted Score'
    }

    # Map other group averages back to VINs
    for key, avg_series in group_averages.items():
        df[key] = df['Series Name'].map(avg_series)

    # Drop only the Fuel Efficiency Score Adjusted column
    df.drop(columns=['Fuel Efficiency Score Adjusted'], inplace=True)

    # Keep only the required columns
    required_columns = [
        'VIN', 'Body Color', 'Model', 'Series Name', 'Sales Date', 'Fuel Type',
        'Connected Data Acquisition Period - Start Date', 'Manufacture Date',
        'Type of Usage', 'Connected Data Acquisition Period - Cut-Off Date',
        'Odometer exchange/rewind possibility detection',
        'Odometer exchange/rewind possibility detection - Distance',
        'Odometer rewind possibility detection - Date', 'Model Code', 'car_age',
        'Odometer Value', 'Average Mileage (Km/Trip)', 'Fuel efficiency (Km/l)',
        'Trip Count', 'Average Trips per Week', 'Average Speed', 'avg time per trip',
        'Speed WRT Mileage < 20km/h', 'Speed WRT Mileage 21 - 60km/h',
        'Speed WRT Mileage 61 - 80km/h', 'Speed WRT Mileage 81 - 100 km/h',
        'Speed WRT Mileage > 100km/h', 'Speed WRT Trips < 20km/h',
        'Speed WRT Trips 21 - 60km/h', 'Speed WRT Trips 61 - 80km/h',
        'Speed WRT Trips 81 - 100km/h', 'Speed WRT Trips >100 km/h',
        'Collision Detection(Yes/No)', 'collision count', 'Collision Detection',
        'Collision Detection Dates', 'Collision Detection Details',
        'Warning Light Illuminates', 'Number of Current Warnings',
        'Warning Current Dates', 'Warning Current Details', 'Warning - Current',
        'Number of Historical Warnings', 'Warning History Dates',
        'Warning History Details', 'Warning - History (has been fixed)',
        'warning count', 'Current Priority DTC Count', 'Historic Priority DTC Count',
        'DTC Count', 'Diagnostic code detected - Priority', 'Unresolved Priority DTC',
        'Resolved Priority DTC', 'Loading level 1 (km)', 'Loading level 2 (km)',
        'Loading level 3 (km)', 'Loading level 4 (km)', 'Loading level 5 (km)',
        'Loading level 6 (km)', 'Road surface category 1 (km)',
        'Road surface category 2 (km)', 'Road surface category 3 (km)',
        'Road surface category 4 (km)', 'Road surface category 5 (km)',
        'Road surface category 6 (km)',
        'Maintenance Data Acquisition Period - Start Date',
        'Maintenance Data Acquisition Period - Last Maintenance/Inspection',
        'Inspection history', 'Periodic Service', 'Service Visits per Year',
        'Engine', 'Recent AC filters serviced date', 'Suspension',
        'Body Work & Paint', 'Service(Repair/Part replacement) History',
        'Group 1 Normalized & Weighted Score',
        'Group 2 Normalized & Weighted Score',
        'Group 3 Normalized & Weighted Score',
        'Group 4 Normalized & Weighted Score',
        'Group 1 Normalized & Weighted Average Score',
        'Group 2 Normalized & Weighted Average Score',
        'Group 3 Normalized & Weighted Average Score',
        'Group 4 Normalized & Weighted Average Score',
        'Weighted Overall Score', 'Normalized Overall Score (Out of 10)',
        'Score Category', 'Create_Timestamp'
    ]

    # Keep only the required columns
    df = df[required_columns]

    print(df.shape)

    return df
    
#Grouped Data PowerBI
def process_grouped_data(df):
    selected_columns = [
        'VIN', 'Model Code', 'Series Name',
        'Odometer Rewind Yes/No Score Normalized & Weighted',
        'Odometer Rewind Score Normalized & Weighted',
        'Odometer Value Score Normalized & Weighted',
        'Warning Score Normalized & Weighted',
        'Current Warning Score Normalized & Weighted',
        'Historical Warning Score Normalized & Weighted',
        'Accident Score Normalized & Weighted',
        'Trip Frequency Score Normalized & Weighted',
        'Fuel Efficiency Score Normalized & Weighted',
        'Usage Distance Score Normalized & Weighted',
        'Speed Score Normalized & Weighted',
        'Health Score Normalized & Weighted',
        'DTC Occurrence Score Normalized & Weighted',
        'Age Score Normalized & Weighted',
        'Percentage Speed WRT Mileage < 20km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage 21 - 60km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage 61 - 80km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage 81 - 100 km/h Score Normalized & Weighted',
        'Percentage Speed WRT Mileage > 100km/h Score Normalized & Weighted',
        'Weighted Overall Score', 'Normalized Overall Score (Out of 10)',
        'Score Category']

    grouped_data = df[selected_columns].copy()

    # Define column groups
    groups = {
        'Group 1 Normalized & Weighted Score': [
            'Odometer Value Score Normalized & Weighted',
            'Usage Distance Score Normalized & Weighted',
            'Trip Frequency Score Normalized & Weighted',
            'Fuel Efficiency Score Normalized & Weighted'
        ],
        'Group 2 Normalized & Weighted Score': [
            'Odometer Rewind Yes/No Score Normalized & Weighted',
            'Odometer Rewind Score Normalized & Weighted',
            'DTC Occurrence Score Normalized & Weighted',
            'Health Score Normalized & Weighted',
            'Accident Score Normalized & Weighted',
            'Current Warning Score Normalized & Weighted',
            'Historical Warning Score Normalized & Weighted'
        ],
        'Group 3 Normalized & Weighted Score': [
            'Percentage Speed WRT Mileage < 20km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage 21 - 60km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage 61 - 80km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage 81 - 100 km/h Score Normalized & Weighted',
            'Percentage Speed WRT Mileage > 100km/h Score Normalized & Weighted'
        ],
        'Group 4 Normalized & Weighted Score': ['Age Score Normalized & Weighted']
    }

    # Compute group scores
    for group_name, cols in groups.items():
        grouped_data[group_name] = grouped_data[cols].sum(axis=1)

    # Compute average scores per Series Name (EXCLUDING Fuel Efficiency)
    group_1_columns_excluding_fuel = [
        'Odometer Value Score Normalized & Weighted',
        'Usage Distance Score Normalized & Weighted',
        'Trip Frequency Score Normalized & Weighted'
    ]
    group_1_avg_excluding_fuel = grouped_data.groupby("Series Name")[group_1_columns_excluding_fuel].mean().sum(axis=1)

    # Compute Fuel Efficiency average at Model Code level
    fuel_eff_avg_by_model = grouped_data.groupby("Model Code")['Fuel Efficiency Score Normalized & Weighted'].mean()

    # Assign Fuel Efficiency back to VINs based on Model Code
    grouped_data['Fuel Efficiency Score Adjusted'] = grouped_data['Model Code'].map(fuel_eff_avg_by_model)

    # Compute final Group 1 Average per VIN (Series Name avg + Model Code Fuel Efficiency avg)
    grouped_data['Group 1 Normalized & Weighted Average Score'] = (
        grouped_data['Series Name'].map(group_1_avg_excluding_fuel) + grouped_data['Fuel Efficiency Score Adjusted']
    )

    # Compute other group averages per Series Name
    group_averages = {
        key.replace("Score", "Average Score"): grouped_data.groupby("Series Name")[cols].mean().sum(axis=1)
        for key, cols in groups.items() if key != 'Group 1 Normalized & Weighted Score'
    }

    # Map other group averages back to VINs
    for key, avg_series in group_averages.items():
        grouped_data[key] = grouped_data['Series Name'].map(avg_series)

    # Reorder columns
    column_order = [
        'VIN', 'Model Code', 'Series Name',
        'Group 1 Normalized & Weighted Score', 'Group 1 Normalized & Weighted Average Score',
        'Group 2 Normalized & Weighted Score', 'Group 2 Normalized & Weighted Average Score',
        'Group 3 Normalized & Weighted Score', 'Group 3 Normalized & Weighted Average Score',
        'Group 4 Normalized & Weighted Score', 'Group 4 Normalized & Weighted Average Score',
        'Weighted Overall Score', 'Normalized Overall Score (Out of 10)', 'Score Category'
    ]

    # Ensure column order does not break if missing columns
    column_order = [col for col in column_order if col in grouped_data.columns]
    grouped_data = grouped_data[column_order]

    # Drop Model Code and Series Name before saving
    grouped_data.drop(columns=['Model Code', 'Series Name'], inplace=True)

    grouped_data.to_csv(dashboard_data, index=False)

    return grouped_data

#Model Dummy File
def file_exists_in_s3(bucket_name, folder_path, file_name):
    """
    Check if file exists in S3 folder
    """
    s3 = boto3.client('s3')
    file_key = f"{folder_path}{file_name}"
    try:
        s3.head_object(Bucket=bucket_name, Key=file_key)
        return True  # File exists
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False  # File does not exist
        else:
            raise  # Re-raise other errors

#Write to RedShift
def write_to_redshift(df, connection_string):
    engine = create_engine(connection_string)
    df.to_sql(name='your_table_name', con=engine, if_exists='append', index=False)
    print("Data written to Redshift successfully")