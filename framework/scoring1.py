import pandas as pd
import numpy as np
from _datetime import datetime
import math
from math import inf

def determine_bins_fd(data):
    # Determine number of bins using Freedman-Diaconis rule
    data = data.dropna()
    n = len(data)
    if n == 0:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (n ** (1 / 3))
    num_bins = int(np.ceil((data.max() - data.min()) / bin_width)) if bin_width > 0 else 1
    return max(1, num_bins)

def dynamic_bins_for_odometer(values):
    values = np.sort(values[values >= 0])  # Drop negatives if any
    if len(values) == 0:
        return 1, np.array([0, 1])  # Default bin when no valid values

    # Separate out zero if present
    non_zero_values = values[values > 0]
    
    if len(non_zero_values) == 0:
        return 1, np.array([0, 1])  # Only zeros exist, return [0, 1] bin

    min_val, max_val = non_zero_values.min(), non_zero_values.max()

    # Ensure bins: [0], [0,100], then apply dynamic binning from 100 onwards
    bin_edges = np.array([0, 100])  # First bin for (0, 100]

    if max_val > 100:
        # Freedman-Diaconis Rule for bin width
        q75, q25 = np.percentile(non_zero_values, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(non_zero_values) ** (1 / 3))
        if bin_width <= 0:
            bin_width = 1  # fallback

        # Handle extreme skew with logarithmic bins
        if max_val / min_val > 1000:
            num_bins = int(np.ceil(np.log(max_val / 100))) + 1
            dynamic_bins = np.geomspace(100, max_val, num_bins + 1)
        else:
            num_bins = int(np.ceil((max_val - 100) / bin_width))
            dynamic_bins = np.linspace(100, max_val, num_bins + 1)

        # Merge custom bins and dynamic bins
        bin_edges = np.concatenate((bin_edges, dynamic_bins[1:]))

    num_bins = len(bin_edges) - 1
    return num_bins, bin_edges

def dynamic_bins_for_dtc(values, max_bins=10):
    values = np.sort(values[values >= 0])  # Remove negatives if any

    if len(values) == 0:
        return 1, np.array([0, 1])  # Fallback for empty input

    # Separate zero values
    zero_count = np.sum(values == 0)
    non_zero_values = values[values > 0]

    if len(non_zero_values) == 0:
        return 1, np.array([0, 1])  # Only zeros exist

    # Compute Freedman-Diaconis bin width
    q75, q25 = np.percentile(non_zero_values, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(non_zero_values) ** (1 / 3))

    # Fallback to Sturges’ rule if FD is too fine
    if bin_width <= 0:
        bin_width = 1

    min_val, max_val = non_zero_values.min(), non_zero_values.max()

    # If the data is highly skewed, use logarithmic binning
    if max_val / min_val > 50:  # Adjusted threshold
        num_bins = min(int(np.ceil(np.log2(max_val / min_val))) + 1, max_bins)
        bin_edges = np.geomspace(min_val, max_val, num_bins)
        if zero_count > 0:
            bin_edges = np.insert(bin_edges, 0, 0)  # Include zero bin
    else:
        # Linear binning
        num_bins = min(int(np.ceil((max_val - min_val) / bin_width)), max_bins)
        bin_edges = np.linspace(min_val, max_val, num_bins)
        if zero_count > 0:
            bin_edges = np.insert(bin_edges, 0, 0)  # Include zero bin

    num_bins = len(bin_edges) - 1  # Adjust bin count

    return num_bins, bin_edges

def dynamic_bins_warnings_fuel_mileage_age_collision(data):
    
    data = data.dropna()
    if len(data) == 0:
        return 1, np.array([0, 1])  # Handle empty data

    min_val, max_val = data.min(), data.max()
    data_range = max_val - min_val

    # Handle single value edge case
    if data_range == 0:
        return 1, np.array([min_val, min_val + 1])

    # Adaptive bin calculation based on range
    if data_range <= 10:
        num_bins = min(int(np.sqrt(len(data))), 5) # Max 5 bins for small ranges
    elif data_range <= 50:
        num_bins = min(int(np.sqrt(len(data))), 10) # Max 10 bins for mid-range
    else:
        num_bins = int(np.log2(len(data)) + 1) # Standard logarithmic bins for large ranges

    # Prevent bins from being less than 1
    num_bins = max(num_bins, 1)

    # Create bins
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    return num_bins, bin_edges

def map_scores(df, score_column):
    # Define the mapping
    score_mapping = [
        (10, inf, "Outstanding: A+"),
        (9.5, 10, "Outstanding: A+"),
        (9, 9.5, "Excellent: A"),
        (8, 9, "Very Good: B+"),
        (7, 8, "Good: B"),
        (6, 7, "Above Average: C+"),
        (5, 6, "Fair: C"),
        (4, 5, "Below Average: D+"),
        (3, 4, "Poor: D"),
        (2.0, 3, "Very Poor: E"),
        (-inf, 2.0, "Critical: F"),
    ]
    
    # Create a function to map grades
    def get_grade(score):
        for lower, upper, grade in score_mapping:
            if lower <= score < upper:
                return grade
        return "Invalid score"  # Fallback for out-of-range scores

    # Apply the mapping
    df['Score Category'] = df[score_column].apply(get_grade)
    return df

def scoring_logic_dynamic(df, config):
    score_columns = []
    weighted_scores = {}

    for column, rules in config['columns'].items():
        source_column = rules['source']
        weight = rules.get('weight', 1)

        if rules['type'] == 'map':
            if 'map' in rules:
                mapping = rules['map']

                def parse_range_key(key):
                    if isinstance(key, str) and ',' in key:
                        parts = key.strip('[]()').split(',')
                        low, high = parts[0].strip(), parts[1].strip()
                        low = float(low) if low.replace('.', '').isdigit() else float('-inf')
                        high = float(high) if high.replace('.', '').isdigit() else float('inf')
                        return (low, high)
                    return key

                parsed_mapping = {parse_range_key(k): v for k, v in mapping.items()}

                if all(isinstance(k, tuple) for k in parsed_mapping.keys()):
                    def map_value(x):
                        for (low, high), score in parsed_mapping.items():
                            if low < x <= high:
                                return score
                        return 0
                    df[column] = df[source_column].apply(map_value)
                else:
                    df[column] = df[source_column].map(mapping).fillna(0)

                min_score = min(parsed_mapping.values())
                max_score = max(parsed_mapping.values())

            else:
                raise KeyError(f"Missing 'map' key for map-type column: {column}")

        elif rules['type'] == 'range':
            if 'scoring_limits' not in rules:
                raise KeyError(f"Missing 'scoring_limits' for range-type column: {column}")

            min_score, max_score = rules['scoring_limits']

            if rules.get('dynamic_bins', False):
                if source_column == 'Odometer exchange/rewind possibility detection - Distance':
                    # Use the custom dynamic binning logic for this special column
                    num_bins, bin_edges = dynamic_bins_for_odometer(df[source_column].values)
                elif source_column == 'DTC Count':
                    # Use the custom dynamic binning logic for this special column
                    num_bins, bin_edges = dynamic_bins_for_dtc(df[source_column].values)
                elif source_column in ['Number of Current Warnings', 'Number of Historical Warnings',
                                       'Fuel efficiency (Km/l)', 'Average Mileage (Km/Trip)', 'car_age', 'collision count']:
                    # Use the custom dynamic binning logic for these two special columns
                    num_bins, bin_edges = dynamic_bins_warnings_fuel_mileage_age_collision(df[source_column])
                else:
                    # Default FD binning for other columns
                    num_bins = determine_bins_fd(df[source_column])
                    bin_edges = np.linspace(df[source_column].min(), df[source_column].max(), num_bins + 1)

                bin_labels = np.linspace(min_score, max_score, num_bins)
                df[column] = pd.cut(df[source_column], bins=bin_edges, labels=bin_labels, include_lowest=True).astype(float)

        # Normalization to 1-10
        actual_min, actual_max = df[column].min(), df[column].max()

        if actual_min == actual_max:
            if min_score < max_score:
                normalized_score = pd.Series([1] * len(df), index=df.index)
            else:
                normalized_score = pd.Series([10] * len(df), index=df.index)

        else:
            if min_score > max_score:
                normalized_score = 1 + 9 * (max_score - df[column]) / (max_score - min_score)
            else:
                normalized_score = 1 + 9 * (df[column] - min_score) / (max_score - min_score)

        normalized_score = normalized_score.clip(1, 10)

        df[f'{column} Normalized & Weighted'] = normalized_score * weight
        
        # If source column was NaN, the weighted score should be zero
        df.loc[df[source_column].isna(), f'{column} Normalized & Weighted'] = 0

        weighted_scores[column] = df[f'{column} Normalized & Weighted']

        del df[column]  # Clean up if necessary

    df['Weighted Overall Score'] = sum(weighted_scores.values())

    overall_min, overall_max = df['Weighted Overall Score'].min(), df['Weighted Overall Score'].max()
    if overall_min == overall_max:
        df['Normalized Overall Score (Out of 10)'] = 1
    else:
        df['Normalized Overall Score (Out of 10)'] = (
            ((df['Weighted Overall Score'] - overall_min) / (overall_max - overall_min)) * 9 + 1
        )
        
    # Call the categorization function
    df = map_scores(df,"Normalized Overall Score (Out of 10)")

    # if 'ranking_categories' in config:
    #     for category, rank_info in config['ranking_categories'].items():
    #         if rank_info.get('dynamic_bins', False):
    #             labels = rank_info.get('labels', None)
    #             num_bins = len(labels) if labels else 5

    #             df[category] = pd.qcut(
    #                 df['Normalized Overall Score (Out of 10)'].dropna(),
    #                 q=num_bins,
    #                 labels=labels if labels else range(1, num_bins + 1),
    #                 duplicates='drop'
    #             )

    return df

def create_summary_table(df, config):
    summary_table = []
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for column, rules in config['columns'].items():
        source_column = rules['source']
        weight = rules.get('weight', 1)

        if source_column not in df.columns:
            print(f"Warning: Source column '{source_column}' not found in DataFrame.")
            continue

        data = df[source_column].dropna()
        feature_name = source_column

        if rules['type'] == 'range':
            if len(data) > 0:
                if source_column == 'Odometer exchange/rewind possibility detection - Distance':
                    # Use custom dynamic binning logic for odometer column
                    num_bins, bin_edges = dynamic_bins_for_odometer(data.values)
                    bin_width = None  # Bin width doesn't apply cleanly here
                    binning_method = "Dynamic (Odometer-specific)"
                elif source_column == 'DTC Count':
                    # Use custom dynamic binning logic for DTC Count column
                    num_bins, bin_edges = dynamic_bins_for_dtc(data.values)
                    bin_width = None  # Bin width doesn't apply cleanly here
                    binning_method = "Dynamic (DTC Count-specific)"
                elif source_column in ['Number of Current Warnings', 'Number of Historical Warnings',
                                       'Fuel efficiency (Km/l)', 'Average Mileage (Km/Trip)', 'car_age', 'collision count']:
                    # Use the custom dynamic binning logic for Fuel Efficiency, Average Mileage, and Car Age
                    num_bins, bin_edges = dynamic_bins_warnings_fuel_mileage_age_collision(df[source_column])
                    bin_width = round((data.max() - data.min()) / num_bins, 2)
                    binning_method = f"Dynamic ({source_column}-specific)"
                else:
                    # Use Freedman-Diaconis for other columns
                    q75, q25 = np.percentile(data, [75, 25])
                    iqr = q75 - q25
                    bin_width = 2 * iqr / (len(data) ** (1 / 3))
                    num_bins = int(np.ceil((data.max() - data.min()) / bin_width)) if bin_width > 0 else 1
                    binning_method = f"Freedman-Diaconis (Bin width: {bin_width:.2f})" if bin_width > 0 else "0 (Single Bin)"

                summary_table.append({
                    "Feature Name": feature_name,
                    "Number of Bins": num_bins,
                    "Bin Width": bin_width if bin_width is not None else binning_method,
                    "Weight": weight,
                    "MIN Value": data.min(),
                    "MAX Value": data.max(),
                    "Updation Date": current_date
                })
            else:
                summary_table.append({
                    "Feature Name": feature_name,
                    "Number of Bins": 0,
                    "Bin Width": None,
                    "Weight": weight,
                    "MIN Value": None,
                    "MAX Value": None,
                    "Updation Date": current_date
                })

        elif rules['type'] == 'map':
            if len(data) > 0:
                if pd.api.types.is_numeric_dtype(data) or data.dtype == 'object':
                    summary_table.append({
                        "Feature Name": feature_name,
                        "Number of Bins": 'Custom (Map)',
                        "Bin Width": 'Custom (Map)',
                        "Weight": weight,
                        "MIN Value": data.min(),
                        "MAX Value": data.max(),
                        "Updation Date": current_date
                    })
                else:
                    summary_table.append({
                        "Feature Name": feature_name,
                        "Number of Bins": len(data.unique()),
                        "Bin Width": 'Categorical',
                        "Weight": weight,
                        "MIN Value": 'Categorical',
                        "MAX Value": 'Categorical',
                        "Updation Date": current_date
                    })
            else:
                summary_table.append({
                    "Feature Name": feature_name,
                    "Number of Bins": 0,
                    "Bin Width": None,
                    "Weight": weight,
                    "MIN Value": None,
                    "MAX Value": None,
                    "Updation Date": current_date
                })

    return pd.DataFrame(summary_table)