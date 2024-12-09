import streamlit as st
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import datetime
import numpy as np
from scipy import stats
#import os
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
#from glob import glob
#import dask.dataframe as dd
#from tqdm import tqdm
#import warnings
#warnings.filterwarnings("ignore")
from zipfile import ZipFile
import vars

# Function to load and select columns from a list of files
def load_and_select(files, columns):
    dfs = [pd.read_csv(file)[ columns] for file in files]
    return pd.concat(dfs)

# Function to filter rows based on t_stamp format
def filter_and_parse_dates(df, date_format):
    # Convert t_stamp with strict parsing and drop invalid rows
    df['t_stamp'] = pd.to_datetime(df['t_stamp'], format=date_format, errors='coerce')
    df = df.dropna(subset=['t_stamp'])  # Keep only rows with valid dates
    return df

#Function to filter data based on metadata file
def filter_columns(df, columns_to_keep):
    # Ensure 't_stamp' is in the list of columns to keep
    if 't_stamp' not in columns_to_keep:
        columns_to_keep.insert(0, 't_stamp')  # Add it back to the beginning if needed

    # Print columns before filtering for debugging
    #print("Columns before filtering:", df.columns.tolist())

    # Filter DataFrame to keep only specified columns
    filtered_df = df[columns_to_keep]  # This will raise an error if columns do not exist
    return filtered_df

def group_data_by_metadata(df, column_groups):
    grouped_data = {}
    
    for group_name, columns in column_groups.items():
        # Filter the main DataFrame for each group and store it
        grouped_data[group_name] = df[columns] if all(col in df.columns for col in columns) else None
    
    return grouped_data

def average_if(row, columns):
    values = [row[col] for col in columns]
    valid_values = [v for v in values if v > 0]
    return pd.Series(valid_values).mean() if valid_values else 0

# Helper function to generate floating point ranges
def frange(start, stop, step):
    while start < stop:
        yield round(start, 2)  # Round to 2 decimal places
        start += step

# Function to loop through reporting condition thresholds
def outdated_loop_rc_threshold(min_rc, max_rc, step_size, rc_poa_total, merged_df):
    # Initialize an empty list to store results before starting the loop
    results = []
    
    # Loop over the range of thresholds (min_rc to max_rc with step_size)
    for reporting_condition_thresold in frange(min_rc, max_rc, step_size):
        
        # Get min and max RC
        reporting_condition_thresold_min = (1 - reporting_condition_thresold) * rc_poa_total
        reporting_condition_thresold_max = (1 + reporting_condition_thresold) * rc_poa_total
        
        # Apply the filter condition to the DF
        merged_df['rc_check'] = merged_df['average_poa_total'].between(reporting_condition_thresold_min, reporting_condition_thresold_max)
        merged_df['secondary_filter'] = merged_df['primary_filters'] * merged_df['rc_check']

        # Count the 'Including' and 'Excluding' values
        count_rc_condition_threshold = merged_df['secondary_filter'].value_counts().rename(index={True: "Including", False: "Excluding"})
        
        # Calculate the percentage of 'Including' above and below the threshold
        total_secondary_filter_true = (merged_df['secondary_filter'] == True).sum()
        if total_secondary_filter_true > 0:
            secondary_above_rc_perc = ((merged_df['secondary_filter'] == True) & (merged_df['average_poa_total'] >= rc_poa_total)).sum() / total_secondary_filter_true * 100
        else:
            secondary_above_rc_perc = 0
        secondary_below_rc_perc = 100 - secondary_above_rc_perc
        
        # Get the total counts for Including and Excluding
        count_including = count_rc_condition_threshold.get("Including", 0)
        count_excluding = count_rc_condition_threshold.get("Excluding", 0)

        # Get daily counts of 'Including' and 'Excluding'
        measured_regression_df = merged_df[merged_df['secondary_filter'] == True]
        count_secondary_filters_per_day = measured_regression_df.groupby(measured_regression_df['t_stamp'].dt.date)['secondary_filter'].value_counts().unstack().fillna(0).rename(columns={True: "Including", False: "Excluding"})
        
        
        #Format results table, but currently not working
        result = {
            'Threshold': reporting_condition_thresold,
            'Total number of points': count_including,
            'Excluding Count': count_excluding,
            'Secondary Above RC %': secondary_above_rc_perc,
            'Secondary Below RC %': secondary_below_rc_perc
        }


        results.append(result)
    results_df = pd.DataFrame(results)

    return results_df