import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import datetime
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import zipfile
from scipy import stats
import os
from glob import glob
import warnings
warnings.filterwarnings("ignore")
#from dateutil.relativedelta import relativedelta
#import statsmodels.api as sm
#import math
#import plotly.io as pio
#import dask.dataframe as dd
import vars, funcs, filters

# Page Setup

st.set_page_config(page_title= "PV Cap Test", page_icon="icon.jpg", layout="wide")
#st.logo("long_logo.jpg")

st.image("long_logo.jpg", width=300)
st.title("PV Cap Test")
#st.header("Date: " + str(datetime.date.today()))
tab1, tab2, tab3 = st.tabs(['Data Upload', 'Inputs', 'Report'])

# Tab 1: Data Upload

# Specify the main directory containing folders of daily CSV files
main_directory = tab1.text_input("Name of ZIP File (do not include .zip)", "2.Raw Data")
metadata_file_path = tab1.text_input("Metadata File Path", "SCADA Tags_Liberty.xlsx")  # Path to your metadata file
date_format = tab1.text_input("Date Format", "%Y-%m-%d %H:%M:%S.%f")

uploaded_zip = tab1.file_uploader("Upload raw data", type='zip')
column_groups = tab1.file_uploader("Upload column groups", type='xlsx')
pvsyst_test_model_path = tab1.file_uploader("Upload PVSyst test model", type='csv')

# Tab 2: Inputs

form1 = tab2.form("inputs form")

form1_col1, form1_col2 = form1.columns(2)
test_start_date = datetime.datetime.combine(form1_col1.date_input("Start Date", pd.to_datetime('2024-10-10'), format='MM/DD/YYYY'), datetime.datetime.min.time())
test_end_date = datetime.datetime.combine(form1_col2.date_input("End Date", pd.to_datetime('2024-10-14'), format='MM/DD/YYYY'), datetime.datetime.min.time())

form1.subheader("Irradiance Inputs:")
form1_col1, form1_col2 = form1.columns(2)
minimum_irradiance = form1_col1.number_input("Minimum Irradiance (W/m^2):", min_value=0, value=400, step=100)
max_irradiance = form1_col2.number_input("Maximum Irradiance (W/m^2):", min_value=minimum_irradiance, value=1500, step=100)
temporal_stability_thresold = form1_col1.number_input("Temporal Stability Threshold:", min_value=0, value=20, step=1)
spatial_stability_thresold = form1_col2.number_input("Spatial Stability Threshold:", min_value=0.0, value=0.05, step=0.10)

form1.subheader("Grid Inputs:")
form1_col1, form1_col2 = form1.columns(2)
minimum_grid = form1_col1.number_input("Minimum Grid Value:", value=0, step=100)
max_gridlimit = form1_col2.number_input("Maximum Grid Value:", value=99600, step=100)
grid_clipping_thresold = form1.number_input("Grid Clipping Threshold:", value=0.98, step=0.01)
grid_clipping = grid_clipping_thresold * max_gridlimit

form1.subheader("RC Inputs:")
form1_col1, form1_col2 = form1.columns(2)
percentile = form1_col1.number_input("Percentile:", min_value=0.0, max_value=1.0, value=0.50, step=0.10)
reporting_condition_thresold = form1_col2.number_input("Reporting Condition Threshold:", value=0.2, min_value=0.0, step=0.1)
min_rc = form1_col1.number_input("Min RC Threshold:", min_value=0.0, max_value=1.0, value=0.1, step=0.1)
max_rc = form1_col2.number_input("Max RC Threshold:", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
step_size = form1.number_input("RC Step Size:", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

form1.subheader("Inverter Inputs:")
form1_col1, form1_col2 = form1.columns(2)
inverter_rating = form1_col1.number_input("Inverter Rating:", min_value=0, value=3600, step=100)
Inverter_limit = form1_col2.number_input("Inverter Limit:", min_value=0, value=118800, step=100)
inverter_clipping_thresold = form1_col1.number_input("Inverter Clipping Threshold:", min_value=0.0, value=0.98, max_value=1.0, step=0.01)
inverter_clipping = inverter_rating * inverter_clipping_thresold

form1.subheader("Other Inputs:")
passing_capacity = form1.number_input("Passing Capacity (Bifacial):", min_value=0.0, value=97.0, max_value=100.0)
pvsyst_shading = form1.number_input("PVSyst Shading:", min_value=0, value=1, step=1)
bifaciality = form1.number_input("Bifaciality", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
availability_min_fpoa = form1.number_input("Availability Minimum FPOA", value=50, min_value=0, step=1)
system_size_dc = form1.number_input("System Size DC", value=134046, min_value=0, step=1)
min_poa_soiling = form1.number_input("Min POA Soiling:", min_value=0, value=150, step=1)
soiling_with_iv_curve = form1.number_input("Soiling with IV Curve:", min_value=0.0, value=1.52, step=0.01) ## from EPC 

form1.form_submit_button("Submit Inputs")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend begin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(".")
if uploaded_zip is None or column_groups is None or pvsyst_test_model_path is None:
    tab2.write('Upload files to proceed.')
    tab3.write('Upload files to proceed.')
    st.stop()

# Initialize an empty Dask DataFrame
all_dfs = []
    
# Gather all file paths
all_files = []
for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)
    if os.path.isdir(folder_path):  # Only process folders
        csv_files = glob(os.path.join(folder_path, "*.csv"))
        all_files.extend(csv_files)


for file in all_files:
    df = pd.read_csv(file)
    df = funcs.filter_and_parse_dates(df, date_format)  # Assuming you have a function to filter and parse dates
    all_dfs.append(df)

# Concatenate all DataFrames vertically
all_data = pd.concat(all_dfs, axis=0, ignore_index=True)

# group by t_stamp
all_data = all_data.groupby('t_stamp').first()

# Step 2: Load the metadata and get columns to keep
metadata_df = pd.read_excel(column_groups, header=None)
columns_to_keep = [col.strip() for col in metadata_df[1].dropna().tolist()]  # Assuming column B has the names

# Ensure 't_stamp' is in the list of columns to keep
if 't_stamp' not in columns_to_keep:
    columns_to_keep.insert(0, 't_stamp')  # Add it back to the beginning if needed

# Step 3: Filter the merged DataFrame
merged_df = funcs.filter_columns(all_data.reset_index(), columns_to_keep).set_index('t_stamp')  # Reset index to keep 't_stamp' as a column
merged_df = merged_df[(merged_df.index >= test_start_date) & (merged_df.index <= test_end_date)]

metadata_df = pd.read_excel(column_groups, header=None)  # Adjust header if necessary
column_groups = {}

current_group = None

# Iterate through the rows of the metadata DataFrame
for index, row in metadata_df.iterrows():
    # Check if the first column (Column A) has a group name (e.g., "FPOA")
    if pd.notna(row[0]):
        current_group = row[0].strip()  # Set the current group
        column_groups[current_group] = []  # Initialize an empty list for this group
    
    # If there's a column name in Column B, add it to the current group
    if pd.notna(row[1]) and current_group:
        column_groups[current_group].append(row[1].strip())

grouped_data = funcs.group_data_by_metadata(merged_df, column_groups)
fpoa_data = grouped_data.get('FPOA', None)
rpoa_data = grouped_data.get('RPOA', None)
temp_data = grouped_data.get('Temp', None)
wind_data = grouped_data.get('Wind', None)
soiling_data = grouped_data.get('Soiling Ratio', None)
inverter_data = grouped_data.get('Inverter', None)
meter_data = grouped_data.get('meter', None)

merged_df['t_stamp'] = pd.to_datetime(merged_df.index)
merged_df['t_stamp_check'] = (merged_df['t_stamp'] >= test_start_date) & (merged_df['t_stamp'] <= test_end_date)
merged_df['data_check_inv'] = merged_df[inverter_data.columns].notna().all(axis=1)

# Apply the functions to each row and create new columns
merged_df['average_fpoa'] = merged_df.apply(lambda row: funcs.average_if(row, fpoa_data), axis=1)
merged_df['average_rpoa'] = merged_df.apply(lambda row: funcs.average_if(row, rpoa_data), axis=1)
merged_df['average_poa_total'] = (merged_df['average_fpoa']+(merged_df['average_rpoa']*bifaciality))
merged_df['average_temp'] = merged_df.apply(lambda row: funcs.average_if(row, temp_data), axis=1)
merged_df['average_wind'] = merged_df.apply(lambda row: funcs.average_if(row, wind_data), axis=1)
merged_df['average_meter_data'] = merged_df.apply(lambda row: funcs.average_if(row, meter_data), axis=1)
merged_df['sp. yield']=(merged_df['average_meter_data']/system_size_dc)
merged_df['average_soiling'] = merged_df.apply(lambda row: funcs.average_if(row, soiling_data), axis=1)

avg_soiling=((merged_df['average_fpoa']>min_poa_soiling)*(merged_df['average_soiling'])).mean()

count_avail_poa = ((merged_df['average_fpoa'] >= availability_min_fpoa)*merged_df['t_stamp_check']).sum()

#Added by KL to count per column how many inverters were did not hit criteria
merged_df['inverter_count'] = merged_df.apply(
    lambda row: sum(
        (row[column] < 50) and
        (row['average_fpoa'] > availability_min_fpoa) and
        row['t_stamp_check'] and
        row['data_check_inv']
        for column in inverter_data), axis = 1)

##Added by KL to calculate lost capacity of each averaging interval and grphing inverter avail for start to end data
merged_df['lost_capac'] = 100 - ((merged_df['inverter_count'] * inverter_rating / max_gridlimit ) / 0.01)
merged_df.loc[merged_df['lost_capac'] < 0, 'lost_capac'] = 0

counts = {}

# Loop through each inverter column
for column in inverter_data.columns:
    # Calculate the difference and count the occurrences where the difference is greater than 150
    counts[column] = (((merged_df[column] > 50) & (merged_df['average_fpoa'] > availability_min_fpoa))).sum()

# Divide the counts by count_avail_poa
for key in counts:
    counts[key] /= count_avail_poa
# Convert the counts dictionary to a dataframe for better readability
avail_counts_df = pd.DataFrame(list(counts.items()), columns=['Inverter', 'Availabiliy'])
avail_average = round(avail_counts_df['Availabiliy'].mean()*100, 2)

# Apply the criteria and create a filtered DataFrame for the heatmap
filtered_data = pd.DataFrame()

# Apply conditions to each inverter column and populate filtered_data
for column in inverter_data:
    filtered_data[column] = ((merged_df[column].fillna(0) < 50) & 
                             (merged_df['average_fpoa'].fillna(0) > availability_min_fpoa)).astype(int)

filtered_data.index = merged_df['t_stamp']

inverter_df = inverter_data

#Define the filters here, calling from functions defined above
filter_registry = [
    ("Meter > 0", filters.filter_meter_greater_zero, [minimum_grid]),  
    ("Grid Clipping", filters.filter_grid_clipping, [grid_clipping]),  
    ("Inverter Clipping", filters.filter_inverter_clipping, [inverter_df, inverter_clipping]),  
    ("Inverter is 0", filters.filter_inverter_zero, [inverter_df]),
    ("FPOA is blank", filters.filter_fpoa_blank, [fpoa_data]),
    ("FPOA is 0", filters.filter_fpoa_zero, [fpoa_data]),  
    ("RPOA is blank", filters.filter_rpoa_blank, [rpoa_data]),
    ("RPOA is zero", filters.filter_rpoa_zero, [rpoa_data]),
    ("Temp Blank", filters.filter_temp_blank, [temp_data]),
    ("Temp is 0", filters.filter_temp_zero, [temp_data]),
    (" Wind Blank", filters.filter_wind_blank, [wind_data]),
    ("Wind is 0", filters.filter_wind_zero, [wind_data]),
    ("FPOA QC", filters.filter_fpoa_qc, [minimum_irradiance, max_irradiance]),
    ("Spatial Stability Check", filters.filter_spatial_stability, [fpoa_data, spatial_stability_thresold]),
    ("Temporal Stability Check", filters.filter_temporal_stability, [temporal_stability_thresold])]

# Initialize the DataFrame to track cumulative conditions
filter_results = []

# Initialize starting points and condition
remaining_condition = pd.Series(True, index=merged_df.index)
remaining_points = len(merged_df)
initial_points = remaining_points

for idx, (filter_name, filter_function, filter_args) in enumerate(filter_registry, start=1):
    # Apply filter to the remaining points
    current_condition = filter_function(merged_df, *filter_args)
    
    # Combine with the remaining condition from previous filters
    combined_condition = remaining_condition & current_condition
    
    # Calculate lost and remaining points
    lost_points = (~combined_condition & remaining_condition).sum()
    remaining_points = combined_condition.sum()
    
    # Add the filter's results to the table
    filter_results.append({
        #"Filter Number": f"Filter {idx}",
        "Filter Description": filter_name,
        "Initial Points": initial_points,
        "Points Lost": lost_points,
        "Remaining Points": remaining_points,
        #"Filter Description": filter_name,
    })
    
    # Update the remaining condition 
    remaining_condition = combined_condition
    initial_points = remaining_points  # Remaining points become initial points for the next filter

# Put results in DF
filter_results_df = pd.DataFrame(filter_results).set_index('Filter Description')

# Apply filters and store the results in new columns
merged_df['meter>0'] = filters.filter_meter_greater_zero(merged_df, minimum_grid)
merged_df['grid_clipping'] = filters.filter_grid_clipping(merged_df, grid_clipping)
merged_df['inverter_clipping_check'] = filters.filter_inverter_clipping(merged_df, inverter_df)
merged_df['inverter_blank'] = filters.filter_inverter_zero(merged_df, inverter_df)
merged_df['inverter_zero'] = filters.filter_inverter_zero(merged_df, inverter_df)
merged_df['fpoa_blank'] = filters.filter_fpoa_blank(merged_df, fpoa_data)
merged_df['fpoa_zero'] = filters.filter_fpoa_zero(merged_df, fpoa_data)
merged_df['temp_blank'] = filters.filter_temp_blank(merged_df, temp_data)
merged_df['temp_zero'] = filters.filter_temp_zero(merged_df, temp_data)
merged_df['wind_blank'] = filters.filter_wind_blank(merged_df, wind_data)
merged_df['wind_zero'] = filters.filter_wind_zero(merged_df, wind_data)
merged_df['fpoa_QC'] = filters.filter_fpoa_qc(merged_df, minimum_irradiance, max_irradiance)
merged_df['spatial_stability_check'] = filters.filter_spatial_stability(merged_df, fpoa_data, spatial_stability_thresold)
merged_df['temporal_stability_check'] = filters.filter_temporal_stability(merged_df, temporal_stability_thresold)

# Calculate the 'primary_filters' column
merged_df['primary_filters'] = (
    merged_df['t_stamp_check'] *
    merged_df['meter>0'] *
    merged_df['grid_clipping'] *
    merged_df['fpoa_QC'] *
    merged_df['spatial_stability_check'] *
    merged_df['temporal_stability_check'] *
    merged_df['inverter_clipping_check'] *
    merged_df['inverter_blank'] *
    merged_df['inverter_zero'] *
    merged_df['fpoa_blank'] *
    merged_df['temp_blank'] *
    merged_df['wind_blank'] *
    merged_df['temp_zero'] *
    merged_df['fpoa_zero'] *
    merged_df['wind_zero']
)

# Count "Including" and "Excluding" for primary filters overall
count_primary_filters = merged_df['primary_filters'].value_counts().rename(index={True: "Including", False: "Excluding"})

# Group by date and calculate counts of "Including" and "Excluding"
count_primary_filters_per_day = merged_df.groupby(merged_df['t_stamp'].dt.date)['primary_filters'].value_counts().unstack().fillna(0).rename(columns={True: "Including", False: "Excluding"})

# Display the table for counts per day
count_primary_filters_per_day_df = count_primary_filters_per_day

including_points_PF = count_primary_filters.get('Including', 0)
excluding_points_PF = count_primary_filters.get('Excluding', 0)