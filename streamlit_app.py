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
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
#from dateutil.relativedelta import relativedelta
#import statsmodels.api as sm
#import math
#import plotly.io as pio
#import dask.dataframe as dd
import vars
import funcs

# Page Setup

st.set_page_config(page_title= "PV Cap Test", page_icon="icon.jpg", layout="wide")
#st.logo("long_logo.jpg")

st.image("long_logo.jpg", width=300)
st.title("PV Cap Test")
#st.header("Date: " + str(datetime.date.today()))
tab1, tab2, tab3 = st.tabs(['Data Upload', 'Inputs', 'Report'])

# Tab 1: Data Upload

# Specify the main directory containing folders of daily CSV files
main_directory = tab1.text_input("Main Directory", "2.Raw Data")
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
#column_groups = [col.strip() for col in metadata_df[1].dropna().tolist()]  # Assuming column B has the names

# Ensure 't_stamp' is in the list of columns to keep
if 't_stamp' not in columns_to_keep:
    columns_to_keep.insert(0, 't_stamp')  # Add it back to the beginning if needed

# Step 3: Filter the merged DataFrame
filtered_df = funcs.filter_columns(all_data.reset_index(), columns_to_keep)  # Reset index to keep 't_stamp' as a column
merged_df_all = filtered_df.set_index('t_stamp')
merged_df = merged_df_all
merged_df = merged_df_all[(merged_df_all.index >= test_start_date) & (merged_df_all.index <= test_end_date)]