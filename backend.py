import streamlit as st
import pandas as pd
import math
import plotly.io as pio
import plotly.graph_objects as go
import datetime
import numpy as np
from scipy import stats
import os
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from glob import glob
#import dask.dataframe as dd
from tqdm import tqdm
#import warnings
#warnings.filterwarnings("ignore")
from zipfile import ZipFile

# Initialize Plotly's notebook mode
pio.renderers.default = 'notebook_connected'  # For interactive plots

# Specify the main directory containing folders of daily CSV files
main_directory = "2.Raw Data"
metadata_file_path = 'SCADA Tags_Liberty.xlsx'  # Path to your metadata file
date_format = "%Y-%m-%d %H:%M:%S.%f"

# Define the start and end dates as datetime objects
test_start_date = pd.to_datetime('2024-10-10')
test_end_date = pd.to_datetime('2024-10-14')

##IRRADIANCE Inputs

minimum_irradiance=400    ## Minimum Irradiance per contract 
max_irradiance=1500      ## Max irradiance allowed, typically higher then 1200W/m2 would see clipping. This can be flexible   
temporal_stability_thresold=20  ## Check if the avgerage_poa value is within 1 min interval along row is stable 
spatial_stability_thresold=0.20

## Grid inputs 

minimum_grid=0                   ## Typically needs all positive value at grid   
grid_clipping_thresold=0.98    ## Per contract but typically this should be 0.98 per contract to get stable irradiance data
max_gridlimit=99600              ## Use max grid limit value per proeject design    
grid_clipping=grid_clipping_thresold*max_gridlimit

## RC value 
            ## Use to calculate the RC Value, by default RC is calculated as averqge value. If using percentile, please change below manually

percentile=0.50
## Secondary Filters

####KL note, start w/ reporting condition_threshold loop 
reporting_condition_thresold=0.20


## Inverter Inputs 

inverter_rating=3600               ## Use inverter rated capacity per design
inverter_clipping_thresold=0.98    ## This is typically standard assumption but can increase to 0.99 if need more points or see lot of clipping
inverter_clipping=inverter_rating*inverter_clipping_thresold
Inverter_limit=118800



pvsyst_shading=1
pvsyst_test_model_path="Test Model/Liberty_Project_VC0_track_PB33-LBD12 off 0.196.csv"
   ## RC value can be adjusted to +/- 50% range to get more points but the regression equation will be unstable with higher values 



bifaciality=0.7

availability_min_fpoa=50
system_size_dc=134046


min_poa_soiling=150
soiling_with_iv_curve=1.52   ## from EPC 

##Set these variables if you want to loop through RC values --KL addition 11/12
min_rc = 0.1
max_rc = 0.6
step_size = 0.05

####################################################################################################################################################
#Optional Inputs / Not used yet

# rc_poa_total=rc_avg_poa_total 
# rc_fpoa=rc_avg_fpoa 
# rc_rpoa=rc_avg_rpoa 
# rc_temp=rc_avg_temp 
# rc_wind=rc_avg_wind 


# rc_poa_total=percentile_avg_poa_total
# rc_fpoa=percentile_avg_fpoa
# rc_rpoa=percentile_avg_rpoa
# rc_temp=percentile_avg_temp
# rc_wind=percentile_avg_wind

# rc_poa_total=700
# rc_fpoa=650
# rc_rpoa=50
# rc_temp=25
# rc_wind=1

###PVSyst Inputs:
#minimum_irradiance=400


# inverter_clipping_limit=0.99
# Inverter_limit_afterclipping=Inverter_limit*inverter_clipping_limit

# grid_limit= 99600
# grid_clipping_limit=0.98
# POI_limit_afterclipping=grid_limit*grid_clipping_limit

####Start the reading of data
# Initialize an empty Dask DataFrame
all_dfs = []
    
# Gather all file paths
all_files = []
for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)
    if os.path.isdir(folder_path):  # Only process folders
        csv_files = glob(os.path.join(folder_path, "*.csv"))
        all_files.extend(csv_files)


for file in tqdm(all_files, desc="Reading files"):
    df = pd.read_csv(file)
    df = funcs.filter_and_parse_dates(df, date_format)  # Assuming you have a function to filter and parse dates
    all_dfs.append(df)

# Concatenate all DataFrames vertically
all_data = pd.concat(all_dfs, axis=0, ignore_index=True)

# Track the progress of computing and group by t_stamp
with tqdm(total=1, desc="Grouping and computing") as pbar:
    # Group by 't_stamp' and keep the first row for each group
    all_data = all_data.groupby('t_stamp').first()
    pbar.update(1)

# Step 2: Load the metadata and get columns to keep
metadata_df = pd.read_excel(metadata_file_path, header=None)
columns_to_keep = [col.strip() for col in metadata_df[1].dropna().tolist()]  # Assuming column B has the names
#column_groups = [col.strip() for col in metadata_df[1].dropna().tolist()]  # Assuming column B has the names

# Ensure 't_stamp' is in the list of columns to keep
if 't_stamp' not in columns_to_keep:
    columns_to_keep.insert(0, 't_stamp')  # Add it back to the beginning if needed

# Step 3: Filter the merged DataFrame
filtered_df = funcs.filter_columns(all_data.reset_index(), columns_to_keep)  # Reset index to keep 't_stamp' as a column
merged_df_all = filtered_df.set_index('t_stamp')
#merged_df = merged_df_all.drop((merged_df_all.index >= test_start_date) & (merged_df_all.index <= test_end_date))
merged_df = merged_df_all
merged_df_all