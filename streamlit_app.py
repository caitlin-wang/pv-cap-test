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
#from glob import glob
#import dask.dataframe as dd
#from tqdm import tqdm
#import warnings
#warnings.filterwarnings("ignore")
import zipfile
import vars
import funcs

# Page Setup

st.set_page_config(page_title= "PV Cap Test", page_icon="icon.jpg", layout="wide")
st.logo("long_logo.jpg", icon_image="icon.jpg")

st.sidebar.subheader("Next Steps:")
st.sidebar.write("- figure out how to process ZIP file")
st.sidebar.subheader("Edit Log:")
st.sidebar.write("- 11/6/24: added inputs tab")
st.sidebar.write("- 11/5/24: created page")

st.title("PV Cap Test")
tab1, tab2, tab3 = st.tabs(['Data Upload', 'Inputs', 'Report'])

# Tab 1: Data Upload

# Specify the main directory containing folders of daily CSV files
main_directory = tab1.text_input("Main Directory", "2.Raw Data")
metadata_file_path = tab1.text_input("Metadata File Path", "SCADA Tags_Liberty.xlsx")  # Path to your metadata file
date_format = tab1.text_input("Date Format", "%Y-%m-%d %H:%M:%S.%f")

uploaded_zip = tab1.file_uploader("Upload raw data", type='zip')
column_groups = tab1.file_uploader("Upload column groups", type=['csv','xlsx'])
pvsyst_test_model_path = tab1.file_uploader("Upload PVSyst test model", type=['csv'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend begin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(".")
        tab1.write(os.listdir('2.Raw Data'))
else:
    tab2.write('Upload files to proceed.')
    tab3.write('Upload files to proceed.')
    st.stop()

# Load and select columns for MET, Inverter, Meter files 
df1_combined = funcs.load_and_select(vars.files_met, vars.met_cols)
df2_combined = funcs.load_and_select(vars.files_inverter, vars.inverter_cols)
df3_combined = funcs.load_and_select(vars.files_meter, vars.meter_cols)

# Merge the combined dataframes 
merged_df = pd.merge(df1_combined, df2_combined)
merged_df=pd.merge(merged_df, df3_combined)

# Assuming merged_df is your DataFrame and t_stamp is your x-axis column
fig = go.Figure()
for col in vars.y_columns:
    fig.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig.update_layout(
    title='Weather Station and Inverter Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Tab 2: Inputs

form1 = tab2.form("inputs form")

form1.subheader("Irradiance Inputs:")
form1_col1, form1_col2 = form1.columns(2)
test_start_date = form1_col1.date_input("Start Date", 'today', format='MM/DD/YYYY')
test_end_date = form1_col2.date_input("End Date", 'today', format='MM/DD/YYYY')
minimum_irradiance = form1_col1.number_input("Minimum Irradiance (W/m^2):", min_value=0, value=400, step=100)
max_irradiance = form1_col2.number_input("Maximum Irradiance (W/m^2):", min_value=minimum_irradiance, value=1500, step=100)
temporal_stability_thresold = form1_col1.number_input("Temporal Stability Threshold:", min_value=0, value=20, step=1)
spatial_stability_thresold = form1_col2.number_input("Spatial Stability Threshold:", min_value=0.0, value=0.20, step=0.10)

form1.subheader("Grid Inputs:")
form1_col1, form1_col2 = form1.columns(2)
minimum_grid = form1_col1.number_input("Minimum Grid Value:", value=0, step=100)
max_gridlimit = form1_col2.number_input("Maximum Grid Value:", value=99600, step=100)
grid_clipping_thresold = form1.number_input("Grid Clipping Threshold:", value=0.98, step=0.01)
grid_clipping = grid_clipping_thresold * max_gridlimit

form1.subheader("RC Inputs:")
form1_col1, form1_col2 = form1.columns(2)
percentile = form1_col1.number_input("Percentile:", min_value=0.0, max_value=1.0, value=0.50, step=0.10)
reporting_condition_thresold = form1_col2.number_input("Reporting Condition Threshold:", value=0.20, min_value=0.0, step=0.01)

form1.subheader("Inverter Inputs:")
form1_col1, form1_col2 = form1.columns(2)
inverter_rating = form1_col1.number_input("Inverter Rating:", min_value=0, value=3600, step=100)
Inverter_limit = form1_col2.number_input("Inverter Limit:", min_value=0, value=118800, step=100)
inverter_clipping_thresold = form1_col1.number_input("Inverter Clipping Threshold:", min_value=0.0, value=0.98, max_value=1.0, step=0.01)
inverter_clipping = inverter_rating * inverter_clipping_thresold

form1.subheader("Other Inputs:")
pvsyst_shading = form1.number_input("PVSyst Shading:", min_value=0, value=1, step=1)
bifaciality = form1.number_input("Bifaciality", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
availability_min_fpoa = form1.number_input("Availability Minimum FPOA", value=50, min_value=0, step=1)
system_size_dc = form1.number_input("System Size DC", value=134046, min_value=0, step=1)

form1.form_submit_button("Submit Inputs")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend begin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

merged_df['t_stamp'] = pd.to_datetime(merged_df['t_stamp'])
merged_df['t_stamp_check'] = (merged_df['t_stamp'] >= test_start_date) & (merged_df['t_stamp'] <= test_end_date)

# Apply the function to each row and create a new column 'average_fpoa'
merged_df['average_fpoa'] = merged_df.apply(lambda row: funcs.average_if(row, vars.fpoa_data), axis=1)

# Apply the function to each row and create a new column 'average_temp'
merged_df['average_rpoa'] = merged_df.apply(lambda row: funcs.average_if(row, vars.rpoa_data), axis=1)

# Apply the function to each row and create a new column 'average_poa_total'
merged_df['average_poa_total'] = (merged_df['average_fpoa']+(merged_df['average_rpoa']*bifaciality))

# Apply the function to each row and create a new column 'average_temp'
merged_df['average_temp'] = merged_df.apply(lambda row: funcs.average_if(row, vars.temp_data), axis=1)

# Apply the function to each row and create a new column 'average_temp'
merged_df['average_wind'] = merged_df.apply(lambda row: funcs.average_if(row, vars.wind_data), axis=1)

merged_df['sp. yield']=(merged_df[vars.meter_data]/system_size_dc)
# Display the DataFrame

# Apply the function to each row and create a new column 'average_fpoa'
merged_df['average_soiling'] = merged_df.apply(lambda row: funcs.average_if(row, vars.soiling_data), axis=1)

avg_soiling=((merged_df['average_fpoa']>vars.min_poa_soiling)*(merged_df['average_soiling'])).mean()
tab3.write(avg_soiling)
avg_soiling_met5=((merged_df['average_fpoa']>vars.min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET05/DustVue/soilingRatio_pct'])).mean()
tab3.write(f"Average soiling for met 5   :{avg_soiling_met5}")
avg_soiling_met15=((merged_df['average_fpoa']>vars.min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET15/DustVue/soilingRatio_pct'])).mean()
tab3.write(f"Average soiling for met 15   :{avg_soiling_met15}")
avg_soiling_met21=((merged_df['average_fpoa']>vars.min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET21/DustVue/soilingRatio_pct'])).mean()
tab3.write(f"Average soiling for met 21   :{avg_soiling_met21}")
avg_soiling_met29=((merged_df['average_fpoa']>vars.min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET29/DustVue/soilingRatio_pct'])).mean()
tab3.write(f"Average soiling for met 29   :{avg_soiling_met29}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Tab 3: Report

tab3.write(merged_df) # merged_df
tab3.plotly_chart(fig) # initial data plot

tab3.write("congrats you passed 🎉")
tab3.write("click button below to access in-depth report :)")
tab3.link_button("Download report as PDF", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")