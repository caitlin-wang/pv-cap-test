import streamlit as st
import pandas as pd
#import math
#import plotly.io as pio
import plotly.graph_objects as go
import datetime
from dateutil.relativedelta import relativedelta 
import numpy as np
from scipy import stats
import os
import plotly.graph_objs as go
import statsmodels.api as sm
import plotly.express as px
#from glob import glob
#import dask.dataframe as dd
#from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import zipfile
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
column_groups = tab1.file_uploader("Upload column groups", type=['csv','xlsx'])
pvsyst_test_model_path = tab1.file_uploader("Upload PVSyst test model", type=['csv', 'xlsx'])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend begin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def filter_meter_greater_zero(df):
    return df['average_meter_data'] > minimum_grid

def filter_grid_clipping(df):
    return df['average_meter_data'] < grid_clipping

def filter_inverter_clipping(df, inverter_data):
    return inverter_data.apply(lambda row: row.max() < inverter_clipping, axis=1)

def filter_inverter_zero(df, inverter_data):
    return ~(inverter_data == 0).any(axis=1)
    
def filter_fpoa_blank(df, fpoa_data):
    return ~(fpoa_data.isnull()).any(axis=1)

def filter_fpoa_zero(df, fpoa_data):
    return ~(fpoa_data == 0).any(axis=1)

def filter_rpoa_blank(df, rpoa_data):
    return ~(rpoa_data.isnull()).any(axis=1)
    
def filter_rpoa_zero(df, rpoa_data):
    return ~(rpoa_data == 0).any(axis=1)

def filter_temp_blank(df, temp_data):
    return ~(temp_data.isnull().any(axis=1))

def filter_temp_zero(df, temp_data):
    return ~(temp_data == 0).any(axis=1)

def filter_wind_blank(df, wind_data):
    return ~(wind_data.isnull().any(axis=1))

def filter_wind_zero(df, wind_data):
    return ~(wind_data == 0).any(axis=1)

def filter_fpoa_qc(df, minimum_irradiance, max_irradiance):
    return df['average_fpoa'].between(minimum_irradiance, max_irradiance)

def filter_spatial_stability(df, fpoa_data, spatial_threshold):
    #df['average_fpoa_pct_change'] = df['average_fpoa'].pct_change() * 100  #Ask Ashish if we need this
    # df['temporal_stability_check'] = df['average_fpoa_pct_change'].abs() <= temporal_threshold #Ask Ashish if we need this
    df['fpoa_spatial_std'] = fpoa_data.std(axis=1)
    df['fpoa_spatial_mean'] = fpoa_data.mean(axis=1)
    df['fpoa_spatial_stability'] = df['fpoa_spatial_std'] / df['fpoa_spatial_mean']
    df['spatial_stability_check'] = df['fpoa_spatial_stability'].abs() <= spatial_threshold
    return df['spatial_stability_check']

def filter_temporal_stability(df, temporal_threshold):
    df['average_fpoa_pct_change']=df['average_fpoa'].pct_change()*100
    df['temporal_stability_check']=df['average_fpoa_pct_change'].abs()<=temporal_threshold
    return df['temporal_stability_check']

if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(".")
        #tab1.write(os.listdir('2.Raw Data'))
if uploaded_zip is None or pvsyst_test_model_path is None:
    tab2.write('Upload files to proceed.')
    tab3.write('Upload files to proceed.')
    st.stop()

# Load and select columns for MET, Inverter, Meter files 
df1_combined = funcs.load_and_select(vars.files_met, vars.met_cols)
df2_combined = funcs.load_and_select(vars.files_inverter, vars.inverter_cols)
df3_combined = funcs.load_and_select(vars.files_meter, vars.meter_cols)

# Merge the combined dataframes 
merged_df = pd.merge(df1_combined, df2_combined)
merged_df = pd.merge(merged_df, df3_combined)

# Assuming merged_df is your DataFrame and t_stamp is your x-axis column
fig = go.Figure()
for col in vars.y_columns:
    fig.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig.update_layout(
    title='Weather Station and Inverter Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1000)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Tab 2: Inputs

form1 = tab2.form("inputs form")

form1.subheader("Irradiance Inputs:")
form1_col1, form1_col2 = form1.columns(2)
test_start_date = datetime.datetime.combine(form1_col1.date_input("Start Date", pd.to_datetime('2024-10-10'), format='MM/DD/YYYY'), datetime.datetime.min.time())
test_end_date = datetime.datetime.combine(form1_col2.date_input("End Date", pd.to_datetime('2024-10-14'), format='MM/DD/YYYY'), datetime.datetime.min.time())
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
passing_capacity = form1.number_input("Passing Capacity (Bifacial):", min_value=0.0, value=0.97, max_value=1.0)
pvsyst_shading = form1.number_input("PVSyst Shading:", min_value=0, value=1, step=1)
bifaciality = form1.number_input("Bifaciality", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
availability_min_fpoa = form1.number_input("Availability Minimum FPOA", value=50, min_value=0, step=1)
system_size_dc = form1.number_input("System Size DC", value=134046, min_value=0, step=1)
min_poa_soiling = form1.number_input("Min POA Soiling:", min_value=0, value=150, step=1)
soiling_with_iv_curve = form1.number_input("Soiling with IV Curve:", min_value=0.0, value=1.52, step=0.01) ## from EPC 

form1.form_submit_button("Submit Inputs")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend begin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

merged_df['t_stamp'] = pd.to_datetime(merged_df['t_stamp'])
merged_df['t_stamp_check'] = (merged_df['t_stamp'] >= test_start_date) & (merged_df['t_stamp'] <= test_end_date)
merged_df['data_check_inv'] = merged_df[vars.inverter_data].notna().all(axis=1)
#st.write(test_start_date)
#st.write(type(test_start_date))
#st.write(merged_df['t_stamp'].dtype)

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

merged_df['average_meter_data'] = merged_df.apply(lambda row: funcs.average_if(row, vars.meter_data), axis=1)
merged_df['sp. yield']=(merged_df['average_meter_data']/system_size_dc)
# Display the DataFrame

# Apply the function to each row and create a new column 'average_fpoa'
merged_df['average_soiling'] = merged_df.apply(lambda row: funcs.average_if(row, vars.soiling_data), axis=1)

avg_soiling=((merged_df['average_fpoa']>min_poa_soiling)*(merged_df['average_soiling'])).mean()
avg_soiling_met5=((merged_df['average_fpoa']>min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET05/DustVue/soilingRatio_pct'])).mean()
avg_soiling_met15=((merged_df['average_fpoa']>min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET15/DustVue/soilingRatio_pct'])).mean()
avg_soiling_met21=((merged_df['average_fpoa']>min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET21/DustVue/soilingRatio_pct'])).mean()
avg_soiling_met29=((merged_df['average_fpoa']>min_poa_soiling)*(merged_df['LBSP1/Device/WeatherStation/MET29/DustVue/soilingRatio_pct'])).mean()

count_avail_poa = ((merged_df['average_fpoa'] >= availability_min_fpoa)*merged_df['t_stamp_check']).sum()
counts = {}

# Loop through each inverter column
for column in merged_df.columns:
    if 'LBSP1/Device/Inverter' in column and 'p3_kW' in column:
        
        # Calculate the difference and count the occurrences where the difference is greater than 150
        counts[column] = (((merged_df[column] > 50) & (merged_df['average_fpoa'] > availability_min_fpoa))*merged_df['t_stamp_check']).sum()

# Divide the counts by count_avail_poa
for key in counts:
    counts[key] /= count_avail_poa
# Convert the counts dictionary to a dataframe for better readability
avail_counts_df = pd.DataFrame(list(counts.items()), columns=['Inverter', 'Availabiliy'])
avail_average=avail_counts_df['Availabiliy'].mean()

## Applying filters to remove all data where meter value is positive and not clipping 

merged_df['meter>0']=merged_df[vars.meter_data]>minimum_grid       ## Using this to filter value greater than zero

count_meter_greaterzero=merged_df['meter>0'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['grid_clipping']=merged_df[vars.meter_data]<grid_clipping        ##Removing all data points at grid clipping to have stable point
count_grid_clipping=merged_df['grid_clipping'].value_counts().rename(index={True:"Including",False:"Excluding"})

count_meter_filter_data=(merged_df['meter>0']&merged_df['grid_clipping']).value_counts().rename(index={True:"Including",False:"Excluding"})

## Applying filters to remove all data when there is zero or blank inverter data and also removing value where any inverter is clipping , 
#around 0.98 to 1 of inverter rated capacity

# Convert inverter_data to a DataFrame
inverter_df = merged_df[vars.inverter_data]

merged_df['inverter_clipping_check'] = inverter_df.apply(lambda row: row.max() < inverter_clipping, axis=1)

count_inverter_clipping_check=(~merged_df['inverter_clipping_check']).value_counts().rename(index={True:"Including",False:"Excluding"})    ##count when there is no blank data in inverter

merged_df['inverter_blank']=~(merged_df[vars.inverter_data]).isnull().any(axis=1)      ##Note: Checking if there are any blank data for inverter. We are reversing the value so True means all inverter data are available, False means there are some data missing
count_inverter_blank=(merged_df['inverter_blank']).value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['inverter_zero']=~(merged_df[vars.inverter_data]==0).any(axis=1)            ##Note: If values are True means inverter data are non zero since we reverse the data
count_inverter_zero=(~merged_df['inverter_zero']).value_counts().rename(index={True:"Including",False:"Excluding"})        ##count when there is no blank data in inverter

count_inverter_filter_data=(merged_df['inverter_clipping_check']&merged_df['inverter_blank']).value_counts().rename(index={True:"Including",False:"Excluding"})

# count_grid_inverter_filter_data=(merged_df['inverter_clipping_check']&merged_df['inverter_blank']&merged_df['meter>0']&merged_df['grid_clipping']).value_counts().rename(index={True:"Including",False:"Excluding"})

## Applying conditions on irradiance, temp and wind data

merged_df['fpoa_blank']=~(merged_df[vars.fpoa_data]).isnull().any(axis=1)
count_fpoa_blank=merged_df['fpoa_blank'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['fpoa_zero']=~(merged_df[vars.fpoa_data]==0).any(axis=1)
count_fpoa_zero=merged_df['fpoa_zero'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['rpoa_blank']=~(merged_df[vars.rpoa_data]).isnull().any(axis=1)
count_rpoa_blank=merged_df['rpoa_blank'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['rpoa_zero']=~(merged_df[vars.rpoa_data]==0).any(axis=1)
count_rpoa_zero=merged_df['rpoa_zero'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['temp_blank']=~(merged_df[vars.temp_data]).isnull().any(axis=1)
count_temp_blank=merged_df['temp_blank'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['temp_zero']=~(merged_df[vars.temp_data]==0).any(axis=1)
count_temp_zero=merged_df['temp_zero'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['wind_blank']=~(merged_df[vars.wind_data]).isnull().any(axis=1)
count_wind_blank=merged_df['wind_blank'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['wind_zero']=~(merged_df[vars.wind_data]==0).any(axis=1)
count_wind_zero=merged_df['wind_zero'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['fpoa_QC']=merged_df['average_fpoa'].between(minimum_irradiance,max_irradiance)    ## Checking if avg poa value are between 400 to 1200 or based on inputs 
count_fpoa_qc=merged_df['fpoa_QC'].value_counts().rename(index={True:"Including",False:"Excluding"})

count_after_all_met_data_filters=(merged_df['fpoa_blank']&merged_df['fpoa_zero']&merged_df['rpoa_blank']&merged_df['rpoa_zero']&merged_df['temp_blank']&
                                  merged_df['temp_zero']&merged_df['wind_blank']&merged_df['wind_zero']&merged_df['fpoa_QC']).value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['average_fpoa_pct_change'] = merged_df['average_fpoa'].pct_change() * 100
merged_df['temporal_stability_check'] = merged_df['average_fpoa_pct_change'].abs() <= temporal_stability_thresold

# Calculate standard deviation and mean for each row across the fpoa_data columns
merged_df['fpoa_spatial_std'] = merged_df[vars.fpoa_data].std(axis=1)
merged_df['fpoa_spatial_mean'] = merged_df[vars.fpoa_data].mean(axis=1)

# Calculate spatial stability by dividing standard deviation by mean
merged_df['fpoa_spatial_stability'] = merged_df['fpoa_spatial_std'] / merged_df['fpoa_spatial_mean']

# Check if the absolute value of fpoa_spatial_stability is less than 0.10
merged_df['spatial_stability_check'] = abs(merged_df['fpoa_spatial_stability']) <=spatial_stability_thresold

# Count the number of True and False values in the new column
spatial_stability_counts = merged_df['spatial_stability_check'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['average_inverter_pct_change'] = merged_df['average_fpoa'].pct_change() * 100
merged_df['temporal_stability_check'] = merged_df['average_fpoa_pct_change'].abs() <= temporal_stability_thresold

##Calculating Reporting Conditions 

##Note: Adding another coloumn as primary filters where taking all True value from all filters from above 

merged_df['primary_filters']=(merged_df['t_stamp_check']*
    merged_df['meter>0']*merged_df['grid_clipping']
                                *merged_df['fpoa_QC']*merged_df['temporal_stability_check']*merged_df['spatial_stability_check']
                                *merged_df['inverter_clipping_check']
                                *merged_df['inverter_blank']*merged_df['inverter_zero']
                              *merged_df['fpoa_blank']*merged_df['temp_blank']
                                *merged_df['wind_blank']*merged_df['temp_zero']*merged_df['fpoa_zero']*merged_df['wind_zero'])
# 
count_primary_filters=merged_df['primary_filters'].value_counts().rename(index={True:"Including",False:"Excluding"})       ##Note: Counting number of primary filters, this should be minimum number of points 

count_primary_filters_per_day = merged_df.groupby(merged_df['t_stamp'].dt.date)['primary_filters'].value_counts().unstack().fillna(0).rename(columns={True: "Including", False: "Excluding"})

rc_conditions=merged_df[merged_df['primary_filters']==True]    ## Note: Reporting conditions are calculated only on True values on primary filters 

## Calculating RC conditions on fpoa, rpoa, temp and wind by taking average values
rc_avg_poa_total=rc_conditions['average_poa_total'].mean()
rc_avg_fpoa=rc_conditions['average_fpoa'].mean()             
rc_avg_rpoa=rc_conditions['average_rpoa'].mean()
rc_avg_temp=rc_conditions['average_temp'].mean()
rc_avg_wind=rc_conditions['average_wind'].mean()

## Calculating RC conditions on fpoa, rpoa, temp and wind by taking percentile values

percentile_avg_poa_total=rc_conditions['average_poa_total'].quantile(percentile)
percentile_avg_fpoa=rc_conditions['average_fpoa'].quantile(percentile)
percentile_avg_rpoa=rc_conditions['average_rpoa'].quantile(percentile)
percentile_avg_temp=rc_conditions['average_temp'].quantile(percentile)
percentile_avg_wind=rc_conditions['average_wind'].quantile(percentile)

## Checking secondary consditions by taking irradiance threshold on primary filters

rc_poa_total=percentile_avg_poa_total
rc_fpoa=percentile_avg_fpoa
rc_rpoa=percentile_avg_rpoa
rc_temp=percentile_avg_temp
rc_wind=1

reporting_condition_thresold_min = (1-reporting_condition_thresold)*rc_poa_total
reporting_condition_thresold_max = (1+reporting_condition_thresold)*rc_poa_total

merged_df['rc_check'] = merged_df['average_poa_total'].between(reporting_condition_thresold_min,reporting_condition_thresold_max)

## Checking the secondary filter where the number of data should be 750 or based on contract with EPC

merged_df['secondary_filter']=merged_df['primary_filters']*merged_df['rc_check']
count_rc_condition_thresold=merged_df['secondary_filter'].value_counts().rename(index={True:"Including",False:"Excluding"})

secondary_above_rc_perc=(((merged_df['secondary_filter']==True)&(merged_df['average_poa_total']>=rc_poa_total)).sum()/((merged_df['secondary_filter']==True)).sum()*100)
secondary_below_rc_perc=100-secondary_above_rc_perc

measured_regression_df = merged_df[merged_df['secondary_filter']==True]
count_secondary_filters_per_day = measured_regression_df.groupby(measured_regression_df['t_stamp'].dt.date)['secondary_filter'].value_counts().unstack().fillna(0).rename(columns={True: "Including", False: "Excluding"})

# Assuming merged_df is your DataFrame and t_stamp is your x-axis column
fig2 = go.Figure()

y_columns_secondary = ['average_fpoa','average_rpoa','average_temp','average_wind']  # Replace with your column names


# Add traces for the primary y-axis
fig2.add_trace(go.Scatter(x=measured_regression_df['t_stamp'], y=measured_regression_df['LBSP1/Device/PowerMeter/MTR/p3_kW'], mode='lines', name='LBSP1/Device/PowerMeter/MTR/p3_kW'))

# Add traces for the secondary y-axis
fig2.add_trace(go.Scatter(x=measured_regression_df['t_stamp'], y=measured_regression_df['average_fpoa'], mode='lines', name='average_fpoa', yaxis='y2'))

# Update layout to include a secondary y-axis
fig2.update_layout(
    title='Meter vs. FPOA after secondary filtering',
    xaxis_title='Timestamp',
    yaxis_title='Meter and Inverter Data',
    yaxis2=dict(
        title='MET Avg Data',
        overlaying='y',
        side='right'),
    hovermode='x unified',
    width = 1000
)

## Adding columns in data frame to use it for regression equation per ASTM 2848
##Power = POA * (fpoa + fpoa_poa_poa*POA + fpoa_temp*Temp + fpoa_wind*Wind)

## Calculating POA xPOA, POAxTemp and POAx Wind 
measured_regression_df['fpoa'] = measured_regression_df['average_poa_total']
measured_regression_df['fpoaxfpoa'] = measured_regression_df['average_poa_total'] * measured_regression_df['average_poa_total']
measured_regression_df['fpoaxtemp'] = measured_regression_df['average_temp'] * measured_regression_df['average_poa_total']
measured_regression_df['fpoaxwind'] = measured_regression_df['average_wind'] * measured_regression_df['average_poa_total']*0

X = measured_regression_df[['fpoa','fpoaxfpoa','fpoaxtemp','fpoaxwind']]
y = measured_regression_df[vars.meter_data]

coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
final_coefficients = coefficients[::-1]

fpoa_wind, fpoa_temp, fpoa_poa_poa, fpoa = final_coefficients

##Power = POA * (fpoa + fpoa_poa_poa*POA + fpoa_temp*Temp + fpoa_wind*Wind)

## Note: Calculating energy 

measured_energy_bifacial=rc_poa_total*(fpoa+fpoa_poa_poa*rc_poa_total+fpoa_temp*rc_temp+fpoa_wind*rc_wind)[0]
measured_energy_monofacial=rc_fpoa*(fpoa+fpoa_poa_poa*rc_fpoa+fpoa_temp*rc_temp+fpoa_wind*rc_wind)[0]

measured_regression_df["Energy Predicted"] = measured_regression_df['average_poa_total']*((fpoa)+fpoa_poa_poa*measured_regression_df['average_poa_total']+fpoa_temp*measured_regression_df['average_temp']+fpoa_wind*1)

fig3 = px.scatter(measured_regression_df, x='Energy Predicted', y=vars.meter_data[0], title='Scatter plot between x and y')

# Update the layout to include (0, 0) in the axes
fig3.update_layout(
    title = "Measured vs. Expected Energy after secondary filtering",
    #xaxis = dict(range=[0, measured_regression_df['Energy Predicted'].max()]),
    #yaxis = dict(range=[0, measured_regression_df[vars.meter_data[0]].max()]),
    width = 1000
)

pvsyst_test_model_df = pd.read_csv(pvsyst_test_model_path,encoding="latin-1")
#pvsyst_test_model_df["date"] = pd.to_datetime(pvsyst_test_model_df["date"]) + datetime.timedelta(days=34*365 + 8)

midpoint_date = test_start_date + (test_end_date - test_start_date) / 2
pvsyst_model_start_date = midpoint_date + datetime.timedelta(days=-45)
pvsyst_model_end_date = midpoint_date + datetime.timedelta(days=45)

pvsyst_selected_column = ["date", "E_Grid", "GlobInc", "WindVel", "FShdBm", "T_Amb", "IL_Pmax", "GlobBak", "BackShd"]

pvsyst_test_model_selected_columns_df = pvsyst_test_model_df[pvsyst_selected_column]

pvsyst_test_model_selected_columns_df['POA_Total_pvsyst'] = (pvsyst_test_model_selected_columns_df['GlobInc'] + ((pvsyst_test_model_selected_columns_df['GlobBak'] + pvsyst_test_model_selected_columns_df['BackShd']) * bifaciality))

# Convert 'date' column to datetime
pvsyst_test_model_selected_columns_df['date'] = pd.to_datetime(pvsyst_test_model_selected_columns_df['date'])


pvsyst_filtered_df = pvsyst_test_model_selected_columns_df.loc[(pvsyst_test_model_selected_columns_df['date'] >= pvsyst_model_start_date)
    & (pvsyst_test_model_selected_columns_df['date'] <= pvsyst_model_end_date)
    & (pvsyst_test_model_selected_columns_df['GlobInc'] > minimum_irradiance) 
    & (pvsyst_test_model_selected_columns_df['E_Grid'] > 0)
    & (pvsyst_test_model_selected_columns_df['E_Grid'] < grid_clipping)
    & (pvsyst_test_model_selected_columns_df['FShdBm'] == pvsyst_shading)
    & (pvsyst_test_model_selected_columns_df['IL_Pmax'] == 0)]

pvsyst_filtered_df.loc[:,'POA_Total'] = pvsyst_filtered_df['POA_Total_pvsyst']
pvsyst_filtered_df.loc[:,'POA_Total*POA_Total'] = pvsyst_filtered_df['POA_Total_pvsyst'] * pvsyst_filtered_df['POA_Total_pvsyst']
pvsyst_filtered_df.loc[:,'POA_Total*Temp'] = pvsyst_filtered_df['POA_Total_pvsyst'] * pvsyst_filtered_df['T_Amb']
pvsyst_filtered_df.loc[:,'POA_Total*Wind'] = pvsyst_filtered_df['POA_Total_pvsyst'] * pvsyst_filtered_df['WindVel']

rc_pvsyst_avg_poa_total=pvsyst_filtered_df['POA_Total_pvsyst'].mean()
rc_pvsyst_avg_fpoa=pvsyst_filtered_df['GlobInc'].mean()
rc_pvsyst_avg_rpoa=pvsyst_filtered_df['GlobBak'].mean()
rc_pvsyst_avg_temp=pvsyst_filtered_df['T_Amb'].mean()
rc_pvsyst_avg_wind=pvsyst_filtered_df['WindVel'].mean()

rc_pvsyst_percentile_poa_total=pvsyst_filtered_df['POA_Total_pvsyst'].quantile(percentile)
rc_pvsyst_percentile_fpoa=pvsyst_filtered_df['GlobInc'].quantile(percentile)
rc_pvsyst_percentile_rpoa=pvsyst_filtered_df['GlobBak'].quantile(percentile)
rc_pvsyst_percentileg_temp=pvsyst_filtered_df['T_Amb'].quantile(percentile)
rc_pvsyst_percentile_wind=pvsyst_filtered_df['WindVel'].quantile(percentile)

reporting_condition_thresold_min=(1-reporting_condition_thresold)*rc_poa_total
reporting_condition_thresold_max=(1+reporting_condition_thresold)*rc_poa_total
pvsyst_filtered_df.loc[:,'rc_pvsyst_check']=pvsyst_filtered_df['POA_Total'].between(reporting_condition_thresold_min,reporting_condition_thresold_max)
expected_regression_df=pvsyst_filtered_df[pvsyst_filtered_df['rc_pvsyst_check']==True]

X=expected_regression_df[['POA_Total','POA_Total*POA_Total','POA_Total*Temp','POA_Total*Wind']]
y=expected_regression_df['E_Grid']

coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
final_coefficients = coefficients[::-1]

pvsyst_fpoa_wind, pvsyst_fpoa_temp, pvsyst_fpoa_poa_poa, pvsyst_fpoa = final_coefficients

expected_energy_monofacial=(pvsyst_fpoa+pvsyst_fpoa_poa_poa*rc_fpoa+pvsyst_fpoa_temp*rc_temp+pvsyst_fpoa_wind*rc_wind)*rc_fpoa
expected_energy_bifacial=(pvsyst_fpoa+pvsyst_fpoa_poa_poa*rc_poa_total+pvsyst_fpoa_temp*rc_temp+pvsyst_fpoa_wind*rc_wind)*rc_poa_total

expected_regression_df.loc[:,"Energy Predicted"]=expected_regression_df['POA_Total']*((fpoa)+fpoa_poa_poa*expected_regression_df['POA_Total']+fpoa_temp*expected_regression_df['T_Amb']+fpoa_wind*expected_regression_df['WindVel'])
# Assuming expected_regression_df is already defined and contains columns 'Energy Predicted' and 'E_Grid'
fig4 = px.scatter(expected_regression_df, x='Energy Predicted', y='E_Grid', title='Scatter plot between Energy P and E_G', width=1000)

# Customize the hover data
fig4.update_traces(marker=dict(size=10), selector=dict(mode='markers'))

Capacity_Ratio_Mono=measured_energy_monofacial/expected_energy_monofacial*100
Capacity_Ratio_Bifacial=measured_energy_bifacial/expected_energy_bifacial*100

#Added by KL to count per column how many inverters were did not hit criteria
merged_df['inverter_count'] = merged_df.apply(
    lambda row: sum(
        (row[column] < 50) and
        (row['average_fpoa'] > availability_min_fpoa) and
        row['t_stamp_check'] and
        row['data_check_inv']
        for column in vars.inverter_data #merged_df.columns
        #if 'INV' in column 
    ),
    axis = 1)

##Added by KL to calculate lost capacity of each averaging interval and grphing inverter avail for start to end data
merged_df['lost_capac'] = 100 - ((merged_df['inverter_count'] * inverter_rating / max_gridlimit ) / .01)
merged_df.loc[merged_df['lost_capac'] < 0, 'lost_capac'] = 0

# Assuming merged_df is your DataFrame and t_stamp is your x-axis column
fig6 = go.Figure()

fig6.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df['lost_capac'], mode='lines', name=col))

# Update layout
fig6.update_layout(
    title='Availability Plot',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width=1000
)

##Define the filters here, calling from functions defined above
#filter_registry = [
#    ("Meter > 0", filter_meter_greater_zero, []),  
#    ("Grid Clipping", filter_grid_clipping, []),  
#    ("Inverter Clipping", filter_inverter_clipping, [inverter_df]),  
#    ("Inverter is 0", filter_inverter_zero, [inverter_df]),
#    ("FPOA is blank", filter_fpoa_blank, [vars.fpoa_data]),
#    ("FPOA is 0", filter_fpoa_zero, [vars.fpoa_data]),  
#    ("RPOA is blank", filter_rpoa_blank, [vars.rpoa_data]),
#    ("RPOA is zero", filter_rpoa_zero, [vars.rpoa_data]),
#    ("Temp Blank", filter_temp_blank, [vars.temp_data]),
#    ("Temp is 0", filter_temp_zero, [vars.temp_data]),
#    (" Wind Blank", filter_wind_blank, [vars.wind_data]),
#    ("Wind is 0", filter_wind_zero, [vars.wind_data]),
#    ("FPOA QC", filter_fpoa_qc, [minimum_irradiance, max_irradiance]),
#    ("Spatial Stability Check", filter_spatial_stability, [vars.fpoa_data, spatial_stability_thresold]),
#    ("Temporal Stability Check", filter_temporal_stability, [temporal_stability_thresold])
#]
## Initialize the DataFrame to track cumulative conditions
##merged_df['cumulative_condition'] = True  
#filter_results = []
#
## Initialize starting points and condition
#remaining_condition = pd.Series(True, index=merged_df.index)
#remaining_points = len(merged_df)
#initial_points = remaining_points
#
#for idx, (filter_name, filter_function, filter_args) in enumerate(filter_registry, start=1):
#    # Apply filter to the remaining points
#    current_condition = filter_function(merged_df, *filter_args)
#    
#    # Combine with the remaining condition from previous filters
#    combined_condition = remaining_condition & current_condition
#    
#    # Calculate lost and remaining points
#    lost_points = (~combined_condition & remaining_condition).sum()
#    remaining_points = combined_condition.sum()
#    
#    # Add the filter's results to the table
#    filter_results.append({
#        #"Filter Number": f"Filter {idx}",
#        "Filter Description": filter_name,
#        "Initial Points": initial_points,
#        "Points Lost": lost_points,
#        "Remaining Points": remaining_points,
#        #"Filter Description": filter_name,
#    })
#    
#    # Update the remaining condition 
#    remaining_condition = combined_condition
#    initial_points = remaining_points  # Remaining points become initial points for the next filter
#
## Put results in DF
#filter_results_df = pd.DataFrame(filter_results)
#
## Display the table, I gave a few options
##print(tabulate(filter_results_df, headers = 'keys', tablefmt = 'github'))

fig7 = go.Figure()
for col in vars.inverter_data:
    fig7.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig7.update_layout(
    title='Inverter Raw Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1000)

fig8 = go.Figure()
for col in vars.fpoa_data:
    fig8.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig8.update_layout(
    title='Irradiance Raw Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1000)

fig9 = go.Figure()
fig9.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df['LBSP1/Device/PowerMeter/MTR/p3_kW'], mode='lines', name=col))
fig9.update_layout(
    title='Meter Power Raw Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1000)

fig10 = go.Figure()
for col in vars.soiling_data:
    fig10.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig10.update_layout(
    title='Soiling Raw Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1000)

results_df = funcs.loop_rc_threshold(min_rc, max_rc, step_size, rc_poa_total, merged_df)

# Plot Total number of points against Threshold using Plotly
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=results_df['Threshold'],
    y=results_df['Total number of points'],
    mode='lines+markers',
    name='Total number of points'
))
fig5.update_layout(
    title="Total Number of Points vs Threshold",
    xaxis_title="Threshold",
    yaxis_title="Total Number of Points",
    template="plotly_white",
    width=1000
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Tab 3: Report

tab3.write("Test Start Date: " + str(test_start_date))
tab3.write("Test End Date : " + str(test_end_date))
tab3.write("Number of Days: " + str(test_end_date-test_start_date))

# add: table of inputs
tab3.header("Inputs")
tab3_col1, tab3_col2 = tab3.columns(2)
tab3_col1.write("Minimum Irradiance: " + str(minimum_irradiance) + "W/m^2")
tab3_col1.write("Maximum Irradiance: " + str(max_irradiance) + "W/m^2")
tab3_col1.write("Temporal Stability Threshold: " + str(temporal_stability_thresold))
tab3_col1.write("Spatial Stability Threshold: " + str(spatial_stability_thresold))
tab3_col1.write("Minimum Grid Value: " + str(minimum_grid))
tab3_col1.write("Maximum Grid Value: " + str(max_gridlimit))
tab3_col1.write("Grid Clipping Threshold: " + str(grid_clipping_thresold))
tab3_col2.write("RC Percentile: " + str(percentile))
tab3_col2.write("RC Threshold: " + str(reporting_condition_thresold))
tab3_col2.write("Inverter Rating: " + str(inverter_rating))
tab3_col2.write("Inverter Limit: " + str(Inverter_limit))
tab3_col2.write("Inverter Clipping Threshold: " + str(inverter_clipping_thresold))

tab3.header("Capacity Test Results:")

# statement of passing or failing w/ percentage
if Capacity_Ratio_Bifacial >= passing_capacity:
    tab3.success("The test passed with a " + str(Capacity_Ratio_Bifacial) + "% capacity.")
else:
    tab3.error("The test failed with a " + str(Capacity_Ratio_Bifacial) + "% capacity.")

tab3.dataframe(pd.DataFrame({"Summary": ["Model Energy", "Measured Energy", "Capacity Ratio %"],
    "Monofacial": [expected_energy_monofacial, measured_energy_monofacial, Capacity_Ratio_Mono],
    "Bifacial": [expected_energy_bifacial, measured_energy_bifacial, Capacity_Ratio_Bifacial]}).set_index("Summary"))

tab3.plotly_chart(fig3) # Measured vs. Expected Energy after secondary filtering

tab3.plotly_chart(fig2) # Meter vs. FPOA after secondary filtering

tab3.header("Availability Test:")
# add: statement of availability calculation tab3.write("This calculation was done with...")

tab3.subheader("Test total availability")
tab3.write("Average Availability of the project is : " + str(avail_average*100) + "%")
tab3.plotly_chart(fig6) # availability plot

# add: graph of all inverter data after secondary filters

tab3.header("Raw Data Graphs")
tab3.plotly_chart(fig7)
tab3.plotly_chart(fig8)
tab3.plotly_chart(fig9)
tab3.plotly_chart(fig10)

tab3.header("Soiling")
tab3.write("Average Soiling: " + str(avg_soiling))
tab3.dataframe(pd.DataFrame({"MET Station": [5, 15, 21, 29],
    "Avg Soiling": [avg_soiling_met5, avg_soiling_met15, avg_soiling_met21, avg_soiling_met29]}).set_index("MET Station"))

# add: heap map of inverters

#tab3.header("Number of Points by Filter")
#tab3.write(filter_results_df)

tab3.header("Filters Per Day")
tab3.write("Primary filters per day:")
tab3.write(count_primary_filters_per_day)
tab3.write("Secondary filters per day:")
tab3.write(count_secondary_filters_per_day)

tab3.header("RC Values")
tab3.dataframe(pd.DataFrame({"RC Value": ["Total POA", "FPOA", "RPOA", "Temp", "Wind"],
    "Measured Average": [rc_avg_poa_total, rc_avg_fpoa, rc_avg_rpoa, rc_avg_temp, rc_avg_wind],
    "Measured Percentile": [percentile_avg_poa_total, percentile_avg_fpoa, percentile_avg_rpoa, percentile_avg_temp, percentile_avg_wind],
    "PVSyst Average": [rc_pvsyst_avg_poa_total, rc_pvsyst_avg_fpoa, rc_pvsyst_avg_rpoa, rc_pvsyst_avg_temp, rc_pvsyst_avg_wind],
    "PVSyst Percentile": [rc_pvsyst_percentile_poa_total, rc_pvsyst_percentile_fpoa, rc_pvsyst_percentile_rpoa, rc_pvsyst_percentileg_temp, rc_pvsyst_percentile_wind]}).set_index("RC Value"))
tab3.write("Percent above RC after secondary filtering: " + str(secondary_above_rc_perc) + "%")
tab3.write("Percent below RC after secondary filtering: " + str(secondary_below_rc_perc) + "%")

tab3.header("Regression Coefficients")
tab3.dataframe(pd.DataFrame({"Regression Coefficients": ["fpoa", "fpoa_poa_poa", "fpoa_temp", "fpoa_wind"],
    "Measured": [fpoa[0], fpoa_poa_poa[0], fpoa_temp[0], fpoa_wind[0]],
    "PVSyst": [pvsyst_fpoa, pvsyst_fpoa_poa_poa, pvsyst_fpoa_temp, pvsyst_fpoa_wind]}).set_index("Regression Coefficients"))

tab3.title("~~~~~~~~~~~~~~~~ PDF Ends Here ~~~~~~~~~~~~~~~~")

tab3.write(f"Number of events POA is greater then minimum irradiance :{count_avail_poa}")
tab3.write(avail_counts_df)

tab3.subheader("Inverter Filters")
tab3.write(count_meter_greaterzero.to_string(dtype=False))
tab3.write(count_grid_clipping.to_string(dtype=False))
tab3.write(count_meter_filter_data.to_string(dtype=False))
tab3.write(count_inverter_clipping_check.to_string(dtype=False))
tab3.write(count_inverter_blank.to_string(dtype=False))
tab3.write(count_inverter_zero.to_string(dtype=False))
tab3.write(count_inverter_filter_data.to_string(dtype=False))

tab3.subheader("MET Filters")
tab3.write(count_fpoa_blank.to_string(dtype=False))
tab3.write(count_fpoa_zero.to_string(dtype=False))
tab3.write(count_rpoa_blank.to_string(dtype=False))
tab3.write(count_rpoa_zero.to_string(dtype=False))
tab3.write(count_temp_blank.to_string(dtype=False))
tab3.write(count_temp_zero.to_string(dtype=False))
tab3.write(count_wind_blank.to_string(dtype=False))
tab3.write(count_wind_zero.to_string(dtype=False))
tab3.write(count_fpoa_qc.to_string(dtype=False))
tab3.write(count_after_all_met_data_filters.to_string(dtype=False))

tab3.subheader("Spatial Stability Counts")
tab3.write(spatial_stability_counts.to_string(dtype=False))

tab3.subheader("RC Threshold Loop")
tab3.write(results_df)
tab3.plotly_chart(fig5)

tab3.subheader("Secondary Filters")
tab3.write(count_rc_condition_thresold.to_string(dtype=False))
tab3.write(secondary_above_rc_perc)
tab3.write(secondary_below_rc_perc)

tab3.plotly_chart(fig4)

tab3.subheader("PVSyst Test Model")
tab3.write(pvsyst_test_model_df)
tab3.write("PVsyst Start Date: " + str(pvsyst_model_start_date))
tab3.write("PVSyst End Date: " + str(pvsyst_model_end_date))

#tab3.header("Detailed Report Below:")
#tab3.write(detailed_report)

#tab3.link_button("Download in-depth report as PDF", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")