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
import funcs, filters, figs

# Page Setup

st.set_page_config(page_title= "PV Cap Test", page_icon="icon.jpg", layout="wide")
#st.logo("long_logo.jpg")

st.image("long_logo.jpg", width=300)
st.title("PV Cap Test")
#st.header("Date: " + str(datetime.date.today()))
tab1, tab2, tab3 = st.tabs(['Main Inputs', 'Detailed Inputs', 'Report'])

# Tab 1: Data Upload

scada_tags = 'SCADA Tags_Liberty.xlsx'
pvsyst_test_model_path = 'PVSyst Model_Liberty.CSV'
# Specify the main directory containing folders of daily CSV files
#main_directory = tab1.text_input("Name of ZIP File (do not include .zip)", '2.Raw Data')
#metadata_file_path = tab1.text_input("Metadata File Path", "SCADA Tags_Liberty.xlsx")  # Path to your metadata file
project = tab1.selectbox(
    "Default project inputs:",
    ("Liberty", "Bayou Galion", "North Fork"))
tab1_col1, tab1_col2 = tab1.columns(2)
test_start_date = datetime.datetime.combine(tab1_col1.date_input("Start Date", 'today', format='MM/DD/YYYY'), datetime.datetime.min.time())
test_end_date = datetime.datetime.combine(tab1_col2.date_input("End Date", 'today', format='MM/DD/YYYY'), datetime.datetime.min.time())
uploaded_zip = tab1.file_uploader("Upload raw data", type='zip')
#scada_tags = tab1.file_uploader("Upload SCADA tags", type='xlsx')
#pvsyst_test_model_path = tab1.file_uploader("Upload PVSyst test model", type='csv')

# Tab 2: Inputs

form1 = tab2.form("inputs form")

form1_col1, form1_col2 = form1.columns(2)

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
meter_units = form1.selectbox(
    "Meter Units:",
    ("MW", "KW")
)
passing_capacity = form1.number_input("Passing Capacity (Bifacial):", min_value=0.0, value=97.0, max_value=100.0)
pvsyst_shading = form1.number_input("PVSyst Shading:", min_value=0, value=1, step=1)
bifaciality = form1.number_input("Bifaciality", value=0.7, min_value=0.0, max_value=1.0, step=0.1)
availability_min_fpoa = form1.number_input("Availability Minimum FPOA", value=50, min_value=0, step=1)
system_size_dc = form1.number_input("System Size DC", value=134046, min_value=0, step=1)
min_poa_soiling = form1.number_input("Min POA Soiling:", min_value=0, value=150, step=1)
soiling_with_iv_curve = form1.number_input("Soiling with IV Curve:", min_value=0.0, value=1.52, step=0.01) ## from EPC 

form1.form_submit_button("Submit Inputs")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend begin ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def loop_rc_threshold(min_rc, max_rc, step_size, rc_poa_total, merged_df):
    # Initialize an empty list to store results before starting the loop
    results = []
    results_day = {}
    
    # Loop over the range of thresholds (min_rc to max_rc with step_size)
    for reporting_condition_thresold in funcs.frange(min_rc, max_rc, step_size):

        
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
        results_day[reporting_condition_thresold] = count_secondary_filters_per_day

        #Calculate regression coefficents
        measured_regression_df['fpoa']=measured_regression_df['average_poa_total']
        measured_regression_df['fpoaxfpoa']=measured_regression_df['average_poa_total']*measured_regression_df['average_poa_total']
        measured_regression_df['fpoaxtemp']=measured_regression_df['average_temp']*measured_regression_df['average_poa_total']
        measured_regression_df['fpoaxwind']=measured_regression_df['average_wind']*measured_regression_df['average_poa_total']
        X=measured_regression_df[['fpoa','fpoaxfpoa','fpoaxtemp','fpoaxwind']]
        y=measured_regression_df['average_meter_data']
        coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        final_coefficients = coefficients[::-1]
        fpoa_wind, fpoa_temp, fpoa_poa_poa, fpoa = final_coefficients

        #Calculate energy both monofacial and bifacial
        measured_energy_bifacial = round(rc_poa_total*(fpoa+fpoa_poa_poa*rc_poa_total+fpoa_temp*rc_temp+fpoa_wind*rc_wind))
        measured_energy_monofacial = round(rc_fpoa*(fpoa+fpoa_poa_poa*rc_fpoa+fpoa_temp*rc_temp+fpoa_wind*rc_wind))
        measured_regression_df["Energy Predicted"]=measured_regression_df['average_poa_total']*((fpoa)+fpoa_poa_poa*measured_regression_df['average_poa_total']+fpoa_temp*measured_regression_df['average_temp']+fpoa_wind*1)

        #Adding Comparison of Site vs Pvsyst data here 
        Capcity_Ratio_Mono = measured_energy_monofacial/expected_energy_monofacial
        Capcity_Ratio_Bifacial = measured_energy_bifacial/expected_energy_bifacial
        
        #Format results table, but currently not working
        result = {
            'Threshold': reporting_condition_thresold,
            'Total # points': count_including,
            'Excluding Count': count_excluding,
            'Secondary Above RC %': secondary_above_rc_perc,
            'Secondary Below RC %': secondary_below_rc_perc,
            'Capacity Ratio Monofacial': Capcity_Ratio_Mono, 
            'Capacity Ratio Bifacial': Capcity_Ratio_Bifacial 
        }
        # 'FPOA Coeff' : fpoa, 
        # 'FPOAxFPOA Coeff' : fpoa_poa_poa, 
        # 'FPOAxTemp Coeff' : fpoa_temp, 
        # 'FPOAxWind Coeff' : fpoa_wind,
        # 'Measured Energy Mono': measured_energy_monofacial, 
        # 'Measured Energy Bi' : measured_energy_bifacial, 


        results.append(result)
    results_df = pd.DataFrame(results).set_index('Threshold')

    #Showing the results here#####################################################################
    # Plot Total # points against Threshold using Plotly
    #fig = go.Figure()
    fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Total Number of Points vs RC Threshold", "Passing % vs RC Threshold"))

    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['Total # points'],
        mode='lines+markers'),
        #name='Total # points',
        row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=results_df.index,
        y=results_df['Capacity Ratio Bifacial'],
        mode='lines+markers'),
        #name='Total # points',
        row=2, col=1
    )
    
    fig.update_layout(
        title="RC Threshold Sensitivity",
        #xaxis_title="Threshold",
        #yaxis_title="Total Number of Points",
        template="plotly_white",
        width=1095
    )

    #print(results_day)
    return results_df, fig

if project == 'Liberty':
    scada_tags = 'SCADA Tags_Liberty.xlsx'
    pvsyst_test_model_path = 'PVSyst Model_Liberty.CSV'
elif project == 'Bayou Galion':
    scada_tags = 'SCADA Tags_BayouGalion.xlsx'
    pvsyst_test_model_path = 'PVSyst Model_BayouGalion.CSV'
    meter_units = 'KW'
elif project == 'North Fork':
    scada_tags = 'SCADA Tags_NorthFork.xlsx'
    pvsyst_test_model_path = 'PVSyst Model_NorthFork1.CSV'

if uploaded_zip is not None:
    with zipfile.ZipFile(uploaded_zip, "r") as z:
        z.extractall(".") # extract zip file to local directory
if uploaded_zip is None or scada_tags is None or pvsyst_test_model_path is None: # if one of the files is not uploaded
    tab2.write('Upload files to proceed.')
    tab3.write('Upload files to proceed.')
    st.stop()

# Initialize an empty Dask DataFrame
all_dfs = []
    
# Gather all file paths
all_files = []
main_directory = uploaded_zip.name[:-4]
for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)
    if os.path.isdir(folder_path):  # Only process folders
        csv_files = glob(os.path.join(folder_path, "*.csv"))
        all_files.extend(csv_files)

for file in all_files:
    df = pd.read_csv(file)
    df = funcs.filter_and_parse_dates(df)  # Assuming you have a function to filter and parse dates
    all_dfs.append(df)

# Concatenate all DataFrames vertically
all_data = pd.concat(all_dfs, axis=0, ignore_index=True)

# group by t_stamp
all_data = all_data.groupby('t_stamp').first()

# Step 2: Load the metadata and get columns to keep
metadata_df = pd.read_excel(scada_tags, header=None)
columns_to_keep = [col.strip() for col in metadata_df[1].dropna().tolist()]  # Assuming column B has the names

# Ensure 't_stamp' is in the list of columns to keep
if 't_stamp' not in columns_to_keep:
    columns_to_keep.insert(0, 't_stamp')  # Add it back to the beginning if needed

# Step 3: Filter the merged DataFrame
merged_df = funcs.filter_columns(all_data.reset_index(), columns_to_keep).set_index('t_stamp')  # create merged_df and reset index to keep 't_stamp' as a column
merged_df = merged_df[(merged_df.index >= test_start_date) & (merged_df.index <= test_end_date)] # restrict merged_df to given dates

metadata_df = pd.read_excel(scada_tags, header=None)  # Adjust header if necessary
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

if meter_units == "KW":
    merged_df[meter_data.columns[0]] *= 1000

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
soiling_cols = []
soiling_avgs = []
for col in soiling_data.columns:
    soiling_cols.append(col)
    soiling_avgs.append(((merged_df['average_fpoa'] > min_poa_soiling) * soiling_data[col]).mean())
avg_soiling_by_day = pd.DataFrame({'MET Stations': soiling_cols,
                                  'Average Soiling (%)': soiling_avgs})
#avg_soiling_by_day = ((merged_df['average_fpoa'] > min_poa_soiling)*soiling_data).mean()

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
    # Calculate the difference and count the occurrences where the difference is greater than 50
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
    ("Inverter is blank", filters.filter_inverter_blank, [inverter_df]),
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
merged_df['inverter_clipping_check'] = filters.filter_inverter_clipping(merged_df, inverter_df, inverter_clipping)
merged_df['inverter_blank'] = filters.filter_inverter_blank(merged_df, inverter_df)
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

rc_conditions = merged_df[merged_df['primary_filters']==True]

# Calculate averages and percentiles for each column and store in the dictionary
columns = ['average_poa_total', 'average_fpoa', 'average_rpoa', 'average_temp', 'average_wind']
results_dict = {}
for col in columns:
    avg_value = rc_conditions[col].mean()
    percentile_value = rc_conditions[col].quantile(percentile)
    
    results_dict[f"{col}_avg"] = avg_value
    results_dict[f"{col}_percentile"] = percentile_value

metric_names = ['POA Total', 'FPOA', 'RPOA', 'Temp', 'Wind'] 
metrics = ['average_poa_total', 'average_fpoa', 'average_rpoa', 'average_temp', 'average_wind'] 
averages = [results_dict[f"{metric}_avg"] for metric in metrics]  
percentiles = [results_dict[f"{metric}_percentile"] for metric in metrics]

# DFe for Average and Percentile
rc_conditions_table = pd.DataFrame({
    'Metric': metric_names,
    'Average': averages,
    f'{percentile*100}th Percentile': percentiles
}).set_index('Metric')

rc_poa_total = results_dict.get("average_poa_total_percentile")
rc_fpoa = results_dict.get("average_fpoa_percentile")
rc_rpoa = results_dict.get("average_rpoa_percentile")
rc_temp = results_dict.get("average_temp_percentile")
#rc_wind = results_dict.get("average_wind_percentile")
rc_wind=1

reporting_condition_thresold_min = (1 - reporting_condition_thresold) * rc_poa_total
reporting_condition_thresold_max = (1 + reporting_condition_thresold) * rc_poa_total

merged_df['rc_check']=merged_df['average_poa_total'].between(reporting_condition_thresold_min,reporting_condition_thresold_max)

## Checking the secondary filter where the number of data should be 750 or based on contract with EPC
merged_df['secondary_filter'] = merged_df['primary_filters'] * merged_df['rc_check']
count_rc_condition_thresold = merged_df['secondary_filter'].value_counts().rename(index={True:"Including", False:"Excluding"})
including_points_SF = count_rc_condition_thresold.get('Including', 0)
secondary_above_rc_perc = round((((merged_df['secondary_filter']==True)&(merged_df['average_poa_total']>=rc_poa_total)).sum()/((merged_df['secondary_filter']==True)).sum()*100), 2)
secondary_below_rc_perc = round(100 - secondary_above_rc_perc, 2)

#print("The secondary filter has a value of:", reporting_condition_thresold)
#print(f"After applying the secondary filter of {reporting_condition_thresold}, the number of points included in the test is now: {including_points_SF}")
#print(f"The percentage above secondary filter is {secondary_above_rc_perc:.2f}%")
#print(f"The percentage below secondary filter is {secondary_below_rc_perc:.2f}%")

data = {"Metric": [f"{reporting_condition_thresold:.2f}", f"{including_points_SF:.2f}", f"{secondary_above_rc_perc:.2f}%", f"{secondary_below_rc_perc:.2f}%"]}
secondary_filter_df = pd.DataFrame(data, index=["Secondary Filter Value", "Included Points", "Percentage Above Filter", "Percentage Below Filter"])

measured_regression_df = merged_df[merged_df['secondary_filter']==True]
count_secondary_filters_per_day = measured_regression_df.groupby(measured_regression_df['t_stamp'].dt.date)['secondary_filter'].value_counts().unstack().fillna(0).rename(columns={True: "Including", False: "Excluding"})

## Adding columns in data frame to use it for regression equation per ASTM 2848
##Power = POA * (fpoa + fpoa_poa_poa*POA + fpoa_temp*Temp + fpoa_wind*Wind)

## Calculating POA xPOA, POAxTemp and POAx Wind 
measured_regression_df['fpoa'] = measured_regression_df['average_poa_total']
measured_regression_df['fpoaxfpoa'] = measured_regression_df['average_poa_total'] * measured_regression_df['average_poa_total']
measured_regression_df['fpoaxtemp'] = measured_regression_df['average_temp'] * measured_regression_df['average_poa_total']
measured_regression_df['fpoaxwind'] = measured_regression_df['average_wind'] * measured_regression_df['average_poa_total']*0

X = measured_regression_df[['fpoa','fpoaxfpoa','fpoaxtemp','fpoaxwind']]
y = measured_regression_df['average_meter_data']

coefficients, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
final_coefficients = coefficients[::-1]

fpoa_wind, fpoa_temp, fpoa_poa_poa, fpoa = final_coefficients

##Power = POA * (fpoa + fpoa_poa_poa*POA + fpoa_temp*Temp + fpoa_wind*Wind)
## Note: Calculating energy 

measured_energy_bifacial = round(rc_poa_total*(fpoa+fpoa_poa_poa*rc_poa_total+fpoa_temp*rc_temp+fpoa_wind*rc_wind))
measured_energy_monofacial = round(rc_fpoa*(fpoa+fpoa_poa_poa*rc_fpoa+fpoa_temp*rc_temp+fpoa_wind*rc_wind))

measured_regression_df["Energy Predicted"] = measured_regression_df['average_poa_total']*((fpoa)+fpoa_poa_poa*measured_regression_df['average_poa_total']+fpoa_temp*measured_regression_df['average_temp']+fpoa_wind*1)

pvsyst_test_model_df = pd.read_csv(pvsyst_test_model_path, encoding="latin-1")
#pvsyst_test_model_df = pd.read_csv(pvsyst_test_model_path,encoding="latin-1",skiprows=12, header=0)

midpoint_date = test_start_date + (test_end_date - test_start_date) / 2
pvsyst_model_start_date = midpoint_date + datetime.timedelta(days=-45)
pvsyst_model_end_date = midpoint_date + datetime.timedelta(days=45)

pvsyst_selected_column = ["date", "E_Grid", "GlobInc", "WindVel", "FShdBm", "T_Amb", "IL_Pmax", "GlobBak", "BackShd"]
pvsyst_test_model_selected_columns_df = pd.DataFrame()
for column in pvsyst_selected_column:
    if column in pvsyst_test_model_df:
        pvsyst_test_model_selected_columns_df[column] = pvsyst_test_model_df[column]
    else:
        pvsyst_test_model_selected_columns_df[column] = 0
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

expected_energy_monofacial = round((pvsyst_fpoa+pvsyst_fpoa_poa_poa*rc_fpoa+pvsyst_fpoa_temp*rc_temp+pvsyst_fpoa_wind*rc_wind)*rc_fpoa)
expected_energy_bifacial = round((pvsyst_fpoa+pvsyst_fpoa_poa_poa*rc_poa_total+pvsyst_fpoa_temp*rc_temp+pvsyst_fpoa_wind*rc_wind)*rc_poa_total)
Capacity_Ratio_Mono = round(measured_energy_monofacial/expected_energy_monofacial*100, 2)
Capacity_Ratio_Bifacial = round(measured_energy_bifacial/expected_energy_bifacial*100, 2)

results_df, fig5 = loop_rc_threshold(min_rc, max_rc, step_size, rc_poa_total, merged_df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ backend end ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Tab 3: Report

# add: table of inputs
tab3.header("Inputs")
tab3_col1, tab3_col2, tab3_col3 = tab3.columns(3)
tab3_col1.write("Test Start Date: " + str(test_start_date))
tab3_col1.write("Test End Date : " + str(test_end_date))
tab3_col1.write("Number of Days: " + str(test_end_date - test_start_date))
tab3_col2.write("Minimum Irradiance: " + str(minimum_irradiance) + "W/m^2")
tab3_col2.write("Maximum Irradiance: " + str(max_irradiance) + "W/m^2")
tab3_col2.write("Temporal Stability Threshold: " + str(temporal_stability_thresold))
tab3_col2.write("Spatial Stability Threshold: " + str(spatial_stability_thresold))
tab3_col2.write("RC Percentile: " + str(percentile))
tab3_col2.write("RC Threshold: " + str(reporting_condition_thresold))
tab3_col3.write("Minimum Grid Value: " + str(minimum_grid))
tab3_col3.write("Maximum Grid Value: " + str(max_gridlimit))
tab3_col3.write("Grid Clipping Threshold: " + str(grid_clipping_thresold))
tab3_col3.write("Inverter Rating: " + str(inverter_rating))
tab3_col3.write("Inverter Limit: " + str(Inverter_limit))
tab3_col3.write("Inverter Clipping Threshold: " + str(inverter_clipping_thresold))

tab3.divider()
tab3.header("Capacity Test Results:")
tab3_col1, tab3_col2 = tab3.columns(2)
# statement of passing or failing w/ percentage
if Capacity_Ratio_Bifacial >= passing_capacity:
    tab3_col1.success("The test passed with a " + str(Capacity_Ratio_Bifacial) + "% capacity. Yippee!")
else:
    tab3_col1.error("The test failed with a " + str(Capacity_Ratio_Bifacial) +
               "% capacity. The target bifacial capacity is " + str(passing_capacity) + "%")

tab3_col2.dataframe(pd.DataFrame({"Summary": ["Model Energy", "Measured Energy", "Capacity Ratio %"],
    "Monofacial": [expected_energy_monofacial, measured_energy_monofacial, Capacity_Ratio_Mono],
    "Bifacial": [expected_energy_bifacial, measured_energy_bifacial, Capacity_Ratio_Bifacial]}).set_index("Summary"))

tab3.plotly_chart(figs.create_fig3(measured_regression_df)) # Measured vs. Expected Energy after secondary filtering
tab3.plotly_chart(figs.create_fig2(measured_regression_df)) # Meter vs. FPOA

tab3.header("Availability Test:")
# add: statement of availability calculation tab3.write("This calculation was done with...")
tab3.write("Average Availability of the project is : " + str(avail_average) + "%")
#tab3.plotly_chart(fig6) # availability plot
tab3.plotly_chart(figs.create_fig11(filtered_data))

tab3.header("Raw Data Graphs")
tab3.plotly_chart(figs.create_fig7(merged_df, inverter_data))
tab3.plotly_chart(figs.create_fig8(merged_df, fpoa_data))
tab3.plotly_chart(figs.create_fig9(merged_df, meter_data))
tab3.plotly_chart(figs.create_fig10(merged_df, soiling_data))

tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.markdown('######')
tab3.header("Number of Points by Filter")
tab3_col1, tab3_col2, tab3_col3 = tab3.columns(3)
tab3_col1.dataframe(filter_results_df, height=563)
tab3_col2.write("Primary filters per day:")
tab3_col2.dataframe(count_primary_filters_per_day, width=400)
tab3_col3.write("Secondary filters per day:")
tab3_col3.dataframe(count_secondary_filters_per_day, width=400)

tab3.divider()
tab3_col1, tab3_col2, tab3_col3 = tab3.columns(3)
tab3_col1.header("Regression Coefficients")
tab3_col1.dataframe(pd.DataFrame({"Regression Coefficients": ["fpoa", "fpoa_poa_poa", "fpoa_temp", "fpoa_wind"],
    "Measured": [fpoa, fpoa_poa_poa, fpoa_temp, fpoa_wind],
    "PVSyst": [pvsyst_fpoa, pvsyst_fpoa_poa_poa, pvsyst_fpoa_temp, pvsyst_fpoa_wind]}).set_index("Regression Coefficients"))
tab3_col2.header("RC Values")
tab3_col2.write(rc_conditions_table)
tab3_col3.write("")
tab3_col3.write("")
tab3_col3.write("")
tab3_col3.write("")
tab3_col3.write("Percent above RC after secondary filtering: " + str(secondary_above_rc_perc) + "%")
tab3_col3.write("Percent below RC after secondary filtering: " + str(secondary_below_rc_perc) + "%")

tab3.subheader("RC Threshold Loop")
tab3.write(results_df)
tab3.plotly_chart(fig5)

tab3.divider()
tab3.header("Soiling")
tab3.write("Average Soiling: " + str(avg_soiling) + "%")
tab3.write(avg_soiling_by_day)
tab3.write("")

#tab3.write(f"Number of events POA is greater then minimum irradiance: {count_avail_poa}")
#tab3.write(avail_counts_df)

# different measured v expected energy plot
#tab3.plotly_chart(fig4)

#tab3.subheader("PVSyst Test Model")
#tab3.write(pvsyst_test_model_df)
#tab3.write("PVsyst Start Date: " + str(pvsyst_model_start_date))
#tab3.write("PVSyst End Date: " + str(pvsyst_model_end_date))

#tab3.header("Detailed Report Below:")
#tab3.write(detailed_report)

#tab3.link_button("Download in-depth report as PDF", "https://www.youtube.com/watch?v=dQw4w9WgXcQ")