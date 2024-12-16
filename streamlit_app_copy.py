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
            'FPOA Coeff' : fpoa, 
            'FPOA x FPOA Coeff' : fpoa_poa_poa, 
            'FPOA x Temp Coeff' : fpoa_temp, 
            'FPOA x Wind Coeff' : fpoa_wind,
            'Measured Energy Monofacial': measured_energy_monofacial, 
            'Measured Energy Bifacial' : measured_energy_bifacial, 
            'Capacity Ratio Monofacial': Capcity_Ratio_Mono, 
            'Capacity Ratio Bifacial': Capcity_Ratio_Bifacial 
        }


        results.append(result)
    results_df = pd.DataFrame(results)

    #Showing the results here#####################################################################
    # Plot Total # points against Threshold using Plotly
    #fig = go.Figure()
    fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Total Number of Points vs RC Threshold", "Passing % vs RC Threshold"))

    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
        y=results_df['Total # points'],
        mode='lines+markers'),
        #name='Total # points',
        row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=results_df['Threshold'],
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

## Applying filters to remove all data where meter value is positive and not clipping 

merged_df['meter>0']=merged_df[meter_data]>minimum_grid       ## Using this to filter value greater than zero

count_meter_greaterzero=merged_df['meter>0'].value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['grid_clipping']=merged_df[meter_data]<grid_clipping        ##Removing all data points at grid clipping to have stable point
count_grid_clipping=merged_df['grid_clipping'].value_counts().rename(index={True:"Including",False:"Excluding"})

count_meter_filter_data=(merged_df['meter>0']&merged_df['grid_clipping']).value_counts().rename(index={True:"Including",False:"Excluding"})

## Applying filters to remove all data when there is zero or blank inverter data and also removing value where any inverter is clipping , 
#around 0.98 to 1 of inverter rated capacity

# Convert inverter_data to a DataFrame
inverter_df = merged_df[inverter_data]

merged_df['inverter_clipping_check'] = inverter_df.apply(lambda row: row.max() < inverter_clipping, axis=1)

count_inverter_clipping_check=(~merged_df['inverter_clipping_check']).value_counts().rename(index={True:"Including",False:"Excluding"})    ##count when there is no blank data in inverter

merged_df['inverter_blank']=~(merged_df[vars.inverter_data]).isnull().any(axis=1)      ##Note: Checking if there are any blank data for inverter. We are reversing the value so True means all inverter data are available, False means there are some data missing
count_inverter_blank=(merged_df['inverter_blank']).value_counts().rename(index={True:"Including",False:"Excluding"})

merged_df['inverter_zero']=~(merged_df[vars.inverter_data]==0).any(axis=1)            ##Note: If values are True means inverter data are non zero since we reverse the data
count_inverter_zero=(~merged_df['inverter_zero']).value_counts().rename(index={True:"Including",False:"Excluding"})        ##count when there is no blank data in inverter

count_inverter_filter_data=(merged_df['inverter_clipping_check']&merged_df['inverter_blank']).value_counts().rename(index={True:"Including",False:"Excluding"})

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
count_primary_filters = merged_df['primary_filters'].value_counts().rename(index={True:"Including",False:"Excluding"})       ##Note: Counting number of primary filters, this should be minimum number of points 

count_primary_filters_per_day = merged_df.groupby(merged_df['t_stamp'].dt.date)['primary_filters'].value_counts().unstack().fillna(0).rename(columns={True: "Including", False: "Excluding"})

rc_conditions = merged_df[merged_df['primary_filters']==True]    ## Note: Reporting conditions are calculated only on True values on primary filters 

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

## Checking the secondary filter where the number of data should be 750 or based on contract with EPC


# Assuming merged_df is your DataFrame and t_stamp is your x-axis column
fig2 = go.Figure()

y_columns_secondary = ['average_fpoa','average_rpoa','average_temp','average_wind']  # Replace with your column names


# Add traces for the primary y-axis
fig2.add_trace(go.Scatter(x=measured_regression_df['t_stamp'], y=measured_regression_df['LBSP1/Device/PowerMeter/MTR/p3_kW'], mode='lines', name='LBSP1/Device/PowerMeter/MTR/p3_kW'))

# Add traces for the secondary y-axis
fig2.add_trace(go.Scatter(x=measured_regression_df['t_stamp'], y=measured_regression_df['average_fpoa'], mode='lines', name='average_fpoa', yaxis='y2'))

# Update layout to include a secondary y-axis
fig2.update_layout(
    title='Meter vs. FPOA',
    xaxis_title='Timestamp',
    yaxis_title='Meter and Inverter Data',
    yaxis2=dict(
        title='MET Avg Data',
        overlaying='y',
        side='right'),
    hovermode='x unified',
    width = 1258
)

# Create the interactive scatter plot
fig3 = px.scatter(measured_regression_df, x='Energy Predicted', y=measured_regression_df['average_meter_data'], title='Scatter plot between Predicted Energy and Site Energy')

# Calculate R² value using scipy's linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(measured_regression_df['Energy Predicted'], measured_regression_df['average_meter_data'])
r_squared = r_value**2  # R² value

# Update the layout to include (0, 0) in the axes
fig3.update_layout(
    xaxis=dict(
        range=[
            measured_regression_df['Energy Predicted'].min() - 0.05 * (measured_regression_df['Energy Predicted'].max() - measured_regression_df['Energy Predicted'].min()),
            measured_regression_df['Energy Predicted'].max() + 0.05 * (measured_regression_df['Energy Predicted'].max() - measured_regression_df['Energy Predicted'].min())
        ]),
    yaxis=dict(
        range=[
            measured_regression_df['average_meter_data'].min() - 0.05 * (measured_regression_df['average_meter_data'].max() - measured_regression_df['average_meter_data'].min()),
            measured_regression_df['average_meter_data'].max() + 0.05 * (measured_regression_df['average_meter_data'].max() - measured_regression_df['average_meter_data'].min())
        ]),
    width=1008
)

fig3.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=max(fig3.data[0].x),
    y1=max(fig3.data[0].x),
    line=dict(color="red", dash="dash")
)

fig3.add_annotation(
    x=0.05, y=0.95, 
    text=f"R² = {r_squared:.2f}", 
    showarrow=False, 
    font=dict(size=14, color="black"), 
    align="left",
    xref="paper", yref="paper"
)

expected_regression_df.loc[:,"Energy Predicted"]=expected_regression_df['POA_Total']*((fpoa)+fpoa_poa_poa*expected_regression_df['POA_Total']+fpoa_temp*expected_regression_df['T_Amb']+fpoa_wind*expected_regression_df['WindVel'])
# Assuming expected_regression_df is already defined and contains columns 'Energy Predicted' and 'E_Grid'
fig4 = px.scatter(expected_regression_df, x='Energy Predicted', y='E_Grid', title='Scatter plot between Energy P and E_G', width=1000)

# Customize the hover data
fig4.update_traces(marker=dict(size=10), selector=dict(mode='markers'))

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

fig7 = go.Figure()
for col in vars.inverter_data:
    fig7.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig7.update_layout(
    title='Inverter Raw Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1270)

fig8 = go.Figure()
for col in vars.fpoa_data:
    fig8.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
fig8.update_layout(
    title='Irradiance Raw Data',
    xaxis_title='Timestamp',
    yaxis_title='Values',
    hovermode='x unified',
    width = 1350)

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
    width = 1385)

results_df = results_df.set_index("Threshold")

## Plot Total number of points against Threshold using Plotly
#fig5 = go.Figure()
#fig5.add_trace(go.Scatter(
#    x=results_df['Threshold'],
#    y=results_df['Total number of points'],
#    mode='lines+markers',
#    name='Total number of points'
#))
#fig5.update_layout(
#    title="Total Number of Points vs Threshold",
#    xaxis_title="Threshold",
#    yaxis_title="Total Number of Points",
#    template="plotly_white",
#    width=1000
#)

fig11 = px.imshow(
    filtered_data.T,  # Transpose to have inverters as rows
    labels=dict(x="Timestamp", y="Inverter", color="Avaiability"),
    x=filtered_data.index,  # Timestamps
    y=filtered_data.columns, 
    color_continuous_scale=['white', 'red'],# Map 0 to white and 1 to red
    zmin = 0, 
    zmax = 1
)

fig11.update_layout(
    title="Inverter Performance Compliance Over Time",
    xaxis=dict(
        showgrid=True,
        gridcolor='lightgray',  
        gridwidth=1  
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='lightgray', 
        gridwidth=1  
    ),
    xaxis_nticks=20,
    width=1100
)

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

tab3.plotly_chart(fig3) # Measured vs. Expected Energy after secondary filtering
tab3.plotly_chart(fig2) # Meter vs. FPOA

tab3.header("Availability Test:")
# add: statement of availability calculation tab3.write("This calculation was done with...")
tab3.write("Average Availability of the project is : " + str(avail_average) + "%")
#tab3.plotly_chart(fig6) # availability plot
tab3.plotly_chart(fig11)

tab3.header("Raw Data Graphs")
tab3.plotly_chart(fig7)
tab3.plotly_chart(fig8)
tab3.plotly_chart(fig9)
tab3.plotly_chart(fig10)

tab3.divider()
tab3.header("Soiling")
tab3.write("Average Soiling: " + str(avg_soiling) + "%")
tab3.dataframe(pd.DataFrame({"MET Station": vars.soiling_data,
    "Avg Soiling (%)": [avg_soiling_met5, avg_soiling_met15, avg_soiling_met21, avg_soiling_met29]}).set_index("MET Station"))
tab3.write("")

tab3.header("Number of Points by Filter")
tab3_col1, tab3_col2 = tab3.columns(2)
tab3_col1.dataframe(filter_results_df, height=563)
tab3_col2.write("Primary filters per day:")
tab3_col2.write(count_primary_filters_per_day)
tab3_col2.write("Secondary filters per day:")
tab3_col2.write(count_secondary_filters_per_day)

tab3.divider()
tab3.header("RC Values")
tab3_col1, tab3_col2 = tab3.columns(2)
tab3_col1.dataframe(pd.DataFrame({"RC Value": ["Total POA", "FPOA", "RPOA", "Temp", "Wind"],
    "Measured Average": [rc_avg_poa_total, rc_avg_fpoa, rc_avg_rpoa, rc_avg_temp, rc_avg_wind],
    "Measured Percentile": [percentile_avg_poa_total, percentile_avg_fpoa, percentile_avg_rpoa, percentile_avg_temp, percentile_avg_wind],
    "PVSyst Average": [rc_pvsyst_avg_poa_total, rc_pvsyst_avg_fpoa, rc_pvsyst_avg_rpoa, rc_pvsyst_avg_temp, rc_pvsyst_avg_wind],
    "PVSyst Percentile": [rc_pvsyst_percentile_poa_total, rc_pvsyst_percentile_fpoa, rc_pvsyst_percentile_rpoa, rc_pvsyst_percentileg_temp, rc_pvsyst_percentile_wind]}).set_index("RC Value").round(2))
tab3_col2.write("Percent above RC after secondary filtering: " + str(secondary_above_rc_perc) + "%")
tab3_col2.write("Percent below RC after secondary filtering: " + str(secondary_below_rc_perc) + "%")

tab3.subheader("RC Threshold Loop")
tab3.write(results_df)
tab3.plotly_chart(fig5)

tab3.header("Regression Coefficients")
tab3.dataframe(pd.DataFrame({"Regression Coefficients": ["fpoa", "fpoa_poa_poa", "fpoa_temp", "fpoa_wind"],
    "Measured": [fpoa[0], fpoa_poa_poa[0], fpoa_temp[0], fpoa_wind[0]],
    "PVSyst": [pvsyst_fpoa, pvsyst_fpoa_poa_poa, pvsyst_fpoa_temp, pvsyst_fpoa_wind]}).set_index("Regression Coefficients"))

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