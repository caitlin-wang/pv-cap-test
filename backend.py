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

detailed_report=f"""

Summary :

Soiling Avg ={avg_soiling}
Capacity Ratio_Bifacial with Soiling from Met Station={Capacity_Ratio_Bifacial*100+avg_soiling}
Capacity Ratio_Monofacial with Soiling from Met Station={Capacity_Ratio_Mono*100+avg_soiling}


Soiling Avg_IV Curve ={soiling_with_iv_curve}
apacity Ratio_Bifacial with Soiling from IV Curve={Capacity_Ratio_Bifacial*100+soiling_with_iv_curve}
Capacity Ratio_Monofacial with Soiling from IV Curve={Capacity_Ratio_Mono*100+soiling_with_iv_curve}



Total number of points on all primary filters=  {count_primary_filters}

Total Number of points on secondary or RC conditions={count_rc_condition_thresold}                                           
Percentage of secondary points above RC={secondary_above_rc_perc}          
Percentage of secondary points below RC={secondary_below_rc_perc} 

RC Values for model  : 

RC POA Total = {rc_poa_total}
RC FPOA =      {rc_fpoa}
RC RPOA =      {rc_rpoa}
RC Temp =      {rc_temp}
RC Wind=       {rc_wind}

Coeff for Measured Data : 
Power = POA * (a1 + a2*POA + a3*Temp + a4*Wind)
a1 ={fpoa}
a2 ={fpoa_poa_poa}
a3 ={fpoa_temp}
a4 ={fpoa_wind}

Coeff for PVSyst: 

Power = POA * (b1 + b2*POA + b3*Temp + b4*Wind)
b1 ={pvsyst_fpoa}
b2 ={pvsyst_fpoa_poa_poa}
b3 ={pvsyst_fpoa_temp}
b4 ={pvsyst_fpoa_wind}


Reporting Conditions Measured Data using Mean:               Reporting Conditions Measured Data using Percentile:

RC T.POA=         {rc_avg_poa_total}                         RC T.POA=         {percentile_avg_poa_total}
RC POA=           {rc_avg_fpoa}                              RC POA=           {percentile_avg_fpoa}
RC RPOA=          {rc_avg_fpoa}                              RC RPOA=          {percentile_avg_rpoa}
RC Temp=          {rc_avg_temp}                              RC Temp=          {percentile_avg_temp}
RC Wind=          {rc_avg_wind}                              RC Wind=          {percentile_avg_wind}

Reporting Conditions PVSyst Data using Mean:                Reporting Conditions PVSyst Data using Percentile:

RC T.POA=         {rc_pvsyst_avg_poa_total}                  RC T.POA=         {rc_pvsyst_percentile_poa_total}
RC POA=           {rc_pvsyst_avg_fpoa}                       RC POA=           {rc_pvsyst_percentile_fpoa}
RC RPOA=          {rc_pvsyst_avg_rpoa}                       RC RPOA=          {rc_pvsyst_percentile_rpoa}
RC Temp=          {rc_pvsyst_avg_temp}                       RC Temp=          {rc_pvsyst_percentileg_temp}
RC Wind=          {rc_pvsyst_avg_wind}                       RC Wind=          {rc_pvsyst_percentile_wind}

##Measured Data Inputs

Minimum POA Irradaince={minimum_irradiance}
Max POA Irradiance={max_irradiance}

Grid Limit={max_gridlimit}kW
Grid Clipping Threshold={grid_clipping_thresold}

Inverter Rating={inverter_rating}
Inverter Clipping Threshold={inverter_clipping_thresold}

Reporting Condition Irradaince Band Range={reporting_condition_thresold}

## Primary Filters
                                                                                                           
Number of points after applying grid>0 =                               
{(count_meter_greaterzero)}                     

Number of points after applying grid clipping limit  =                
{count_grid_clipping}                            


 Number of points after applying grid filters=                      
{count_meter_filter_data}                        

Number of points after checking inverter clipping=                    


Number of points after checking inverter data is blank=                
{count_inverter_blank}                             

Number of points after checking inverter data is Zero                  
{count_inverter_zero}                            

#  Number of points after applying Inverter filters=                  
# {count_inverter_filter_data}

               
                                      
Number of points after removing fpoa blank data=                       
{count_fpoa_blank}             

Number of points after removing fpoa zero data=                        
{count_fpoa_zero}

Number of points after removing rpoa blank data=                       
{count_rpoa_blank}

Number of points after removing rpoa zero data=                        
{count_rpoa_zero}

Number of points after removing temp blank data=                       
{count_temp_blank}

Number of points after removing temp zero data=                        
{count_temp_zero}

Number of points after removing wind blank data=                       
{count_wind_blank}

Number of points after removing wind zero data=                        
{count_wind_zero}

Number of points after removin unstable irradiance data=               
{count_fpoa_qc}

Total number of points after all met data filtering=
{count_after_all_met_data_filters}

Total number of points on all primary filters= 
{count_primary_filters}

Reporting Condition Irradaince Band Range={reporting_condition_thresold}

Total Number of points on secondary or RC conditions={count_rc_condition_thresold}

##PVSyst Inputs

PVSyst Minimum Irradiance= {minimum_irradiance}
Minimum Shading (FShBm)= {pvsyst_shading}

PVSyst Inverter Capacity= {Inverter_limit}
PVSyst Inverter Clipping= {inverter_clipping}

PVSyst Grid Capacity= {grid_clipping}
PVSyst Grid Clipping ={grid_clipping}

Bifaciality Factor = {bifaciality}

PVSyst Test Data start date = {pvsyst_model_start_date}
PVSyst Test Data end date = {pvsyst_model_end_date}

Number of PVSyst points after filters = {pvsyst_filtered_df['E_Grid'].count()}

Availability: 
Average Availability for the project is                                          {avail_average}
Here are list of inverters availability for the test period: 

{avail_counts_df}

"""