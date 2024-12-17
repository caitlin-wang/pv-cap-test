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

def create_fig2(measured_regression_df):
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
    return fig2

def create_fig3(measured_regression_df):
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

    return fig3

def create_fig11(filtered_data):
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
    fig11

def create_fig7(merged_df):
    fig7 = go.Figure()
    for col in vars.inverter_data:
        fig7.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
    fig7.update_layout(
        title='Inverter Raw Data',
        xaxis_title='Timestamp',
        yaxis_title='Values',
        hovermode='x unified',
        width = 1270)
    return fig7

def create_fig8(merged_df):
    fig8 = go.Figure()
    for col in vars.fpoa_data:
        fig8.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
    fig8.update_layout(
        title='Irradiance Raw Data',
        xaxis_title='Timestamp',
        yaxis_title='Values',
        hovermode='x unified',
        width = 1350)
    return fig8

def create_fig9(merged_df):
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df['LBSP1/Device/PowerMeter/MTR/p3_kW'], mode='lines', name=col))
    fig9.update_layout(
        title='Meter Power Raw Data',
        xaxis_title='Timestamp',
        yaxis_title='Values',
        hovermode='x unified',
        width = 1000)
    return fig9

def create_fig10(merged_df):
    fig10 = go.Figure()
    for col in vars.soiling_data:
        fig10.add_trace(go.Scatter(x=merged_df['t_stamp'], y=merged_df[col], mode='lines', name=col))
    fig10.update_layout(
        title='Soiling Raw Data',
        xaxis_title='Timestamp',
        yaxis_title='Values',
        hovermode='x unified',
        width = 1385)
    return fig10