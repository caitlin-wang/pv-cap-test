def filter_meter_greater_zero(df, minimum_grid):
    return df['average_meter_data'] > minimum_grid

def filter_grid_clipping(df, grid_clipping):
    return df['average_meter_data'] < grid_clipping

def filter_inverter_clipping(df, inverter_data, inverter_clipping):
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
    df['fpoa_spatial_std'] = df[fpoa_data].std(axis=1) # changed fpoa_data to df[fpoa_data]
    df['fpoa_spatial_mean'] = df[fpoa_data].mean(axis=1) # changed fpoa_data to df[fpoa_data]
    df['fpoa_spatial_stability'] = df['fpoa_spatial_std'] / df['fpoa_spatial_mean']
    df['spatial_stability_check'] = df['fpoa_spatial_stability'].abs() <= spatial_threshold
    return df['spatial_stability_check']

def filter_temporal_stability(df, temporal_threshold):
    df['average_fpoa_pct_change']=df['average_fpoa'].pct_change()*100
    df['temporal_stability_check']=df['average_fpoa_pct_change'].abs()<=temporal_threshold
    return df['temporal_stability_check']