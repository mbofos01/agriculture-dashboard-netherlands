import pandas as pd


def transform_daily_data(merged_data):
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    # 1. Group by 'date'
    daily_netherlands_data = merged_data.groupby('date')[['Frost days', 'Wet days', 'Precipitation rate',
                                                          'Minimum 2m temperature', 'Mean 2m temperature',
                                                          'Maximum 2m temperature', 'wind_speed_10m_max', "potential evapo-transpiration"]].mean().reset_index()

    daily_netherlands_data['Frost days'] = (
        daily_netherlands_data['Frost days'] > 0).astype(int)
    daily_netherlands_data['Wet days'] = (
        daily_netherlands_data['Wet days'] > 0).astype(int)

    daily_netherlands_data['year'] = daily_netherlands_data['date'].dt.year
    daily_netherlands_data['month'] = daily_netherlands_data['date'].dt.month

    # 2. Group by year and month
    monthly_netherlands_data = daily_netherlands_data.groupby(['year', 'month']).agg({
        'Frost days': 'sum',  # Sum for Frost days
        'Wet days': 'sum',  # Sum for Wet days
        'Precipitation rate': 'mean',  # Average for other variables
        'Minimum 2m temperature': 'mean',
        'Mean 2m temperature': 'mean',
        'Maximum 2m temperature': 'mean',
        'wind_speed_10m_max': 'mean',
        'potential evapo-transpiration': 'mean'
    }).reset_index()

    # 1. Create 'Month-Year' column
    monthly_netherlands_data['Month-Year'] = pd.to_datetime(monthly_netherlands_data['month'].astype(
        str) + '-' + monthly_netherlands_data['year'].astype(str), format='%m-%Y').dt.strftime('%m-%Y')
    monthly_netherlands_data = monthly_netherlands_data[[
        'Month-Year'] + [col for col in monthly_netherlands_data.columns if col != 'Month-Year']]
    # 2. Drop original columns
    monthly_netherlands_data = monthly_netherlands_data.drop(
        ['year', 'month'], axis=1)

    return monthly_netherlands_data


def transform_hourly_data(merged_data, monthly_netherlands_data):
    # sum hourly data to monthly
    hourly_merged_data = merged_data  # pd.read_csv('/hourlyAPI.csv')

    # comvert kpa to hpa for consistency with DB values
    hourly_merged_data['Vapour pressure'] = hourly_merged_data['Vapour pressure'] * 10

    # Convert 'date' column to datetime objects
    hourly_merged_data['date'] = pd.to_datetime(hourly_merged_data['date'])

    # Group by date and city and calculate the daily averages
    daily_data = hourly_merged_data.groupby([pd.Grouper(key='date', freq='D'), 'city'])[
        ['Cloud cover', 'Vapour pressure']].mean().reset_index()

    daily_data = daily_data.groupby(pd.Grouper(key='date', freq='D'))[
        ['Cloud cover', 'Vapour pressure']].mean().reset_index()

    daily_data['Month-Year'] = daily_data['date'].dt.strftime('%m-%Y')
    daily_data = daily_data[[
        'Month-Year'] + [col for col in daily_data.columns if col != 'Month-Year']]
    daily_data = daily_data.drop('date', axis=1)

    # Group by 'Month-Year' and calculate the mean for other columns
    monthly_avg_data = daily_data.groupby(
        'Month-Year')[['Cloud cover', 'Vapour pressure']].mean().reset_index()

    # Display the new DataFrame
    # print(monthly_avg_data.head(20))

    # Merge dayly and hourly data after summarisign
    final_merged_data = pd.merge(
        monthly_netherlands_data, monthly_avg_data, on='Month-Year', how='inner')

    # EDO EINAI MONTHLY
    monthly_merged_data__ = final_merged_data.copy()
    monthly_merged_data__.drop(columns=['wind_speed_10m_max'])
    # DISPATCH TO LOAD

    # Add season column
    df = final_merged_data

    def get_season(month_year_str):
        # Extract month from 'Month-Year' string
        month = int(month_year_str[:2])
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Unknown"  # Handle invalid months if necessary

    # Apply the function to create the 'Season' column
    df['Season'] = df['Month-Year'].apply(get_season)

    # Display the updated DataFrame
    # print(df.head())

    # calclate seasonal avges

    # Step 1: Convert 'Month-Year' to datetime and extract 'Year'
    df['Month-Year'] = pd.to_datetime(df['Month-Year'], format='%m-%Y')
    df['Year'] = df['Month-Year'].dt.year

    # Step 2: Calculate yearly summaries for each feature
    # Group by 'Year' and calculate mean for all features except 'Frost days' and 'Wet days', where we calculate the sum
    yearly_summary = df.groupby('Year').agg({
        'Frost days': 'sum',
        'Wet days': 'sum',
        'Precipitation rate': 'mean',
        'Minimum 2m temperature': 'mean',
        'Mean 2m temperature': 'mean',
        'Maximum 2m temperature': 'mean',
        'wind_speed_10m_max': 'mean',
        'potential evapo-transpiration': 'mean',
        'Cloud cover': 'mean',
        'Vapour pressure': 'mean'
    }).reset_index()

    # Step 3: Calculate seasonal averages for each feature within each year
    # Group by 'Year' and 'Season' and calculate mean for all features except 'Frost days' and 'Wet days', where we calculate the sum
    seasonal_summary = df.groupby(['Year', 'Season']).agg({
        'Frost days': 'sum',
        'Wet days': 'sum',
        'Precipitation rate': 'mean',
        'Minimum 2m temperature': 'mean',
        'Mean 2m temperature': 'mean',
        'Maximum 2m temperature': 'mean',
        'wind_speed_10m_max': 'mean',
        'potential evapo-transpiration': 'mean',
        'Cloud cover': 'mean',
        'Vapour pressure': 'mean'
    }).reset_index()

    # Display both summaries
    # print("Yearly Summary:")
    # print(yearly_summary)
    # print("\nSeasonal Summary:")
    # print(seasonal_summary)

    # Step 4: Pivot the seasonal summary to create Feature_Season columns
    # Create a new column for each feature-season combination
    seasonal_summary_melted = seasonal_summary.melt(id_vars=['Year', 'Season'],
                                                    var_name='Feature',
                                                    value_name='Value')
    seasonal_summary_melted['Feature_Season'] = seasonal_summary_melted['Feature'] + \
        '_' + seasonal_summary_melted['Season']

    # Pivot to have each Feature_Season as a column
    seasonal_summary_pivot = seasonal_summary_melted.pivot(
        index='Year', columns='Feature_Season', values='Value')

    # Reset index for a clean DataFrame
    seasonal_summary_pivot.reset_index(inplace=True)

    # Display the reshaped seasonal summary with Feature_Season columns
    # print("Reshaped Seasonal Summary:")
    # print(seasonal_summary_pivot)

    # Drop the 'Feature_Season' column if it exists in the pivoted DataFrame
    # Note: In seasonal_summary_pivot, the 'Feature_Season' column does not exist as it's already in columns.
    # So, no action needed here for dropping.

    # Merge the two DataFrames on 'Year'
    final_summary = pd.merge(
        yearly_summary, seasonal_summary_pivot, on='Year', how='inner')

    # Display the final summary
    # print("Final Merged Summary:")
    # print(final_summary)
    final_summary_yearly = final_summary

    # Final dataframe to be loaded to the db
    # Change col names to be concistent with our data in the DB
    rename_mapping = {
        'Cloud cover_Spring': 'cld_MAM', 'Cloud cover_Summer': 'cld_JJA', 'Cloud cover_Autumn': 'cld_SON', 'Cloud cover_Winter': 'cld_DJF',
        'potential evapo-transpiration_Spring': 'pet_MAM', 'potential evapo-transpiration_Summer': 'pet_JJA',
        'potential evapo-transpiration_Autumn': 'pet_SON', 'potential evapo-transpiration_Winter': 'pet_DJF',
        'Precipitation rate_Spring': 'pre_MAM', 'Precipitation rate_Summer': 'pre_JJA', 'Precipitation rate_Autumn': 'pre_SON',
        'Precipitation rate_Winter': 'pre_DJF',
        'Minimum 2m temperature_Spring': 'tmn_MAM', 'Minimum 2m temperature_Summer': 'tmn_JJA',
        'Minimum 2m temperature_Autumn': 'tmn_SON', 'Minimum 2m temperature_Winter': 'tmn_DJF',
        'Mean 2m temperature_Spring': 'tmp_MAM', 'Mean 2m temperature_Summer': 'tmp_JJA',
        'Mean 2m temperature_Autumn': 'tmp_SON', 'Mean 2m temperature_Winter': 'tmp_DJF',
        'Maximum 2m temperature_Spring': 'tmx_MAM', 'Maximum 2m temperature_Summer': 'tmx_JJA',
        'Maximum 2m temperature_Autumn': 'tmx_SON', 'Maximum 2m temperature_Winter': 'tmx_DJF',
        'Vapour pressure_Spring': 'vap_MAM', 'Vapour pressure_Summer': 'vap_JJA', 'Vapour pressure_Autumn': 'vap_SON',
        'Vapour pressure_Winter': 'vap_DJF',
        'Frost days_Spring': 'frs_MAM', 'Frost days_Summer': 'frs_JJA', 'Frost days_Autumn': 'frs_SON', 'Frost days_Winter': 'frs_DJF',
        'Wet days_Spring': 'wet_MAM', 'Wet days_Summer': 'wet_JJA', 'Wet days_Autumn': 'wet_SON', 'Wet days_Winter': 'wet_DJF'
    }

    # Rename the columns
    final_summary_yearly.rename(columns=rename_mapping, inplace=True)

    # Drop the specified columns
    final_summary_yearly.drop(columns=['wind_speed_10m_max_Autumn', 'wind_speed_10m_max_Spring',
                                       'wind_speed_10m_max_Summer', 'wind_speed_10m_max_Winter'], inplace=True)

    return final_summary_yearly, monthly_merged_data__
