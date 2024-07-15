import yfinance as yf
import pandas as pd
import ta
import os

save_path = r'C:\Users\Home\enhanced_index_data_with_tech_indicators_and_season.csv'

start_date = '1979-12-26'
end_date = '2024-06-21'

# List of indices
indices = {
    'S&P 500': '^GSPC',
    'Dow Jones Industrial Average': '^DJI',
    'Nasdaq-100': '^NDX',
    'Russell 2000': '^RUT',
    'XAUUSD': 'GC=F',
    'US02Y': '^IRX',  # 2-Year Treasury Yield Index
    'US10Y': '^TNX',  # 10-Year Treasury Yield Index
    'DXY': 'DX-Y.NYB',  # US Dollar Index
    'USIRYY': '^FVX'  # 5-Year Treasury Yield Index
}

# Function to download data with error handling
def download_data(indices):
    data = {}
    for index_name, index_ticker in indices.items():
        try:
            df = yf.download(index_ticker, start=start_date, end=end_date, auto_adjust=True)
            df['Sector'] = 'Index'
            data[index_ticker] = df
        except Exception as e:
            print(f"Failed to download {index_name}: {e}")
    return data

# Download historical data for each index
data = download_data(indices)

# Combine the data into a single DataFrame
combined_data = pd.concat(data, keys=data.keys(), axis=1)

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA_200'] = ta.trend.ema_indicator(df['Close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_Hist'] = ta.trend.macd_diff(df['Close'])  # MACD Histogram
    return df

# Apply the function to calculate indicators for each index
for ticker in combined_data.columns.levels[0]:
    df = combined_data[ticker].copy()
    df = calculate_technical_indicators(df)
    for col in df.columns:
        combined_data[(ticker, col)] = df[col]

# Flatten the multi-level column index
combined_data.columns = ['_'.join(col).strip() for col in combined_data.columns.values]

# Ensure the combined_data index is a DateTimeIndex
combined_data.index = pd.to_datetime(combined_data.index)

# Extract month from the index
combined_data['Month'] = combined_data.index.month

# Define the function to assign seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Apply the season function to the Month column
combined_data['Season'] = combined_data['Month'].apply(get_season)

# Convert the 'Season' column to a categorical type
combined_data['Season'] = combined_data['Season'].astype('category')

# Drop the 'Month' column if it's not needed
combined_data.drop('Month', axis=1, inplace=True)

# Save the enhanced data to a CSV file
combined_data.to_csv(save_path)

print(f"CSV file with technical indicators and seasonal effect saved to {save_path}")
