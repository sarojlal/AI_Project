import yfinance as yf
import pandas as pd
import ta
import os

# Define the save path
save_path = r'C:\Users\Home'

# Define the date range
start_date = '1979-12-26'
end_date = '2024-06-21'

# List of top 50 stock tickers from each specified sector
tickers = {
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'ORCL', 'IBM', 'AMD', 'QCOM', 'TXN', 'MU', 'AVGO', 'CRM', 'PYPL', 'INTC', 'CSCO', 'HPQ', 'ADSK', 'MSI', 'NTAP', 'ANSS', 'FTNT', 'WDAY', 'PANW', 'TEAM', 'SNOW', 'TWLO', 'NOW', 'CDNS', 'VMW', 'SQ', 'SPOT', 'OKTA', 'MDB', 'DDOG', 'ZS', 'VEEV', 'CHKP', 'DOCU', 'CRWD', 'PLTR', 'NET', 'ASAN', 'BILL', 'APPN', 'ZM'],
    'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'UNH', 'BMY', 'GILD', 'AMGN', 'MDT', 'DHR', 'SYK', 'ISRG', 'BSX', 'ZBH', 'ABT', 'HUM', 'CI', 'VRTX', 'REGN', 'BIIB', 'MCK', 'WAT', 'A', 'MTD', 'BDX', 'XRAY', 'PKI', 'BIO', 'IDXX', 'HOLX', 'BAX', 'STE', 'ILMN', 'EW', 'ALGN', 'TECH', 'RMD', 'LH', 'DGX', 'INCY', 'ACN', 'IQV', 'LHCG', 'NEO', 'CRL', 'QGEN', 'BIO', 'SYNH'],
    'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'BX', 'AMT', 'PLD', 'SPG', 'CCI', 'EQIX', 'PSA', 'EXR', 'WPC', 'VICI', 'O', 'STOR', 'IRM', 'MAA', 'INVH', 'ESS', 'EQR', 'AVB', 'CPT', 'AIV', 'MGM', 'WY', 'DRE', 'ARE', 'AMH', 'CBRE', 'SLG', 'BXP', 'VNO', 'PEB', 'HST', 'XHR', 'PK', 'APLE', 'DRH', 'SOHO', 'AHT', 'RHP', 'RLJ', 'FCH'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TJX', 'TGT', 'BKNG', 'MAR', 'HLT', 'ROST', 'DG', 'EBAY', 'EXPE', 'AZO', 'ORLY', 'AAP', 'ULTA', 'DRI', 'CMG', 'YUM', 'DHI', 'LEN', 'PHM', 'TOL', 'NVR', 'MHK', 'WHR', 'NWL', 'KMX', 'GPC', 'LKQ', 'APTV', 'GM', 'F', 'TSN', 'HRL', 'MKC', 'ADM', 'BG', 'SJM', 'CAG', 'CPB', 'GIS', 'K', 'KHC', 'POST'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'CL', 'KMB', 'HSY', 'MDLZ', 'EL', 'STZ', 'CLX', 'HRL', 'CPB', 'GIS', 'K', 'KHC', 'MKC', 'TSN', 'ADM', 'BG', 'SJM', 'CAG', 'POST', 'FMX', 'TAP', 'BUD', 'DEO', 'BF.B', 'MNST', 'KDP', 'CHD', 'TGT', 'WBA', 'CVS', 'RAD', 'UN', 'UL', 'HENKY', 'NSRGY', 'ULVR', 'REYN', 'HAIN', 'FLO', 'ARMK', 'GNLN'],
    'Telecommunication Services': ['T', 'VZ', 'TMUS', 'CHTR', 'CMCSA', 'NFLX', 'DIS', 'DISH', 'VOD', 'BCE', 'RCI', 'TEF', 'AMX', 'TU', 'SKM', 'KT', 'CHL', 'NOK', 'ERIC', 'MBT', 'VEON', 'ORAN', 'TDS', 'USM', 'LBTYA', 'LBRDA', 'CABO', 'ATUS', 'GSAT', 'IRDM', 'VSAT', 'INTC', 'AMD', 'TXN', 'AVGO', 'QCOM', 'NVDA', 'MU', 'SWKS', 'MPWR', 'NXPI', 'ADI', 'MRVL', 'LSCC', 'MCHP', 'CY', 'AMD', 'ON', 'AMKR']
} 

# List of indices
indices = {
    'S&P 500': '^GSPC',
    'Dow Jones Industrial Average': '^DJI',
    'Nasdaq-100': '^NDX',
    'Russell 2000': '^RUT'
}

# Function to download data with error handling
def download_data(tickers, indices):
    data = {}
    for sector, sector_tickers in tickers.items():
        for ticker in sector_tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                df['Sector'] = sector
                data[ticker] = df
            except Exception as e:
                print(f"Failed to download {ticker}: {e}")
    for index_name, index_ticker in indices.items():
        try:
            df = yf.download(index_ticker, start=start_date, end=end_date, auto_adjust=True)
            df['Sector'] = 'Index'
            data[index_ticker] = df
        except Exception as e:
            print(f"Failed to download {index_name}: {e}")
    return data

# Download historical data for each ticker and index
data = download_data(tickers, indices)

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

# Apply the function to calculate indicators for each ticker
for ticker in combined_data.columns.levels[0]:
    if ticker not in indices.values():
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
file_path = os.path.join(save_path, 'enhanced_stock_data_with_tech_indicators_and_season.csv')
combined_data.to_csv(file_path)
print(f"CSV file with technical indicators and seasonal effect saved to {file_path}")
