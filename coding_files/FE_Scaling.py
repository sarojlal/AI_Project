import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

# Step 1: Load the dataset into a dataframe
df = pd.read_csv("C://Users/jesel sequeira/Downloads/finalest.csv")

# Printing top 5 rows of the dataset
print(df.head())

# Printing last 5 rows of the dataset
print(df.tail())

# Step 2: Initial Exploration
# Get basic information about the dataset
print(df.info())

# Get summary statistics for numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Dropping the null values
df.dropna(inplace=True)

# Check for missing values
print(df.isnull().sum())

# Printing datatypes
print(df.dtypes)

# Data types conversion
df['Date'] = pd.to_datetime(df['Date'])

# Printing the information after type conversion
print(df.info())

# Checking the shape of the dataset
print(df.shape)

# Step 3: Feature Engineering

# 1. Date Features
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
# df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year

# 2. Price Movement Indicators
# Computes relative differences between Close vs. Open and High vs. Low, which can indicate intraday volatility.
df['Close_vs_Open'] = (df['Close'] - df['Open']) / df['Open']
df['High_vs_Low'] = (df['High'] - df['Low']) / df['Low']

# 3. Technical Indicators (Example: RSI-based feature)
# Generates an RSI_Signal based on the Relative Strength Index (RSI), categorizing it as 'Overbought', 'Oversold', or 'Normal'.
df['RSI_Signal'] = np.where(df['RSI'] > 70, 'Overbought', np.where(df['RSI'] < 30, 'Oversold', 'Normal'))

# Create a new feature for the previous day's closing price (Feature Engineering)
df['Previous_Close'] = df['Close'].shift(1)
df.head(15)

# Checking for null values
df.isnull().sum()

# Drop the first row with NaN value introduced by the shift operation
df = df.dropna()
df.head(15)

# Step 4: Encoding Categorical Variables
label_encoders = {}
for col in ['Company', 'Sector', 'Season']:
    label_encoders[col] = LabelEncoder()
    df[col + '_Encoded'] = label_encoders[col].fit_transform(df[col])

all_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'Day_of_Week', 'Month', 'Year', 'Close_vs_Open', 'High_vs_Low', 'Company_Encoded', 'Sector_Encoded', 'Season_Encoded','Previous_Close']


# Compute the correlation matrix
correlation_matrix = df[all_cols].corr()

# Plot the correlation matrix using seaborn
plt.figure(figsize=(15, 9))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('First Correlation Matrix of Numeric Features')
plt.show()

# Drop the unnecessary columns
final_df = df.drop(columns=['Date','Company','Open','High','Low','Volume','Day_of_Week','Month','Year','Close_vs_Open','High_vs_Low','Company_Encoded','Sector_Encoded','Season_Encoded','Sector','Season','RSI_Signal'])

# Use all columns except 'Close' as features for X
# features = df.columns.tolist()
features = ['EMA_50', 'EMA_200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist']
# features.remove('Close')

# Prepare X and Y
X = df[features].values
Y = df['Close'].values

### LSTM are sensitive to the scale of the data. so we apply MinMax scaler
# Normalize the dataset
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

#  Step 6: Data Splitting
from sklearn.model_selection import train_test_split
Target_Variable='Close'
# Assuming 'final_df' contains your selected features and target variable
X = final_df.drop(columns=[Target_Variable])  # Features
y = final_df[Target_Variable]  # Target variable

# Split data into training and test sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
