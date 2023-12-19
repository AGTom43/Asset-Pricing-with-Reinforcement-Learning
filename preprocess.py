import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
# C:\Users\rohit\OneDrive\Documents\Asset-Pricing-with-Reinforcement-Learning\XOM_30_minute_6_month_data.csv
df = pd.read_csv("XOM_30_minute_6_month_data.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

# Check for missing values
# df.isnull().sum()

# Fill missing values if any
# df.fillna(method='ffill', inplace=True)

# Normalize
scaler = MinMaxScaler()
df[["Last Price", "Volume", "SMAVG (15)"]] = scaler.fit_transform(
    df[["Last Price", "Volume", "SMAVG (15)"]]
)

# Split into training and testing sets
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

print(df)
