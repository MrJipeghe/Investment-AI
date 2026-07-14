import yfinance as yf
import pandas as pd

# 1. Download 5 years of daily data for the S&P 500 (SPY)
print("Downloading data...")
ticker = yf.Ticker("AAPL")
# We only want the historical price data
df = ticker.history(period="5y")

# 2. Clean it up (Remove columns we don't need right now)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print(df.tail()) # Prints the last 5 days of data so you can see it

# 3. Create new columns for Moving Averages
# The rolling() function looks back at the last 'X' days and calculates the average
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# 4. Drop the rows that have empty data (the first 50 days won't have a 50-day average!)
df = df.dropna()

# 5. Create a 'Target' column.
# Shift(-1) grabs TOMORROW'S closing price.
# We check if Tomorrow's Close is greater than Today's Close.
# astype(int) turns True/False into 1 (Up) or 0 (Down).

df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# We drop the very last row because we don't know tomorrow's price yet!
df = df.dropna()

# 6. Define our Features (the inputs) and the Target (the output)
features = ['Close', 'Volume', 'SMA_10', 'SMA_50']
X = df[features] # Inputs
y = df['Target'] # Output

# 7. Split the data. Let's use the first 80% of time for training, and the last 20% for testing.
split_index = int(len(df) * 0.8)

# Everything before the split index
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

# Everything after the split index
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# 8. Create the model.
# n_estimators=100 means we are building 100 decision trees.
# random_state=42 just ensures we get the same results if we run it twice.
model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)

print("Training the AI. Please wait...")
# 9. Train it! This is where the math happens.
model.fit(X_train, y_train)

# 10. Ask the model to predict the test data
predictions = model.predict(X_test)

# 11. Calculate how accurate it was.
# Precision asks: "When the AI said the stock would go UP, how often was it actually right?"
precision = precision_score(y_test, predictions)

print(f"AI Precision Score: {precision * 100:.2f}%")



# 12. Create a bridge to the second file
# We will save the test dates, actual prices, and predictions to a CSV file.
print("Saving data for the visualizer...")
output_df = pd.DataFrame({
    'Real Price': X_test['Close'],
    'AI_Prediction': predictions
})
output_df.to_csv("ai_prediction_results.csv")
print("Data saved successfully to 'ai_prediction_results.csv'!")