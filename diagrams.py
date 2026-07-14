import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

print("Loading data from the training model...")

# 1. Load the data we saved from the first script
# parse_dates=True and index_col=0 ensures the dates are formatted correctly for the chart
try:
    df = pd.read_csv("ai_prediction_results.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("Error: Could not find 'ai_prediction_results.csv'. Make sure you run the first file to generate it!")
    exit()

# 2. Set up the diagram canvas
fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle("AAPL Price vs. AI Directional Predictions", fontsize=16)

# 3. Plot the actual price line (Real Values)
ax.plot(df.index, df['Real Price'], color='black', label='Actual Price', linewidth=1.5)

# 4. Overlay the prediction backgrounds (Green = Expected Up, Red = Expected Down)
# We loop through the dates to draw a colored box between each day
for i in range(len(df) - 1):
    current_date = df.index[i]
    next_date = df.index[i + 1]

    # Check what the AI predicted for this time period
    prediction_for_tomorrow = df['AI_Prediction'].iloc[i]

    if prediction_for_tomorrow == 1:
        color = 'green'  # AI predicted price would go UP
    else:
        color = 'red'  # AI predicted price would go DOWN

    # Draw the colored block on the background
    ax.axvspan(current_date, next_date, color=color, alpha=0.15)

# 5. Formatting and Legend to make it readable
ax.set_ylabel("Share Price ($)")
ax.set_xlabel("Date")
ax.grid(True, linestyle=':', alpha=0.5)

# Build a clean legend so the user understands the colors
legend_elements = [
    Patch(facecolor='green', alpha=0.15, label='AI Expected UP'),
    Patch(facecolor='red', alpha=0.15, label='AI Expected DOWN'),
    plt.Line2D([0], [0], color='black', label='Actual Price')
]
ax.legend(handles=legend_elements, loc='upper left')

print("Opening Diagram...")
plt.tight_layout()
plt.show()