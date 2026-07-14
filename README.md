# Investment AI

A small Python project that uses machine learning to predict whether **AAPL** (Apple) stock will close higher the next trading day, then visualizes those predictions against actual prices.

> ⚠️ **Disclaimer:** This is an educational project, not financial advice. Directional stock prediction from price/volume alone is notoriously hard, and results should not be used for real trading decisions.

## What it does

1. Downloads 5 years of daily AAPL price data from Yahoo Finance.
2. Builds features: closing price, volume, and 10-day / 50-day simple moving averages (SMA).
3. Labels each day `1` (up) or `0` (down/flat) based on the *next* day's close.
4. Trains a `RandomForestClassifier` on the first 80% of the timeline and tests on the last 20% (chronological split — no look-ahead).
5. Prints a **precision score** for the "UP" predictions and saves results to `ai_prediction_results.csv`.
6. Plots actual price with green/red background shading for predicted up/down days.

## Project structure

| File | Role |
| :--- | :--- |
| `ai_stock.py` | Fetches data, engineers features, trains the model, saves predictions. |
| `diagrams.py` | Loads the saved predictions and renders the price-vs-prediction chart. |
| `ai_prediction_results.csv` | Output data passed from the model script to the chart script. |
| `main.py` | Placeholder / scratch file (not part of the pipeline). |
| `PROJECT_CONTEXT.md` | Detailed technical notes on the codebase. |

## Requirements

- Python 3.9+
- Packages: `yfinance`, `pandas`, `scikit-learn`, `matplotlib`

Install them with:

```bash
pip install yfinance pandas scikit-learn matplotlib
```

## Usage

Run the model to generate predictions:

```bash
python ai_stock.py
```

This prints the precision score and writes `ai_prediction_results.csv`.

Then generate the chart:

```bash
python diagrams.py
```

A Matplotlib window opens showing the actual AAPL price with green shading where the model predicted "up" and red where it predicted "down".

## How it works

The task is framed as **binary classification** (direction: up vs. down), not regression (exact price). Data is split chronologically so the model never trains on future data. Features are intentionally simple — price, volume, and two moving averages — making this a clear end-to-end example of an ML pipeline for time-series classification.

## Ideas for extending it

- Make the ticker and date range configurable instead of hardcoded.
- Add more technical indicators (RSI, MACD, Bollinger Bands).
- Compare against a naive baseline (e.g. "always predict up") to sanity-check precision.
- Use walk-forward / rolling backtesting instead of a single train/test split.
