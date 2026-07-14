# Investment AI Project Context

This document provides a comprehensive overview of the Investment AI project, designed for AI models (like Claude) to understand the codebase, architecture, and workflows efficiently.

## 1. Project Overview
The **Investment AI** project is a Python-based predictive tool that uses historical stock market data to predict whether a specific stock's price (specifically AAPL) will go up or down on the following day. It leverages machine learning to identify patterns in price movements and technical indicators.

### General Use Case
- **Financial Analysis:** Automating the prediction of stock direction.
- **Visual Validation:** Providing a clear, color-coded visual representation of AI predictions against actual price movements.
- **Educational:** Demonstrating a basic end-to-end ML pipeline for time-series classification.

---

## 2. Architecture & Pipeline
The project follows a linear pipeline split across two main scripts:

### Pipeline Stages:
1.  **Data Acquisition:** Downloads 5 years of historical OHLCV (Open, High, Low, Close, Volume) data for AAPL using the Yahoo Finance API.
2.  **Feature Engineering:**
    *   Calculates **Simple Moving Averages (SMA)** for 10-day and 50-day windows.
    *   Cleans data by removing rows with NaN values (resulting from the rolling window).
3.  **Labeling:** Creates a binary 'Target' column where `1` indicates the price went up the next day, and `0` indicates it went down or stayed the same.
4.  **Model Training:**
    *   Uses a **RandomForestClassifier** (100 trees, `min_samples_split=50`).
    *   Splits data chronologically (80% training, 20% testing) to avoid data leakage (look-ahead bias).
5.  **Evaluation:** Calculates a **Precision Score** to measure how often the "UP" predictions were correct.
6.  **Persistence:** Saves the test results (Date, Real Price, AI Prediction) to `ai_prediction_results.csv`.
7.  **Visualization:** Loads the CSV and generates a chart with colored background spans (Green for UP predictions, Red for DOWN) overlaid on the actual price line.

---

## 3. Libraries & Tools
| Library | Purpose | Why? |
| :--- | :--- | :--- |
| `yfinance` | Data Sourcing | Provides a reliable and free way to access historical market data directly from Yahoo Finance. |
| `pandas` | Data Manipulation | Industry standard for handling tabular data (DataFrames), performing rolling calculations, and CSV I/O. |
| `scikit-learn` | Machine Learning | Used for the `RandomForestClassifier` and precision metrics due to its robust and easy-to-use API. |
| `matplotlib` | Visualization | Used to create the final chart with customized background spans (`axvspan`) and legends. |

---

## 4. File Structure & Descriptions
- **`ai_stock.py`**: The "Brain". Handles data fetching, feature engineering, model training, and saving results.
- **`diagrams.py`**: The "Eyes". Handles loading results and generating the interactive plot.
- **`main.py`**: Currently a placeholder script.
- **`ai_prediction_results.csv`**: The "Bridge". A data exchange file between the training script and the visualizer.

---

## 5. Inputs & Outputs

### Inputs:
- **Ticker:** `AAPL` (hardcoded in `ai_stock.py`).
- **Period:** 5 years of daily historical data.
- **Features:** `Close`, `Volume`, `SMA_10`, `SMA_50`.

### Outputs:
- **Metric:** AI Precision Score (printed to console).
- **CSV Data:** `ai_prediction_results.csv` with the following columns:
    - `Date`: (Index) Timestamp of the trading day.
    - `Real Price`: The actual closing price of the stock.
    - `AI_Prediction`: `1` (UP) or `0` (DOWN).
- **Visual:** A Matplotlib window showing the price trend with prediction-based background coloring.

---

## 6. Key Logic Details (For Claude)
- **Time-Series Integrity:** The project uses `.iloc[:split_index]` for training and `.iloc[split_index:]` for testing, which is crucial for time-series data to ensure the model isn't "seeing the future" during training.
- **Binary Classification:** This is NOT a regression problem (predicting exact price). It is a binary classification problem (predicting direction).
- **Visualization Logic:** In `diagrams.py`, the `axvspan` is used to color the *interval* between two dates based on the prediction made for that period.

---
*Created by Gemini CLI on June 11, 2026.*
