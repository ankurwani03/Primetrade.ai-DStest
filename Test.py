import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)

base_path = Path(r"D:\KJSCE\Primetrade.ai Intern test")
trader_file = base_path / "historical_data.csv"
sentiment_file = base_path / "fear_greed_index.csv"

# Load CSV files
try:
    trader_data = pd.read_csv(trader_file)
    sentiment_data = pd.read_csv(sentiment_file)
    print(" Data successfully loaded.")
except Exception as e:
    raise FileNotFoundError(f" Failed to load files: {e}")

trader_data['date'] = pd.to_datetime(trader_data['Timestamp IST'], format="%d-%m-%Y %H:%M").dt.date

daily_trader_perf = trader_data.groupby('date').agg({
    'Closed PnL': 'sum',
    'Execution Price': 'mean',
    'Size USD': 'sum',
    'Fee': 'sum'
}).reset_index()

daily_trader_perf.columns = ['date', 'total_pnl', 'avg_price', 'total_volume_usd', 'total_fees']

sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
sentiment_data_clean = sentiment_data[['date', 'value', 'classification']]
sentiment_data_clean.columns = ['date', 'sentiment_score', 'sentiment_label']

merged_data = pd.merge(daily_trader_perf, sentiment_data_clean, on='date', how='inner')
correlation_matrix = merged_data[[
    'total_pnl', 'avg_price', 'total_volume_usd', 'total_fees', 'sentiment_score'
]].corr()
fig, axs = plt.subplots(2, 2, figsize=(18, 12))

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axs[0, 0])
axs[0, 0].set_title("Correlation Heatmap")
axs[0, 0].set_xticklabels(axs[0, 0].get_xticklabels(), rotation=45)
axs[0, 0].set_yticklabels(axs[0, 0].get_yticklabels(), rotation=0)

#  PnL by sentiment
sns.boxplot(data=merged_data, x='sentiment_label', y='total_pnl',
            order=["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"], ax=axs[0, 1])
axs[0, 1].set_title("Trader PnL by Market Sentiment")
axs[0, 1].set_xlabel("Market Sentiment")
axs[0, 1].set_ylabel("Total PnL")

# Sentiment Score and PnL
axs[1, 0].plot(merged_data['date'], merged_data['sentiment_score'], label='Sentiment Score', color='orange', linewidth=2)
axs[1, 0].plot(merged_data['date'], merged_data['total_pnl'], label='Total PnL', color='blue', linewidth=2)
axs[1, 0].set_title("Sentiment Score vs Trader PnL Over Time")
axs[1, 0].set_xlabel("Date")
axs[1, 0].set_ylabel("Value")
axs[1, 0].legend()

# Volume vs Sentiment
axs[1, 1].plot(merged_data['date'], merged_data['total_volume_usd'], label='Total Volume (USD)', color='green', linewidth=2)
axs[1, 1].plot(merged_data['date'], merged_data['sentiment_score'], label='Sentiment Score', color='orange', linewidth=2)
axs[1, 1].set_title("Trading Volume vs Sentiment Score Over Time")
axs[1, 1].set_xlabel("Date")
axs[1, 1].set_ylabel("Value")
axs[1, 1].legend()

plt.tight_layout()
plt.show()
