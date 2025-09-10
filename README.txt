# Options Flow Summarizer

This Streamlit app analyzes option flow data with:
- Weighted bias scoring for bullish/bearish signals
- Overall and per-level dial gauges
- Data visualization with Plotly

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Start the Streamlit app:
   streamlit run streamlit_app.py

3. Upload a CSV of option flow data.

## Features
- Normalizes and cleans raw option flow data
- Calculates weighted bias scores
- Visual gauges for net market sentiment
- Ranks most bullish and bearish levels
