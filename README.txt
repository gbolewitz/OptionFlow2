Options Flow Summarizer — Cloud Deployment (Full Version)

Deploy on Streamlit Cloud:
1) Create a new GitHub repo and add:
   - streamlit_app.py
   - requirements.txt
2) Go to https://share.streamlit.io (Streamlit Cloud) → New app
3) Select your repo, branch = main, file = streamlit_app.py
4) Deploy, open the app URL, upload your CSV, and download the summary.

Notes
- This version computes: strike distance zones, OI vs Volume vs Size, DTE concentration,
  key levels narrative, and strategy considerations (non-advice).
- Works best with Barchart-like exports; columns are normalized flexibly.