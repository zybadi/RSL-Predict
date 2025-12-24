# ⚽ SPL Match Score Predictor – Streamlit App

This application predicts Saudi Pro League football match scores using machine‑learning models trained on historical results, rolling team statistics, tier strength, and Elo ratings.

The app is updated round‑by‑round and always reflects the latest available league data.

---

## 🔮 Features

- Select **Home Team** and **Away Team** from dropdowns  
- Click **Predict** to get:
  - Predicted score (rounded integers)
  - Match outcome (Home Win / Away Win / Tie)
- Expand **Details** to view:
  - Raw decimal predictions
  - Outcome from decimal model vs rounded score
  - Elo ratings used for each team
  - Latest round used in training
  - Full feature row passed to the model

---

## 🗂 Project Structure

```
spl_streamlit_app/
│
├── app.py
├── requirements.txt
├── README.md
├── spl_results.xlsx
│
└── assets/
    ├── home_model.json
    ├── away_model.json
    ├── meta.json
    └── elo_latest.csv
```

---

## 🧠 Model Overview

- Two XGBoost regressors:
  - `home_model.json` → predicts home goals
  - `away_model.json` → predicts away goals
- Feature engineering includes:
  - Tier strength
  - Rolling averages (goals scored, conceded, win rate)
  - Home advantage bias
  - Optional Elo features
- The app automatically detects the required feature order from `meta.json`.

---

## 🚀 Deploying on Streamlit Community Cloud

### 1️⃣ Push project to GitHub

Create a new repository and upload the full project folder.

Make sure these files exist in the root:
- `app.py`
- `requirements.txt`
- `spl_results.xlsx`
- `assets/` folder (with models and metadata)

---

### 2️⃣ requirements.txt

Ensure it contains:

```
streamlit
pandas
numpy
openpyxl
xgboost
```

---

### 3️⃣ Deploy

1. Go to https://share.streamlit.io  
2. Click **New app**  
3. Choose your GitHub repo  
4. Set main file: `app.py`  
5. Click **Deploy**

You will receive a public URL you can share.

---

## 🔁 Updating Each Round

Whenever a new round finishes:

1. Update your notebook and retrain models
2. Replace:
   - `spl_results.xlsx`
   - `assets/home_model.json`
   - `assets/away_model.json`
   - `assets/elo_latest.csv`
   - `assets/meta.json`
3. Commit & push to GitHub

Streamlit will automatically redeploy your updated model.

---

## 📬 Support

This project is maintained by Ziyad Albaadi.
