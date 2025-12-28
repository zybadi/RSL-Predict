import pandas as pd
import numpy as np
import os, json, xgboost as xgb

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "spl_results.xlsx")
ASSETS = os.path.join(BASE, "assets")
META = os.path.join(ASSETS, "meta.json")
HOME_MODEL = os.path.join(ASSETS, "home_model.json")
AWAY_MODEL = os.path.join(ASSETS, "away_model.json")
ELO_OUT = os.path.join(ASSETS, "elo_latest.csv")

# ---------------- LOAD ----------------
df = pd.read_excel(RESULTS)
with open(META) as f:
    meta = json.load(f)

# ---------------- APPEND ROUND 11 ----------------
r11 = pd.DataFrame({
    "Round":[11]*9,
    "Home Team":["Al Fayha","Al Riyadh","Neom S.C.","Al Fateh","Al Khlood","Al Hilal","Al Qadsiah","Al Nassr","Al Ittihad"],
    "Away Team":["Al Hazem","Al Ettifaq","Al Najmah","Al Ahli","Al Taawoun","Al Khaleej","Damac","Al Okhdood","Al Shabab"],
    "Home Score":[0,0,2,2,0,3,1,3,2],
    "Away Score":[0,2,1,1,2,2,1,0,0],
})
r11["Result_Code"] = np.sign(r11["Home Score"] - r11["Away Score"]).astype(int)

df = pd.concat([df, r11]).drop_duplicates(subset=["Round","Home Team","Away Team"])
df.to_excel(RESULTS, index=False)

# ---------------- BUILD TRAIN MATRIX ----------------
from app import build_training_matrix   # reuse your real logic

X, y_home, y_away = build_training_matrix(df, meta)

dtrain_h = xgb.DMatrix(X, label=y_home)
dtrain_a = xgb.DMatrix(X, label=y_away)

home_model = xgb.train({"objective":"reg:squarederror"}, dtrain_h, 250)
away_model = xgb.train({"objective":"reg:squarederror"}, dtrain_a, 250)

home_model.save_model(HOME_MODEL)
away_model.save_model(AWAY_MODEL)

# ---------------- RECOMPUTE ELO ----------------
elo = {t:1500 for t in pd.unique(df[["Home Team","Away Team"]].values.ravel())}

for _,r in df.sort_values("Round").iterrows():
    h,a,rc = r["Home Team"], r["Away Team"], r["Result_Code"]
    Rh,Ra = elo[h], elo[a]
    Eh = 1/(1+10**(-((Rh+50-Ra)/400)))
    Sh = 1 if rc==1 else 0 if rc==-1 else 0.5
    elo[h] = Rh + 20*(Sh-Eh)
    elo[a] = Ra + 20*((1-Sh)-(1-Eh))

pd.DataFrame({"Team":elo.keys(),"Elo":elo.values()}).to_csv(ELO_OUT,index=False)

print("Round 11 successfully applied.")
