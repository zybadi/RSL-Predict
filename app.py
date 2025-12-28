# app.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# =========================
# PATHS (match your folder)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_XLSX = os.path.join(BASE_DIR, "spl_results.xlsx")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

HOME_MODEL_PATH = os.path.join(ASSETS_DIR, "home_model.json")
AWAY_MODEL_PATH = os.path.join(ASSETS_DIR, "away_model.json")
META_PATH = os.path.join(ASSETS_DIR, "meta.json")
ELO_LATEST_PATH = os.path.join(ASSETS_DIR, "elo_latest.csv")

# A tiny marker so we don't retrain repeatedly
TRAIN_MARKER_PATH = os.path.join(ASSETS_DIR, "trained_round.txt")

os.makedirs(ASSETS_DIR, exist_ok=True)

st.set_page_config(page_title="SPL Score Predictor", layout="centered")
st.title("⚽ SPL Score Predictor")


# =========================
# LOADERS (cached)
# =========================
@st.cache_data
def load_meta():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Missing file: {META_PATH}")
    with open(META_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def load_results_excel():
    if not os.path.exists(RESULTS_XLSX):
        raise FileNotFoundError(f"Missing file: {RESULTS_XLSX}")

    df = pd.read_excel(RESULTS_XLSX)

    required = {"Round", "Home Team", "Away Team", "Home Score", "Away Score", "Result_Code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"spl_results.xlsx is missing columns: {sorted(list(missing))}")

    # ensure int round
    df["Round"] = df["Round"].astype(int)
    df["Home Score"] = df["Home Score"].astype(int)
    df["Away Score"] = df["Away Score"].astype(int)
    df["Result_Code"] = df["Result_Code"].astype(int)

    return df


@st.cache_resource
def load_models():
    # Use Booster since you're loading .json models
    if not (os.path.exists(HOME_MODEL_PATH) and os.path.exists(AWAY_MODEL_PATH)):
        return None, None

    home_model = xgb.Booster()
    away_model = xgb.Booster()
    home_model.load_model(HOME_MODEL_PATH)
    away_model.load_model(AWAY_MODEL_PATH)
    return home_model, away_model


# =========================
# Helpers
# =========================
def normalize_name(x, normalize_map: dict):
    if pd.isna(x):
        return x
    s = " ".join(str(x).strip().split())
    return normalize_map.get(s.lower(), s)


def build_tiers_and_strength(meta: dict):
    # Defaults (from your notebook)
    default_tier1 = {"Al Hilal", "Al Nassr", "Al Ahli", "Al Ittihad"}
    default_tier2 = {"Al Shabab", "Al Taawoun", "Al Fateh", "Al Qadsiah"}
    default_tier3 = {"Al Ettifaq", "Al Khaleej", "Damac", "Neom S.C.", "Al Khlood"}
    default_strength = {1: 1.6, 2: 1.3, 3: 1.1, 4: 1.0}

    tier1 = set(meta.get("tier1", default_tier1))
    tier2 = set(meta.get("tier2", default_tier2))
    tier3 = set(meta.get("tier3", default_tier3))

    raw_ts = meta.get("tier_strength", default_strength)
    tier_strength = {int(k): float(v) for k, v in raw_ts.items()}
    for k, v in default_strength.items():
        tier_strength.setdefault(k, v)

    return tier1, tier2, tier3, tier_strength


def get_strength(team: str, tier1: set, tier2: set, tier3: set, tier_strength: dict) -> float:
    if team in tier1:
        t = 1
    elif team in tier2:
        t = 2
    elif team in tier3:
        t = 3
    else:
        t = 4
    return float(tier_strength.get(t, 1.0))


def build_histories_for_rolling(df_completed: pd.DataFrame):
    work = df_completed.sort_values(["Round"]).reset_index(drop=True)
    teams = pd.unique(work[["Home Team", "Away Team"]].values.ravel("K"))
    hist = {t: {"matches": 0, "gs": 0, "gc": 0, "wins": 0} for t in teams}

    for _, r in work.iterrows():
        h, a = r["Home Team"], r["Away Team"]
        hist[h]["matches"] += 1
        hist[a]["matches"] += 1

        hist[h]["gs"] += int(r["Home Score"])
        hist[h]["gc"] += int(r["Away Score"])
        hist[a]["gs"] += int(r["Away Score"])
        hist[a]["gc"] += int(r["Home Score"])

        rc = int(r["Result_Code"])
        if rc == 1:
            hist[h]["wins"] += 1
        elif rc == -1:
            hist[a]["wins"] += 1

    return hist


def avg_from_hist(hist: dict, team: str, key: str) -> float:
    h = hist.get(team, {"matches": 0, "gs": 0, "gc": 0, "wins": 0})
    m = h["matches"]
    if m <= 0:
        return 0.0
    if key == "gs":
        return h["gs"] / m
    if key == "gc":
        return h["gc"] / m
    if key == "wins":
        return h["wins"] / m
    return 0.0


# =========================
# Elo (used for training + display)
# =========================
ELO_START = 1500.0
ELO_K = 20.0
ELO_HFA = 50.0

def elo_expected(r_home, r_away, hfa=ELO_HFA):
    return 1.0 / (1.0 + 10 ** (-(((r_home + hfa) - r_away) / 400.0)))

def elo_actual_from_rc(rc: int):
    if rc == 1:
        return 1.0, 0.0
    if rc == -1:
        return 0.0, 1.0
    return 0.5, 0.5

def recompute_elo_latest(df_matches: pd.DataFrame, start=ELO_START, k=ELO_K, hfa=ELO_HFA):
    work = df_matches.sort_values(["Round"]).reset_index(drop=True)
    teams = pd.unique(work[["Home Team","Away Team"]].values.ravel("K"))
    rating = {t: float(start) for t in teams}

    # track last elo per team per round for "end-of-round snapshot"
    per_team_round = {}  # (team, round) -> elo

    for _, r in work.iterrows():
        rd = int(r["Round"])
        h = r["Home Team"]
        a = r["Away Team"]
        rc = int(r["Result_Code"])

        Rh = rating.get(h, start)
        Ra = rating.get(a, start)

        exp_h = elo_expected(Rh, Ra, hfa=hfa)
        exp_a = 1.0 - exp_h

        Sh, Sa = elo_actual_from_rc(rc)

        dh = k * (Sh - exp_h)
        da = k * (Sa - exp_a)

        rating[h] = Rh + dh
        rating[a] = Ra + da

        per_team_round[(h, rd)] = rating[h]
        per_team_round[(a, rd)] = rating[a]

    # latest round available (not assuming consecutive rounds)
    rounds = sorted(work["Round"].unique())
    latest_round = rounds[-1]
    prev_round = rounds[-2] if len(rounds) >= 2 else None

    rows = []
    for t in sorted(rating.keys()):
        elo_now = float(per_team_round.get((t, latest_round), rating[t]))
        elo_prev = float(per_team_round.get((t, prev_round), np.nan)) if prev_round is not None else np.nan
        delta = (elo_now - elo_prev) if (prev_round is not None and not np.isnan(elo_prev)) else np.nan
        rows.append({"Team": t, "Elo": elo_now, "Delta": delta, "Round": int(latest_round), "Prev_Round_Used": prev_round})

    elo_latest = pd.DataFrame(rows).sort_values("Elo", ascending=False).reset_index(drop=True)
    elo_latest.insert(0, "Rank", np.arange(1, len(elo_latest) + 1))
    return elo_latest, rating  # rating dict is "current" elo


# =========================
# Feature builder (same style as your notebook)
# =========================
def build_feature_row(home_team: str, away_team: str, predict_round: int, df_completed: pd.DataFrame, meta: dict, elo_dict: dict):
    feature_cols = meta["feature_cols"]
    tier1, tier2, tier3, tier_strength = build_tiers_and_strength(meta)

    hist = build_histories_for_rolling(df_completed)

    hs = get_strength(home_team, tier1, tier2, tier3, tier_strength)
    aas = get_strength(away_team, tier1, tier2, tier3, tier_strength)

    row = {
        "Round": int(predict_round),
        "home_strength": hs,
        "away_strength": aas,
        "strength_diff": hs - aas,
        "home_avg_scored_pre": avg_from_hist(hist, home_team, "gs"),
        "home_avg_conceded_pre": avg_from_hist(hist, home_team, "gc"),
        "home_win_rate_pre": avg_from_hist(hist, home_team, "wins"),
        "away_avg_scored_pre": avg_from_hist(hist, away_team, "gs"),
        "away_avg_conceded_pre": avg_from_hist(hist, away_team, "gc"),
        "away_win_rate_pre": avg_from_hist(hist, away_team, "wins"),
        "home_advantage": 1.0,
    }

    need_elo = any(c in feature_cols for c in ["home_elo_pre", "away_elo_pre", "elo_diff"])
    if need_elo:
        h_elo = float(elo_dict.get(home_team, 1500.0))
        a_elo = float(elo_dict.get(away_team, 1500.0))
        row["home_elo_pre"] = h_elo
        row["away_elo_pre"] = a_elo
        row["elo_diff"] = h_elo - a_elo

    feats = pd.DataFrame([row])

    for c in feature_cols:
        if c not in feats.columns:
            feats[c] = 0.0

    feats = feats[feature_cols]
    X = feats.values
    return feats, X


# =========================
# Round 11 auto-append
# =========================
def ensure_round_11_exists(df: pd.DataFrame) -> (pd.DataFrame, bool):
    """
    If Round 11 is missing from spl_results.xlsx, append it using the known fixtures/scores.
    Returns (df_updated, changed_bool).
    """
    if (df["Round"] == 11).any():
        return df, False

    fixtures_r11 = pd.DataFrame({
        "Round": [11] * 9,
        "Home Team": [
            "Al Fayha",
            "Al Riyadh",
            "Neom S.C.",
            "Al Fateh",
            "Al Khlood",
            "Al Hilal",
            "Al Qadsiah",
            "Al Nassr",
            "Al Ittihad",
        ],
        "Away Team": [
            "Al Hazem",
            "Al Ettifaq",
            "Al Najmah",
            "Al Ahli",
            "Al Taawoun",
            "Al Khaleej",
            "Damac",
            "Al Okhdood",
            "Al Shabab",
        ],
        "Home Score": [0, 0, 2, 2, 0, 3, 1, 3, 2],
        "Away Score": [0, 2, 1, 1, 2, 2, 1, 0, 0],
    })
    fixtures_r11["Result_Code"] = np.sign(fixtures_r11["Home Score"] - fixtures_r11["Away Score"]).astype(int)

    df2 = pd.concat([df, fixtures_r11], ignore_index=True)
    df2 = df2.sort_values(["Round"]).reset_index(drop=True)
    return df2, True


def save_results_excel(df: pd.DataFrame):
    df.to_excel(RESULTS_XLSX, index=False)


def read_trained_round_marker() -> int | None:
    if not os.path.exists(TRAIN_MARKER_PATH):
        return None
    try:
        txt = open(TRAIN_MARKER_PATH, "r").read().strip()
        return int(txt)
    except Exception:
        return None


def write_trained_round_marker(r: int):
    with open(TRAIN_MARKER_PATH, "w") as f:
        f.write(str(int(r)))


# =========================
# Training (home/away goals)
# =========================
def build_training_matrix(df_all: pd.DataFrame, meta: dict):
    """
    Build per-match pre-game features in chronological order and targets:
      y_home = Home Score
      y_away = Away Score
    Elo used here is computed dynamically as "pre" per match.
    """
    feature_cols = meta["feature_cols"]
    tier1, tier2, tier3, tier_strength = build_tiers_and_strength(meta)

    work = df_all.sort_values(["Round"]).reset_index(drop=True)

    teams = pd.unique(work[["Home Team","Away Team"]].values.ravel("K"))
    hist = {t: {"matches": 0, "gs": 0, "gc": 0, "wins": 0} for t in teams}
    elo = {t: float(ELO_START) for t in teams}

    X_rows = []
    y_home = []
    y_away = []

    for _, r in work.iterrows():
        rd = int(r["Round"])
        h = r["Home Team"]
        a = r["Away Team"]

        hs = get_strength(h, tier1, tier2, tier3, tier_strength)
        aas = get_strength(a, tier1, tier2, tier3, tier_strength)

        row = {
            "Round": rd,
            "home_strength": hs,
            "away_strength": aas,
            "strength_diff": hs - aas,
            "home_avg_scored_pre": (hist[h]["gs"] / hist[h]["matches"]) if hist[h]["matches"] > 0 else 0.0,
            "home_avg_conceded_pre": (hist[h]["gc"] / hist[h]["matches"]) if hist[h]["matches"] > 0 else 0.0,
            "home_win_rate_pre": (hist[h]["wins"] / hist[h]["matches"]) if hist[h]["matches"] > 0 else 0.0,
            "away_avg_scored_pre": (hist[a]["gs"] / hist[a]["matches"]) if hist[a]["matches"] > 0 else 0.0,
            "away_avg_conceded_pre": (hist[a]["gc"] / hist[a]["matches"]) if hist[a]["matches"] > 0 else 0.0,
            "away_win_rate_pre": (hist[a]["wins"] / hist[a]["matches"]) if hist[a]["matches"] > 0 else 0.0,
            "home_advantage": 1.0,
        }

        # Elo pre
        if any(c in feature_cols for c in ["home_elo_pre","away_elo_pre","elo_diff"]):
            h_elo = float(elo.get(h, ELO_START))
            a_elo = float(elo.get(a, ELO_START))
            row["home_elo_pre"] = h_elo
            row["away_elo_pre"] = a_elo
            row["elo_diff"] = h_elo - a_elo

        feats = pd.DataFrame([row])
        for c in feature_cols:
            if c not in feats.columns:
                feats[c] = 0.0
        feats = feats[feature_cols]

        X_rows.append(feats.iloc[0].values.astype(float))
        y_home.append(int(r["Home Score"]))
        y_away.append(int(r["Away Score"]))

        # After recording features, update histories + Elo using actual result
        hs_sc = int(r["Home Score"])
        aw_sc = int(r["Away Score"])
        rc = int(r["Result_Code"])

        # history update
        hist[h]["matches"] += 1
        hist[a]["matches"] += 1
        hist[h]["gs"] += hs_sc
        hist[h]["gc"] += aw_sc
        hist[a]["gs"] += aw_sc
        hist[a]["gc"] += hs_sc
        if rc == 1:
            hist[h]["wins"] += 1
        elif rc == -1:
            hist[a]["wins"] += 1

        # Elo update
        Rh = float(elo.get(h, ELO_START))
        Ra = float(elo.get(a, ELO_START))
        exp_h = elo_expected(Rh, Ra, hfa=ELO_HFA)
        exp_a = 1.0 - exp_h
        Sh, Sa = elo_actual_from_rc(rc)
        elo[h] = Rh + ELO_K * (Sh - exp_h)
        elo[a] = Ra + ELO_K * (Sa - exp_a)

    X = np.vstack(X_rows) if len(X_rows) else np.zeros((0, len(meta["feature_cols"])))
    return X, np.array(y_home), np.array(y_away)


def train_and_save_models(df_all: pd.DataFrame, meta: dict):
    X, y_h, y_a = build_training_matrix(df_all, meta)

    # Default training params (can be overridden from meta.json)
    default_params = {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "eta": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": 42,
    }
    home_params = meta.get("home_xgb_params", default_params)
    away_params = meta.get("away_xgb_params", default_params)
    num_boost_round = int(meta.get("num_boost_round", 300))

    dtrain_h = xgb.DMatrix(X, label=y_h)
    dtrain_a = xgb.DMatrix(X, label=y_a)

    home_model = xgb.train(home_params, dtrain_h, num_boost_round=num_boost_round)
    away_model = xgb.train(away_params, dtrain_a, num_boost_round=num_boost_round)

    home_model.save_model(HOME_MODEL_PATH)
    away_model.save_model(AWAY_MODEL_PATH)

    return home_model, away_model


meta = load_meta()
df = load_results_excel()

normalize_map = meta.get("normalize_map", {})
for c in ["Home Team", "Away Team"]:
    df[c] = df[c].map(lambda x: normalize_name(x, normalize_map))

home_model, away_model = load_models()
elo_latest_df = pd.read_csv(ELO_LATEST_PATH)
elo_dict_current = dict(zip(elo_latest_df.Team, elo_latest_df.Elo))


# =========================
# HEADER INFO
# =========================
max_round_loaded = int(df["Round"].max())
next_round_default = max_round_loaded + 1

st.caption(
    f"✅ This model is trained on results up to **Round {max_round_loaded}**. "
    f"Next predicted round default: **{next_round_default}**."
)

# Optional: show Elo table
with st.expander("Elo Table"):
    if not os.path.exists(ELO_LATEST_PATH):
        st.write("No `assets/elo_latest.csv` found.")
    else:
        try:
            st.dataframe(pd.read_csv(ELO_LATEST_PATH), use_container_width=True)
        except Exception:
            st.write("Couldn't read `assets/elo_latest.csv`.")


# =========================
# MAIN UI (with SWAP)
# =========================
teams = sorted(pd.unique(df[["Home Team", "Away Team"]].values.ravel("K")).tolist())

if "home_team" not in st.session_state:
    st.session_state.home_team = teams[0] if len(teams) else ""
if "away_team" not in st.session_state:
    st.session_state.away_team = teams[1] if len(teams) > 1 else (teams[0] if len(teams) else "")
if "predict_round" not in st.session_state:
    st.session_state.predict_round = int(next_round_default)

if len(teams):
    if st.session_state.home_team not in teams:
        st.session_state.home_team = teams[0]
    if st.session_state.away_team not in teams:
        st.session_state.away_team = teams[1] if len(teams) > 1 else teams[0]

def do_swap():
    st.session_state.home_team, st.session_state.away_team = (
        st.session_state.away_team,
        st.session_state.home_team,
    )

c1, c2, c3 = st.columns([5, 1, 5])

with c1:
    st.selectbox("Home Team", teams, key="home_team")

with c2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.button("🔁", on_click=do_swap, help="Swap Home/Away", use_container_width=True)

with c3:
    st.selectbox("Away Team", teams, key="away_team")

st.number_input("Round to predict", min_value=1, step=1, key="predict_round")

submitted = st.button("Predict", type="primary")

home_team = st.session_state.home_team
away_team = st.session_state.away_team
predict_round = int(st.session_state.predict_round)

if home_team == away_team:
    st.warning("Home Team and Away Team are the same. Please select two different teams.")

if submitted and home_team != away_team:
    # Elo dict to use for prediction should be current latest Elo
    elo_dict = {k: float(v) for k, v in elo_dict_current.items()}

    feats, X = build_feature_row(home_team, away_team, predict_round, df, meta, elo_dict)

    dmat = xgb.DMatrix(X)
    pred_home_decimal = float(np.clip(home_model.predict(dmat)[0], 0, None))
    pred_away_decimal = float(np.clip(away_model.predict(dmat)[0], 0, None))

    pred_home_int = int(np.clip(np.rint(pred_home_decimal), 0, None))
    pred_away_int = int(np.clip(np.rint(pred_away_decimal), 0, None))

    outcome_from_rounded = (
        "Tie" if pred_home_int == pred_away_int else ("Home Win" if pred_home_int > pred_away_int else "Away Win")
    )

    st.subheader("Prediction")
    st.markdown(f"### **{home_team} {pred_home_int} - {pred_away_int} {away_team}**")

    if outcome_from_rounded == "Tie":
        st.info("🤝 Rounded score is a **Tie**.")
    else:
        st.success(f"✅ Rounded outcome: **{outcome_from_rounded}**")

    details = {
        "pred_home_decimal": pred_home_decimal,
        "pred_away_decimal": pred_away_decimal,
        "outcome_from_rounded_score": outcome_from_rounded,
        "df_max_round_loaded": max_round_loaded,
        "elo_strength_home_used": float(feats.loc[0, "home_elo_pre"]) if "home_elo_pre" in feats.columns else None,
        "elo_strength_away_used": float(feats.loc[0, "away_elo_pre"]) if "away_elo_pre" in feats.columns else None,
    }

    with st.expander("Details"):
        st.json(details)
        st.caption("Feature row used by Streamlit:")
        st.dataframe(feats, use_container_width=True)


# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div style="margin-top: 30px; text-align: center; font-size: 12px; opacity: 0.7;">
        Created by <b>Ziyad Albaadi</b>, with a little help from ChatGPT 👀
    </div>
    """,
    unsafe_allow_html=True,
)
