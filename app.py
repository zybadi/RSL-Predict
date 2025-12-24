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
ELO_LATEST_PATH = os.path.join(ASSETS_DIR, "elo_latest.csv")  # optional


st.set_page_config(page_title="SPL Score Predictor", layout="centered")
st.title("⚽ SPL Score Predictor")


# =========================
# LOADERS (cached)
# =========================
@st.cache_resource
def load_models_and_meta():
    for p in (HOME_MODEL_PATH, AWAY_MODEL_PATH, META_PATH):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    # Use Booster since you're loading .json models
    home_model = xgb.Booster()
    away_model = xgb.Booster()
    home_model.load_model(HOME_MODEL_PATH)
    away_model.load_model(AWAY_MODEL_PATH)

    return home_model, away_model, meta


@st.cache_data
def load_results_excel():
    if not os.path.exists(RESULTS_XLSX):
        raise FileNotFoundError(f"Missing file: {RESULTS_XLSX}")

    df = pd.read_excel(RESULTS_XLSX)

    required = {"Round", "Home Team", "Away Team", "Home Score", "Away Score", "Result_Code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"spl_results.xlsx is missing columns: {sorted(list(missing))}")

    return df


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

        hist[h]["gs"] += r["Home Score"]
        hist[h]["gc"] += r["Away Score"]
        hist[a]["gs"] += r["Away Score"]
        hist[a]["gc"] += r["Home Score"]

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


@st.cache_data
def load_elo_latest_dict(normalize_map: dict):
    """
    Load latest Elo ratings from assets/elo_latest.csv and return team->elo dict.
    This is the Elo source of truth in Streamlit.
    """
    if not os.path.exists(ELO_LATEST_PATH):
        return {}

    try:
        df_elo = pd.read_csv(ELO_LATEST_PATH)
    except Exception:
        return {}

    # Try to detect columns
    team_col = None
    elo_col = None
    for c in df_elo.columns:
        if c.lower() in ["team", "club", "name"]:
            team_col = c
        if c.lower() in ["elo", "rating"]:
            elo_col = c

    if team_col is None and "Team" in df_elo.columns:
        team_col = "Team"
    if elo_col is None and "Elo" in df_elo.columns:
        elo_col = "Elo"

    if team_col is None or elo_col is None:
        return {}

    elo_dict = {}
    for _, r in df_elo.iterrows():
        t = normalize_name(r[team_col], normalize_map)
        try:
            elo_dict[t] = float(r[elo_col])
        except Exception:
            continue

    return elo_dict


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

    # Ensure all expected columns exist
    for c in feature_cols:
        if c not in feats.columns:
            feats[c] = 0.0

    feats = feats[feature_cols]
    X = feats.values
    return feats, X


# =========================
# LOAD EVERYTHING
# =========================
try:
    home_model, away_model, meta = load_models_and_meta()
    df = load_results_excel()
except Exception as e:
    st.error(str(e))
    st.stop()

normalize_map = meta.get("normalize_map", {})
for c in ["Home Team", "Away Team"]:
    df[c] = df[c].map(lambda x: normalize_name(x, normalize_map))

elo_dict = load_elo_latest_dict(normalize_map)

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
# MAIN UI (with SWAP) ✅ works even before Predict
# =========================
teams = sorted(pd.unique(df[["Home Team", "Away Team"]].values.ravel("K")).tolist())

# ---- Session-state initialization (prevents warnings) ----
if "home_team" not in st.session_state:
    st.session_state.home_team = teams[0] if len(teams) else ""
if "away_team" not in st.session_state:
    st.session_state.away_team = teams[1] if len(teams) > 1 else (teams[0] if len(teams) else "")
if "predict_round" not in st.session_state:
    st.session_state.predict_round = int(next_round_default)

# Ensure stored values still exist in options (safety for future updates)
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
    # No st.rerun(): Streamlit reruns automatically after a click.

# ---- Controls row: Home | 🔁 | Away ----
c1, c2, c3 = st.columns([5, 1, 5])

with c1:
    st.selectbox("Home Team", teams, key="home_team")

with c2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.button("🔁", on_click=do_swap, help="Swap Home/Away", use_container_width=True)


with c3:
    st.selectbox("Away Team", teams, key="away_team")

# Round input (outside form so it updates immediately too)
st.number_input("Round to predict", min_value=1, step=1, key="predict_round")

# Predict button (you can keep as normal button; form not needed)
submitted = st.button("Predict", type="primary")

home_team = st.session_state.home_team
away_team = st.session_state.away_team
predict_round = int(st.session_state.predict_round)

if home_team == away_team:
    st.warning("Home Team and Away Team are the same. Please select two different teams.")

if submitted and home_team != away_team:
    feats, X = build_feature_row(home_team, away_team, predict_round, df, meta, elo_dict)

    dmat = xgb.DMatrix(X)
    pred_home_decimal = float(np.clip(home_model.predict(dmat)[0], 0, None))
    pred_away_decimal = float(np.clip(away_model.predict(dmat)[0], 0, None))

    pred_home_int = int(np.clip(np.rint(pred_home_decimal), 0, None))
    pred_away_int = int(np.clip(np.rint(pred_away_decimal), 0, None))

    outcome_from_rounded = (
        "Tie" if pred_home_int == pred_away_int else ("Home Win" if pred_home_int > pred_away_int else "Away Win")
    )
    outcome_decimal = (
        "Tie" if pred_home_decimal == pred_away_decimal else ("Home Win" if pred_home_decimal > pred_away_decimal else "Away Win")
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
        "outcome_decimal_model": outcome_decimal,
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
