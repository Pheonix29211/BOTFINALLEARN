import os
import time
import sqlite3
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import beta

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Configuration
DB_FILE = "trade_logs.db"
MODEL_FILE = "trade_model.pkl"
HIST_CACHE = "historical_ohlcv.pkl"
MEXC_BASE = "https://contract.mexc.com/api/v1/contract"
SYMBOL = "BTC_USDT"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("new_utils")

# In-memory EMA states for adaptive (still kept but not used for fixed TP/SL)
VOL_EMA = None
LIQ_EMA = None

# === Database helpers ===
def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            direction TEXT,
            entry_price REAL,
            result TEXT,
            exit_price REAL,
            exit_time TEXT,
            rsi REAL,
            wick_percent REAL,
            liquidation_usd REAL,
            score REAL,
            tp_pct REAL,
            sl_pct REAL,
            win_prob REAL,
            reward_to_risk REAL,
            volume_z REAL,
            liq_ratio REAL,
            news_sentiment REAL
        )"""
    )
    conn.commit()
    return conn

def store_trade(trade):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO trades (
            time, direction, entry_price, result, exit_price, exit_time,
            rsi, wick_percent, liquidation_usd, score, tp_pct, sl_pct,
            win_prob, reward_to_risk, volume_z, liq_ratio, news_sentiment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            trade.get("time"),
            trade.get("direction"),
            trade.get("entry_price"),
            trade.get("result", "open"),
            trade.get("exit_price"),
            trade.get("exit_time"),
            trade.get("rsi"),
            trade.get("wick_percent"),
            trade.get("liquidation_usd"),
            trade.get("score"),
            trade.get("tp_pct"),
            trade.get("sl_pct"),
            trade.get("win_prob"),
            trade.get("reward_to_risk"),
            trade.get("volume_z"),
            trade.get("liq_ratio"),
            trade.get("news_sentiment"),
        ),
    )
    conn.commit()
    conn.close()

def get_last_trades(n=30):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trades ORDER BY id DESC LIMIT ?", conn, params=(n,))
    conn.close()
    if df.empty:
        return "No trades yet."
    lines = []
    for _, r in df.iterrows():
        lines.append(f'{r["time"]} | {r["direction"].upper()} @ {r["entry_price"]} | Result: {r["result"]} | Score: {r["score"]} | WinProb: {r["win_prob"]} | TP%: {r["tp_pct"]} | SL%: {r["sl_pct"]}')
    return "\n".join(lines)

# === Historical data backfill ===
def fetch_mexc_ohlcv(symbol=SYMBOL, interval="Min5", limit=100):
    try:
        r = requests.get(f"{MEXC_BASE}/kline/{symbol}", params={"interval": interval, "limit": limit}, timeout=10)
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success", False):
            logger.warning("OHLCV fetch failed: %s", resp)
            return []
        data = resp.get("data", {})
        times = data.get("time", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        candles = []
        for i in range(len(times)):
            candles.append({
                "open_time": times[i] * 1000,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]) if i < len(volumes) else None,
            })
        return candles
    except Exception as e:
        logger.warning("fetch_mexc_ohlcv error: %s", e)
        return []

def fetch_mexc_ohlcv_range(symbol=SYMBOL, interval="Min5", start=None, end=None):
    try:
        params = {"interval": interval}
        if start is not None:
            params["start"] = int(start)
        if end is not None:
            params["end"] = int(end)
        r = requests.get(f"{MEXC_BASE}/kline/{symbol}", params=params, timeout=10)
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success", False):
            logger.warning("range fetch failed: %s", resp)
            return []
        data = resp.get("data", {})
        times = data.get("time", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        volumes = data.get("volume", [])
        candles = []
        for i in range(len(times)):
            candles.append({
                "ts": times[i] * 1000,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]) if i < len(volumes) else None,
            })
        return candles
    except Exception as e:
        logger.warning("fetch range error: %s", e)
        return []

def backfill_one_year():
    now = int(time.time())
    one_year_ago = now - 365 * 24 * 3600
    all_candles = []
    cursor = one_year_ago
    chunk_seconds = 300 * 300
    while cursor < now:
        end = min(cursor + chunk_seconds, now)
        chunk = fetch_mexc_ohlcv_range(start=cursor, end=end)
        if chunk:
            all_candles.extend(chunk)
        time.sleep(0.2)
        cursor = end
    unique = {c["ts"]: c for c in all_candles if "ts" in c}
    sorted_candles = sorted(unique.values(), key=lambda x: x["ts"])
    joblib.dump(sorted_candles, HIST_CACHE)
    return sorted_candles

def load_cached_history():
    if os.path.exists(HIST_CACHE):
        try:
            return joblib.load(HIST_CACHE)
        except Exception as e:
            logger.warning("load cache error: %s", e)
    return backfill_one_year()

# === Technical indicators ===
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def compute_atr(candles, period=14):
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, period + 1):
        high = candles[i].get("high", 0)
        low = candles[i].get("low", 0)
        prev = candles[i - 1].get("close", 0)
        tr = max(high - low, abs(high - prev), abs(low - prev))
        trs.append(tr)
    return sum(trs) / period

# === Liquidation inference ===
def infer_liquidation_pressure_from_mexc():
    try:
        r = requests.get(f"{MEXC_BASE}/ticker", params={"symbol": SYMBOL}, timeout=5)
        r.raise_for_status()
        data = r.json().get("data", {})
        open_interest = float(data.get("holdVol", 0))
        funding_rate = float(data.get("fundingRate", 0))
        if open_interest <= 0:
            return 0.0, "no_oi"
        pressure = (open_interest / 1e9) * (1 + abs(funding_rate) * 10)
        return pressure * 1_000_000, "mexc_inferred"
    except Exception as e:
        logger.warning("liquidation fetch error: %s", e)
        return 0.0, "error"

# === Scoring ===
def score_long(rsi, lower_wick_pct, liquidation_usd, funding_rate=1.0):
    score = 0.0
    if rsi is not None and rsi < 35:
        score += (35 - rsi) * 0.05
    score += min(lower_wick_pct, 5) * 0.2
    score += min(liquidation_usd / 5_000_000, 2) * 0.5
    score *= funding_rate
    return round(score, 2)

def score_short(rsi, upper_wick_pct, liquidation_usd, funding_rate=1.0):
    score = 0.0
    if rsi is not None and rsi > 65:
        score += (rsi - 65) * 0.03
    score += min(upper_wick_pct, 5) * 0.2
    score += min(liquidation_usd / 5_000_000, 2) * 0.5
    score *= funding_rate
    return round(score, 2)

# === Learning model ===
def _load_labeled_trades_df():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trades WHERE result IN ('TP HIT','SL HIT')", conn)
    conn.close()
    if df.empty:
        return None, None, None
    df["label"] = df["result"].apply(lambda r: 1 if r == "TP HIT" else 0)
    df["is_long"] = (df["direction"] == "long").astype(int)
    df["liq_scaled"] = df["liquidation_usd"] / 1_000_000
    X = df[["rsi", "wick_percent", "score", "liq_scaled", "is_long"]]
    y = df["label"]
    times = pd.to_datetime(df["time"])
    return X, y, times

def _compute_sample_weights(times, half_life_hours=72):
    now = pd.Timestamp.utcnow()
    hours = (now - times).dt.total_seconds() / 3600.0
    return 0.5 ** (hours / half_life_hours)

def train_model_incremental():
    data = _load_labeled_trades_df()
    if data is None:
        return "Not enough labeled closed trades yet."
    X, y, times = data
    if len(y) < 30:
        return f"Need at least 30 closed trades; have {len(y)}."
    weights = _compute_sample_weights(times)
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = make_pipeline(StandardScaler(), SGDClassifier(loss="log", max_iter=1000, tol=1e-3))
    try:
        model.named_steps["sgdclassifier"].partial_fit(X, y, classes=[0,1], sample_weight=weights)
    except Exception:
        model.fit(X, y, sample_weight=weights)
    joblib.dump(model, MODEL_FILE)
    return f"Incrementally trained model on {len(y)} samples."

def predict_win_prob(signal):
    if not os.path.exists(MODEL_FILE):
        return 1.0
    try:
        model = joblib.load(MODEL_FILE)
        is_long = 1 if signal.get("direction") == "long" else 0
        liq_scaled = signal.get("liquidation_usd", 0) / 1_000_000
        feat = [
            signal.get("rsi", 50),
            signal.get("wick_percent", 0),
            signal.get("score", 0),
            liq_scaled,
            is_long,
        ]
        X = pd.DataFrame([feat], columns=["rsi","wick_percent","score","liq_scaled","is_long"])
        prob = model.predict_proba(X)[0][1]
        return prob
    except Exception as e:
        logger.warning("predict error: %s", e)
        return 1.0

# === Simulation / backtest ===
def simulate_history(candles, lookahead_bars=12, tp_pct=0.015, sl_pct=0.01):
    results = []
    for i in range(15, len(candles) - lookahead_bars):
        window = candles[i - 15 : i + 1]
        closes = [c.get("close") for c in window]
        rsi = compute_rsi(closes) if len(closes) >= 15 else None
        last = window[-1]
        open_p = last.get("open")
        close_p = last.get("close")
        high = last.get("high")
        low = last.get("low")
        total_range = high - low if high - low != 0 else 1
        lower_wick = (min(open_p, close_p) - low) / total_range * 100
        upper_wick = (high - max(open_p, close_p)) / total_range * 100
        liquidation, source = infer_liquidation_pressure_from_mexc()
        long_score = score_long(rsi, lower_wick, liquidation)
        short_score = score_short(rsi, upper_wick, liquidation)
        direction = None
        wick_pct = 0
        score = 0
        if short_score > long_score and short_score >= 1.2:
            direction = "short"
            wick_pct = upper_wick
            score = short_score
        elif long_score >= short_score and long_score >= 1.2:
            direction = "long"
            wick_pct = lower_wick
            score = long_score
        else:
            continue
        entry_price = close_p
        outcome = "SL HIT"
        for future in candles[i+1:i+1+lookahead_bars]:
            if direction == "long":
                tp = entry_price * (1 + tp_pct)
                sl = entry_price * (1 - sl_pct)
                if future.get("high",0) >= tp:
                    outcome = "TP HIT"
                    break
                if future.get("low",0) <= sl:
                    outcome = "SL HIT"
                    break
            else:
                tp = entry_price * (1 - tp_pct)
                sl = entry_price * (1 + sl_pct)
                if future.get("low",0) <= tp:
                    outcome = "TP HIT"
                    break
                if future.get("high",0) >= sl:
                    outcome = "SL HIT"
                    break
        label = 1 if outcome == "TP HIT" else 0
        result = {
            "time": datetime.utcfromtimestamp(candles[i].get("ts",0)/1000).strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "entry_price": entry_price,
            "rsi": rsi,
            "wick_percent": round(wick_pct,2),
            "liquidation_usd": liquidation,
            "score": score,
            "result": outcome,
            "label": label,
        }
        results.append(result)
    return results

# === Similarity & strength estimation ===
def _make_feature_vector(trade):
    is_long = 1 if trade.get("direction") == "long" else 0
    liq_scaled = (trade.get("liquidation_usd", 0) or 0) / 1_000_000
    return np.array([
        trade.get("rsi", 50),
        trade.get("wick_percent", 0),
        trade.get("score", 0),
        liq_scaled,
        is_long,
        trade.get("reward_to_risk", 1.0),
    ], dtype=float)

def get_similar_closed_trades(signal, k=50):
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM trades WHERE result IN ('TP HIT','SL HIT')", conn)
    conn.close()
    if df.empty:
        return []
    past_vectors = []
    records = []
    for _, row in df.iterrows():
        trade_dict = dict(row)
        fv = _make_feature_vector(trade_dict)
        past_vectors.append(fv)
        records.append((trade_dict, fv, row["result"]))
    X = np.vstack(past_vectors)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    X_norm = (X - mu) / sigma
    qv = _make_feature_vector(signal)
    qv_norm = (qv - mu) / sigma
    dists = np.linalg.norm(X_norm - qv_norm, axis=1)
    idxs = np.argsort(dists)[:k]
    similar = []
    for i in idxs:
        trade_dict, fv, result = records[i]
        similar.append({
            "trade": trade_dict,
            "result": result,
            "distance": float(dists[i]),
        })
    return similar

def get_setup_strength(signal, k=50, prior_alpha=1, prior_beta=1):
    similar = get_similar_closed_trades(signal, k=k)
    if not similar:
        return {
            "empirical_win_rate": None,
            "count": 0,
            "ci_lower": None,
            "ci_upper": None,
        }
    wins = sum(1 for s in similar if s["result"] == "TP HIT")
    total = len(similar)
    post_a = prior_alpha + wins
    post_b = prior_beta + (total - wins)
    empirical = post_a / (post_a + post_b)
    ci_lower = beta.ppf(0.05, post_a, post_b)
    ci_upper = beta.ppf(0.95, post_a, post_b)
    return {
        "empirical_win_rate": empirical,
        "count": total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }

# === Full-year simulation & training ===
def simulate_and_store_full_history():
    candles = load_cached_history()
    if not candles:
        return "Failed to load historical candles."
    results = simulate_history(candles)
    if not results:
        return "No signals simulated during full-year backfill."
    wins = sum(1 for r in results if r.get("label") == 1)
    total = len(results)
    for r in results:
        entry = r.get("entry_price")
        direction = r.get("direction")
        if r.get("result") == "TP HIT":
            if direction == "long":
                exit_price = entry * 1.015
            else:
                exit_price = entry * (1 - 0.015)
        else:
            if direction == "long":
                exit_price = entry * (1 - 0.01)
            else:
                exit_price = entry * (1 + 0.01)
        r["exit_price"] = round(exit_price,2)
        r["exit_time"] = r.get("time")
        r["tp_pct"] = 1.5
        r["sl_pct"] = 1.0
        r["win_prob"] = 1.0
        r["reward_to_risk"] = 3.0
        r["volume_z"] = 1.0
        r["liq_ratio"] = 1.0
        r["news_sentiment"] = 0.0
        store_trade(r)
    train_msg = train_model_incremental()
    return f"Full-year simulation stored {total} trades, wins: {wins}/{total} ({wins/total:.1%}). {train_msg}"
