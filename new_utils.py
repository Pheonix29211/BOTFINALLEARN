# new_utils.py
import os
import time
import sqlite3
import threading
import joblib
import math
import requests
import numpy as np
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta

# Constants / cache paths
DB_PATH = "trade_logs.db"
MODEL_PATH = "trade_model.pkl"
HIST_CACHE = "historical_ohlcv.pkl"

# TP/SL hardcoded for simulation consistency; can be overridden externally if needed
TP_POINTS = 600
SL_POINTS = 200

# Thread safety
_db_lock = threading.Lock()

# Ensure database and schema
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    with _db_lock:
        conn = get_conn()
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            exit_time TEXT,
            result TEXT,
            rsi REAL,
            wick_percent REAL,
            liquidation_usd REAL,
            score REAL,
            win_prob REAL,
            tp_pct REAL,
            sl_pct REAL,
            reward_to_risk REAL,
            volume_z REAL,
            liq_ratio REAL,
            liquidation_source TEXT
        )''')
        conn.commit()
        conn.close()

init_db()

def store_trade(trade: dict):
    """Thread-safe insert of a trade dict into DB."""
    with _db_lock:
        conn = get_conn()
        c = conn.cursor()
        c.execute(
            '''INSERT INTO trades (
                time,direction,entry_price,exit_price,exit_time,result,
                rsi,wick_percent,liquidation_usd,score,win_prob,
                tp_pct,sl_pct,reward_to_risk,volume_z,liq_ratio,liquidation_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (
                trade.get("time"),
                trade.get("direction"),
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("exit_time"),
                trade.get("result"),
                trade.get("rsi"),
                trade.get("wick_percent"),
                trade.get("liquidation_usd"),
                trade.get("score"),
                trade.get("win_prob"),
                trade.get("tp_pct"),
                trade.get("sl_pct"),
                trade.get("reward_to_risk"),
                trade.get("volume_z"),
                trade.get("liq_ratio"),
                trade.get("liquidation_source"),
            ),
        )
        conn.commit()
        conn.close()

def get_last_trades(n=10):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT time,direction,entry_price,exit_price,result,win_prob FROM trades ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "No trades stored."
    lines = []
    for t, direction, entry, exit_p, result, win_prob in rows:
        lines.append(f"{t} | {direction.upper()} at {entry} | Exit: {exit_p} | {result} | WinProb: {win_prob}")
    return "Last trades:\n" + "\n".join(lines)

# Feature computations
def compute_rsi(prices, period=14):
    prices = np.array(prices, dtype=float)
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    ups = np.where(deltas > 0, deltas, 0)
    downs = np.where(deltas < 0, -deltas, 0)
    roll_up = np.convolve(ups, np.ones(period) / period, mode='valid')[-1]
    roll_down = np.convolve(downs, np.ones(period) / period, mode='valid')[-1]
    if roll_down == 0:
        return 100.0
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def compute_atr(candles, period=14):
    # candles: list of dicts with 'high','low','close','open'
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    atr = np.mean(trs[-period:])
    return round(atr, 4)

# Data fetching
def fetch_mexc_ohlcv(limit=50):
    """
    Fetch latest 5m OHLCV from MEXC. Fallback to Binance if failure.
    MEXC contract endpoint example; adjust symbol as needed.
    """
    try:
        # This endpoint pattern may vary; adjust symbol syntax if needed.
        params = {
            "symbol": "BTC_USDT",
            "interval": "5m",
            "limit": limit
        }
        resp = requests.get("https://contract.mexc.com/api/v1/contract/kline", params=params, timeout=10)
        j = resp.json()
        data = j.get("data") or []
        candles = []
        for entry in data:
            # Format may be [timestamp, open, high, low, close, volume, ...]
            ts = entry[0]
            open_p = float(entry[1])
            high = float(entry[2])
            low = float(entry[3])
            close_p = float(entry[4])
            volume = float(entry[5])
            candles.append({
                "ts": ts,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close_p,
                "volume": volume
            })
        if candles:
            return candles
    except Exception:
        pass

    # Fallback to Binance
    try:
        resp = requests.get("https://api.binance.com/api/v3/klines",
                            params={"symbol": "BTCUSDT", "interval": "5m", "limit": limit}, timeout=10)
        data = resp.json()
        candles = []
        for entry in data:
            open_p = float(entry[1])
            high = float(entry[2])
            low = float(entry[3])
            close_p = float(entry[4])
            volume = float(entry[5])
            ts = entry[0]
            candles.append({
                "ts": ts,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close_p,
                "volume": volume
            })
        return candles
    except Exception:
        return []

def fetch_mexc_ohlcv_range(start, end, interval_minutes=5):
    """Fetch historical range; will chunk if needed. start/end are epoch seconds."""
    # MEXC might require milliseconds
    try:
        params = {
            "symbol": "BTC_USDT",
            "interval": f"{interval_minutes}m",
            # some APIs expect startTime/endTime in ms
            "startTime": int(start * 1000),
            "endTime": int(end * 1000),
            "limit": 1000
        }
        resp = requests.get("https://contract.mexc.com/api/v1/contract/kline", params=params, timeout=10)
        j = resp.json()
        data = j.get("data") or []
        candles = []
        for entry in data:
            ts = entry[0]
            open_p = float(entry[1])
            high = float(entry[2])
            low = float(entry[3])
            close_p = float(entry[4])
            volume = float(entry[5])
            candles.append({
                "ts": ts,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close_p,
                "volume": volume
            })
        return candles
    except Exception:
        # fallback: empty
        return []

def load_cached_history():
    if os.path.exists(HIST_CACHE):
        try:
            return joblib.load(HIST_CACHE)
        except Exception:
            pass
    # On miss, backfill one year quickly (simple fallback)
    now = int(time.time())
    one_year_ago = now - 365 * 24 * 3600
    bucket = fetch_mexc_ohlcv_range(start=one_year_ago, end=now)
    if bucket:
        joblib.dump(bucket, HIST_CACHE)
        return bucket
    return []

def infer_liquidation_pressure_from_mexc():
    """Infer liquidation pressure from available MEXC contract ticker / open interest."""
    try:
        resp = requests.get("https://contract.mexc.com/api/v1/contract/ticker", timeout=5)
        j = resp.json()
        data = j.get("data", {})
        # Example inference: use turnover or open interest if available
        liquidation = float(data.get("volume", 0))
        return liquidation, "mexc_inferred"
    except Exception:
        return 0.0, "fallback"

# Simple scoring heuristics
def score_long(rsi, lower_wick_pct, liquidation):
    s = 0.0
    if rsi is not None:
        s += max(0, (50 - rsi) / 50)  # oversold component
    s += min(lower_wick_pct / 100, 1.0)  # wick strength
    s += min(liquidation / 1_000_000, 1.0)  # liquidity pressure bonus
    return round(s, 2)

def score_short(rsi, upper_wick_pct, liquidation):
    s = 0.0
    if rsi is not None:
        s += max(0, (rsi - 50) / 50)  # overbought
    s += min(upper_wick_pct / 100, 1.0)
    s += min(liquidation / 1_000_000, 1.0)
    return round(s, 2)

# Model training & prediction
def train_model_incremental():
    """Re-train model from closed trades in DB."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT direction, rsi, wick_percent, liquidation_usd, result FROM trades WHERE result IS NOT NULL AND result != 'open'")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "No closed trades to train on."

    X = []
    y = []
    for direction, rsi, wick, liq, result in rows:
        features = [
            rsi or 50,
            wick or 0,
            liq or 0,
            1 if direction == "long" else 0,
        ]
        X.append(features)
        # Simple label: TP HIT considered win, otherwise loss
        label = 1 if "TP" in (result or "").upper() else 0
        y.append(label)
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
    clf.fit(Xs, y)

    # Save combined pipeline: scaler + classifier
    joblib.dump({"scaler": scaler, "clf": clf}, MODEL_PATH)
    return f"Trained on {len(y)} samples."

def predict_win_prob(signal):
    if not os.path.exists(MODEL_PATH):
        return 0.5  # no model yet
    try:
        obj = joblib.load(MODEL_PATH)
        scaler = obj["scaler"]
        clf = obj["clf"]
        direction = signal.get("direction")
        rsi = signal.get("rsi") or 50
        wick = signal.get("wick_percent") or 0
        liq = signal.get("liquidation_usd") or 0
        features = np.array([[rsi, wick, liq, 1 if direction == "long" else 0]])
        Xs = scaler.transform(features)
        proba = clf.predict_proba(Xs)[0][1]
        return round(float(proba), 2)
    except Exception:
        return 0.5

# Empirical strength: find k-nearest historical closed trades
def fetch_closed_trades():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT direction, rsi, wick_percent, liquidation_usd, result FROM trades WHERE result IS NOT NULL AND result != 'open'")
    rows = c.fetchall()
    conn.close()
    trades = []
    for direction, rsi, wick, liq, result in rows:
        label = 1 if "TP" in (result or "").upper() else 0
        trades.append({
            "direction": direction,
            "rsi": rsi or 50,
            "wick_percent": wick or 0,
            "liquidation_usd": liq or 0,
            "label": label
        })
    return trades

def get_setup_strength(signal, k=50):
    """Returns empirical win rate and confidence interval from nearest neighbors."""
    base_features = np.array([
        signal.get("rsi") or 50,
        signal.get("wick_percent") or 0,
        signal.get("liquidation_usd") or 0,
        1 if signal.get("direction") == "long" else 0,
    ], dtype=float)

    trades = fetch_closed_trades()
    if not trades:
        return {"empirical_win_rate": None, "count": 0}

    # Compute Euclidean distance with feature scaling
    X = []
    labels = []
    for t in trades:
        f = np.array([t["rsi"], t["wick_percent"], t["liquidation_usd"], 1 if t["direction"] == "long" else 0], dtype=float)
        X.append(f)
        labels.append(t["label"])
    X = np.array(X)
    labels = np.array(labels)

    # Simple normalization per feature (to avoid huge liquidation dominating)
    # scale rsi [0,100], wick_percent [0,100], liquidation_usd (divide by 1e6), direction as binary
    X_norm = np.column_stack([
        X[:, 0] / 100.0,
        X[:, 1] / 100.0,
        X[:, 2] / 1_000_000.0,
        X[:, 3]
    ])
    base_norm = np.array([
        base_features[0] / 100.0,
        base_features[1] / 100.0,
        base_features[2] / 1_000_000.0,
        base_features[3]
    ])

    dists = np.linalg.norm(X_norm - base_norm, axis=1)
    idx = np.argsort(dists)[:k]
    nearest_labels = labels[idx]
    count = len(nearest_labels)
    wins = int(nearest_labels.sum())

    # Bayesian smoothing (Beta prior) to avoid 0/1 extremes
    alpha = 1
    beta_param = 1
    empirical_rate = (wins + alpha) / (count + alpha + beta_param)
    # Confidence interval (e.g., 90%)
    ci_lower, ci_upper = beta.ppf([0.05, 0.95], wins + alpha, count - wins + beta_param)
    return {
        "empirical_win_rate": round(empirical_rate, 2),
        "count": count,
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
    }

# Simulation over history
def simulate_history(candles):
    """Simulate signals over a list of candles with TP/SL logic."""
    results = []
    for i in range(15, len(candles) - 10):  # leave some lookahead
        window = candles[i - 15 : i + 1]
        closes = [c["close"] for c in window]
        rsi = compute_rsi(closes) or 50
        last = candles[i]
        open_p = last["open"]
        close_p = last["close"]
        high = last["high"]
        low = last["low"]
        total_range = high - low if high - low != 0 else 1
        lower_wick_pct = (min(open_p, close_p) - low) / total_range * 100
        upper_wick_pct = (high - max(open_p, close_p)) / total_range * 100

        liquidation, source = infer_liquidation_pressure_from_mexc()
        long_score = score_long(rsi, lower_wick_pct, liquidation)
        short_score = score_short(rsi, upper_wick_pct, liquidation)

        direction = "long" if long_score >= short_score else "short"
        entry_price = close_p
        tp_price = entry_price + TP_POINTS if direction == "long" else entry_price - TP_POINTS
        sl_price = entry_price - SL_POINTS if direction == "long" else entry_price + SL_POINTS

        # Look ahead up to 20 candles for TP/SL
        result_label = "NO SIGNAL"
        exit_price = None
        for future in candles[i+1 : i+21]:
            if direction == "long":
                if future["high"] >= tp_price:
                    result_label = "TP HIT"
                    exit_price = tp_price
                    break
                if future["low"] <= sl_price:
                    result_label = "SL HIT"
                    exit_price = sl_price
                    break
            else:
                if future["low"] <= tp_price:
                    result_label = "TP HIT"
                    exit_price = tp_price
                    break
                if future["high"] >= sl_price:
                    result_label = "SL HIT"
                    exit_price = sl_price
                    break
        if result_label == "NO SIGNAL":
            continue

        trade = {
            "time": datetime.utcfromtimestamp(last["ts"]/1000).strftime("%Y-%m-%d %H:%M:%S"),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": round(exit_price, 2),
            "exit_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result_label,
            "rsi": rsi,
            "wick_percent": round(lower_wick_pct if direction == "long" else upper_wick_pct, 2),
            "liquidation_usd": liquidation,
            "score": long_score if direction == "long" else short_score,
            "win_prob": 1.0,
            "tp_pct": (TP_POINTS / entry_price) * 100,
            "sl_pct": (SL_POINTS / entry_price) * 100,
            "reward_to_risk": TP_POINTS / SL_POINTS,
            "volume_z": 1.0,
            "liq_ratio": 1.0,
            "liquidation_source": source,
        }
        # Label for get_setup_strength uses label=1 for TP HIT
        trade["label"] = 1 if result_label == "TP HIT" else 0
        results.append(trade)
    return results

# Full-year backfill + simulation with progress
def backfill_one_year_with_progress(progress_callback=None):
    now = int(time.time())
    one_year_ago = now - 365 * 24 * 3600
    chunk_seconds = 300 * 300  # ~25h per chunk
    total_chunks = ((now - one_year_ago) + chunk_seconds - 1) // chunk_seconds
    all_candles = []
    cursor = one_year_ago
    chunk_index = 0
    while cursor < now:
        end = min(cursor + chunk_seconds, now)
        chunk = fetch_mexc_ohlcv_range(start=cursor, end=end)
        if chunk:
            all_candles.extend(chunk)
        cursor = end
        chunk_index += 1
        if progress_callback:
            pct = int(chunk_index / total_chunks * 100)
            progress_callback(f"🟦 Backfill progress: {pct}% ({chunk_index}/{total_chunks} chunks)")
        time.sleep(0.2)
    unique = {c["ts"]: c for c in all_candles if "ts" in c}
    sorted_candles = sorted(unique.values(), key=lambda x: x["ts"])
    joblib.dump(sorted_candles, HIST_CACHE)
    if progress_callback:
        progress_callback(f"✅ Backfill complete, cached {len(sorted_candles)} candles.")
    return sorted_candles

def simulate_and_store_full_history_with_progress(progress_callback=None):
    if progress_callback:
        progress_callback("🔁 Starting full-year backfill + simulation...")
    candles = backfill_one_year_with_progress(progress_callback=progress_callback)
    if not candles:
        return "Failed to load historical candles."
    if progress_callback:
        progress_callback("🧠 Simulating signals over history...")
    results = simulate_history(candles)
    if not results:
        return "No signals simulated during full-year backfill."
    wins = sum(1 for r in results if r.get("label") == 1)
    total = len(results)
    if progress_callback:
        progress_callback(f"📊 Simulation complete: {wins}/{total} wins. Storing trades...")
    for i, r in enumerate(results, start=1):
        store_trade(r)
        if progress_callback and i % max(1, total // 10) == 0:
            pct = int(i / total * 100)
            progress_callback(f"💾 Storing trades: {pct}% ({i}/{total})")
    if progress_callback:
        progress_callback("📈 Training model on stored history...")
    train_msg = train_model_incremental()
    if progress_callback:
        progress_callback(f"✅ Training complete: {train_msg}")
    return f"Full-year simulation stored {total} trades, wins: {wins}/{total} ({wins/total:.1%}). {train_msg}"
