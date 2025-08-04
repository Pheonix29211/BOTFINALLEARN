import os
import json
import requests
import sqlite3
import logging
import time
from datetime import datetime, timedelta
import pytz

# --- Configuration / files ---
DB_FILE = "trade_logs.db"
STRATEGY_FILE = "strategy.json"
PERF_FILE = "performance_stats.json"

# --- API keys ---
COINGLASS_API_KEY = os.getenv("COINGLASS_API_KEY", "").strip()
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()

# --- Logging ---
logger = logging.getLogger("liquidbot_utils")
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# --- Live data tracking ---
_LAST_MEXC_OHLCV_SUCCESS = None
_LAST_MEXC_TICKER_SUCCESS = None
_LAST_COINGLASS_SUCCESS = None
_LAST_LIQ_SOURCE = "none"
_LAST_MEXC_OI = None  # for delta in enhanced inference

# --- Timezones ---
UTC = pytz.UTC
IST = pytz.timezone("Asia/Kolkata")

# --- DB helpers ---
def _get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
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
            tp_points INTEGER,
            sl_points INTEGER,
            liquidation_source TEXT,
            win_prob REAL,
            session TEXT
        )"""
    )
    conn.commit()
    return conn

# --- Performance tracking ---
def _init_perf():
    default = {
        "score_buckets": {
            "low": {"wins": 0, "losses": 0, "total": 0},
            "mid": {"wins": 0, "losses": 0, "total": 0},
            "high": {"wins": 0, "losses": 0, "total": 0},
        },
        "sessions": {
            "Asia": {"wins": 0, "losses": 0, "total": 0},
            "USA": {"wins": 0, "losses": 0, "total": 0},
            "other": {"wins": 0, "losses": 0, "total": 0},
        },
        "last_updated": None,
    }
    return default

def load_performance():
    if not os.path.exists(PERF_FILE):
        perf = _init_perf()
        save_performance(perf)
        return perf
    try:
        with open(PERF_FILE, "r") as f:
            return json.load(f)
    except Exception:
        perf = _init_perf()
        save_performance(perf)
        return perf

def save_performance(perf):
    perf["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(PERF_FILE, "w") as f:
        json.dump(perf, f, indent=2)

def bucket_for_score(score):
    if score >= 1.5:
        return "high"
    elif score >= 1.0:
        return "mid"
    else:
        return "low"

def session_for_time(utc_dt):
    # Convert to IST to decide session
    ist_dt = utc_dt.astimezone(IST)
    hour = ist_dt.hour
    # Define Asia session early part, USA evening in IST
    if 0 <= hour < 12:
        return "Asia"
    elif 16 <= hour <= 23:
        return "USA"
    else:
        return "other"

def update_performance_from_closed_trades(days=180):
    perf = _init_perf()
    cutoff = datetime.utcnow() - timedelta(days=days)
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE result IN ('TP HIT','SL HIT')")
    rows = c.fetchall()
    conn.close()
    for r in rows:
        # r schema: id, time, direction, entry_price, result, exit_price, exit_time, rsi, wick, liq, score, tp_points, sl_points, source, win_prob, session
        time_str = r[1]
        score = r[10]
        result = r[4]
        session = r[15] if len(r) > 15 else "other"
        try:
            trade_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            trade_dt = UTC.localize(trade_dt)
        except Exception:
            trade_dt = datetime.utcnow()
        if trade_dt < cutoff:
            continue
        bucket = bucket_for_score(score)
        # Update bucket
        if result == "TP HIT":
            perf["score_buckets"][bucket]["wins"] += 1
        elif result == "SL HIT":
            perf["score_buckets"][bucket]["losses"] += 1
        perf["score_buckets"][bucket]["total"] += 1
        # Update session
        sess = session if session in perf["sessions"] else session_for_time(trade_dt)
        if result == "TP HIT":
            perf["sessions"][sess]["wins"] += 1
        elif result == "SL HIT":
            perf["sessions"][sess]["losses"] += 1
        perf["sessions"][sess]["total"] += 1
    save_performance(perf)
    return perf

def record_closed_trade_in_perf(trade):
    perf = load_performance()
    bucket = bucket_for_score(trade["score"])
    session = trade.get("session", "other")
    # bucket
    if trade["result"] == "TP HIT":
        perf["score_buckets"][bucket]["wins"] += 1
    elif trade["result"] == "SL HIT":
        perf["score_buckets"][bucket]["losses"] += 1
    perf["score_buckets"][bucket]["total"] += 1
    # session
    if session not in perf["sessions"]:
        session = session_for_time(UTC.localize(datetime.utcnow()))
    if trade["result"] == "TP HIT":
        perf["sessions"][session]["wins"] += 1
    elif trade["result"] == "SL HIT":
        perf["sessions"][session]["losses"] += 1
    perf["sessions"][session]["total"] += 1
    save_performance(perf)

def get_bucket_win_rate(bucket_name):
    perf = load_performance()
    bucket = perf["score_buckets"].get(bucket_name, {"wins": 0, "losses": 0, "total": 0})
    # Laplace smoothing
    wins = bucket["wins"] + 1
    total = bucket["total"] + 2
    return wins / total

def get_session_win_rate(session_name):
    perf = load_performance()
    sess = perf["sessions"].get(session_name, {"wins": 0, "losses": 0, "total": 0})
    wins = sess["wins"] + 1
    total = sess["total"] + 2
    return wins / total

def format_performance():
    perf = load_performance()
    lines = []
    lines.append("=== Score Bucket Performance (Laplace-smoothed) ===")
    for name in ["high", "mid", "low"]:
        b = perf["score_buckets"][name]
        win_rate = get_bucket_win_rate(name)
        lines.append(f"{name.title():5}: wins={b['wins']} losses={b['losses']} total={b['total']} win_rate={win_rate:.2%}")
    lines.append("\n=== Session Performance ===")
    for sess in ["Asia", "USA", "other"]:
        s = perf["sessions"][sess]
        win_rate = get_session_win_rate(sess)
        lines.append(f"{sess:5}: wins={s['wins']} losses={s['losses']} total={s['total']} win_rate={win_rate:.2%}")
    return "\n".join(lines)

# --- MEXC / market data ---
MEXC_BASE = "https://contract.mexc.com/api/v1/contract"
SYMBOL = os.getenv("MEXC_SYMBOL", "BTC_USDT")  # default

def fetch_mexc_ohlcv(symbol=SYMBOL, interval="Min5", limit=50):
    global _LAST_MEXC_OHLCV_SUCCESS
    interval_map = {
        "Min1": 60,
        "Min5": 300,
        "Min15": 900,
        "Min30": 1800,
        "Min60": 3600,
        "Hour4": 4 * 3600,
        "Hour8": 8 * 3600,
        "Day1": 24 * 3600,
    }
    step = interval_map.get(interval, 300)
    end = int(time.time())
    start = end - step * limit
    params = {"interval": interval, "start": start, "end": end}
    try:
        url = f"{MEXC_BASE}/kline/{symbol}"
        r = requests.get(url, params=params, timeout=8, headers={"User-Agent": "LiquidBot/1.0"})
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success"):
            logging.warning("MEXC kline returned not success: %s", resp)
            return []
        data = resp.get("data", {})
        times = data.get("time", [])
        opens = data.get("open", [])
        highs = data.get("high", [])
        lows = data.get("low", [])
        closes = data.get("close", [])
        candles = []
        for i in range(len(times)):
            candles.append(
                {
                    "open_time": times[i] * 1000,
                    "open": float(opens[i]),
                    "high": float(highs[i]),
                    "low": float(lows[i]),
                    "close": float(closes[i]),
                }
            )
        _LAST_MEXC_OHLCV_SUCCESS = time.time()
        return candles
    except Exception as e:
        logging.warning("Failed to fetch MEXC OHLCV: %s", e)
        return []

def fetch_mexc_ticker(symbol=SYMBOL):
    global _LAST_MEXC_TICKER_SUCCESS
    try:
        url = f"{MEXC_BASE}/ticker"
        r = requests.get(url, params={"symbol": symbol}, timeout=5, headers={"User-Agent": "LiquidBot/1.0"})
        r.raise_for_status()
        resp = r.json()
        if not resp.get("success"):
            logging.warning("MEXC ticker not success: %s", resp)
            return {}
        _LAST_MEXC_TICKER_SUCCESS = time.time()
        return resp.get("data", {})
    except Exception as e:
        logging.warning("Failed to fetch MEXC ticker: %s", e)
        return {}

# --- Liquidation inference ---
def infer_liquidation_pressure_from_mexc_enhanced():
    global _LAST_MEXC_OI, _LAST_LIQ_SOURCE
    try:
        ticker = fetch_mexc_ticker()
        if not ticker:
            return 0.0, "mexc_failed"
        open_interest = float(ticker.get("holdVol", 0))
        funding_rate = float(ticker.get("fundingRate", 0))

        oi_change = 0.0
        if _LAST_MEXC_OI and _LAST_MEXC_OI > 0:
            oi_change = (open_interest - _LAST_MEXC_OI) / _LAST_MEXC_OI
        _LAST_MEXC_OI = open_interest

        oi_factor = 1 + max(0.0, oi_change)
        funding_factor = 1 + min(abs(funding_rate), 0.005) * 50

        composite_pressure = (open_interest / 1e9) * oi_factor * funding_factor * 1_000_000

        source = f"mexc_enhanced(oi={int(open_interest):,}, Δoi={oi_change:.2%}, fund_rate={funding_rate:.4f})"
        _LAST_LIQ_SOURCE = source
        return composite_pressure, source
    except Exception as e:
        logging.warning("Enhanced MEXC pressure failed: %s", e)
        return infer_liquidation_pressure_from_mexc_simple()

def infer_liquidation_pressure_from_mexc_simple():
    global _LAST_LIQ_SOURCE
    try:
        ticker = fetch_mexc_ticker()
        if not ticker:
            return 0.0, "mexc_failed"
        open_interest = float(ticker.get("holdVol", 0))
        funding_rate = float(ticker.get("fundingRate", 0))
        if open_interest <= 0:
            return 0.0, "mexc_no_oi"
        pressure = (open_interest / 1e9) * (1 + abs(funding_rate) * 10)
        fallback_liq = pressure * 1_000_000
        source = "mexc_inferred"
        _LAST_LIQ_SOURCE = source
        return fallback_liq, source
    except Exception as e:
        logging.warning("Infer simple MEXC pressure failed: %s", e)
        return 0.0, "fallback"

def fetch_coinglass_liquidation():
    global _LAST_COINGLASS_SUCCESS, _LAST_LIQ_SOURCE
    if not COINGLASS_API_KEY:
        return 0, "coinglass_missing"
    try:
        headers = {"accept": "application/json", "coinglassSecret": COINGLASS_API_KEY}
        url = "https://open-api.coinglass.com/public/v2/liquidation/chart?symbol=BTC"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            logger.warning("CoinGlass HTTP %s: %s", resp.status_code, resp.text[:200])
            return 0, "coinglass_error"
        data = resp.json()
        total = 0
        if isinstance(data.get("data"), list):
            for entry in data["data"]:
                total += entry.get("sumAmount", entry.get("liquidationAmount", 0))
        _LAST_COINGLASS_SUCCESS = time.time()
        _LAST_LIQ_SOURCE = "coinglass"
        return total, "coinglass"
    except Exception as e:
        logging.warning("CoinGlass fetch failed: %s", e)
        return 0, "coinglass_failed"

def fetch_combined_liquidation():
    cg, src = fetch_coinglass_liquidation()
    if cg and cg > 0:
        return cg, src
    enhanced, src_enh = infer_liquidation_pressure_from_mexc_enhanced()
    if enhanced and enhanced > 0:
        return enhanced, src_enh
    simple, src_simple = infer_liquidation_pressure_from_mexc_simple()
    if simple and simple > 0:
        return simple, src_simple
    return 0, "none"

# --- Price fallback ---
def fetch_coingecko_price_candle():
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "usd"},
            timeout=6,
            headers={"User-Agent": "LiquidBot/1.0"},
        )
        resp.raise_for_status()
        data = resp.json()
        price = float(data.get("bitcoin", {}).get("usd", 0))
        if price > 0:
            t_ms = int(time.time() * 1000)
            return [{"open_time": t_ms, "open": price, "high": price, "low": price, "close": price}]
    except Exception as e:
        logging.warning("CoinGecko fallback failed: %s", e)
    return []

# --- Indicators / scoring ---
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_score(rsi, wick_pct, liquidation_usd, funding_rate=1.0):
    score = 0
    if rsi is not None:
        if rsi < 35:
            score += (35 - rsi) * 0.05
        elif rsi > 65:
            score += (rsi - 65) * 0.03
    score += min(wick_pct, 5) * 0.2
    score += min(liquidation_usd / 5_000_000, 2) * 0.5
    score *= funding_rate
    return round(score, 2)

def compute_adaptive_tp_sl(entry_price, ohlcv):
    # baseline points from env
    base_tp = int(os.getenv("TP_POINTS", "600"))
    base_sl = int(os.getenv("SL_POINTS", "200"))
    # volatility-based scaling (last 12 candles)
    if ohlcv and len(ohlcv) >= 1:
        ranges = [c["high"] - c["low"] for c in ohlcv[-12:]] if len(ohlcv) >= 12 else [c["high"] - c["low"] for c in ohlcv]
        avg_range = sum(ranges) / len(ranges) if ranges else 0
        tp_points = max(base_tp, int(avg_range * 4))
        sl_points = max(base_sl, int(avg_range * 1.5))
    else:
        tp_points = base_tp
        sl_points = base_sl
    return tp_points, sl_points

def check_tp_sl_points(entry_price, current_price, direction, tp_points, sl_points):
    if direction == "long":
        if current_price >= entry_price + tp_points:
            return "TP HIT"
        if current_price <= entry_price - sl_points:
            return "SL HIT"
    else:
        if current_price <= entry_price - tp_points:
            return "TP HIT"
        if current_price >= entry_price + sl_points:
            return "SL HIT"
    return "open"

# --- Signal generation with learning gating ---
def generate_trade_signal():
    ohlcv = fetch_mexc_ohlcv()
    if not ohlcv:
        ohlcv = fetch_coingecko_price_candle()
    if not ohlcv:
        return None
    closes = [c["close"] for c in ohlcv]
    rsi = compute_rsi(closes[-15:]) if len(closes) >= 15 else None
    last = ohlcv[-1]
    open_p = last["open"]
    close_p = last["close"]
    high = last["high"]
    low = last["low"]
    total_range = high - low if high - low != 0 else 1
    lower_wick_pct = ((min(open_p, close_p) - low) / total_range) * 100
    upper_wick_pct = ((high - max(open_p, close_p)) / total_range) * 100

    liquidation, source = fetch_combined_liquidation()
    funding_rate = 1.0  # placeholder

    direction = None
    wick_pct = 0
    if rsi is None:
        return None
    if rsi < 35 and lower_wick_pct > 0.5:
        direction = "long"
        wick_pct = lower_wick_pct
    elif rsi > 65 and upper_wick_pct > 0.5:
        direction = "short"
        wick_pct = upper_wick_pct
    else:
        return None

    score = calculate_score(rsi, wick_pct, liquidation, funding_rate)

    # Empirical win probability from score bucket
    bucket = bucket_for_score(score)
    win_prob = get_bucket_win_rate(bucket)

    # Session label
    now_utc = UTC.localize(datetime.utcnow())
    session = session_for_time(now_utc)

    # Adaptive TP/SL in points
    tp_points, sl_points = compute_adaptive_tp_sl(close_p, ohlcv)

    # Build signal
    entry_price = close_p
    signal = {
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": direction,
        "entry_price": entry_price,
        "rsi": rsi,
        "wick_percent": round(wick_pct, 2),
        "liquidation_usd": liquidation,
        "score": score,
        "result": "open",
        "tp_points": tp_points,
        "sl_points": sl_points,
        "liquidation_source": source,
        "win_prob": round(win_prob, 3),
        "session": session,
    }

    return signal

# --- Persistence / evaluation ---
def store_trade(trade):
    conn = _get_conn()
    c = conn.cursor()
    c.execute(
        """INSERT INTO trades 
           (time, direction, entry_price, result, exit_price, exit_time,
            rsi, wick_percent, liquidation_usd, score, tp_points, sl_points, liquidation_source, win_prob, session)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            trade.get("tp_points"),
            trade.get("sl_points"),
            trade.get("liquidation_source"),
            trade.get("win_prob"),
            trade.get("session"),
        ),
    )
    conn.commit()
    conn.close()

def evaluate_open_trades():
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trades WHERE result = 'open'")
    rows = c.fetchall()
    if not rows:
        conn.close()
        return
    ohlcv = fetch_mexc_ohlcv()
    if not ohlcv:
        ohlcv = fetch_coingecko_price_candle()
    if not ohlcv:
        conn.close()
        return
    current_price = ohlcv[-1]["close"]
    updated = False
    for r in rows:
        trade_id = r[0]
        direction = r[2]
        entry_price = r[3]
        tp_points = r[10]
        sl_points = r[11]
        status = check_tp_sl_points(entry_price, current_price, direction, tp_points, sl_points)
        if status in ("TP HIT", "SL HIT"):
            exit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                "UPDATE trades SET result = ?, exit_price = ?, exit_time = ? WHERE id = ?",
                (status, current_price, exit_time, trade_id),
            )
            # record performance immediately
            updated_trade = {
                "score": r[10] if False else r[9],  # score field at index 9
                "result": status,
                "session": r[15] if len(r) > 15 else "other",
            }
            updated = True
    if updated:
        conn.commit()
        # refresh performance stats from closed trades (incremental)
        update_performance_from_closed_trades(days=180)
    conn.close()

# --- Reporting ---
def format_trade_row(r):
    (
        _,
        time_str,
        direction,
        entry_price,
        result,
        exit_price,
        exit_time,
        rsi,
        wick,
        liq,
        score,
        tp_points,
        sl_points,
        source,
        win_prob,
        session,
    ) = r
    # show time in IST
    try:
        dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        dt = UTC.localize(dt).astimezone(IST)
        time_fmt = dt.strftime("%Y-%m-%d %H:%M:%S IST")
    except:
        time_fmt = time_str
    s = (
        f"{time_fmt} | {direction.upper()} @ {entry_price:.1f} | "
        f"RSI={rsi} | Wick={wick:.2f}% | Liq=${liq:,} ({source}) | "
        f"Score={score} | WinProb={win_prob:.0%} | TP+{tp_points} | SL-{sl_points} | Session={session}"
    )
    if result and result != "open":
        s += f" | {result} @ {exit_price:.1f} ({exit_time})"
    return s

def get_last_trades(limit=30):
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    if not rows:
        return "No recent trades."
    return "\n".join(format_trade_row(r) for r in rows)

def get_logs(limit=20):
    return get_last_trades(limit)

def get_status():
    default = {"rsi_threshold": 35, "wick_threshold": 0.5, "liq_threshold": 2_000_000, "min_win_rate": 0.5}
    if os.path.exists(STRATEGY_FILE):
        try:
            with open(STRATEGY_FILE, "r") as f:
                default = json.load(f)
        except:
            pass
    return (
        f"Thresholds: RSI<{default.get('rsi_threshold')} / >{100 - default.get('rsi_threshold')}, "
        f"Wick>{default.get('wick_threshold')}%, Liq>${default.get('liq_threshold'):,}, "
        f"MinWinRate={default.get('min_win_rate'):.2f}"
    )

def run_backtest(days=7):
    cutoff = datetime.utcnow() - timedelta(days=days)
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT * FROM trades ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    filtered = []
    for r in rows:
        try:
            t = datetime.strptime(r[1], "%Y-%m-%d %H:%M:%S")
        except:
            continue
        if t >= cutoff:
            filtered.append(format_trade_row(r))
    if not filtered:
        return f"No trades in last {days} days."
    return "📉 Backtest:\n" + "\n".join(filtered[:30])

def get_results_summary():
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT result, score FROM trades WHERE result IN ('TP HIT','SL HIT')")
    closed = c.fetchall()
    conn.close()
    if not closed:
        return "No closed trades yet."
    wins = sum(1 for r in closed if r[0] == "TP HIT")
    losses = sum(1 for r in closed if r[0] == "SL HIT")
    total = wins + losses
    win_rate = round((wins / total) * 100, 2) if total else 0
    conn = _get_conn()
    c = conn.cursor()
    c.execute("SELECT score FROM trades")
    all_scores = [r[0] for r in c.fetchall()]
    conn.close()
    avg_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    return (
        f"Closed trades: {total}\n"
        f"Wins (TP): {wins}\n"
        f"Losses (SL): {losses}\n"
        f"Win rate: {win_rate}%\n"
        f"Avg score: {avg_score}"
    )

# --- News ---
def fetch_news():
    if not NEWS_API_KEY:
        return ["No news API key set."]
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={NEWS_API_KEY}&currencies=BTC"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return [f"CryptoPanic HTTP {r.status_code}: {r.text[:200]}"]
        data = r.json()
        items = data.get("results", [])[:5]
        if not items:
            return ["No recent news found."]
        headlines = []
        for it in items:
            title = it.get("title", "No title")
            link = it.get("url", "")
            headlines.append(f"• {title}\n{link}")
        return headlines
    except Exception as e:
        return [f"News fetch error: {e}"]

# --- Live status ---
def get_live_data_status():
    def fmt(ts):
        if not ts:
            return "never"
        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC")

    return {
        "mexc_ohlcv_last_success": fmt(_LAST_MEXC_OHLCV_SUCCESS),
        "mexc_ticker_last_success": fmt(_LAST_MEXC_TICKER_SUCCESS),
        "coinglass_last_success": fmt(_LAST_COINGLASS_SUCCESS),
        "last_liq_source": _LAST_LIQ_SOURCE,
    }
