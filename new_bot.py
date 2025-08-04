import os
import logging
import time
from threading import Thread
from datetime import datetime
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler
from dotenv import load_dotenv
import pytz

# Load env
load_dotenv()

# Environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")
MIN_WIN_RATE = float(os.getenv("MIN_WIN_RATE", "0.5"))

if not TELEGRAM_TOKEN or not WEBHOOK_URL:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or WEBHOOK_URL in environment.")

# Setup
bot = Bot(token=TELEGRAM_TOKEN)
app = Flask(__name__)
dispatcher = Dispatcher(bot, None, workers=4, use_context=True)
os.environ["TZ"] = "Asia/Kolkata"  # for any legacy uses

# Logging
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("LiquidBot")

# Timezones
UTC = pytz.UTC
IST = pytz.timezone("Asia/Kolkata")

# Utils imports
from utils import (
    generate_trade_signal,
    store_trade,
    evaluate_open_trades,
    get_last_trades,
    get_results_summary,
    run_backtest,
    get_status,
    get_logs,
    fetch_combined_liquidation,
    fetch_news,
    get_live_data_status,
    fetch_mexc_ohlcv,
    fetch_mexc_ticker,
    format_trade_row,
    format_performance,
    update_performance_from_closed_trades,
)

# --- Safe OHLCV fetch with backoff ---
_last_good_ohlcv = None
_live_failures = 0
_backoff_seconds = 60  # initial backoff
_last_attempt_time = 0
MAX_BACKOFF = 8 * 60  # 8 minutes
ALERT_THRESHOLD = 4  # after this many consecutive failures, notify

def safe_fetch_ohlcv():
    global _last_good_ohlcv, _live_failures, _backoff_seconds, _last_attempt_time
    now = time.time()
    if _live_failures > 0 and (now - _last_attempt_time) < _backoff_seconds:
        return _last_good_ohlcv or []
    _last_attempt_time = now
    ohlcv = fetch_mexc_ohlcv()
    if not ohlcv:
        # fallback to coingecko inside utils if implemented
        try:
            from utils import fetch_coingecko_price_candle
            ohlcv = fetch_coingecko_price_candle()
        except ImportError:
            ohlcv = []
    if ohlcv:
        _live_failures = 0
        _backoff_seconds = 60
        _last_good_ohlcv = ohlcv
        return ohlcv
    _live_failures += 1
    _backoff_seconds = min(MAX_BACKOFF, _backoff_seconds * 2)
    if _live_failures >= ALERT_THRESHOLD:
        try:
            bot.send_message(
                chat_id=OWNER_CHAT_ID,
                text=(
                    f"⚠️ Live OHLCV fetch failing {_live_failures} times; "
                    f"using last snapshot if available. Backoff {_backoff_seconds//60}m."
                ),
            )
        except Exception:
            pass
    return _last_good_ohlcv or []

# --- Command handlers ---

def start(update: Update, context):
    update.message.reply_text("🚀 LiquidBot live. Use /menu to see commands.")

def menu(update: Update, context):
    update.message.reply_text(
        "/menu\n"
        "/start\n"
        "/backtest\n"
        "/last30\n"
        "/results\n"
        "/status\n"
        "/logs\n"
        "/liqcheck\n"
        "/news\n"
        "/scan\n"
        "/envcheck\n"
        "/debug_sources\n"
        "/learn\n"
        "/performance"
    )

def backtest_cmd(update: Update, context):
    update.message.reply_text(run_backtest())

def last30_cmd(update: Update, context):
    update.message.reply_text(get_last_trades())

def results_cmd(update: Update, context):
    update.message.reply_text(get_results_summary())

def status_cmd(update: Update, context):
    strategy = get_status()
    live = get_live_data_status()
    extra = (
        f"\n\n📡 Live Data Status:\n"
        f" • MEXC OHLCV last success: {live.get('mexc_ohlcv_last_success')}\n"
        f" • MEXC ticker last success: {live.get('mexc_ticker_last_success')}\n"
        f" • CoinGlass last success: {live.get('coinglass_last_success')}\n"
        f" • Last liquidation source: {live.get('last_liq_source')}"
    )
    update.message.reply_text(strategy + extra)

def logs_cmd(update: Update, context):
    update.message.reply_text(get_logs())

def liqcheck(update: Update, context):
    liq, source = fetch_combined_liquidation()
    update.message.reply_text(f"Liquidation proxy: ${liq:,.0f} (source: {source})")

def news_cmd(update: Update, context):
    headlines = fetch_news()
    if isinstance(headlines, list):
        update.message.reply_text("\n\n".join(headlines))
    else:
        update.message.reply_text(str(headlines))

def envcheck(update: Update, context):
    required = ["TELEGRAM_BOT_TOKEN", "WEBHOOK_URL", "OWNER_CHAT_ID"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        update.message.reply_text("Missing env vars: " + ", ".join(missing))
    else:
        update.message.reply_text("All primary env vars present.")

def debug_sources(update: Update, context):
    ohlcv = safe_fetch_ohlcv()
    last_close = ohlcv[-1]["close"] if ohlcv else "n/a"
    liq, source = fetch_combined_liquidation()
    mexc_ticker = fetch_mexc_ticker()
    hold_vol = mexc_ticker.get("holdVol", "n/a")
    funding = mexc_ticker.get("fundingRate", "n/a")
    update.message.reply_text(
        f"MEXC last close: {last_close}\n"
        f"Liquidation proxy: ${liq:,.0f} (source: {source})\n"
        f"MEXC holdVol: {hold_vol}\n"
        f"MEXC fundingRate: {funding}"
    )

def learn_cmd(update: Update, context):
    perf = update_performance_from_closed_trades(days=180)
    update.message.reply_text("Rebuilt performance stats from last 6 months.\n" + format_performance())

def performance_cmd(update: Update, context):
    update.message.reply_text(format_performance())

def send_signal_message(signal):
    direction = signal["direction"].upper()
    score = signal["score"]
    entry = signal["entry_price"]
    rsi = signal["rsi"]
    wick = signal["wick_percent"]
    liq = signal["liquidation_usd"]
    source = signal.get("liquidation_source", "unknown")
    tp = signal.get("tp_points")
    sl = signal.get("sl_points")
    win_prob = signal.get("win_prob", 0)
    session = signal.get("session", "unknown")

    strength = "Strong" if score >= 1.5 else ("Moderate" if score >= 1.0 else "Weak")
    confidence_tag = f"{win_prob:.0%} historical win rate"
    if win_prob >= 0.6:
        leverage_hint = "Higher leverage"
    elif win_prob >= 0.4:
        leverage_hint = "Cautious / low leverage"
    else:
        leverage_hint = "Skip / very low confidence"

    msg = (
        f"🚨 {direction} Signal ({strength})\n"
        f"Entry: {entry:.1f}\n"
        f"RSI: {rsi} | Wick%: {wick:.2f}% | Liq: ${liq:,.0f} ({source})\n"
        f"Score: {score} | {confidence_tag} | Session: {session}\n"
        f"TP: +{tp} pts | SL: -{sl} pts\n"
        f"Leverage hint: {leverage_hint}"
    )
    try:
        if OWNER_CHAT_ID:
            bot.send_message(chat_id=OWNER_CHAT_ID, text=msg)
        else:
            logger.warning("OWNER_CHAT_ID not set; cannot send signal.")
    except Exception as e:
        logger.error("Failed to send signal message: %s", e)

def scan_cmd(update: Update, context):
    try:
        evaluate_open_trades()
    except Exception as e:
        update.message.reply_text(f"Error evaluating open trades: {e}")

    ohlcv = safe_fetch_ohlcv()
    if not ohlcv:
        update.message.reply_text("🔍 Scan: failed to fetch price data.")
        return

    # Show debug info similar to previous versions
    closes = [c["close"] for c in ohlcv]
    # compute RSI via utils
    from utils import compute_rsi, calculate_score
    rsi = compute_rsi(closes[-15:]) if len(closes) >= 15 else None
    last = ohlcv[-1]
    open_p = last["open"]
    close_p = last["close"]
    high = last["high"]
    low = last["low"]
    total_range = high - low if high - low != 0 else 1
    lower_wick_pct = ((min(open_p, close_p) - low) / total_range) * 100
    upper_wick_pct = ((high - max(open_p, close_p)) / total_range) * 100
    liq, source = fetch_combined_liquidation()
    score_long = calculate_score(rsi, lower_wick_pct, liq) if rsi is not None else None
    score_short = calculate_score(rsi, upper_wick_pct, liq) if rsi is not None else None

    debug_msg = (
        f"🛠️ Debug Info:\n"
        f"RSI: {rsi}\n"
        f"Lower wick %: {lower_wick_pct:.2f}\n"
        f"Upper wick %: {upper_wick_pct:.2f}\n"
        f"Liquidation proxy: ${liq:,.0f} (source: {source})\n"
        f"Score Long: {score_long} | Score Short: {score_short}"
    )
    update.message.reply_text(debug_msg)

    signal = generate_trade_signal()
    if signal:
        # gating based on empirical win rate
        win_prob = signal.get("win_prob", 0)
        if win_prob < MIN_WIN_RATE:
            update.message.reply_text(f"Signal suppressed due to low historical win rate ({win_prob:.0%} < {MIN_WIN_RATE:.0%})")
        else:
            try:
                store_trade(signal)
            except Exception as e:
                update.message.reply_text(f"Failed to store signal: {e}")
            send_signal_message(signal)
            update.message.reply_text("🔍 Scan: real signal processed.")
    else:
        update.message.reply_text("🔍 Scan: no high-confidence real signal.")

# Register handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("menu", menu))
dispatcher.add_handler(CommandHandler("backtest", backtest_cmd))
dispatcher.add_handler(CommandHandler("last30", last30_cmd))
dispatcher.add_handler(CommandHandler("results", results_cmd))
dispatcher.add_handler(CommandHandler("status", status_cmd))
dispatcher.add_handler(CommandHandler("logs", logs_cmd))
dispatcher.add_handler(CommandHandler("liqcheck", liqcheck))
dispatcher.add_handler(CommandHandler("news", news_cmd))
dispatcher.add_handler(CommandHandler("scan", scan_cmd))
dispatcher.add_handler(CommandHandler("envcheck", envcheck))
dispatcher.add_handler(CommandHandler("debug_sources", debug_sources))
dispatcher.add_handler(CommandHandler("learn", learn_cmd))
dispatcher.add_handler(CommandHandler("performance", performance_cmd))

# Webhook route
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

@app.route("/")
def index():
    return "Bot is running."

# Scheduled background job
def scheduled_tasks():
    try:
        evaluate_open_trades()
        signal = generate_trade_signal()
        if signal:
            win_prob = signal.get("win_prob", 0)
            if win_prob >= MIN_WIN_RATE:
                store_trade(signal)
                send_signal_message(signal)
            else:
                logger.info(f"Skipped signal due to low win_prob {win_prob:.2f}")
    except Exception as e:
        logger.error("Scheduled task error: %s", e)

def start_scheduler():
    while True:
        scheduled_tasks()
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    logging.info("Starting bot with webhook URL: %s", f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}")
    try:
        bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}")
    except Exception as e:
        logger.warning("Failed to set webhook: %s", e)
    Thread(target=start_scheduler, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
