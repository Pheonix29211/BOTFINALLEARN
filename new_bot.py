import os
import logging
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler
from dotenv import load_dotenv
from new_utils import (
    fetch_mexc_ohlcv, compute_rsi, compute_atr,
    infer_liquidation_pressure_from_mexc,
    score_long, score_short,
    train_model_incremental, predict_win_prob,
    load_cached_history, simulate_history,
    store_trade, get_last_trades,
    simulate_and_store_full_history,
    get_setup_strength
)
from datetime import datetime

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
OWNER_CHAT_ID = os.getenv("OWNER_CHAT_ID")

# Fixed TP/SL points
TP_POINTS = int(os.getenv("TP_POINTS", "600"))
SL_POINTS = int(os.getenv("SL_POINTS", "200"))
WIN_PROB_THRESHOLD = 0.6

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learner_bot")

bot = Bot(token=TOKEN)
app = Flask(__name__)
dispatcher = Dispatcher(bot, None, use_context=True)

def start(update, context):
    update.message.reply_text("""Welcome to LearnerBot Ultimate!
Available commands:
/menu - list
/scan - current signal
/train - incremental retrain
/train_full - full-year backfill + learning (heavy)
/backtest - last 7 days simulation
/last30 - show last 30 stored trades""")

def menu(update, context):
    start(update, context)

def scan(update, context):
    ohlcv = fetch_mexc_ohlcv()
    if not ohlcv:
        update.message.reply_text("Failed to fetch price data.")
        return
    closes = [c.get("close") for c in ohlcv]
    rsi = compute_rsi(closes[-15:]) if len(closes) >= 15 else None
    atr = compute_atr(ohlcv[-15:]) if len(ohlcv) >= 15 else None
    last = ohlcv[-1]
    open_p = last.get("open")
    close_p = last.get("close")
    high = last.get("high")
    low = last.get("low")
    total_range = high - low if high - low != 0 else 1
    lower_wick_pct = (min(open_p, close_p) - low) / total_range * 100
    upper_wick_pct = (high - max(open_p, close_p)) / total_range * 100

    liquidation, source = infer_liquidation_pressure_from_mexc()
    long_score = score_long(rsi, lower_wick_pct, liquidation)
    short_score = score_short(rsi, upper_wick_pct, liquidation)

    if short_score > long_score:
        direction = "short"
        wick_pct = upper_wick_pct
        base_score = short_score
    else:
        direction = "long"
        wick_pct = lower_wick_pct
        base_score = long_score

    entry_price = close_p

    # Fixed TP/SL in absolute points
    if direction == "long":
        tp_price = entry_price + TP_POINTS
        sl_price = entry_price - SL_POINTS
    else:
        tp_price = entry_price - TP_POINTS
        sl_price = entry_price + SL_POINTS

    tp_pct = abs(tp_price / entry_price - 1) * 100
    sl_pct = abs(sl_price / entry_price - 1) * 100
    reward_to_risk = TP_POINTS / SL_POINTS

    signal = {
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "direction": direction,
        "entry_price": entry_price,
        "rsi": rsi,
        "wick_percent": round(wick_pct,2),
        "liquidation_usd": liquidation,
        "score": base_score,
        "tp_pct": round(tp_pct,2),
        "sl_pct": round(sl_pct,2),
        "tp_price": round(tp_price,2),
        "sl_price": round(sl_price,2),
        "reward_to_risk": round(reward_to_risk,2),
        "volume_z": None,
        "liq_ratio": None,
        "liquidation_source": source,
    }

    win_prob = predict_win_prob(signal)
    signal["win_prob"] = round(win_prob,2)

    # Empirical strength
    strength = get_setup_strength(signal, k=50)
    empirical = strength.get("empirical_win_rate")
    count = strength.get("count")
    ci_lo = strength.get("ci_lower") or 0
    ci_hi = strength.get("ci_upper") or 0

    composite_strength = signal["win_prob"]
    if empirical is not None:
        weight_empirical = min(0.7, count / (count + 30))
        weight_model = 1 - weight_empirical
        composite_strength = round(weight_model * signal["win_prob"] + weight_empirical * empirical,2)

    if composite_strength < WIN_PROB_THRESHOLD:
        update.message.reply_text(f"Signal suppressed: low composite strength {composite_strength:.2f}")
        return

    signal["result"] = "open"
    store_trade(signal)

    # Craft message
    emp_display = f"{empirical:.2%}" if empirical is not None else "N/A"
    ci_display = f"{ci_lo:.2%}-{ci_hi:.2%}" if empirical is not None else "N/A"

    msg = (
        f"🚨 {direction.upper()} Signal\n"
        f"Entry: {entry_price:.1f}\n"
        f"RSI: {rsi} | Wick%: {signal['wick_percent']}% | Liq: ${liquidation:,.0f} ({source})\n"
        f"Base Score: {base_score} | WinProb: {signal['win_prob']} | Empirical: {emp_display} ({count} similar, CI {ci_display})\n"
        f"Composite Strength: {composite_strength} | RR: {signal['reward_to_risk']}\n"
        f"TP: +{signal['tp_pct']}% @ {signal['tp_price']:.1f} | SL: -{signal['sl_pct']}% @ {signal['sl_price']:.1f}"
    )
    bot.send_message(chat_id=OWNER_CHAT_ID, text=msg)
    update.message.reply_text("Signal sent.")

def train(update, context):
    res = train_model_incremental()
    update.message.reply_text(res)

def train_full(update, context):
    update.message.reply_text("Starting full-year simulation and learning; this can take a while...")
    result = simulate_and_store_full_history()
    update.message.reply_text(result)

def backtest(update, context):
    candles = load_cached_history()
    now_ms = int(datetime.utcnow().timestamp() * 1000)
    seven_days_ms = 7 * 24 * 3600 * 1000
    recent = [c for c in candles if c.get("ts",0) >= now_ms - seven_days_ms]
    if not recent:
        update.message.reply_text("No historical data for backtest.")
        return
    results = simulate_history(recent)
    if not results:
        update.message.reply_text("No signals simulated in last 7 days.")
        return
    wins = sum(1 for r in results if r.get("label") == 1)
    total = len(results)
    for r in results:
        entry = r.get("entry_price")
        direction = r.get("direction")
        if r.get("result") == "TP HIT":
            exit_price = entry + TP_POINTS if direction == "long" else entry - TP_POINTS
        else:
            exit_price = entry - SL_POINTS if direction == "long" else entry + SL_POINTS
        r["exit_price"] = round(exit_price,2)
        r["exit_time"] = r.get("time")
        r["tp_pct"] = (TP_POINTS / entry) * 100
        r["sl_pct"] = (SL_POINTS / entry) * 100
        r["win_prob"] = 1.0
        r["reward_to_risk"] = TP_POINTS / SL_POINTS
        r["volume_z"] = 1.0
        r["liq_ratio"] = 1.0
        r["news_sentiment"] = 0.0
        store_trade(r)
    summary = f"Backtest last 7d: {wins}/{total} wins ({wins/total:.1%}) stored {total} trades."
    update.message.reply_text(summary)

def last30(update, context):
    res = get_last_trades(30)
    update.message.reply_text(res)

# Handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("menu", menu))
dispatcher.add_handler(CommandHandler("scan", scan))
dispatcher.add_handler(CommandHandler("train", train))
dispatcher.add_handler(CommandHandler("train_full", train_full))
dispatcher.add_handler(CommandHandler("backtest", backtest))
dispatcher.add_handler(CommandHandler("last30", last30))

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

@app.route("/")
def index():
    return "LearnerBot Ultimate running."

if __name__ == "__main__":
    logging.info("Starting LearnerBot Ultimate with webhook %s", f"{WEBHOOK_URL}/{TOKEN}")
    bot.set_webhook(url=f"{WEBHOOK_URL}/{TOKEN}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
