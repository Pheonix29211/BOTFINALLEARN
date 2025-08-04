# LearnerBot Ultimate (Fixed TP/SL + Strength Scoring)

## Features
- Full-year historical MEXC backfill and signal simulation.
- Fixed TP = 600 points, SL = 200 points (3:1 RR) for live signals.
- Direction-aware scoring (RSI, wick, inferred liquidation).
- Empirical setup strength via nearest similar past trades with Bayesian smoothing.
- Supervised incremental model predicting win probability.
- Composite strength combining model and empirical.
- Telegram commands: /start, /menu, /scan, /train, /train_full, /backtest, /last30.

## Setup
1. Copy `.env.example` to `.env` and fill tokens & variables.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Push to GitHub and deploy. Run `/train_full` once to seed model.
4. Use `/scan` for live signals.

## Git push example
```bash
git init
git remote add origin https://github.com/yourusername/LIQUIDBOT-LEARN.git
git add new_utils.py new_bot.py requirements.txt .gitignore .env.example README.md
git commit -m "Add final learning bot with fixed TP/SL and strength scoring"
git branch -M main
git push -u origin main
```
