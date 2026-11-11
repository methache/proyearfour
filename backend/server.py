"""
Supervisor entrypoint — exposes the FastAPI app defined in main.py
as `server:app`, matching the /etc/supervisor/conf.d/*.conf config.
"""
from main import app  # noqa: F401

# Register the /api/prices/live endpoint + startup hook.
import live  # noqa: E402, F401
live.register(app)

# Register auth endpoints (Emergent-managed Google OAuth).
import auth  # noqa: E402, F401
auth.register(app)

# Register authenticated chat history and conversation management.
import chat_history  # noqa: E402, F401
chat_history.register(app)

# Register aggregated portfolio news endpoint (Yahoo Finance).
import news  # noqa: E402, F401
news.register(app)

# Register detail-page analysis endpoints (MPT + Regime real data).
import analysis  # noqa: E402, F401
analysis.register(app)

# Register My Portfolio historical price endpoint.
import myportfolio  # noqa: E402, F401
myportfolio.register(app)

# Register per-user data persistence (portfolio, watchlist, custom tickers, alerts).
import user_data  # noqa: E402, F401
user_data.register(app)

# Register multi-portfolio + dividend tracking.
import portfolio_manager  # noqa: E402, F401
portfolio_manager.register(app)

# Register rebalancing suggestions (MPT + regime tilt).
import rebalancing  # noqa: E402, F401
rebalancing.register(app)

# Register backtesting engine (buy & hold vs MPT vs regime-based).
import backtest  # noqa: E402, F401
backtest.register(app)

# Register trading-terminal data (candlestick OHLCV + watchlist quotes).
import terminal  # noqa: E402, F401
terminal.register(app)

# Register OpenMarket crypto data proxy (candles + markets) for /terminal.
import omarket  # noqa: E402, F401
omarket.register(app)
