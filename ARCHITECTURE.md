# Algobot v2 Architecture

This document describes the architecture of algobot_v2. It is maintained and updated after every significant change.

## Overview

Algobot v2 is a modular trading algorithm system that interfaces with Godel Terminal for market data/signals (MOST screening, PRT probability model) and executes trades through Interactive Brokers via `ib_async`. The system is designed with maintainability and organization as core principles.

## Directory Structure

```
algobot_v2/
├── constants/          # Global constants and parameters
│   ├── __init__.py
│   ├── parameters.py  # All configuration values, thresholds, hyperparameters
│   └── blacklist.py   # Commission blacklist for stocks
│
├── ibkr/               # Interactive Brokers integration (single socket API - no OAuth/token package needed)
│   ├── __init__.py
│   ├── connection.py   # IBConnection - owns the shared IB() instance, connect/reconnect/paper-live guard
│   ├── account.py      # AccountClient (positions, account values, open orders)
│   ├── orders.py       # OrderManager (order creation, cancellation, live status)
│   └── market_data.py  # MarketDataClient (batch quotes, market hours)
│
├── database/          # Database management
│   ├── __init__.py
│   ├── schema.sql     # SQLite database schema
│   └── init.py        # Database initialization
│
├── core/              # Core framework
│   ├── __init__.py
│   └── controller.py  # Godel Terminal controller and base classes
│
├── commands/          # Godel Terminal commands
│   ├── __init__.py
│   ├── des_command.py
│   ├── prt_command.py
│   ├── most_command.py
│   └── ...            # Other command implementations
│
├── strategy/          # Trading strategy
│   ├── __init__.py
│   ├── rolling_strategy.py  # Rolling strategy implementation
│   ├── trade_tracker.py     # Trade time tracking
│   ├── position_sizing.py   # Position sizing calculations
│   └── trade_executor.py    # Trade execution with limit orders
│
├── monitoring/        # System monitoring
│   ├── __init__.py
│   └── account_size.py      # Account size monitoring and CSV logging
│
├── scripts/           # Utility scripts
│   ├── __init__.py
│   ├── startup.py           # System startup and main trading loop
│   ├── algo_loop.py         # Main algorithm loop (alternative entry point)
│   └── close_all_positions.py  # Emergency position closer
│
├── quant/             # Quant research framework (see quant/README.md)
│   ├── config.py            # Resolves QUANT_* constants + storage paths
│   ├── pipeline.py          # CLI: ingest -> universe -> train -> backtest -> report
│   ├── data/                # Market/fundamental/macro/alt data + Godel PRT signals
│   ├── universe/            # Tradeable-universe filters (liquidity, ADDV, spread)
│   ├── features/            # Technical, price action, volatility, regime, microstructure
│   ├── models/              # Baselines, gradient boosting, HMM regimes, Kalman, registry
│   ├── probability/         # P(target-before-stop), expected return, Godel blend
│   ├── sizing/              # Fixed fractional, vol targeting, fractional Kelly, caps
│   ├── risk/                # Daily loss/max DD kill switches, corr/sector limits, circuit breakers
│   ├── backtest/            # Event-driven engine (costs+slippage), walk-forward validation
│   ├── metrics/             # CAGR, Sharpe, Sortino, maxDD, Calmar, PF, alpha/beta, IR
│   ├── execution/           # Limit-first executor, TWAP slicing, reconciliation
│   ├── monitor/             # Console dashboard + persisted snapshots
│   ├── journal/             # Audit trail: predictions, orders, fills, model versions
│   └── retrain/             # Challenger-vs-champion retraining, drift detection, rollback
│
├── tests/             # Offline tests (synthetic data, no network/IBKR)
│   └── test_quant_smoke.py
│
├── quant_data/        # Generated storage (gitignored): parquet, DuckDB, models, journal
│
├── RULES.md           # Architecture rules and guidelines
├── ARCHITECTURE.md    # This file
├── requirements.txt   # Python dependencies
├── requirements-quant.txt  # Quant framework dependencies
└── README.md          # Project documentation
```

## Module Descriptions

### Constants Package (`constants/`)

**Purpose**: Centralized storage of all configuration values, parameters, and hyperparameters.

**Key Files**:
- `parameters.py`: Contains all global constants including:
  - Godel Terminal configuration (URL, credentials)
  - Interactive Brokers configuration (host, port, client ID, paper/live flag)
  - Database configuration (type, name, path)
  - Trading parameters (stop loss, take profit percentages, max position size)
  - PRT strategy parameters (prob_up thresholds, edge thresholds)
  - MOST command parameters (tab, limit)
  - Market hours parameters
  - Order parameters (limit order settings, price offsets)
  - Account monitoring parameters (check interval)
  - Timeouts and browser settings

- `blacklist.py`: Commission blacklist for stocks that have trading commissions

**Design Principle**: No magic numbers or hardcoded values in business logic. All constants are defined here.

### Interactive Brokers Package (`ibkr/`)

**Purpose**: Complete IBKR integration via `ib_async`. Flat, one-module-per-responsibility - unlike Schwab's REST API, IBKR is a single authenticated socket connection (TWS/IB Gateway), so there is no separate OAuth/token subpackage and no split between a "Trader API" and a "Market Data API".

**Why native `ib_async` objects, not REST-style dicts**: `ib_async` keeps positions, account values, and order status live in memory via a background reader thread. Reading them is a local attribute access, not a network round trip. `ibkr/` returns native types (`Position`, `Trade`, `Ticker`) directly instead of re-wrapping them into a REST-shaped dict, so callers get that speed for free.

**Modules**:
1. **`connection.py`**: `IBConnection` class
   - Owns the single shared `ib_async.IB()` instance used by every other module in the package
   - `connect()`: dials TWS/IB Gateway, retries once after 5s on failure
   - Hard guard at construction time: aborts if `PAPER_TRADING` doesn't match `IBKR_PORT` (7497/4002 = paper, 7496/4001 = live)
   - `ensure_connected()`: reconnects if the socket has dropped
   - `sleep(seconds)`: sleeps via `ib.sleep()` instead of `time.sleep()`, so the event loop keeps processing incoming messages (fills, position/account updates) during the wait

2. **`account.py`**: `AccountClient` class
   - `get_positions()`: `ib.positions()` - live, in-memory
   - `get_account_info()`: flat dict of `ib.accountSummary()` tags (`NetLiquidation`, `BuyingPower`, `TotalCashValue`, `EquityWithLoanValue`, ...)
   - `get_all_open_orders()`: `ib.openTrades()`

3. **`orders.py`**: `OrderManager` class
   - `create_market_order()` / `create_limit_order()` / `create_stop_loss_order()`: qualify the contract, place the order, return the live `Trade` object
   - `cancel_order(trade)`: cancels via the `Trade`'s underlying `Order`
   - `get_order_status(trade)`: reads `trade.orderStatus.status` directly - no network call, unlike Schwab's REST-polled `get_order_status()`
   - IBKR has no separate `SELL_SHORT`/`BUY_TO_COVER` instruction - a plain `SELL` on a flat/long position opens/closes a long, a plain `BUY`/`SELL` on an existing short covers/opens it

4. **`market_data.py`**: `MarketDataClient` class
   - `get_quotes_batch(symbols)`: one `ib.reqTickers()` snapshot call for many symbols at once (auto-cancels its market data line afterwards, so it doesn't count against the account's concurrent-streaming-line limit even for the full `PRT_MAX_TICKERS` list)
   - `get_quote(symbol)`: single-symbol convenience wrapper
   - `is_market_open()`: parses the reference contract's (`SPY`) `liquidHours` from `ib.reqContractDetails()` - same signature as the old Schwab market-hours check, but sourced from IBKR's own contract data instead of a separate API call

**Key Methods**:
- `AccountClient.get_account_info()`: Get account net liquidation, buying power, cash, equity
- `AccountClient.get_positions()`: Get current positions
- `OrderManager.create_market_order()` / `create_limit_order()` / `create_stop_loss_order()`: Place orders
- `OrderManager.cancel_order()`: Cancel an order
- `OrderManager.get_order_status()`: Read live order status/fill information
- `MarketDataClient.get_quotes_batch()` / `get_quote()`: Get current price(s)
- `MarketDataClient.is_market_open()`: Check if market is open with delay check

### Database Package (`database/`)

**Purpose**: Manages local SQLite database for trade and PRT data storage.

**Modules**:
1. **`schema.sql`**: Database schema definition
   - Tables: trades, trade_parameters, prt_data, des_data, etc.
   - Indexes for performance
   - Foreign key relationships

2. **`init.py`**: Database initialization
   - Creates database file if it doesn't exist
   - Executes schema to create tables
   - Provides connection management

**Database Structure**:
- **trades**: Main trade records
- **trade_parameters**: Trade-specific parameters (stop loss, take profit)
- **prt_data**: PRT (Probability Return Table) data associated with trades
- **des_data**: Stock description data from Godel Terminal
- **dashboard_snapshots**: Historical dashboard statistics
- **active_trades**: Real-time active trade tracking
- **portfolio_value_history**: Portfolio value over time
- **beta_over_time**: Portfolio beta tracking
- **spy_prt_snapshots**: SPY PRT data over time

### Core Package (`core/`)

**Purpose**: Core framework for Godel Terminal interaction.

**Modules**:
1. **`controller.py`**: Main controller and base classes
   - `DOMMonitor`: Monitors DOM for new windows
   - `BaseCommand`: Abstract base class for all commands
   - `GodelTerminalController`: Main controller class

**Key Classes**:

#### `DOMMonitor`
- Monitors DOM for new command windows
- Tracks window creation and loading states
- Provides methods to wait for windows and loading completion

#### `BaseCommand` (Abstract)
- Base class for all Godel Terminal commands
- Handles command execution flow:
  1. Get command string
  2. Send command to terminal
  3. Wait for new window
  4. Wait for loading
  5. Extract data
- Provides window closing functionality

#### `GodelTerminalController`
- Manages browser automation
- Handles login and authentication
- Manages command registration and execution
- Provides methods:
  - `connect()`: Initialize browser
  - `login()`: Authenticate with Godel Terminal
  - `load_layout()`: Load specific layout
  - `open_terminal()`: Open terminal interface
  - `send_command()`: Send command string
  - `execute_command()`: Execute registered command
  - `disconnect()`: Clean shutdown

**Control Flow**:
```
Controller.connect() → Browser opens → Navigate to Godel Terminal
Controller.login() → Enter credentials → Wait for main page
Controller.load_layout() → Select layout → Wait for layout load
Controller.open_terminal() → Press backtick → Terminal opens
Controller.execute_command() → Send command → Wait for window → Extract data
```

### Commands Package (`commands/`)

**Purpose**: Implementations of specific Godel Terminal commands.

**Design**: Each command inherits from `BaseCommand` and implements:
- `get_command_string()`: Returns the command string to send
- `extract_data()`: Extracts data from the command window

**Implemented Commands**:
- **DES**: Description command - gets stock information
- **PRT**: Probability Return Table command - batch analysis of tickers
- **MOST**: Most Active Stocks command - gets active/gainers/losers lists
- **G**: Generic command
- **GIP**: Generic IP command
- **QM**: Query Market command

### Strategy Package (`strategy/`)

**Purpose**: Trading strategy implementation.

**Modules**:
1. **`rolling_strategy.py`**: RollingStrategy class
   - Implements rolling (continuous) strategy (not batch-based)
   - Collects tickers from MOST
   - Sorts by prob_up thresholds
   - Creates positions based on PRT predictions
   - Maintains positions by re-running PRT
   - **Position Direction Checking**:
     - Compares PRT direction (prob_up > 50% = LONG, < 50% = SHORT) with current position direction
     - If PRT direction matches position direction → Keep position (log it)
     - If PRT direction is opposite → Close position
   - **Boxed Position Prevention**:
     - Checks existing positions before opening new ones
     - Only opens positions for tickers not already in portfolio
     - Prevents attempting to open positions that would create boxed positions
   - **Position Closing**:
     - Uses limit orders first when closing positions
     - Falls back to market orders if limit orders don't fill
     - Checks order fills and retries if needed
   - **Rolling Cycle Logic**:
     - First iteration: MOST tickers only → PRT → Open positions
     - Subsequent iterations: (Active positions from previous) + MOST → PRT → Maintain/close/open
   - **Key Methods**:
     - `get_position_details()`: Get position direction (LONG/SHORT) and quantity
     - `should_close_position()`: Determine if position should be closed based on PRT
     - `execute_cycle()`: Main strategy execution cycle

2. **`trade_tracker.py`**: TradeTracker class
   - Tracks trade entry times
   - Calculates trade ages
   - Identifies trades exceeding maximum hold time
   - Ensures trades don't exceed MAX_HOLD_TIME_MINUTES

3. **`position_sizing.py`**: PositionSizer class
   - Retrieves buying power from the `BuyingPower` tag in `AccountClient.get_account_info()`
   - Gets current market prices via `MarketDataClient.get_quote()`
   - Calculates position sizes based on:
     - Maximum position size constraint (MAX_POSITION_SIZE_DOLLARS)
     - Available buying power
   - Sizes trades to ensure each position is under max size limit

4. **`trade_executor.py`**: TradeExecutor class
   - Executes trades using limit orders
   - Gets bid/ask prices from a batched `Ticker` snapshot (falls back to last/close if bid/ask are thin)
   - Uses optimal pricing:
     - **Longs**: Uses bid price (or last/close if bid unavailable)
     - **Shorts**: Uses ask price (or last/close if ask unavailable)
   - **Order Fill Verification**:
     - Reads fill status straight off each order's live `trade.orderStatus` (no REST polling)
     - If not filled after a check, cancels the order and retries at a fresh price
     - Retries up to `MAX_FILL_CHECKS` times, waiting via `IBConnection.sleep()` (which pumps `ib_async`'s event loop) between checks
     - Falls back to a market order on the last attempt
   - Handles order execution with error handling

**Strategy Flow**:
```
Get tickers from MOST → Get open positions with directions → Combine tickers
→ Run PRT analysis → Compare PRT direction with position direction
→ Keep positions where PRT matches position direction
→ Close positions where PRT direction is opposite (using limit orders, fallback to market)
→ Filter new candidates (exclude tickers already in positions to avoid boxed positions)
→ Size positions (PositionSizer) → Execute trades (TradeExecutor)
→ Verify order fills → Retry unfilled orders with new prices
→ Track trade times → Close expired trades
→ Next iteration: Active positions feed into PRT analysis
```

### Monitoring Package (`monitoring/`)

**Purpose**: System monitoring and performance tracking.

**Modules**:
1. **`account_size.py`**: AccountSizeMonitor class
   - Periodically checks account size and metrics
   - Logs to CSV file (`account_size_history.csv`) for easy graphing
   - Tracks: account_value, buying_power, intraday_buying_power, cash_balance, equity, liquidation_value
   - Configurable check interval (ACCOUNT_SIZE_CHECK_INTERVAL_MINUTES)
   - CSV format with timestamp for easy visualization

**Key Methods**:
- `get_account_metrics()`: Retrieves current account metrics from `AccountClient.get_account_info()`
- `log_account_size()`: Appends account metrics to CSV file with timestamp

### Scripts Package (`scripts/`)

**Purpose**: Utility and startup scripts.

**Modules**:
1. **`startup.py`**: Main system startup and trading loop
   - Checks prerequisites (database)
   - Connects to IBKR (`IBConnection`, retries once after 5s on failure)
   - Initializes IBKR clients (AccountClient, MarketDataClient, OrderManager)
   - Initializes Godel Terminal controller
   - Initializes AccountSizeMonitor for periodic account tracking
   - Runs supervisor loop (when market closed)
   - Runs trading loop (when market open)
   - Executes rolling strategy cycles continuously
   - Monitors market hours and closes positions before market close
   - Logs account size periodically during trading
   - Handles cleanup on exit (positions closed, IBKR connection disconnected)

2. **`algo_loop.py`**: Alternative algorithm loop entry point
   - Similar functionality to startup.py (no AccountSizeMonitor)
   - Can be used as alternative entry point

3. **`close_all_positions.py`**: Emergency position closer
   - Standalone script for emergency closure
   - Cancels all open orders
   - Closes all positions with market orders
   - Includes confirmation prompts for safety

**Control Flow**:
```
startup.py → Check prerequisites → Initialize database → Connect to IBKR
→ Initialize IBKR clients → Initialize Godel controller
→ Supervisor loop (market closed) OR Trading loop (market open)
→ Rolling strategy cycles → Position sizing → Trade execution
→ Monitor market hours → Close positions before market close
→ Cleanup on exit
```

## Class Hierarchy

```
BaseCommand (ABC)
    ├── DESCommand
    ├── PRTCommand
    ├── MOSTCommand
    ├── GCommand
    ├── GIPCommand
    └── QMCommand

IBConnection
    ├── Owns: the shared ib_async.IB() instance
    └── Provides: connect(), ensure_connected(), disconnect(), sleep()

AccountClient
    ├── Uses: IBConnection
    └── Provides: get_positions(), get_account_info(), get_all_open_orders()

MarketDataClient
    ├── Uses: IBConnection
    └── Provides: get_quotes_batch(), get_quote(), is_market_open()

GodelTerminalController
    ├── Uses: DOMMonitor
    ├── Manages: BaseCommand instances
    └── Provides: execute_command()

RollingStrategy
    ├── Uses: GodelTerminalController
    ├── Uses: IBConnection
    ├── Uses: AccountClient
    ├── Uses: MarketDataClient
    ├── Uses: TradeTracker
    ├── Uses: OrderManager
    └── Provides: execute_cycle()

TradeTracker
    └── Provides: track trades, check expiration, track each position's stop-loss Trade

PositionSizer
    ├── Uses: AccountClient
    ├── Uses: MarketDataClient
    └── Provides: size_trades(), get_buying_power(), get_current_price()

TradeExecutor
    ├── Uses: IBConnection
    ├── Uses: AccountClient
    ├── Uses: MarketDataClient
    ├── Uses: OrderManager
    └── Provides: execute_trades(), execute_trade(), get_limit_price(), get_bid_ask_prices()

OrderManager
    ├── Uses: IBConnection
    └── Provides: create_market_order(), create_limit_order(), create_stop_loss_order(), cancel_order(), get_order_status()

AccountSizeMonitor
    ├── Uses: AccountClient
    └── Provides: log_account_size(), get_account_metrics()
```

## Dependencies

### External Dependencies
- `selenium`: Browser automation
- `webdriver-manager`: Chrome driver management
- `pandas`: Data manipulation
- `ib_async`: Interactive Brokers API client (order execution, positions, account values, market data)
- `loguru`: Logging
- `yfinance`: Market data (if needed)

### Internal Dependencies
- `constants.parameters` → Used by all modules
- `constants.blacklist` → Used by `strategy.rolling_strategy`
- `core.controller` → Used by `commands.*`, `strategy.rolling_strategy`
- `database.init` → Used by `scripts.startup`
- `ibkr.connection` → Used by `ibkr.account`, `ibkr.orders`, `ibkr.market_data`, `strategy.*`, `scripts.*`
- `ibkr.account` → Used by `strategy.*`, `monitoring.account_size`, `scripts.*`
- `ibkr.market_data` → Used by `strategy.*`, `scripts.*`
- `ibkr.orders` → Used by `strategy.trade_executor`, `strategy.rolling_strategy`, `scripts.*`

## Data Flow

### Command Execution Flow
```
Application → GodelTerminalController.execute_command()
  → BaseCommand.execute()
  → Controller.send_command()
  → DOMMonitor.get_new_window()
  → DOMMonitor.wait_for_loading()
  → BaseCommand.extract_data()
  → Return data to application
```

### Database Flow
```
Application → database.init.get_connection()
  → SQLite connection
  → Insert/query trades, PRT data, etc.
```

### Trading Flow
```
RollingStrategy.execute_cycle()
  → Get tickers from MOST
  → Get active positions with directions (get_position_details(), reads ib.positions() directly)
  → Combine and run PRT
  → Compare PRT direction with position direction
  → Keep positions where directions match
  → Close positions where directions are opposite
    → TradeExecutor.get_limit_price() for closing price
    → OrderManager.create_limit_order() (limit order first, action BUY/SELL - no BUY_TO_COVER)
    → Check order fills via trade.orderStatus (conn.sleep() pumps the event loop between checks) → Fallback to market order if needed
  → Filter new candidates (exclude existing positions)
  → PositionSizer.size_trades()
    → Get buying power (BuyingPower tag from accountSummary)
    → Get current prices (MarketDataClient.get_quote(), batched reqTickers under the hood)
    → Calculate position sizes
  → TradeExecutor.execute_trades()
    → Get bid/ask prices from a batched Ticker snapshot (fallback to last/close)
    → Determine limit prices (bid for longs, ask for shorts, or last/close)
    → OrderManager.create_limit_order()
    → Place orders via IBKR (ib_async)
    → Check order fills (trade.orderStatus.status, no network call)
    → Retry unfilled orders with new prices
    → Fallback to market orders if limit orders don't fill after MAX_FILL_CHECKS attempts
```

### Account Monitoring Flow
```
Trading loop → AccountSizeMonitor initialized
  → Every ACCOUNT_SIZE_CHECK_INTERVAL_MINUTES:
    → AccountSizeMonitor.get_account_metrics()
      → AccountClient.get_account_info() (ib.accountSummary() tags)
      → Extract account_value, buying_power, etc.
    → AccountSizeMonitor.log_account_size()
      → Append to account_size_history.csv
      → Format: timestamp, account_value, buying_power, intraday_buying_power, cash_balance, equity, liquidation_value
```

## Design Patterns

1. **Abstract Base Class**: `BaseCommand` - defines interface for Godel Terminal commands
2. **Factory Pattern**: `GodelTerminalController.register_command()` - registers command types
3. **Shared-connection Pattern**: `IBConnection` - owns the single `ib_async.IB()` socket; `AccountClient`/`OrderManager`/`MarketDataClient` are thin readers/writers over it rather than each managing their own connection

## Recent Updates

### Order Fill Verification
- TradeExecutor now verifies all orders are filled before reporting completion
- Checks order status after placement
- Retries unfilled orders with updated prices
- Handles "Order in state FILLED cannot be canceled" error correctly
- Falls back to market orders if limit orders don't fill after retries

### Boxed Position Prevention
- RollingStrategy now checks existing positions before opening new ones
- Compares PRT direction with current position direction
- Only opens positions for tickers not already in portfolio
- Prevents attempting to create boxed positions

### Position Direction Matching
- Compares PRT prediction direction (prob_up > 50% = LONG, < 50% = SHORT) with current position
- Keeps positions where PRT direction matches position direction
- Closes positions where PRT direction is opposite
- Uses limit orders first when closing, with market order fallback

### Interactive Brokers Migration
- Replaced the Schwab REST/OAuth broker layer (`schwab/`) with `ibkr/`, built on `ib_async`
- Positions, account values, and order status are read as live in-memory objects (`ib.positions()`, `ib.accountSummary()`, `trade.orderStatus`) instead of REST-polled - no per-call network round trip
- Batch quotes via a single `ib.reqTickers()` snapshot call instead of one REST call per symbol
- `IBConnection.sleep()` replaces `time.sleep()` everywhere the code waits on live IBKR state, so `ib_async`'s event loop keeps processing incoming messages during the wait
- IBKR has no separate `SELL_SHORT`/`BUY_TO_COVER` instruction - all order placement uses plain `BUY`/`SELL`
- Godel Terminal / PRT / MOST signal generation is unchanged

### Account Size Monitoring
- Monitoring package tracks account size periodically
- Logs to CSV file for easy graphing
- Tracks account value, buying power, cash balance, equity, etc. (sourced from `AccountClient.get_account_info()`)
- Configurable check interval

### Quant Framework (`quant/`)

**Purpose**: Institutional-style research stack (data → universe → features →
models → probability → sizing → risk → backtest → metrics → execution →
monitoring → journal → retraining) that complements the live Godel+IBKR bot.
Full component map and usage in `quant/README.md`.

**Key design decisions**:
- All tunables live in `constants/parameters.py` (QUANT_* section) per rule #4;
  `quant/config.py` resolves them with defaults and derives storage paths.
- Storage is Parquet + DuckDB under `quant_data/` (gitignored).
- Godel Terminal PRT `prob_up` is a first-class signal: historical rows are read
  from the bot's SQLite DB (`quant/data/godel_signals.py`) and blended with model
  probabilities in `quant/probability/engine.py` (QUANT_GODEL_BLEND_WEIGHT).
- Execution follows house rule #8: limit orders with bounded repricing, market
  escalation only afterwards and always logged; `SmartExecutor.reconcile()`
  flags any position without a working stop.
- Model lifecycle is challenger-vs-champion: retraining only promotes a new
  version if it beats the active one on embargoed walk-forward AUC; drift
  detection can auto-rollback (`quant/retrain/scheduler.py`).
- Paid-data capabilities (ticks, L2, options flow, IV, economic calendar,
  borrow) are explicit `NotImplementedError` stubs naming the provider needed.

**Control flow (research loop)**:
```
python -m quant.pipeline ingest    # yfinance -> parquet lake
python -m quant.pipeline universe  # liquidity/price/ADDV filters
python -m quant.pipeline train     # features -> boosted model -> walk-forward -> registry
python -m quant.pipeline backtest  # event-driven sim w/ costs (in-sample warning printed)
python -m quant.pipeline report    # metrics scorecard
```

## Future Enhancements

1. **Database Manager**: Add database manager class for trade operations
2. **Error Handling**: Enhanced error handling and recovery
3. **Live wiring**: Drive `strategy/rolling_strategy.py` entries through
   `quant.probability` blended estimates and `quant.execution.SmartExecutor`

## Maintenance Notes

- This document should be updated after every significant architectural change
- When adding new modules, update the directory structure section
- When adding new classes, update the class hierarchy section
- When changing control flow, update the relevant control flow sections

