# Algobot v2 — RESEARCH LAB COPY

> ## ⚠️ THIS IS THE EXPERIMENTAL LAB (`V4.2algobot-master-lab`)
>
> **Do not use this folder for live or paper observation.** The frozen
> paper-observation system of record is `V4.1algobot-master` (no `-lab`) —
> its scheduled task, journal, and paper reports must never point here.
> This lab exists for research experiments only (see
> `LAB_RESEARCH_ROADMAP.md`). Its `quant_data/` is research scratch, not
> observation evidence. `scripts/setup_btc_daily_task.ps1` refuses to run
> here (LAB_MODE guard). Nothing in this folder may be promoted without a
> fresh out-of-sample judge. Live trading is disabled, same as the original.

A modular, maintainable trading algorithm system that interfaces with Godel Terminal for signals and Interactive Brokers (via `ib_async`) for execution.

## Overview

Algobot v2 is a complete redesign of the original godel_api, built with maintainability and organization as core principles. The system is structured into clear, modular components that are easy to understand and maintain.

## Features

- **Modular Architecture**: Clear separation of concerns with dedicated packages
- **Interactive Brokers Execution**: Event-driven order placement and account/position tracking via `ib_async` - no REST polling, state is kept live by IBKR's own event loop
- **Database**: Local SQLite database for trade and PRT data storage
- **Godel Terminal Integration**: Framework for executing Godel Terminal commands (MOST/PRT screening)
- **Maintainable Code**: Well-organized, documented, and following best practices

## Quick Start

### 1. Set Up Virtual Environment

```bash
cd algobot_v2
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize System

```bash
python scripts/startup.py
```

This will:
- Initialize the database (if needed)
- Connect to IBKR (TWS or IB Gateway must already be running with API access enabled)
- Connect to Godel Terminal
- Prepare the system for use

## Project Structure

```
algobot_v2/
├── constants/      # Global parameters and configuration
├── ibkr/           # Interactive Brokers integration (connection, account, orders, market data)
├── database/       # Database initialization and schema
├── core/           # Core framework (Godel Terminal controller)
├── commands/       # Godel Terminal command implementations (DES, PRT, MOST, ...)
├── strategy/       # Rolling PRT strategy, position sizing, trade execution, trade tracking
├── monitoring/     # Account size monitoring/logging
└── scripts/        # Startup and utility scripts
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Configuration

All configuration values are in `constants/parameters.py`. Key settings include:

- Godel Terminal credentials
- IBKR connection settings (host, port, client ID, paper/live flag)
- Database settings
- Trading parameters (stop loss, max positions, position sizing, etc.)

## Usage

### Interactive Brokers

**Connect and read account state**:
```python
from ibkr.connection import IBConnection
from ibkr.account import AccountClient

conn = IBConnection()
conn.connect()
account_client = AccountClient(conn)
positions = account_client.get_positions()
```

### Database

**Initialize database**:
```python
from database.init import init_database
init_database()
```

**Get database connection**:
```python
from database.init import get_connection
conn = get_connection()
```

### Godel Terminal

**Use the controller**:
```python
from core.controller import GodelTerminalController
from constants.parameters import GODEL_USERNAME, GODEL_PASSWORD

controller = GodelTerminalController()
controller.connect()
controller.login(GODEL_USERNAME, GODEL_PASSWORD)
controller.load_layout("dev")
controller.open_terminal()

# Execute commands (once commands are implemented)
# result, cmd = controller.execute_command("DES", "AAPL", "EQ")
```

## Architecture Rules

This project follows strict architecture rules defined in [RULES.md](RULES.md):

1. **Organization**: Related components grouped in folders
2. **Modularity**: Separate files for different tasks
3. **Documentation**: Architecture markdown maintained after changes
4. **Constants**: All parameters in dedicated constants file

## Development

### Adding New Commands

1. Create a new command class in `commands/`
2. Inherit from `core.controller.BaseCommand`
3. Implement `get_command_string()` and `extract_data()`
4. Register with controller: `controller.register_command("COMMAND", CommandClass)`

### Adding New Features

1. Follow the modular structure
2. Update `ARCHITECTURE.md` after changes
3. Keep constants in `constants/parameters.py`
4. Maintain separation of concerns

## Requirements

- Python 3.8+
- Chrome browser (for Selenium)
- ChromeDriver (managed by webdriver-manager)
- TWS or IB Gateway running locally with API access enabled (paper trading by default - see `IBKR_PORT` in `constants/parameters.py`)

## License

[Your License Here]

