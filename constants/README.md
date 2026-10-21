# Configuration Setup

## Setting up parameters.py

1. Copy the template file:
   ```bash
   cp constants/parameters.py.template constants/parameters.py
   ```

2. Edit `constants/parameters.py` and fill in your actual values:
   - `GODEL_USERNAME`: Your Godel Terminal email
   - `GODEL_PASSWORD`: Your Godel Terminal password
   - `IBKR_HOST`/`IBKR_PORT`/`IBKR_CLIENT_ID`: Your TWS/IB Gateway connection settings
   - `PAPER_TRADING`: Must match `IBKR_PORT` (paper ports: 7497/4002, live ports: 7496/4001)

3. The `parameters.py` file is gitignored and will not be committed to the repository.

## Security Note

Never commit `parameters.py` to version control. It contains your Godel Terminal credentials.

