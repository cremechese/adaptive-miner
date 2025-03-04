import numpy as np
import pandas as pd
import bittensor as bt
import os
from datetime import datetime, timedelta
import requests

async def get_historical_prices(asset="BTC", lookback_minutes=60):
    """
    Retrieves historical price data from the local price file.
    
    Args:
        asset (str): Asset symbol (only BTC is supported)
        lookback_minutes (int): How far back to fetch data in minutes
        
    Returns:
        numpy.ndarray: Array of historical close prices
    """
    if asset != "BTC":
        bt.logging.warning(f"Asset '{asset}' not supported")
        return None
        
    try:
        # Path to the price file
        price_file = "/root/fetch_btc_price/btc_price_minute_new.txt"
        
        # Check if file exists
        if not os.path.exists(price_file):
            bt.logging.error(f"Price file not found: {price_file}")
            return None
            
        # Read the file into a pandas DataFrame
        # Format: timestamp open high low close
        df = pd.read_csv(price_file, sep=' ', header=None, 
                         names=['timestamp', 'open', 'high', 'low', 'close'])
        
        # Convert timestamps to datetime objects
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # Sort by datetime in descending order (newest first)
        df = df.sort_values('datetime', ascending=False)
        
        # Get current time
        current_time = datetime.now()
        
        # Calculate cutoff time
        cutoff_time = current_time - timedelta(minutes=lookback_minutes)
        
        # Filter data newer than cutoff time
        recent_data = df[df['datetime'] >= cutoff_time]
        
        # If we don't have enough recent data, take the most recent available
        if len(recent_data) < 30:
            recent_data = df.head(40)  # Take at least 40 points
            
        # Extract close prices
        close_prices = recent_data['close'].values
        
        # Ensure we have the prices in chronological order (oldest first)
        close_prices = close_prices[::-1]
        
        bt.logging.info(f"Retrieved {len(close_prices)} historical price points")
        return close_prices
        
    except Exception as e:
        bt.logging.error(f"Error fetching historical prices from file: {e}")
        return None

def get_asset_price(asset="BTC"):
    """
    Retrieves the current price of the specified asset.
    Currently, supports BTC via Pyth Network.

    Returns:
        float: Current asset price.
    """
    if asset == "BTC":
        btc_price_id = (
            "e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43"
        )
        endpoint = f"https://hermes.pyth.network/api/latest_price_feeds?ids[]={btc_price_id}"  # TODO: this endpoint is deprecated
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            if not data or len(data) == 0:
                raise ValueError("No price data received")
            price_feed = data[0]
            price = float(price_feed["price"]["price"]) / (10**8)
            return price
        except Exception as e:
            print(f"Error fetching {asset} price: {str(e)}")
            return None
    else:
        # For other assets, implement accordingly
        print(f"Asset '{asset}' not supported.")
        return None

def simulate_single_price_path(
    current_price, time_increment, time_length, sigma
):
    """
    Simulate a single crypto asset price path.
    """
    one_hour = 3600
    dt = time_increment / one_hour
    num_steps = int(time_length / time_increment)
    std_dev = sigma * np.sqrt(dt)
    price_change_pcts = np.random.normal(0, std_dev, size=num_steps)
    cumulative_returns = np.cumprod(1 + price_change_pcts)
    cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    price_path = current_price * cumulative_returns
    return price_path


def simulate_crypto_price_paths(
    current_price, time_increment, time_length, num_simulations, sigma
):
    """
    Simulate multiple crypto asset price paths.
    """

    price_paths = []
    for _ in range(num_simulations):
        price_path = simulate_single_price_path(
            current_price, time_increment, time_length, sigma
        )
        price_paths.append(price_path)

    return np.array(price_paths)

def simulate_single_price_path_adaptive(
    current_price,
    time_increment,
    time_length,
    price_history=None,
    base_scale=3.0,
    volatility_window=30,
    volatility_multiplier=1.0,
    mean_reversion=0.01,
    fat_tails=True
):
    """
    Simulate a single crypto asset price path using an adaptive random walk approach.
    
    Args:
        current_price: Starting price for simulation
        time_increment: Time step in seconds
        time_length: Total time to simulate in seconds
        price_history: Historical price data for volatility calculation (optional)
        base_scale: Base scale parameter for random noise
        volatility_window: Window size for calculating recent volatility
        volatility_multiplier: How strongly to adjust based on recent volatility
        mean_reversion: Strength of mean reversion to recent average
        fat_tails: Whether to use Student's t-distribution for fat tails
        
    Returns:
        numpy.ndarray: Array of simulated prices
    """
    
    num_steps = int(time_length / time_increment)

    price_path = np.zeros(num_steps + 1)
    price_path[0] = current_price

    if price_history is not None and len(price_history) > 1:
        if volatility_window > (len(price_history) - 1):
            bt.logging.warning("price history is shorter than volatility window, inaccurate predictions may occur.")
            bt.logging.warning(f"window_size = {window_size}, volatility_window = {volatility_window}")
        
        window_size = min(volatility_window, len(price_history) - 1)

        recent_changes = np.diff(price_history[-window_size-1:])

        recent_volatility = np.std(recent_changes)
        adjusted_scale = base_scale * (1.0 + volatility_multiplier * recent_volatility / base_scale)

        recent_avg = np.mean(price_history[-window_size:])
    else:
        bt.logging.warning("price_history is None or only has one data point. inaccurate predictions may occur.")
        adjusted_scale = base_scale
        recent_avg = current_price

    for i in range(1, num_steps + 1):
        reversion = mean_reversion * (recent_avg - price_path[i-1])

        if fat_tails:
            shock = np.random.standard_t(df=5) * adjusted_scale / np.sqrt(3)
        else:
            shock = np.random(0, adjusted_scale)
        
        price_path[i] = price_path[i-1] + reversion + shock
    
    price_path = np.round(price_path, 8)

    return price_path

def simulate_crypto_price_paths_adaptive(
        current_price,
        time_increment,
        time_length,
        num_simulations,
        price_history=None,
        base_scale=3.0,
        volatility_window=30,
        volatility_multiplier=1.0,
        mean_reversion=0.01,
        fat_tails=True,
): 
    """
    Simulate multiple crypto asset price paths using adaptive random walk approach.
    
    Args:
        current_price: Starting price for simulation
        time_increment: Time step in seconds
        time_length: Total time to simulate in seconds
        num_simulations: Number of paths to generate
        price_history: Historical price data for volatility calculation (optional)
        base_scale: Base scale parameter for random noise
        volatility_window: Window size for calculating recent volatility
        volatility_multiplier: How strongly to adjust based on recent volatility
        mean_reversion: Strength of mean reversion to recent average
        fat_tails: Whether to use Student's t-distribution for fat tails
        
    Returns:
        numpy.ndarray: Array of simulated price paths
    """
    price_paths = []
    for _ in range(num_simulations):
        price_path = simulate_single_price_path_adaptive(
            current_price,
            time_increment,
            time_length,
            price_history,
            base_scale,
            volatility_window,
            volatility_multiplier,
            mean_reversion,
            fat_tails
        )
        price_paths.append(price_path)

    return np.array(price_paths)