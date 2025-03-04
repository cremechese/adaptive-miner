from synth.miner.price_simulation import (
    simulate_crypto_price_paths,
    get_asset_price,
    simulate_crypto_price_paths_adaptive,
    get_historical_prices
)
from synth.utils.helpers import (
    convert_prices_to_time_format,
)

async def generate_simulations(
    start_time=None,
    asset="BTC",
    time_increment=300,
    time_length=86400,
    num_simulations=100,
    sigma=0.01,
    model="adaptive",
    price_history=None,
    base_scale=3.0,
    volatility_window=30,
    volatility_multiplier=1.0,
    mean_reversion=0.01,
    fat_tails=True
):
    """
    Generate simulated price paths.

    Parameters:
        start_time (str): The start time of the simulation. Defaults to current time.
        asset (str): The asset to simulate. Default is 'BTC'.
        time_increment (int): Time increment in seconds.
        time_length (int): Total time length in seconds.
        num_simulations (int): Number of simulation runs.
        sigma (float): Standard deviation for legacy GBM model.
        model (str): Simulation model to use ('gbm' or 'adaptive').
        price_history (numpy.ndarray): Historical price data for adaptive model.
        base_scale (float): Base scale parameter for adaptive model.
        volatility_window (int): Window size for calculating recent volatility.
        volatility_multiplier (float): How strongly to adjust based on recent volatility.
        mean_reversion (float): Strength of mean reversion.
        fat_tails (bool): Whether to use Student's t-distribution for fat tails.

    Returns:
        list: List of simulated price paths with timestamps.
    """
    if start_time is None:
        raise ValueError("Start time must be provided.")
    
    current_price = get_asset_price(asset)
    if current_price is None:
        raise ValueError(f"Failed to fetch current price for asset: {asset}")
    
    price_history = await get_historical_prices()

    if model.lower() == "adaptive":
        simulations = simulate_crypto_price_paths_adaptive(
            current_price=current_price,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            price_history=price_history,
            base_scale=base_scale,
            volatility_window=volatility_window,
            volatility_multiplier=volatility_multiplier,
            mean_reversion=mean_reversion,
            fat_tails=fat_tails
        )
    else: 
        simulations = simulate_crypto_price_paths(
            current_price=current_price,
            time_increment=time_increment,
            time_length=time_length,
            num_simulations=num_simulations,
            sigma=sigma
        )

    predictions = convert_prices_to_time_format (
        simulations.tolist(), start_time, time_increment
    )

    return predictions