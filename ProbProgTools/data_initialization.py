import os
import numpy as np
from jax import random,jit,vmap
from scipy.stats import levy_stable
import time
import jax
import jax.numpy as jnp
import pandas as pd
from jax.scipy.linalg import cholesky
from jax.scipy.stats import norm
from typing import Dict, List, Tuple
from functools import partial 

class StockMarketSimulator:
    """
    A simulator for stock market prices using Levy processes with JAX for computation and
    Pareto-distributed initial prices using NumPy.

    Attributes:
        n_industries (int): Number of industries.
        n_stocks_per_industry (int): Number of stocks per industry.
        base_stock_price (float): Base stock price for scaling initial prices.
        industries (List[str]): List of industry names.
        stocks (List[str]): List of stock symbols.
        stock_prices (pd.DataFrame): DataFrame to store simulated stock prices.
        seed (int): Int derived through os for true random state
        key (jax.random.PRNGKey): JAX PRNG key for random number generation.
        industry_map (Dict[str, str]): Mapping of stocks to their respective industries.
        alpha_params (Dict[str, float]): Alpha parameter for each industry.
        beta_params (Dict[str, float]): Beta parameter for each industry.
        pareto_shapes (Dict[str, float]): Pareto shape parameter for each industry.
    """

    def __init__(self, n_industries: int = 8, n_stocks_per_industry: int = 10, base_stock_price: float = 100) -> None:
        """
        Initialize the stock market simulator with specified parameters.

        Args:
            n_industries (int): The number of industries to simulate.
            n_stocks_per_industry (int): The number of stocks per industry.
            base_stock_price (float): The base stock price for scaling initial prices.
        """
        self.n_industries: int = n_industries
        self.n_stocks_per_industry: int = n_stocks_per_industry
        self.n_stocks: int = self.n_industries*self.n_stocks_per_industry
        self.base_stock_price: float = base_stock_price
        self.industries: List[str] = ["Energy", "Materials", "Industrials", "Consumer Discretionary",
                                      "Consumer Staples", "Health Care", "Financials", "Information Technology"]
        self.stocks: List[str] = list(set([f"{industry[0]}{i + 1}" for industry in self.industries for i in range(n_stocks_per_industry)]))
        self.stock_prices: pd.DataFrame = pd.DataFrame()
        self.seed: int = int.from_bytes(os.urandom(4), 'big')
        self.key: jax.random.PRNGKey = jax.random.PRNGKey(self.seed)
        self.industry_map: Dict[str, str] = {stock: industry for industry in self.industries for stock in self.stocks if stock.startswith(industry[0])}
        self.keys = random.split(self.key, num=len(self.industries) * 3)  # Split the key for parameters
        self.alpha_params = {industry: 1.75 for industry in self.industries}  # Example alpha parameters
        self.beta_params = {industry: 0.5 for industry in self.industries}  # Example beta parameters
        self.pareto_shapes = {industry: random.uniform(self.keys[i * 3 + 2], minval=1.5, maxval=3, shape=()) for i, industry in enumerate(self.industries)}
        self._initialize_prices()

    def _initialize_prices(self) -> None:
        """Initialize the stock prices using Pareto distribution for each industry."""
        np.random.seed(self.seed)
        initial_prices: List[float] = []

        for industry in self.industries:
            industry_stocks = [stock for stock in self.stocks if self.industry_map[stock] == industry]
            if not industry_stocks:
                continue  # Skip if there are no stocks for this industry

            pareto_shape = self.pareto_shapes[industry]
            industry_market_caps = np.random.pareto(pareto_shape, len(industry_stocks)) + 1
            min_market_cap = np.min(industry_market_caps) if industry_market_caps.size > 0 else 1
            scaled_prices = (industry_market_caps / min_market_cap) * self.base_stock_price
            initial_prices.extend(scaled_prices)

        self.stock_prices = pd.DataFrame([initial_prices], columns=self.stocks)

    def simulate_stock_prices(self, n_days: int = 252) -> pd.DataFrame:
        """
        Simulate stock prices over a given number of days.

        Args:
            n_days (int): The number of days to simulate stock prices.

        Returns:
            pd.DataFrame: A DataFrame containing the simulated stock prices.
        """
        for day in range(1, n_days):
            try:
                day_prices = self.stock_prices.iloc[-1].values.copy()
                
                all_increments: List[np.ndarray] = []
                industry_indices: List[int] = []
                
                for industry in self.industries:
                    industry_stocks = [stock for stock in self.stocks if self.industry_map[stock] == industry]
                    industry_alpha = self.alpha_params[industry]
                    industry_beta = self.beta_params[industry]

                    np.random.seed(day)
                    levy_increments = levy_stable.rvs(industry_alpha, industry_beta, size=len(industry_stocks), random_state=np.random.RandomState())
                    
                    all_increments.append(levy_increments)
                    industry_indices.extend([self.stocks.index(stock) for stock in industry_stocks])
                
                all_increments = np.concatenate(all_increments)
                correlated_increments = self._apply_correlation(all_increments)
                
                day_prices[industry_indices] *= (1 + correlated_increments / 100)
                
                self.stock_prices = pd.concat([self.stock_prices, pd.DataFrame([day_prices], columns=self.stocks)], ignore_index=True)

            except Exception as e:
                print(f"An error occurred on day {day}: {e}")
                continue

        return self.stock_prices

    def _apply_correlation(self, increments: np.ndarray) -> np.ndarray:
        """
        Apply a correlation matrix to the increments using Cholesky decomposition.

        Args:
            increments (np.ndarray): An array of increments to apply correlation to.

        Returns:
            np.ndarray: Correlated increments after applying the correlation matrix.
        """
        intra_industry_corr = 0.7
        inter_industry_corr = 0.3
        corr_matrix = np.full((len(increments), len(increments)), inter_industry_corr)
        for i in range(self.n_industries):
            start_idx = i * self.n_stocks_per_industry
            end_idx = (i + 1) * self.n_stocks_per_industry
            corr_matrix[start_idx:end_idx, start_idx:end_idx] = intra_industry_corr
        np.fill_diagonal(corr_matrix, 1)
        cholesky_decomp = jnp.linalg.cholesky(jnp.array(corr_matrix))
        correlated_increments = jnp.dot(cholesky_decomp, jnp.array(increments))
        return np.array(correlated_increments)

    #################################################################################################################
    #                                               JAX IMPLEMENTATION (in test)                                    #
    #################################################################################################################
    
    # def create_correlation_matrix(self, n_stocks: int, n_industries: int, n_stocks_per_industry: int) -> jnp.array:
    #     """
    #     Create a correlation matrix for the stock market simulation.

    #     Args:
    #         n_stocks (int): Total number of stocks.
    #         n_industries (int): Number of industries.
    #         n_stocks_per_industry (int): Number of stocks per industry.

    #     Returns:
    #         jnp.array: The correlation matrix.
    #     """
    #     intra_industry_corr = 0.7
    #     inter_industry_corr = 0.3
    #     corr_matrix = jnp.full((n_stocks, n_stocks), inter_industry_corr)

    #     # Efficiently set intra-industry correlations
    #     for i in range(n_industries):
    #         start_idx = i * n_stocks_per_industry
    #         end_idx = start_idx + n_stocks_per_industry
    #         corr_matrix = corr_matrix.at[start_idx:end_idx, start_idx:end_idx].set(intra_industry_corr)

    #     # Correctly fill the diagonal without in-place modification
    #     diag_indices = jnp.arange(n_stocks)
    #     corr_matrix = corr_matrix.at[diag_indices, diag_indices].set(1.0)

    #     return corr_matrix


    # @staticmethod
    # def apply_correlation(increments: jnp.array, corr_matrix: jnp.array) -> jnp.array:
    #     """
    #     Apply the correlation matrix to increments.

    #     Args:
    #         increments (jnp.array): Stock increments.
    #         corr_matrix (jnp.array): Correlation matrix.

    #     Returns:
    #         jnp.array: Correlated stock increments.
    #     """
    #     assert corr_matrix.ndim == 2, "corr_matrix must be two-dimensional"
    #     assert corr_matrix.shape[0] == corr_matrix.shape[1], "corr_matrix must be square"

    #     cholesky_decomp = cholesky(corr_matrix)
    #     return jnp.dot(cholesky_decomp, increments)

    # @staticmethod
    # def simulate_stock_price_change(alpha: float, beta: float, key: jnp.array) -> float:
    #     """
    #     Simulate a single stock price change using Levy distribution.

    #     Args:
    #         alpha (float): Alpha parameter for the Levy distribution.
    #         beta (float): Beta parameter for the Levy distribution.
    #         key (jnp.array): JAX random key.

    #     Returns:
    #         float: Simulated stock price change.
    #     """
    #     # Here we would use JAX's random generation if it supported Levy stable directly.
    #     # As a placeholder, let's return a normal distribution sample for the sake of example.
    #     return random.normal(key)  # Placeholder for Levy stable distribution

    # def simulate_stock_prices(self, n_days: int = 252) -> pd.DataFrame:
    #     """
    #     Simulate stock prices over a given number of days using Levy distribution.

    #     Args:
    #         n_days (int): The number of days to simulate stock prices.

    #     Returns:
    #         pd.DataFrame: A DataFrame containing the simulated stock prices.
    #     """
    #     for day in range(1, n_days):
    #         try:
    #             day_prices = self.stock_prices.iloc[-1].values.copy()
    #             all_increments = []

    #             for industry in self.industries:
    #                 industry_stocks = [stock for stock in self.stocks if self.industry_map[stock] == industry]
    #                 industry_alpha = self.alpha_params[industry]
    #                 industry_beta = self.beta_params[industry]

    #                 # Use SciPy to generate Levy increments
    #                 increments = levy_stable.rvs(alpha=industry_alpha, beta=industry_beta, size=len(industry_stocks))
    #                 increments_jax = jnp.array(increments, dtype=jnp.float32)  # Specify a numeric dtype explicitly if needed

    #                 all_increments.append(increments_jax)

    #             all_increments_jax = jnp.concatenate(all_increments)
    #             corr_matrix = self.create_correlation_matrix(self.n_stocks, self.n_industries, self.n_stocks_per_industry)
    #             correlated_increments = self.apply_correlation(all_increments_jax,corr_matrix)

    #             # Apply the correlated increments to update day_prices
    #             day_prices *= (1 + correlated_increments / 100)

    #             # Update the stock_prices DataFrame
    #             self.stock_prices = pd.concat([self.stock_prices, pd.DataFrame([day_prices], columns=self.stocks)], ignore_index=True)

    #         except Exception as e:
    #             print(f"An error occurred on day {day}: {e}")
    #             continue

    #     return self.stock_prices




def main() -> None:
    """Main function to initialize and run the stock market simulation."""
    start = time.time()
    simulator = StockMarketSimulator()
    # Assuming n_stocks = n_industries * n_stocks_per_industry
    n_stocks = simulator.n_industries * simulator.n_stocks_per_industry

    # Ensure alpha_params and beta_params are correctly sized
    alpha_params = jnp.ones(n_stocks) * 1.5  # Example: All stocks have an alpha of 1.5
    beta_params = jnp.ones(n_stocks) * 0.5   # Example: All stocks have a beta of 0.5
    base_prices = jnp.ones(n_stocks) * 100   # Initial prices for all stocks

    # Pass the correctly sized parameters to the simulation
    simulated_data = simulator.simulate_stock_prices()
    #     252,  # Number of days to simulate
    #     n_stocks,  # Total number of stocks
    #     alpha_params,  # Alpha parameters for each stock
    #     beta_params,  # Beta parameters for each stock
    #     base_prices,  # Initial prices of stocks
    #     simulator.n_industries,  # Number of industries
    #     simulator.n_stocks_per_industry,  # Number of stocks per industry
    #     jax.random.PRNGKey(0)  # Random key
    # 

    end = time.time()
    print(simulated_data)
    print(end-start)

if __name__ == "__main__":
    main()
