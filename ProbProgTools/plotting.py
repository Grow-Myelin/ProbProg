import os
import matplotlib.pyplot as plt
import seaborn as sns
import data_initialization as DI


def plot_stock_prices(simulator: DI.StockMarketSimulator) -> None:
    """
    Plot the simulated stock prices for all stocks in the simulation.

    Args:
        simulator (DI.StockMarketSimulator): An instance of the StockMarketSimulator class.

    This function generates a line plot for each stock across the simulated days and saves the plot
    to a file named 'simulated_stock_prices.png' in the 'plots' directory.
    """
    plt.figure(figsize=(15, 8))
    plt.clf()
    plotted_stocks = set()
    for stock in simulator.stocks:
        if stock not in plotted_stocks:
            plt.plot(simulator.stock_prices.index, simulator.stock_prices[stock], label=stock)
            plotted_stocks.add(stock)
    plt.title('Simulated Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Ensure directory exists
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/simulated_stock_prices.png')


def plot_stock_prices_by_industry(simulator: DI.StockMarketSimulator) -> None:
    """
    Plot the simulated stock prices for each industry separately.

    Args:
        simulator (DI.StockMarketSimulator): An instance of the StockMarketSimulator class.

    This function generates a line plot for each stock within an industry across the simulated days and saves
    each industry's plot to a separate file in the 'plots' directory, named 'simulated_stock_price_by_[industry].png'.
    """
    industries = set(simulator.industry_map.values())
    for industry in industries:
        plt.figure(figsize=(15, 8))
        plt.clf()
        industry_stocks = [stock for stock, ind in simulator.industry_map.items() if ind == industry]
        plotted_stocks = set()
        for stock in industry_stocks:
            if stock not in plotted_stocks:
                plt.plot(simulator.stock_prices.index, simulator.stock_prices[stock], label=stock)
                plotted_stocks.add(stock)
        plt.title(f'Simulated Stock Prices for {industry}')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        # Ensure directory exists
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/simulated_stock_price_by_{industry}.png')


def main() -> None:
    """
    Main function to initialize the simulator, run the stock price simulation, and plot the results.
    """
    simulator = DI.StockMarketSimulator()
    simulator.simulate_stock_prices()
    plot_stock_prices(simulator)
    plot_stock_prices_by_industry(simulator)


if __name__ == "__main__":
    main()
