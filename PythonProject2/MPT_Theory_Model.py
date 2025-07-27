import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


def run_mpt_simulation(tickers, start_date, end_date, num_simulations, risk_free_rate):
    """
    Performs a Markowitz Portfolio Optimization using Monte Carlo simulation.
    """
    # --- 1. Download Historical Data ---
    with st.spinner(f"Downloading data for {len(tickers)} stocks..."):
        adj_close_df = pd.DataFrame()
        for ticker in tickers:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            adj_close_df[ticker] = data['Close']

    returns = adj_close_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # --- 2. Run Monte Carlo Simulation ---
    with st.spinner(f"Running {num_simulations} simulations..."):
        results = np.zeros((3 + len(tickers), num_simulations))

        for i in range(num_simulations):
            weights = np.random.random(len(tickers))
            weights /= np.sum(weights)

            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

            results[0, i] = portfolio_return
            results[1, i] = portfolio_volatility
            results[2, i] = sharpe_ratio
            for j in range(len(weights)):
                results[j + 3, i] = weights[j]

    return pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe'] + tickers)


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")

st.title('Markowitz Portfolio Optimization Dashboard')

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header('Configuration')

    tickers_string = st.text_area(
        'Stock Tickers (comma-separated)',
        'MSFT, AAPL, UNH, JNJ, JPM, V, AMZN, TSLA, GOOGL, META, CAT, RTX, PG, WMT, XOM, CVX, NEE, DUK'
    )

    start_date = st.date_input('Start Date', pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End Date', pd.to_datetime('2024-12-31'))

    num_simulations = st.slider('Number of Simulations', 1000, 50000, 20000)
    risk_free_rate = st.slider('Risk-Free Rate (%)', 0.0, 5.0, 2.0) / 100

    run_button = st.button('Run Optimization')

# --- Main Panel for Outputs ---
if run_button:
    tickers = [ticker.strip().upper() for ticker in tickers_string.split(',')]

    results_df = run_mpt_simulation(tickers, start_date, end_date, num_simulations, risk_free_rate)

    # --- Identify Optimal Portfolios ---
    max_sharpe_portfolio = results_df.iloc[results_df['Sharpe'].idxmax()]
    min_volatility_portfolio = results_df.iloc[results_df['Volatility'].idxmin()]

    st.header('Optimal Portfolio Allocations')

    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Max Sharpe Ratio Portfolio')
        st.write(f"**Annual Return:** {max_sharpe_portfolio['Return'] * 100:.2f}%")
        st.write(f"**Annual Volatility:** {max_sharpe_portfolio['Volatility'] * 100:.2f}%")
        st.write(f"**Sharpe Ratio:** {max_sharpe_portfolio['Sharpe']:.2f}")

        allocation_df = max_sharpe_portfolio.drop(['Return', 'Volatility', 'Sharpe']).sort_values(ascending=False)
        st.dataframe(allocation_df[allocation_df > 0].apply(lambda x: f"{x * 100:.2f}%"),
                     column_config={"value": "Allocation"})

    with col2:
        st.subheader('Minimum Volatility Portfolio')
        st.write(f"**Annual Return:** {min_volatility_portfolio['Return'] * 100:.2f}%")
        st.write(f"**Annual Volatility:** {min_volatility_portfolio['Volatility'] * 100:.2f}%")
        st.write(f"**Sharpe Ratio:** {min_volatility_portfolio['Sharpe']:.2f}")

        allocation_df_min_vol = min_volatility_portfolio.drop(['Return', 'Volatility', 'Sharpe']).sort_values(
            ascending=False)
        st.dataframe(allocation_df_min_vol[allocation_df_min_vol > 0].apply(lambda x: f"{x * 100:.2f}%"),
                     column_config={"value": "Allocation"})

    st.header('Efficient Frontier')

    # --- Create Visualization ---
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(results_df.Volatility, results_df.Return, c=results_df.Sharpe, cmap='viridis')
    plt.colorbar(scatter, label='Sharpe Ratio')

    ax.scatter(max_sharpe_portfolio.Volatility, max_sharpe_portfolio.Return, marker='*', color='r', s=500,
               label='Max Sharpe Ratio')
    ax.scatter(min_volatility_portfolio.Volatility, min_volatility_portfolio.Return, marker='*', color='orange', s=500,
               label='Min Volatility')

    ax.set_title('Monte Carlo Simulation of Portfolios')
    ax.set_xlabel('Annualized Volatility (Risk)')
    ax.set_ylabel('Annualized Return')
    ax.legend(labelspacing=0.8)

    st.pyplot(fig)