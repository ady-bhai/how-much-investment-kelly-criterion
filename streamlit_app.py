import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm
from alpha_vantage.timeseries import TimeSeries

# Set up the Alpha Vantage API
API_KEY = '9R87QX5PYLTNVJI1'  # Replace with your API key
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Function to get current price of a stock
def get_current_price(symbol):
    data, meta_data = ts.get_quote_endpoint(symbol=symbol)
    return float(data['05. price'][0])

# Function to calculate the probability distribution and Kelly criterion
def calculate_distribution(curr_price, mus_stds, weights, max_price, min_price=0, resolution=10000, bins=100):
    x = np.linspace(min_price, max_price, resolution)
    y = np.zeros(resolution)
    weights = np.asarray(weights) / sum(weights)
    
    for mu_std, weight in zip(mus_stds, weights):
        mu, std = mu_std
        temp = norm.pdf(x, mu, std)
        y += temp * weight

    y = y / sum(y)

    bar_centers = np.linspace(min_price, max_price, bins)
    y_bars = np.asarray(np.split(y, bins)).sum(axis=1)

    # Kelly Criterion Calculation
    win_mask = x > curr_price
    win_prob = sum(y[win_mask])
    normalized_win_probabilities = y[win_mask] / sum(y[win_mask])
    expected_win = sum((x[win_mask] - curr_price) * normalized_win_probabilities)
    win_fraction = expected_win / curr_price

    loss_mask = x <= curr_price
    loss_prob = sum(y[loss_mask])
    normalized_loss_probabilities = y[loss_mask] / sum(y[loss_mask])
    expected_loss = sum((x[loss_mask] - curr_price) * normalized_loss_probabilities)
    loss_fraction = -(expected_loss / curr_price)

    f_star = (win_prob / loss_fraction) - (loss_prob / win_fraction)
    
    return x, y, bar_centers, y_bars, f_star, win_prob, expected_win, loss_prob, expected_loss

# Streamlit app UI components
st.title('Stock Price Distribution and Kelly Criterion Calculator')

st.markdown("""
This app uses a **Gaussian Mixture Model** to predict the probability distribution of a stock's future price based on user-defined parameters. 
It then calculates the **Kelly Criterion**, which helps determine the optimal fraction of your net worth to invest in the stock, maximizing long-term growth while managing risk.
""")


# Sidebar inputs for user interaction
st.sidebar.header("Stock Parameters")

symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., 'AAPL')", 'AAPL')

# Fetch current stock price using Alpha Vantage API
curr_price = get_current_price(symbol)
st.sidebar.write(f"Current price of {symbol}: ${curr_price:.2f}")

# Slider for the maximum price range (using the sidebar)
max_price = st.sidebar.slider('Select Maximum Price Range', 
                              min_value=float(curr_price), 
                              max_value=1000.0,  # Ensure this is a float
                              value=500.0,  # Make sure the default value is also a float
                              step=1.0)  # Use a float for step size

# Input fields for the Gaussian mixture parameters (mean and std)
mus_stds_input = st.sidebar.text_area("Enter the mean and standard deviation for each Gaussian (comma separated, e.g. '420,50;100,20;200,40')", 
                                      '420,50;100,20;200,40')
weights_input = st.sidebar.text_area("Enter the weights for each Gaussian (comma separated, e.g. '0.3,0.2,0.7')", '0.3,0.2,0.7')

weights = [float(w) for w in weights_input.split(',')]
mus_stds = [(float(mu), float(std)) for mu, std in (item.split(',') for item in mus_stds_input.split(';'))]

# Price range and other parameters
min_price = 0
resolution = 10000
bins = 100

# Calculate the distribution and Kelly Criterion
x, y, bar_centers, y_bars, f_star, win_prob, expected_win, loss_prob, expected_loss = calculate_distribution(
    curr_price, mus_stds, weights, max_price, min_price, resolution, bins)

# Display the plots
st.subheader("Probability Distribution")
fig, ax = plt.subplots()
ax.bar(bar_centers, y_bars, width=7)
ax.axvline(curr_price, color='red', linestyle='dashed', label=f"Current Price: ${curr_price:.2f}")
ax.set_xlabel("Price ($)")
ax.set_ylabel("Probability (%)")
ax.set_title("Stock Price Probability Distribution")
ax.legend()
st.pyplot(fig)

# Display Kelly Criterion and investment recommendation
st.subheader("Kelly Criterion Investment Recommendation")
st.write(f"Win Probability: {win_prob:.3f}")
st.write(f"Expected Win Amount: ${expected_win:.2f}")
st.write(f"Loss Probability: {loss_prob:.3f}")
st.write(f"Expected Loss Amount: ${expected_loss:.2f}")
st.write(f"Fraction of Net Worth to Invest: {f_star:.3f}")
st.write(f"Recommended Investment Amount (for a net worth of $100,000): ${100000 * (f_star - 1):.2f}")

