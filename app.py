import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go 

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# APPL - Apple, GOOG - Google, MSFT - Microsoft, GME - Gamestop
stocks = ("AAPL", "GOOG", "MSFT", "GME")

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of Prediction:", 1, 4)
periods = n_years * 365

def load_data(ticker):
    data = yf.download(ticker.strip(), START, TODAY)
    data = data.reset_index(inplace = True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data done.")

st.subheader("Raw Data")
st.write(data.tail())

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.update_layout(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()
