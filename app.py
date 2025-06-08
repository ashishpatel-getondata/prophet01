# https://davidmarquis.hashnode.dev/create-a-basic-forecasting-app-using-streamlit-and-prophet
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

def main():
    # Set Streamlit app title and description
    st.title('Time Series Forecasting App')

    # Load prophet example data into a pandas dataframe
    df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

    # Create and fit the Prophet model
    m = Prophet()
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Display the forecast output
    st.write('Forecasted Data')
    st.dataframe(forecast)

    # Display forecast plot
    st.write('Forecasting Plot')
    fig_forecast = plot_plotly(m, forecast)
    st.plotly_chart(fig_forecast)

if __name__ == '__main__':
    main()
