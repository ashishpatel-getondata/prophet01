# https://davidmarquis.hashnode.dev/create-a-basic-forecasting-app-using-streamlit-and-prophet
import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

def main():
    # Set Streamlit app title and description
    st.title('Time Series Forecasting App')

    # Load prophet example data into a pandas dataframe
    df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

    # Display the input data
    st.write('Input Data')
    st.dataframe(df)
    
    # Create and fit the Prophet model
    m = Prophet()
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    # Display the forecast data
    st.write('Forecast Data')
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Display the forecast plot
    st.write('Forecast Plot')
    fig_forecast = plot_plotly(m, forecast)
    st.plotly_chart(fig_forecast)

    # Display the forecast components
    st.write('Forecast Components')
    fig_components = plot_components_plotly(m, forecast)
    st.plotly_chart(fig_components)

if __name__ == '__main__':
    main()
