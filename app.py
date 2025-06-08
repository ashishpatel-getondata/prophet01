import streamlit as st
import pandas as pd
from prophet import Prophet


#Load prophet example data into a pandas dataframe
df = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')

def forecast(df):
    # Create a Prophet model
    m = Prophet()

    # Fit the model to the data
    m.fit(df)

    # Make future predictions
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)

    #Return the forecast DataFrame
    return forecast


def main():
  # Set Streamlit app title and description
  st.title('Time Series Forecasting App')
  st.write('My Forecast Data.')

  forecast_data = forecast(df)

  # Display the forecast output
  st.dataframe(forecast_data)
  
if __name__ == '__main__':
   main()
