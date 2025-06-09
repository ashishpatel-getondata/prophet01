# Reference 1: https://davidmarquis.hashnode.dev/create-a-basic-forecasting-app-using-streamlit-and-prophet
# Reference 2: https://facebook.github.io/prophet/docs/quick_start.html#python-api

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

def main():
    st.set_page_config(page_title='Time Series Forecasting App', layout='centered')
    st.title('Time Series Forecasting App')

    st.markdown("""
    Welcome to the Time Series Forecasting App.
    
    This app helps you create forecasts based on your historical data using the **Prophet** model.
    
    To get started:
    - Upload a CSV file with your data.
    - Select the date column and the value you want to forecast.
    - Choose how many days into the future you’d like to predict.
    
    The app will generate a forecast and display charts to help you visualize future trends and patterns.
    """)
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully. Please fill the form below to configure the forecast.")

    with st.form(key='config_form'):
        date_column = st.selectbox("Select the Date or TimeStamp Column", df.columns)
        value_column = st.selectbox("Select the Metric to Forecast", df.columns)
		
	freq_options = {
	    'D: calendar day': 'D',
	    'W: weekly': 'W',
	    'h: hourly': 'h',
	    'min: minutely': 'min',
	    's: secondly': 's',
	    'MS: month start frequency': 'MS',
	    'ME: month end frequency': 'ME',
	    'YS: year start frequency': 'YS',
	    'YE: year end frequency': 'YE'
	}
	freq_label = st.selectbox("Select the frequency", list(freq_options.keys()))
	freq_input = freq_options[freq_label]	
        periods_input = st.number_input('Period to Forecast into the Future:', min_value=1, max_value=730, value=365)
        submitted = st.form_submit_button("Run Forecast")

    if not submitted:
        st.stop()

    try:
        df_subset = df[[date_column, value_column]].copy()
        df_subset = df_subset.rename(columns={date_column: 'ds', value_column: 'y'})
        df_subset['ds'] = pd.to_datetime(df_subset['ds'])
        df_subset['y'] = pd.to_numeric(df_subset['y'], errors='coerce')
        df_subset.dropna(subset=['ds', 'y'], inplace=True)
    except Exception as e:
        st.error(f"Error processing selected columns: {e}")
        st.stop()

    st.subheader('Input Data')
    st.dataframe(df_subset.sort_values(by='ds'))

    st.subheader('Forecast Results')
    model = Prophet()
    model.fit(df_subset)

    future = model.make_future_dataframe(periods=periods_input, freq=freq_input)
    forecast = model.predict(future)

    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    st.subheader('Forecast Plot')
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    st.subheader('Forecast Components')
    fig2 = plot_components_plotly(model, forecast)
    st.plotly_chart(fig2)

if __name__ == '__main__':
    main()
