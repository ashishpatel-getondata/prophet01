# Reference 1: https://davidmarquis.hashnode.dev/create-a-basic-forecasting-app-using-streamlit-and-prophet
# Reference 2: https://facebook.github.io/prophet/docs/quick_start.html#python-api

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

def main():
    st.set_page_config(page_title='Time Series Forecasting App', layout='centered')
    st.title('Time Series Forecasting App using Prophet')

    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        st.stop()

    # Load uploaded CSV
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully. Please select the required columns.")

    # Step 1: Select columns and submit
    with st.form(key='select_columns'):
        date_column = st.selectbox("Select the Date Column", df.columns)
        value_column = st.selectbox("Select the Metric to Forecast", df.columns)
        submitted_columns = st.form_submit_button("Submit Columns")

    if not submitted_columns:
        st.stop()

    # Prepare dataframe and show input data
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
    st.dataframe(df_subset)

    # Step 2: Ask for forecast days and submit
    with st.form(key='forecast_config'):
        periods_input = st.number_input(
            'Days to forecast into the future:',
            min_value=1, max_value=730, value=365
        )
        submitted_forecast = st.form_submit_button("Run Forecast")

    if not submitted_forecast:
        st.stop()

    # Fit and forecast
    st.subheader('Forecast Results')
    model = Prophet()
    model.fit(df_subset)

    future = model.make_future_dataframe(periods=periods_input)
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
