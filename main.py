import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Function to load and format data
def load_and_format_data(path, date_format=None):
    df = pd.read_csv(path)
    if date_format:
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    else:
        if df['Date'].apply(lambda x: len(str(x))).iloc[0] == 4:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y')
            df['Date'] = df['Date'].apply(lambda x: pd.Timestamp(year=x.year, month=1, day=1))
        else:
            df['Date'] = pd.to_datetime(df['Date'])
    return df

# Placeholder function for forecasting (linear regression example)
def simple_linear_regression(train_x, train_y, test_x):
    model = LinearRegression()
    model.fit(train_x, train_y)
    return model.predict(test_x)

# Main function to run the app
def run():
    st.set_page_config(page_title="Oil Price Forecast Viz App", page_icon="🛢️📈")
    st.write("# Oil Price Forecast Viz App 🛢️📈")

    # Sidebar for user input
    oil_type = st.sidebar.selectbox("Select Oil Type", ["Brent", "WTI"])
    data_frequency = st.sidebar.selectbox("Select Data Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])

    # Model Selection and parameters input
    model_name = st.sidebar.selectbox(
        "Select Forecasting Algorithm",
        ["Linear Regression"]  # Simplified for this example; add other models as needed
    )
    
    training_percentage = st.sidebar.slider("Training Window Percentage", min_value=10, max_value=90, value=70, step=5)
    forecast_period = st.sidebar.number_input("Forecast Period (in data points)", min_value=1, value=30, step=1)

    # Paths to your datasets
    paths = {
        "Brent": {
            "Daily": 'data/brent-day.csv',
            "Weekly": 'data/brent-week.csv',
            "Monthly": 'data/brent-month.csv',
            "Yearly": 'data/brent-year.csv'
        },
        "WTI": {
            "Daily": 'data/wti-day.csv',
            "Weekly": 'data/wti-week.csv',
            "Monthly": 'data/wti-month.csv',
            "Yearly": 'data/wti-year.csv'
        }
    }

    file_path = paths[oil_type][data_frequency]
    df = load_and_format_data(file_path)
    st.write(f"## {oil_type} Crude Oil Prices - {data_frequency}")

    # Convert 'Date' to ordinal for linear regression example
    df['DateOrdinal'] = df['Date'].apply(lambda x: x.toordinal())

    # Calculate the indices for the training and testing periods based on percentages
    training_size = int(len(df) * (training_percentage / 100.0))

    # Extend df with future dates if forecast_period > 0
    last_date = df['Date'].max()
    future_dates = [last_date + pd.Timedelta(days=x+1) for x in range(forecast_period)]
    future_df = pd.DataFrame(future_dates, columns=['Date'])

    # Convert future dates to ordinal for consistency with the model input
    future_df['DateOrdinal'] = future_df['Date'].apply(lambda x: x.toordinal())

    # Combine future_df with df for plotting purposes (not used for training)
    combined_df = pd.concat([df, future_df], ignore_index=True)


    # Separate training and test data
    train_df = df.iloc[:training_size]
    test_df = df.iloc[training_size:].copy()  # This includes all data points beyond the training set

    # Placeholder for model logic based on the user's selection
    if model_name == "Linear Regression":
        predictions = simple_linear_regression(
            train_df[['DateOrdinal']], train_df['Brent Spot Price' if oil_type == 'Brent' else 'WTI Spot Price'],
            test_df[['DateOrdinal']]
        )

        # Now, it's safe to assign predictions to test_df without causing a warning
        test_df['Predictions'] = predictions
        
        # Make predictions for the future dates
        future_predictions = simple_linear_regression(
            df[['DateOrdinal']], df['Brent Spot Price' if oil_type == 'Brent' else 'WTI Spot Price'],
            future_df[['DateOrdinal']]
        )
        future_df['Predictions'] = future_predictions
        

    # Plotting
    fig = px.line(df, x='Date', y='Brent Spot Price' if oil_type == 'Brent' else 'WTI Spot Price', title=f'{oil_type} Spot Price Over Time ({data_frequency})')

    # Highlighting the training and test periods
    fig.add_vrect(x0=train_df['Date'].min(), x1=train_df['Date'].max(), fillcolor="blue", opacity=0.2, layer="below", line_width=0, annotation_text="Training", annotation_position="top left")

    if not test_df.empty:
        test_period_end_date = test_df['Date'].iloc[-1]  # Ensures the rectangle goes to the end of the dataset
        
        # Highlighting the test period correctly
        fig.add_vrect(x0=test_df['Date'].min(), x1=test_period_end_date, fillcolor="green", opacity=0.5, layer="below", line_width=0, annotation_text="Test", annotation_position="top left")

    # Plot predictions if available
    if model_name == "Linear Regression":
        fig.add_scatter(x=test_df['Date'], y=test_df['Predictions'], mode='lines', name='Test Prediction')
        fig.add_scatter(x=future_df['Date'], y=future_df['Predictions'], mode='lines', name='Future Prediction')
        
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    run()
