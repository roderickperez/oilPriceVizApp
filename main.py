import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load and format data
def load_and_format_data(path, date_format=None):
    df = pd.read_csv(path)
    if date_format:
        df['Date'] = pd.to_datetime(df['Date'], format=date_format)
    else:
        # Check if the data is likely to be yearly based on the length of the date column's first entry
        # Assuming that a 4-character string in 'Date' implies a year-only format
        if df['Date'].apply(lambda x: len(str(x))).iloc[0] == 4:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y')
            # Optionally, assign the first day of the year to each entry
            df['Date'] = df['Date'].apply(lambda x: pd.Timestamp(year=x.year, month=1, day=1))
        else:
            df['Date'] = pd.to_datetime(df['Date'])
    return df

# Main function to run the app
def run():
    st.set_page_config(
        page_title="Oil Price Forecast Viz App",
        page_icon="🛢️📈",
    )

    st.write("# Oil Price Forecast Viz App 🛢️📈")

    # Sidebar for user input
    oil_type = st.sidebar.selectbox("Select Oil Type", ["Brent", "WTI"])
    data_frequency = st.sidebar.selectbox("Select Data Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])

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

    # Load and plot the selected dataset
    file_path = paths[oil_type][data_frequency]
    df = load_and_format_data(file_path)
    st.write(f"## {oil_type} Crude Oil Prices - {data_frequency}")

    # Dynamically set the y-axis label based on the oil type
    y_axis_label = f'{oil_type} Spot Price' if oil_type == 'Brent' else f'{oil_type} Spot Price'

    # Use Plotly to create an interactive chart
    fig = px.line(df, x='Date', y=y_axis_label, title=f'{oil_type} Spot Price Over Time ({data_frequency})')

    # Update layout to add a date range slider
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ])
            ),
            type="date"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    run()

