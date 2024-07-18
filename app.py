import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np



import streamlit as st

# Custom CSS for the top black background with logo and title
st.markdown("""
    <style>
    .top-bar {
        background-color: black;
        height: 2cm;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: orange;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Create the top black background div with logo and title
st.markdown("""
    <div class="top-bar">
        MEDIA MONITOR DASHBOARD
    </div>
    """, unsafe_allow_html=True)




# Function to load and process data
def load_data():
    df = pd.read_csv('consolidated_file.csv')
    df['Published Date'] = pd.to_datetime(df['Published Date'], format='%d-%b-%Y')
    return df

# Function to get article counts and sentiment breakdown for a given date
def get_daily_stats(df, date):
    daily_data = df[df['Published Date'].dt.date == date]
    total_count = len(daily_data)
    sentiment_counts = daily_data['Sentiment Label'].value_counts()
    return total_count, sentiment_counts

# Function to create sentiment breakdown chart
def create_sentiment_chart(current_counts, previous_counts):
    categories = ['Neutral', 'Positive', 'Negative']
    
    fig = go.Figure(data=[
        go.Bar(name='Current Day', x=categories, y=[current_counts.get(cat, 0) for cat in categories]),
        go.Bar(name='Previous Day', x=categories, y=[previous_counts.get(cat, 0) for cat in categories])
    ])
    
    fig.update_layout(barmode='group', title='Sentiment Comparison')
    return fig

# Function to safely convert numpy types to Python types
def safe_convert(value):
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    return value

# Function to get headlines for a given date
def get_headlines(df, date):
    daily_data = df[df['Published Date'].dt.date == date]
    return daily_data[['Portal', 'Published Date', 'Author', 'Headline', 'URL Link', 'Sentiment Label']]

# Function to create sentiment trend line chart
def create_sentiment_trend(df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_trend = pd.DataFrame(index=date_range, columns=['Neutral', 'Positive', 'Negative'])
    
    for date in date_range:
        _, sentiments = get_daily_stats(df, date.date())
        sentiment_trend.loc[date] = [sentiments.get(cat, 0) for cat in ['Neutral', 'Positive', 'Negative']]
    
    fig = px.line(sentiment_trend, x=sentiment_trend.index, y=['Neutral', 'Positive', 'Negative'],
                  title='Sentiment Trend', labels={'value': 'Count', 'variable': 'Sentiment'})
    fig.update_xaxes(tickformat='%d-%b-%Y')
    return fig

# Main page function
def main_page(df):
    #st.title('News Article Dashboard')

    # Date selection
    max_date = df['Published Date'].max().date()
    selected_date = st.date_input("Select a Date", max_value=max_date, value=max_date, format="DD/MM/YYYY")

    if selected_date:
        # Get stats for selected date and previous date
        current_total, current_sentiments = get_daily_stats(df, selected_date)
        previous_date = selected_date - timedelta(days=1)
        previous_total, previous_sentiments = get_daily_stats(df, previous_date)

        # Display total article count
        st.metric(
            label="Total Articles",
            value=safe_convert(current_total),
            delta=safe_convert(current_total - previous_total)
        )
        # Display sentiment breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Neutral Articles",
                value=safe_convert(current_sentiments.get('Neutral', 0)),
                delta=safe_convert(current_sentiments.get('Neutral', 0) - previous_sentiments.get('Neutral', 0))
            )
        
        with col2:
            st.metric(
                label="Positive Articles",
                value=safe_convert(current_sentiments.get('Positive', 0)),
                delta=safe_convert(current_sentiments.get('Positive', 0) - previous_sentiments.get('Positive', 0))
            )
        
        with col3:
            st.metric(
                label="Negative Articles",
                value=safe_convert(current_sentiments.get('Negative', 0)),
                delta=safe_convert(current_sentiments.get('Negative', 0) - previous_sentiments.get('Negative', 0))
            )

        # Display headlines
        st.subheader(f"Headlines for {selected_date.strftime('%d-%b-%Y')}")
        headlines = get_headlines(df, selected_date)

        # Reset index to start from 1
        headlines = headlines.reset_index(drop=True)
        headlines.index = headlines.index + 1

        # Create clickable links
        headlines['URL Link'] = headlines['URL Link'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')

        # Custom CSS for table styling
        st.markdown("""
        <style>
        table {
            font-size: 14px;
            border-collapse: collapse;
            width: 100%;
        }
        th {
            background-color: #f2f2f2;
            color: #333;
            font-weight: bold;
            text-align: left;
            padding: 12px;
            border-bottom: 2px solid #ddd;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        a {
            color: #1e90ff;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        .sentiment {
            font-weight: bold;
        }
        .sentiment-Neutral { color: #808080; }
        .sentiment-Positive { color: #008000; }
        .sentiment-Negative { color: #ff0000; }
        </style>
        """, unsafe_allow_html=True)

        # Apply sentiment color styling
        headlines['Sentiment Label'] = headlines['Sentiment Label'].apply(
            lambda x: f'<span class="sentiment sentiment-{x}">{x}</span>'
        )

        # Convert DataFrame to HTML with custom styling
        headlines_html = headlines.to_html(escape=False, index=True, classes='dataframe')

        # Display the styled table
        st.markdown(headlines_html, unsafe_allow_html=True)
        


        # Create and display sentiment chart
        chart = create_sentiment_chart(current_sentiments, previous_sentiments)
        st.plotly_chart(chart, config={'displayModeBar': False})

# Compare page function
def compare_page(df):
    st.title('News Article Comparison')

    # Date range selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_value=df['Published Date'].min().date(), max_value=df['Published Date'].max().date(), format="DD/MM/YYYY")
    with col2:
        end_date = st.date_input("End Date", min_value=start_date, max_value=df['Published Date'].max().date(), format="DD/MM/YYYY")

    if start_date and end_date:
        # Filter data for selected date range
        mask = (df['Published Date'].dt.date >= start_date) & (df['Published Date'].dt.date <= end_date)
        filtered_df = df.loc[mask]

        # Search functionality
        search_query = st.text_input("Search Headlines")
        if search_query:
            filtered_df = filtered_df[filtered_df['Headline'].str.contains(search_query, case=False)]

        # Display top headlines
        st.subheader("Top Headlines")
        top_headlines = filtered_df['Headline'].head(10)
        for headline in top_headlines:
            st.write(headline)

        # Calculate total articles and sentiment counts
        total_articles = len(filtered_df)
        sentiment_counts = filtered_df['Sentiment Label'].value_counts()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Articles", total_articles)
        col2.metric("Neutral Articles", sentiment_counts.get('Neutral', 0))
        col3.metric("Positive Articles", sentiment_counts.get('Positive', 0))
        col4.metric("Negative Articles", sentiment_counts.get('Negative', 0))

        # Create and display sentiment trend line chart
        trend_chart = create_sentiment_trend(df, start_date, end_date)
        st.plotly_chart(trend_chart)

        # Create and display interactive sentiment breakdown chart
        breakdown_chart = px.bar(
            sentiment_counts,
            x=sentiment_counts.index,
            y=sentiment_counts.values,
            title="Sentiment Breakdown",
            labels={'index': 'Sentiment', 'value': 'Count'}
        )
        st.plotly_chart(breakdown_chart)

# Main Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main Dashboard", "Compare"])

    # Load data
    df = load_data()

    if page == "Main Dashboard":
        main_page(df)
    elif page == "Compare":
        compare_page(df)

if __name__ == "__main__":
    main()