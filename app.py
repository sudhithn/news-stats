import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import calendar
from pandas.io.formats import style


# Set page configuration
st.set_page_config(
    page_title="Media Monitor Dashboard",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling (including hide Streamlit branding)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .reportview-container .main footer {visibility: hidden;}
    #stConnectionStatus {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ... rest of your existing CSS ... */
    </style>
    """, unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        margin-top: -2cm;
    }
    .view-selector {
        margin-top: 0.5cm;
    }
    .stMetric {
        text-align: center;
    }
    .stMetric .metric-label {
        font-size: 1em;
    }
    .stMetric .metric-value {
        font-size: 2em;
        font-weight: bold;
    }
    .stMetric .metric-delta {
        font-size: 0.8em;
    }
    .metric-subtext {
        font-size: 0.7em;
        color: #888;
        text-align: center;
        margin-top: -10px;
    }
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

# Function to create horizontal bar chart of portals
def create_portal_chart(df, date):
    portal_counts = df[df['Published Date'].dt.date == date]['Portal'].value_counts()
    fig = px.bar(portal_counts, x=portal_counts.values, y=portal_counts.index, orientation='h',
                 title='Articles by Portal', labels={'x': '# of Articles', 'y': 'Portal'})
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

# Main dashboard function
def main_dashboard(df):
    # Date and Portal selection in the same row
    col1, col2 = st.columns(2)
    with col1:
        max_date = df['Published Date'].max().date()
        selected_date = st.date_input("Select a Date", max_value=max_date, value=max_date, format="DD/MM/YYYY")
    with col2:
        portals = ['All'] + sorted(df['Portal'].unique().tolist())
        selected_portal = st.selectbox("Select Portal", portals)

    if selected_portal != 'All':
        df = df[df['Portal'] == selected_portal]

    if selected_date:
        # Get stats for selected date and previous date
        current_total, current_sentiments = get_daily_stats(df, selected_date)
        previous_date = selected_date - timedelta(days=1)
        previous_total, previous_sentiments = get_daily_stats(df, previous_date)

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Articles", safe_convert(current_total), safe_convert(current_total - previous_total))
            st.markdown("<p class='metric-subtext' style='text-align: left'>From Previous Day</p>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Neutral Articles", safe_convert(current_sentiments.get('Neutral', 0)), 
                      safe_convert(current_sentiments.get('Neutral', 0) - previous_sentiments.get('Neutral', 0)))
            st.markdown("<p class='metric-subtext' style='text-align: left'>From Previous Day</p>", unsafe_allow_html=True)
        
        with col3:
            st.metric("Positive Articles", safe_convert(current_sentiments.get('Positive', 0)), 
                      safe_convert(current_sentiments.get('Positive', 0) - previous_sentiments.get('Positive', 0)))
            st.markdown("<p class='metric-subtext' style='text-align: left'>From Previous Day</p>", unsafe_allow_html=True)
        
        with col4:
            st.metric("Negative Articles", safe_convert(current_sentiments.get('Negative', 0)), 
                      safe_convert(current_sentiments.get('Negative', 0) - previous_sentiments.get('Negative', 0)))
            st.markdown("<p class='metric-subtext' style='text-align: left'>From Previous Day</p>", unsafe_allow_html=True)

        # Display headlines
        st.subheader(f"Headlines for {selected_date.strftime('%d-%b-%Y')}")
        headlines = get_headlines(df, selected_date)

        # Sort by Sentiment Label instead of Sentiment Score
        sentiment_order = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        headlines['Sentiment Order'] = headlines['Sentiment Label'].map(sentiment_order)
        headlines = headlines.sort_values(by='Sentiment Order', ascending=False)
        headlines = headlines.drop('Sentiment Order', axis=1)  # Remove the temporary column

        headlines = headlines.reset_index(drop=True)
        headlines.index = headlines.index + 1

        # Create clickable links
        headlines['URL Link'] = headlines['URL Link'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')

        # Apply sentiment color styling
        headlines['Sentiment Label'] = headlines['Sentiment Label'].apply(
            lambda x: f'<span style="color: {"green" if x == "Positive" else "red" if x == "Negative" else "gray"}">{x}</span>'
        )

        # Display the styled table with pagination
        if len(headlines) > 10:
            show_all = st.checkbox("Show all headlines")
            if show_all:
                st.markdown(headlines.to_html(escape=False, index=True), unsafe_allow_html=True)
            else:
                st.markdown(headlines.head(10).to_html(escape=False, index=True), unsafe_allow_html=True)
        else:
            st.markdown(headlines.to_html(escape=False, index=True), unsafe_allow_html=True)
        # Create and display sentiment chart
        sentiment_chart = create_sentiment_chart(current_sentiments, previous_sentiments)
        st.plotly_chart(sentiment_chart, use_container_width=True, config={'displayModeBar': False})

        # Create and display portal chart
        portal_chart = create_portal_chart(df, selected_date)
        st.plotly_chart(portal_chart, use_container_width=True, config={'displayModeBar': False})

    # Adding new charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily Articles Chart
        daily_articles_chart = create_daily_articles_chart(df)
        st.plotly_chart(daily_articles_chart, use_container_width=True, config={'displayModeBar': False})

    # with col2:
    #     # Sentiment by Portal Chart (previously Sentiment by Topic)
    #     sentiment_by_portal_chart = create_sentiment_by_topic_chart(df)
    #     st.plotly_chart(sentiment_by_portal_chart, use_container_width=True, config={'displayModeBar': False})

    # Portal Statistics Chart
    portal_stats_chart = create_portal_statistics_chart(df)
    st.plotly_chart(portal_stats_chart, use_container_width=True, config={'displayModeBar': False})

# New functions to create additional charts

def create_daily_articles_chart(df):
    daily_counts = df.groupby('Published Date').size().reset_index(name='count')
    fig = px.line(daily_counts, x='Published Date', y='count', title='Daily Article Count')
    return fig

# def create_sentiment_by_topic_chart(df):
#     # Assuming you have a 'Topic' column in your dataframe
#     topic_sentiment = df.groupby('Topic')['Sentiment Label'].value_counts(normalize=True).unstack()
#     fig = px.bar(topic_sentiment, title='Sentiment Distribution by Topic', barmode='stack')
#     return fig

def create_portal_statistics_chart(df):
    portal_counts = df['Portal'].value_counts()
    fig = px.bar(portal_counts, x=portal_counts.index, y=portal_counts.values, title='Articles per Portal')
    return fig

#Comppare news stats
def compare_dashboard(df):
    # Date selection
    col1, col2, col3, col4 = st.columns([2, 2, 1, 3])
    with col1:
        start_date = st.date_input("Start Date", value=None, min_value=df['Published Date'].min().date(), max_value=df['Published Date'].max().date(), format="DD/MM/YYYY", key="start_date")
    with col2:
        end_date = st.date_input("End Date", value=None, min_value=df['Published Date'].min().date(), max_value=df['Published Date'].max().date(), format="DD/MM/YYYY", key="end_date")
    with col3:
        st.markdown("<p style='text-align: center; margin-top: 30px;'>OR</p>", unsafe_allow_html=True)
    with col4:
        # Create a list of month-year options
        month_year_options = ["Select Month"] + [f"{calendar.month_name[d.month]} {d.year}" for d in pd.date_range(start=df['Published Date'].min(), end=df['Published Date'].max(), freq='MS')]
        selected_month_year = st.selectbox("Select Month", month_year_options, key="select_month_year")

    # Portal selection
    portals = ['All'] + sorted(df['Portal'].unique().tolist())
    selected_portal = st.selectbox("Select Portal", portals, key="portal_selector")
    
    if selected_portal != 'All':
        df = df[df['Portal'] == selected_portal]

    # Filter data based on selected date range or month
    if start_date and end_date:
        mask = (df['Published Date'].dt.date >= start_date) & (df['Published Date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        date_range_str = f"{start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}"
        selected_month_year = "Select Month"  # Reset month selection
    elif selected_month_year != "Select Month":
        month, year = selected_month_year.split()
        month_num = list(calendar.month_name).index(month)
        start_date = pd.Timestamp(f"{year}-{month_num:02d}-01")
        end_date = start_date + pd.offsets.MonthEnd(0)
        mask = (df['Published Date'].dt.date >= start_date.date()) & (df['Published Date'].dt.date <= end_date.date())
        filtered_df = df.loc[mask]
        date_range_str = selected_month_year
        start_date = start_date.date()  # Convert to date object for consistency
        end_date = end_date.date()
    else:
        st.warning("Please select either a date range or a specific month.")
        return

    # Calculate total articles and sentiment counts
    total_articles = len(filtered_df)
    sentiment_counts = filtered_df['Sentiment Label'].value_counts()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Articles", total_articles)
        st.markdown("<p class='metric-subtext' style='text-align: left'>For Selected Period</p>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Neutral Articles", sentiment_counts.get('Neutral', 0))
        st.markdown("<p class='metric-subtext'>For Selected Period</p>", unsafe_allow_html=True)
    
    with col3:
        st.metric("Positive Articles", sentiment_counts.get('Positive', 0))
        st.markdown("<p class='metric-subtext'>For Selected Period</p>", unsafe_allow_html=True)
    
    with col4:
        st.metric("Negative Articles", sentiment_counts.get('Negative', 0))
        st.markdown("<p class='metric-subtext'>For Selected Period</p>", unsafe_allow_html=True)

    # Display top headlines
    st.subheader(f"Top Headlines for {date_range_str}")
    
    # Search functionality with placeholder text
    search_query = st.text_input("Search Headlines", placeholder="Enter keywords and press Enter to search", key="headline_search")
    if search_query:
        filtered_df = filtered_df[filtered_df['Headline'].str.contains(search_query, case=False)]

    top_headlines = filtered_df[['Portal', 'Published Date', 'Author', 'Headline', 'URL Link', 'Sentiment Label']].head(10)
    top_headlines['URL Link'] = top_headlines['URL Link'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')
    top_headlines['Sentiment Label'] = top_headlines['Sentiment Label'].apply(
        lambda x: f'<span class="sentiment sentiment-{x}">{x}</span>'
    )
    st.markdown(top_headlines.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Create and display sentiment trend line chart
    trend_chart = create_sentiment_trend(filtered_df, start_date, end_date)
    st.plotly_chart(trend_chart, use_container_width=True, config={'displayModeBar': False})

    # Create and display interactive sentiment breakdown chart
    breakdown_chart = px.bar(
        sentiment_counts,
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Sentiment Breakdown",
        labels={'index': 'Sentiment', 'value': 'Count'}
    )
    st.plotly_chart(breakdown_chart, use_container_width=True, config={'displayModeBar': False})

    # Daily Statistics
    daily_stats = filtered_df.groupby(filtered_df['Published Date'].dt.date).size().reset_index(name='count')
    daily_chart = px.line(daily_stats, x='Published Date', y='count', title='Daily Article Count')
    st.plotly_chart(daily_chart, use_container_width=True, config={'displayModeBar': False})

    # Portal Statistics
    portal_stats = filtered_df['Portal'].value_counts().reset_index()
    portal_stats.columns = ['Portal', 'Count']
    portal_chart = px.bar(portal_stats, x='Count', y='Portal', orientation='h', title='Articles by Portal')
    st.plotly_chart(portal_chart, use_container_width=True, config={'displayModeBar': False})

    # Tone by Percentage
    tone_percentages = (sentiment_counts / sentiment_counts.sum() * 100).round(1)
    tone_chart = px.pie(values=tone_percentages.values, names=tone_percentages.index, title='Tone Distribution')
    st.plotly_chart(tone_chart, use_container_width=True, config={'displayModeBar': False})

    # Tone by Topic (assuming you have a 'Topic' column in your dataframe)
    if 'Topic' in filtered_df.columns:
        tone_by_topic = filtered_df.groupby('Topic')['Sentiment Label'].value_counts(normalize=True).unstack() * 100
        tone_by_topic_chart = px.bar(tone_by_topic, title='Tone by Topic')
        st.plotly_chart(tone_by_topic_chart, use_container_width=True, config={'displayModeBar': False})

    # Display additional statistics
    st.subheader("Additional Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Authors", filtered_df['Author'].nunique())
    with col2:
        st.metric("Total Outlets", filtered_df['Portal'].nunique())

def create_sentiment_trend(df, start_date, end_date):
    # Ensure we have at least two days of data
    if start_date == end_date:
        end_date += timedelta(days=1)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    sentiment_over_time = df.groupby([df['Published Date'].dt.date, 'Sentiment Label']).size().unstack(fill_value=0)
    
    # Reindex to include all dates in the range, filling missing values with 0
    sentiment_over_time = sentiment_over_time.reindex(date_range, fill_value=0)
    
    fig = px.line(sentiment_over_time, x=sentiment_over_time.index, y=sentiment_over_time.columns,
                  title="Sentiment Trend Over Time", labels={'value': 'Count', 'variable': 'Sentiment'})
    return fig


# Main Streamlit app
def main():
    # Load data
    df = load_data()

    # Radio buttons for selecting the view
    col1, col2 = st.columns(2)
    with col1:
        view = st.radio("Select Stats", ["Today's News Stats", "Compare News Stats"], horizontal=True, key="view_selector")

    if view == "Today's News Stats":
        main_dashboard(df)
    else:
        compare_dashboard(df)

if __name__ == "__main__":
    main()