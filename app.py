import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import calendar
from pandas.io.formats import style
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Hindumisia.ai",
    page_icon="🗞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# # Custom CSS for styling (including hide Streamlit branding)
# st.markdown("""
#     <style>
#     #MainMenu {visibility: hidden;}
#     footer {visibility: hidden;}
#     header {visibility: hidden;}
#     .reportview-container .main footer {visibility: hidden;}
#     #stConnectionStatus {visibility: hidden;}
#     .stDeployButton {display: none;}
#     </style>
#     """, unsafe_allow_html=True)


# Custom CSS for styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .reportview-container .main footer {visibility: hidden;}
    #stConnectionStatus {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .top-bar {
        background-color: black;
        height: 150px;
        width: 100vw;
        display: flex;
        align-items: center;
        justify-content: center;
        color: orange;
        font-size: 60px;
        font-weight: bold;
        position: absolute;
        top: -120px;
        left: -86px
    }
    
    .dashboard-title {
        text-align: center;
        width: 100%;
    }
    
    .logo {
        position: absolute;
        left: 10px;
        top: 50%;
        transform: translateY(-50%);
        height: 60px;
    }
            
    .st-emotion-cache-1kyxreq {
        height: 0px;
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
        color: black;
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
    .sentiment-Neutral { color: #A5A5A5; }
    .sentiment-Positive { color: #70AD47; }
    .sentiment-Negative { color: #FF0000; }
    .metric-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 8px;
        text-align: center;
        height: auto;
        width: 90%;
        margin: 0 auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }

    .metric-header {
        font-size: 0.9em;
        font-weight: bold;
        padding: 3px 0;
        margin-bottom: 5px;
        width: 100%;
    }

    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #FF6B00;
        margin-bottom: 1px; /* Reduce space between number and line */
    }

    .metric-separator {
        width: 90%;
        height: 1px;
        margin: 0 auto 12px; /* Reduce space above, keep some below */
        background-color: #ddd;
    }

    .metric-delta {
        font-size: 0.8em;
        margin-bottom: 9px; /* Add space between delta and subtext */
    }

    .metric-subtext {
        font-size: 0.7em;
        color: #888;
    }

    .metric-equation {
        color: #cccccc;
        font-size: 24px;
        position: absolute;
        right: -15px;
        top: 50%;
        transform: translateY(-50%);
    }    
            
    .total-articles .metric-header { background-color: #87CEEB; }
    .negative-articles .metric-header { background-color: #FF0000; }
    .neutral-articles .metric-header { background-color: #A5A5A5; }
    .positive-articles .metric-header { background-color: #70AD47; }

    .delta-positive { color: green; }
    .delta-negative { color: red; }
            
    .footer {
        left: -86px ;
        bottom: -250px;
        width: 100vw;
        background-color: black;
        color: white;
        text-align: left;
        padding: 20px 0;
        height: 4cm;
        position: absolute;
    }
    
    @media screen and (max-width: 430px) {
        body {
            max-width: 100vw;
            overflow: hidden
            }
        .dashboard-title {
            margin-top: 50px;
            }
        .top-bar {
            background-color: black;
            height: 150px;
            width: 100vw;
            display: flex;
            align-items: center;
            justify-content: center;
            color: orange;
            font-size: 20px;
            font-weight: bold;
            position: absolute;
            top: -120px;
            left: -20px
        }
        img {
            position: relative;
            left: 100px;
            top: 0px;
            height: 80px;
            width: 15px;
        }
        .footer {
            left: -20px ;
            bottom: -250px;
            width: 100vw;
            background-color: black;
            color: white;
            text-align: left;
            padding: 10px 0;
            height: 3cm;
            position: absolute;
            overflow: hidden; 
        }
        .footer-column {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .footer-content {
            display: flex;
            font-size: 8px;
            justify-content: space-around;
            max-width: 100%;
            margin: 0 auto;
            padding: 0 10px;
        }
        .social-icons img {
            width: 20px;
            height: 20px;
            margin-right: 5px;
        }
        .phone-social-icons {
            position: relative;
            display: flex;
            bottom: 10px;
            left: 0px;
            transform: translateX(0);
        }
        .copyright {
            text-align: center;
            color: grey;
            margin-top: 30px;
            font-size: 10px;
        }
        table {
            font-size: 14px;
            border-collapse: collapse;
            width: 100%;
            overflow-x: auto;
            display: block;
        }
    }
            
    .footer-content {
        display: flex;
        justify-content: space-around;
        max-width: 1200px;
        margin: 0 auto;
    }
    .footer-column {
        display: flex;
        flex-direction: column;
            gap: 8px;
    }
    .footer-column a {
        color: white;
        text-decoration: none;
        margin-bottom: -10px;
    }
    .footer-column a:hover {
        text-decoration: underline;
    }
    .social-icons img {
        width: 24px;
        height: 24px;
        margin-right: 10px;
        transform: translate(20px, 10px);
    }
    .phone-social-icons{
        display: none;
    }
    .copyright {
        text-align: center;
        color: grey;
        margin-top: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_footer():
    footer_html = """
    <div class="footer">
        <div class="footer-content">
            <div class="footer-column">
                <a href="#">Home</a>
                <a href="#">About Hindumisia.ai</a>
                <a href="#">Benefits</a>
            </div>
            <div class="footer-column">
                <a href="#">Anti-Hindu Hate</a>
                <a href="#">Media References</a>
                <a href="#">Partners</a>
            </div>
            <div class="footer-column">
                <a href="#">Privacy</a>
                <a href="#">Ethics</a>
                <a href="#">Contacts</a>
            </div>
            <div class="footer-column">
                <span>Connect with us</span>
                <div class="social-icons">
                    <a href="#"><img src="https://upload.wikimedia.org/wikipedia/commons/5/57/X_logo_2023_%28white%29.png" alt="X (Twitter)"></a>
                    <a href="#"><img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg" alt="Instagram"></a>
                </div>
                <div class="phone-social-icons">
                    <a href="#"><img src="https://upload.wikimedia.org/wikipedia/commons/5/57/X_logo_2023_%28white%29.png" alt="X (Twitter)"></a>
                    <a href="#"><img src="https://upload.wikimedia.org/wikipedia/commons/e/e7/Instagram_logo_2016.svg" alt="Instagram"></a>
                </div>
            </div>
        </div>
        <div class="copyright">
            ⓒ 2024 hindumisia.ai All Rights Reserved
        </div>
    </div>
    """
    return footer_html


# Load and display the logo
logo = Image.open("hindumisia_lo_go.png")

# Create the top banner with logo and title
st.markdown("""
    <div class="top-bar">
        <div class="dashboard-title">MEDIA MONITOR DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)

# Display the logo image
st.image(logo, width=200, use_column_width=False)

# Add custom CSS to position the logo
st.markdown("""
    <style>
    img {
        position: relative;
        z-index: 1100;
        left: 0px
        top: 0px;
        transform: translate(-50px, -120px);
    }
    </style>
    """, unsafe_allow_html=True)
# st.markdown("""
#     <style>
#     [data-testid="stImage"] {
#         position: relative;
#         z-index: 1100;
#         left: 20px;
#         top: -100px;
#         margin-bottom: -100px;
#     }
#     </style>
#     """, unsafe_allow_html=True)


# Function to load and process data
def load_data():
    df = pd.read_csv('news.csv')
    df['published_date'] = pd.to_datetime(df['published_date'], format='%d-%m-%Y')
    return df

# Function to get article counts and sentiment breakdown for a given date
def get_daily_stats(df, date):
    daily_data = df[df['published_date'].dt.date == date]
    total_count = len(daily_data)
    sentiment_counts = daily_data['sentiment_label'].value_counts()
    return total_count, sentiment_counts

# Function to safely convert numpy types to Python types
def safe_convert(value):
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    return value

# Function to get headlines for a given date
def get_headlines(df, date):
    daily_data = df[df['published_date'].dt.date == date]
    columns = ['portal', 'published_date', 'author', 'headline', 'url_link', 'sentiment_label']
    return daily_data[columns]

def create_stacked_sentiment_graph(df, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    
    filtered_df = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)]
    
    portal_sentiment = filtered_df.groupby('portal')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0)
    portal_sentiment = portal_sentiment.sort_values('Negative', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=-portal_sentiment['Negative'],
        name='Negative',
        orientation='h',
        marker=dict(color='#FF0000'),
        text=((portal_sentiment['Negative']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=portal_sentiment['Neutral'],
        name='Neutral',
        orientation='h',
        marker=dict(color='#A5A5A5'),
        text=((portal_sentiment['Neutral']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=portal_sentiment['Positive'],
        name='Positive',
        orientation='h',
        marker=dict(color='#70AD47'),
        text=((portal_sentiment['Positive']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.update_layout(
        barmode='relative',
        title='Sentiment Distribution by Portal',
        yaxis={'title': '', 'categoryorder':'total ascending'},
        xaxis={'title': '', 'tickformat': '.0%', 'range': [-1, 1]},
        height=400,
        legend_title_text='Sentiment',
        bargap=0.1
    )
    
    return fig

# Modify the create_portal_chart function
def create_portal_chart(df, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    
    filtered_df = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)]
    portal_counts = filtered_df['portal'].value_counts().reset_index()
    portal_counts.columns = ['portal', 'count']
    portal_counts = portal_counts.sort_values('count', ascending=True)
    
    fig = px.bar(portal_counts, x='count', y='portal', orientation='h',
                 title='Articles by Portal')
    fig.update_layout(
        yaxis={'title': '', 'categoryorder':'total ascending'},
        xaxis={'title': '# of Articles'},
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# Main dashboard function
def main_dashboard(df):
    # Date and portal selection in the same row
    col1, col2 = st.columns(2)
    with col1:
        max_date = df['published_date'].max().date()
        selected_date = st.date_input("Select a Date", max_value=max_date, value=max_date, format="DD/MM/YYYY")
    with col2:
        portals = ['All'] + sorted(df['portal'].unique().tolist())
        selected_portal = st.selectbox("Select portal", portals)

    if selected_portal != 'All':
        df = df[df['portal'] == selected_portal]

    if selected_date:
        # Get stats for selected date and previous date
        current_total, current_sentiments = get_daily_stats(df, selected_date)
        previous_date = selected_date - timedelta(days=1)
        previous_total, previous_sentiments = get_daily_stats(df, previous_date)

        if current_total == 0:
            st.warning("No articles found for this selected date. Please select a different date.")
        else:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            metric_data = [
                ("Total Articles", current_total, current_total - previous_total, "total-articles"),
                ("Negative", current_sentiments.get('Negative', 0), current_sentiments.get('Negative', 0) - previous_sentiments.get('Negative', 0), "negative-articles"),
                ("Neutral", current_sentiments.get('Neutral', 0), current_sentiments.get('Neutral', 0) - previous_sentiments.get('Neutral', 0), "neutral-articles"),
                ("Positive", current_sentiments.get('Positive', 0), current_sentiments.get('Positive', 0) - previous_sentiments.get('Positive', 0), "positive-articles")
            ]

            for col, (label, value, delta, class_name) in zip([col1, col2, col3, col4], metric_data):
                with col:
                    delta_class = "delta-positive" if delta > 0 else "delta-negative"
                    delta_symbol = "▲" if delta > 0 else "▼"
                    st.markdown(f"""
                    <div class="metric-box {class_name}">
                        <div class="metric-header">{label}</div>
                        <div class="metric-value">{safe_convert(value)}</div>
                        <div class="metric-separator"></div>
                        <div class="metric-delta {delta_class}">{delta_symbol} {abs(safe_convert(delta))}</div>
                        <div class="metric-subtext">From Previous Day</div>
                        <div class="metric-equation">{'=' if label == 'Total Articles' else '+' if label != 'Positive' else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # After the metric boxes code
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

            # Display headlines
            st.subheader(f"Headlines for {selected_date.strftime('%d-%b-%Y')}")

            headlines = get_headlines(df, selected_date)

            # Add filter and search options
            col1, col2 = st.columns(2)
            with col1:
                search_query = st.text_input("Search headlines", placeholder="Enter keywords and press Enter to search")
            with col2:
                sentiment_filter = st.selectbox("Filter headlines by sentiment", ["All", "Negative", "Neutral", "Positive"])

            # Apply filters
            if search_query:
                headlines = headlines[headlines['headline'].str.contains(search_query, case=False)]
            if sentiment_filter != "All":
                headlines = headlines[headlines['sentiment_label'] == sentiment_filter]

            headlines = headlines.reset_index(drop=True)
            headlines.index = headlines.index + 1

            # Create clickable links
            if 'url_link' in headlines.columns:
                headlines['url_link'] = headlines['url_link'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')
            elif 'url' in headlines.columns:
                headlines['url'] = headlines['url'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')

            # Apply sentiment color styling
            headlines['sentiment_label'] = headlines['sentiment_label'].apply(
                lambda x: f'<span style="color: {"green" if x == "Positive" else "red" if x == "Negative" else "gray"}">{x}</span>'
            )

            # Rename columns
            headlines.columns = ['Portal', 'Published Date', 'Author', 'Headline', 'URL Link', 'Sentiment']

            # Display the styled table with pagination
            if len(headlines) > 10:
                if 'show_all' not in st.session_state:
                    st.session_state.show_all = False

                if st.session_state.show_all:
                    st.markdown(headlines.to_html(escape=False, index=True), unsafe_allow_html=True)
                    if st.button("Show Less"):
                        st.session_state.show_all = False
                        st.rerun()
                else:
                    st.markdown(headlines.head(10).to_html(escape=False, index=True), unsafe_allow_html=True)
                    if st.button("Show more"):
                        st.session_state.show_all = True
                        st.rerun()
            else:
                st.markdown(headlines.to_html(escape=False, index=True), unsafe_allow_html=True)

            # Create and display stacked sentiment graph
            stacked_sentiment_chart = create_stacked_sentiment_graph(df, selected_date, selected_date)
            if not stacked_sentiment_chart.data:
                st.warning("No data available to create the sentiment distribution chart.")
            else:
                st.plotly_chart(stacked_sentiment_chart, use_container_width=True, config={'displayModeBar': False})

            # Create two columns for the remaining charts
            col1, col2 = st.columns(2)

            with col1:
                # Daily Articles Chart
                daily_articles_chart = create_daily_articles_chart(df, selected_date)
                st.plotly_chart(daily_articles_chart, use_container_width=True, config={'displayModeBar': False})

            with col2:
                # portal Statistics Chart (horizontal bar chart)
                portal_chart = create_portal_chart(df, selected_date, selected_date)
                st.plotly_chart(portal_chart, use_container_width=True, config={'displayModeBar': False})

def create_daily_articles_chart(df, selected_date):
    start_date = selected_date - timedelta(days=30)
    daily_counts = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= selected_date)]
    daily_counts = daily_counts.groupby(daily_counts['published_date'].dt.date).size().reset_index(name='articles')
    fig = px.line(daily_counts, x='published_date', y='articles', title='Daily Article Count')
    fig.update_layout(
        height=300, 
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_title="",
        yaxis_title="# of Articles"
    )
    return fig

#Range of news stats
def range_dashboard(df):
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=None, min_value=df['published_date'].min().date(), max_value=df['published_date'].max().date(), format="DD/MM/YYYY", key="start_date")
    with col2:
        end_date = st.date_input("End Date", value=None, min_value=df['published_date'].min().date(), max_value=df['published_date'].max().date(), format="DD/MM/YYYY", key="end_date")
    with col3:
        portals = ['All'] + sorted(df['portal'].unique().tolist())
        selected_portal = st.selectbox("Select portal", portals, key="portal_selector_range")
    
    if selected_portal != 'All':
        df = df[df['portal'] == selected_portal]

    if start_date and end_date:
        mask = (df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        date_range_str = f"{start_date.strftime('%d %b %Y')} and {end_date.strftime('%d %b %Y')}"

        # Calculate total articles and sentiment counts
        total_articles = len(filtered_df)
        sentiment_counts = filtered_df['sentiment_label'].value_counts()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        metric_data = [
            ("Total Articles", total_articles, "For Selected Period", "total-articles"),
            ("Negative", sentiment_counts.get('Negative', 0), "For Selected Period", "negative-articles"),
            ("Neutral", sentiment_counts.get('Neutral', 0), "For Selected Period", "neutral-articles"),
            ("Positive", sentiment_counts.get('Positive', 0), "For Selected Period", "positive-articles")
        ]

        for col, (label, value, subtext, class_name) in zip([col1, col2, col3, col4], metric_data):
            with col:
                st.markdown(f"""
                <div class="metric-box {class_name}">
                    <div class="metric-header">{label}</div>
                    <div class="metric-value">{safe_convert(value)}</div>
                    <div class="metric-separator"></div>
                    <div class="metric-subtext">{subtext}</div>
                    <div class="metric-equation">{'=' if label == 'Total Articles' else '+' if label != 'Positive' else ''}</div>
                </div>
                """, unsafe_allow_html=True)

        # Display headlines
        st.subheader(f"Headlines between {date_range_str}")
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search headlines", placeholder="Enter keywords and press Enter to search", key="headline_search_range")
        with col2:
            sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"], key="sentiment_filter_range")

        # Apply filters
        if search_query:
            filtered_df = filtered_df[filtered_df['headline'].str.contains(search_query, case=False)]
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter]

        # Display headlines
        headlines = filtered_df[['portal', 'published_date', 'author', 'headline', 'url_link', 'sentiment_label']]
        headlines['url_link'] = headlines['url_link'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')
        headlines['sentiment_label'] = headlines['sentiment_label'].apply(
            lambda x: f'<span style="color: {"green" if x == "Positive" else "red" if x == "Negative" else "gray"}">{x}</span>'
        )
        headlines.columns = ['Portal', 'Published Date', 'Author', 'Headline', 'URL Link', 'Sentiment']

        if len(headlines) > 10:
            if 'show_all_range' not in st.session_state:
                st.session_state.show_all_range = False

            if st.session_state.show_all_range:
                st.markdown(headlines.to_html(escape=False, index=False), unsafe_allow_html=True)
                if st.button("Show Less", key="show_less_range"):
                    st.session_state.show_all_range = False
                    st.rerun()
            else:
                st.markdown(headlines.head(10).to_html(escape=False, index=False), unsafe_allow_html=True)
                if st.button("Show more", key="show_more_range"):
                    st.session_state.show_all_range = True
                    st.rerun()
        else:
            st.markdown(headlines.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Create and display charts
        col1, col2 = st.columns(2)

        with col1:
            # Sentiment Trend Over Time
            trend_chart = create_sentiment_trend(filtered_df, start_date, end_date)
            st.plotly_chart(trend_chart, use_container_width=True, config={'displayModeBar': False})

        with col2:
            # Sentiment Distribution by Portal
            sentiment_distribution = create_stacked_sentiment_graph(filtered_df, start_date, end_date)
            st.plotly_chart(sentiment_distribution, use_container_width=True, config={'displayModeBar': False})

        # Articles by Portal
        portal_chart = create_portal_chart(filtered_df, start_date, end_date)
        st.plotly_chart(portal_chart, use_container_width=True, config={'displayModeBar': False})

    else:
        st.warning("Please select a date range to view statistics.")

def create_sentiment_trend(df, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    sentiment_over_time = df.groupby([df['published_date'].dt.date, 'sentiment_label']).size().unstack(fill_value=0)
    
    sentiment_over_time = sentiment_over_time.reindex(date_range, fill_value=0)
    
    # Reset the index and rename the date column
    sentiment_over_time = sentiment_over_time.reset_index()
    sentiment_over_time = sentiment_over_time.rename(columns={'index': 'date'})
    
    # Melt the DataFrame
    sentiment_over_time = sentiment_over_time.melt(id_vars='date', var_name='sentiment', value_name='count')
    
    fig = px.line(sentiment_over_time, x='date', y='count', color='sentiment',
                  title="Sentiment Trend Over Time", labels={'count': 'Count', 'date': 'Date'},
                  color_discrete_map={'Neutral': 'grey', 'Positive': 'green', 'Negative': 'red'})
    return fig

def create_stacked_sentiment_graph(df, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    
    filtered_df = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)]
    
    portal_sentiment = filtered_df.groupby('portal')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0)
    portal_sentiment = portal_sentiment.sort_values('Negative', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=-portal_sentiment['Negative'],
        name='Negative',
        orientation='h',
        marker=dict(color='#FF0000'),
        text=((portal_sentiment['Negative']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=portal_sentiment['Neutral'],
        name='Neutral',
        orientation='h',
        marker=dict(color='#A5A5A5'),
        text=((portal_sentiment['Neutral']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=portal_sentiment['Positive'],
        name='Positive',
        orientation='h',
        marker=dict(color='#70AD47'),
        text=((portal_sentiment['Positive']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.update_layout(
        barmode='relative',
        title='Sentiment Distribution by Portal',
        yaxis={'title': '', 'categoryorder':'total ascending'},
        xaxis={'title': '', 'tickformat': '.0%', 'range': [-1, 1]},
        height=400,
        legend_title_text='Sentiment',
        bargap=0.1
    )
    
    return fig

def create_portal_chart(df, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    
    filtered_df = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)]
    portal_counts = filtered_df['portal'].value_counts().reset_index()
    portal_counts.columns = ['portal', 'count']
    portal_counts = portal_counts.sort_values('count', ascending=True)
    
    fig = px.bar(portal_counts, x='count', y='portal', orientation='h',
                 title='Articles by Portal')
    fig.update_layout(
        yaxis={'title': '', 'categoryorder':'total ascending'},
        xaxis={'title': '# of Articles'},
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

#Monthly dashboard
def monthly_dashboard(df):
    col1, col2 = st.columns(2)
    with col1:
        month_year_options = ["Select Month"] + [f"{calendar.month_name[d.month]} {d.year}" for d in pd.date_range(start=df['published_date'].min(), end=df['published_date'].max(), freq='MS')]
        selected_month_year = st.selectbox("Select Month", month_year_options, key="select_month_year_monthly")
    with col2:
        portals = ['All'] + sorted(df['portal'].unique().tolist())
        selected_portal = st.selectbox("Select portal", portals, key="portal_selector_monthly")
    
    if selected_portal != 'All':
        df = df[df['portal'] == selected_portal]

    if selected_month_year != "Select Month":
        month, year = selected_month_year.split()
        month_num = list(calendar.month_name).index(month)
        start_date = pd.Timestamp(f"{year}-{month_num:02d}-01")
        end_date = start_date + pd.offsets.MonthEnd(0)
        mask = (df['published_date'].dt.date >= start_date.date()) & (df['published_date'].dt.date <= end_date.date())
        filtered_df = df.loc[mask]
        date_range_str = selected_month_year

        # Calculate total articles and sentiment counts
        total_articles = len(filtered_df)
        sentiment_counts = filtered_df['sentiment_label'].value_counts()

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        metric_data = [
            ("Total Articles", total_articles, "For Selected Month", "total-articles"),
            ("Negative", sentiment_counts.get('Negative', 0), "For Selected Month", "negative-articles"),
            ("Neutral", sentiment_counts.get('Neutral', 0), "For Selected Month", "neutral-articles"),
            ("Positive", sentiment_counts.get('Positive', 0), "For Selected Month", "positive-articles")
        ]

        for col, (label, value, subtext, class_name) in zip([col1, col2, col3, col4], metric_data):
            with col:
                st.markdown(f"""
                <div class="metric-box {class_name}">
                    <div class="metric-header">{label}</div>
                    <div class="metric-value">{safe_convert(value)}</div>
                    <div class="metric-separator"></div>
                    <div class="metric-subtext">{subtext}</div>
                    <div class="metric-equation">{'=' if label == 'Total Articles' else '+' if label != 'Positive' else ''}</div>
                </div>
                """, unsafe_allow_html=True)

        # Display headlines
        st.subheader(f"Headlines for {date_range_str}")
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_query = st.text_input("Search headlines", placeholder="Enter keywords and press Enter to search", key="headline_search_monthly")
        with col2:
            sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"], key="sentiment_filter_monthly")

        # Apply filters
        if search_query:
            filtered_df = filtered_df[filtered_df['headline'].str.contains(search_query, case=False)]
        if sentiment_filter != "All":
            filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter]

        # Display headlines
        headlines = filtered_df[['portal', 'published_date', 'author', 'headline', 'url_link', 'sentiment_label']]
        headlines['url_link'] = headlines['url_link'].apply(lambda x: f'<a href="{x}" target="_blank">Read More</a>')
        headlines['sentiment_label'] = headlines['sentiment_label'].apply(
            lambda x: f'<span style="color: {"green" if x == "Positive" else "red" if x == "Negative" else "gray"}">{x}</span>'
        )
        headlines.columns = ['Portal', 'Published Date', 'Author', 'Headline', 'URL Link', 'Sentiment']

        if len(headlines) > 10:
            if 'show_all_monthly' not in st.session_state:
                st.session_state.show_all_monthly = False

            if st.session_state.show_all_monthly:
                st.markdown(headlines.to_html(escape=False, index=False), unsafe_allow_html=True)
                if st.button("Show Less", key="show_less_monthly"):
                    st.session_state.show_all_monthly = False
                    st.rerun()
            else:
                st.markdown(headlines.head(10).to_html(escape=False, index=False), unsafe_allow_html=True)
                if st.button("Show more", key="show_more_monthly"):
                    st.session_state.show_all_monthly = True
                    st.rerun()
        else:
            st.markdown(headlines.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Create and display charts
        col1, col2 = st.columns(2)

        with col1:
            # Sentiment Trend Over Time
            trend_chart = create_sentiment_trend(filtered_df, start_date, end_date)
            st.plotly_chart(trend_chart, use_container_width=True, config={'displayModeBar': False})

        with col2:
            # Sentiment Distribution by Portal
            sentiment_distribution = create_stacked_sentiment_graph(filtered_df, start_date, end_date)
            st.plotly_chart(sentiment_distribution, use_container_width=True, config={'displayModeBar': False})

        # Articles by Portal
        portal_chart = create_portal_chart(filtered_df, start_date, end_date)
        st.plotly_chart(portal_chart, use_container_width=True, config={'displayModeBar': False})

    else:
        st.warning("Please select a month to view statistics.")

def create_stacked_sentiment_graph(df, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    
    filtered_df = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)]
    
    portal_sentiment = filtered_df.groupby('portal')['sentiment_label'].value_counts(normalize=True).unstack(fill_value=0)
    portal_sentiment = portal_sentiment.sort_values('Negative', ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=-portal_sentiment['Negative'],
        name='Negative',
        orientation='h',
        marker=dict(color='#FF0000'),
        text=((portal_sentiment['Negative']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=portal_sentiment['Neutral'],
        name='Neutral',
        orientation='h',
        marker=dict(color='#A5A5A5'),
        text=((portal_sentiment['Neutral']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.add_trace(go.Bar(
        y=portal_sentiment.index,
        x=portal_sentiment['Positive'],
        name='Positive',
        orientation='h',
        marker=dict(color='#70AD47'),
        text=((portal_sentiment['Positive']*100).round(1).astype(str) + '%'),
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.update_layout(
        barmode='relative',
        title='Sentiment Distribution by Portal',
        yaxis={'title': '', 'categoryorder':'total ascending'},
        xaxis={'title': '', 'tickformat': '.0%', 'range': [-1, 1]},
        height=400,
        legend_title_text='Sentiment',
        bargap=0.1
    )
    
    return fig

def create_portal_chart(df, start_date, end_date):
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    
    filtered_df = df[(df['published_date'].dt.date >= start_date) & (df['published_date'].dt.date <= end_date)]
    portal_counts = filtered_df['portal'].value_counts().reset_index()
    portal_counts.columns = ['portal', 'count']
    portal_counts = portal_counts.sort_values('count', ascending=True)
    
    fig = px.bar(portal_counts, x='count', y='portal', orientation='h',
                 title='Articles by Portal')
    fig.update_layout(
        yaxis={'title': '', 'categoryorder':'total ascending'},
        xaxis={'title': '# of Articles'},
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

# Main Streamlit app
def main():
    # Load data
    df = load_data()

    # Radio buttons for selecting the view
    col1, col2 = st.columns(2)
    with col1:
        view = st.radio("Select Statistics", ["Daily", "Monthly", "Range"], horizontal=True, key="view_selector")

    if view == "Daily":
        main_dashboard(df)
    elif view == "Monthly":
        monthly_dashboard(df)
    else:
        range_dashboard(df)

    st.markdown(create_footer(), unsafe_allow_html=True)

if __name__ == "__main__":
    main()