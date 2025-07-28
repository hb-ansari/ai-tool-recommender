import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random

# Sentiment Analysis Libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    SENTIMENT_LIB = "vader"
except ImportError:
    try:
        from textblob import TextBlob
        SENTIMENT_LIB = "textblob"
    except ImportError:
        SENTIMENT_LIB = None

# Page Configuration
st.set_page_config(
    page_title="AI Tools Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .section-header {
        color: #2c3e50;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
    
    .stMetric > div > div > div > div {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Sentiment Analysis Functions
def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    if not text or pd.isna(text):
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    if not text or pd.isna(text):
        return 0.0
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

def get_sentiment_label(score):
    """Convert sentiment score to label"""
    if score >= 0.05:
        return "😊 Positive"
    elif score <= -0.05:
        return "😞 Negative"
    else:
        return "😐 Neutral"

def get_sentiment_color(score):
    """Get color based on sentiment score"""
    if score >= 0.05:
        return "#27ae60"  # Green
    elif score <= -0.05:
        return "#e74c3c"  # Red
    else:
        return "#f39c12"  # Orange

# Data Generation Function
@st.cache_data
def generate_sample_data():
    """Generate sample data with user feedback for sentiment analysis"""
    
    # AI Tools data
    tools = [
        "ChatGPT", "Claude", "Gemini", "Copilot", "Midjourney", 
        "Stable Diffusion", "Notion AI", "Jasper", "Copy.ai", "Grammarly"
    ]
    
    # Sample feedback comments (mix of positive, negative, neutral)
    feedback_samples = [
        "Amazing tool! Really helps with productivity",
        "Love using this for creative writing",
        "Sometimes gives inaccurate results",
        "Great for brainstorming ideas",
        "Interface could be better",
        "Excellent AI capabilities",
        "Not worth the subscription cost",
        "Perfect for my daily workflow",
        "Takes too long to generate responses",
        "Revolutionary technology!",
        "Good but has room for improvement",
        "Outstanding performance",
        "Frequently crashes on mobile",
        "Best AI tool I've used",
        "Limited functionality for free users",
        "Incredible results every time",
        "Customer support needs work",
        "Game-changing for content creation",
        "Too expensive for small businesses",
        "Intuitive and user-friendly"
    ]
    
    # Generate data
    np.random.seed(42)
    data = []
    
    for _ in range(200):  # Generate 200 records
        tool = np.random.choice(tools)
        date = datetime.now() - timedelta(days=np.random.randint(0, 90))
        adoption_rate = np.random.normal(65, 15)  # Normal distribution around 65%
        adoption_rate = max(10, min(95, adoption_rate))  # Clamp between 10-95%
        
        user_feedback = np.random.choice(feedback_samples)
        
        data.append({
            'tool_name': tool,
            'date': date,
            'adoption_rate': round(adoption_rate, 1),
            'user_feedback': user_feedback,
            'users_count': np.random.randint(100, 5000),
            'satisfaction_rating': np.random.uniform(2.5, 4.8)
        })
    
    df = pd.DataFrame(data)
    
    # Add sentiment analysis
    if SENTIMENT_LIB == "vader":
        df['sentiment_score'] = df['user_feedback'].apply(analyze_sentiment_vader)
    elif SENTIMENT_LIB == "textblob":
        df['sentiment_score'] = df['user_feedback'].apply(analyze_sentiment_textblob)
    else:
        # Fallback: simple keyword-based sentiment
        positive_words = ['amazing', 'love', 'great', 'excellent', 'perfect', 'outstanding', 'revolutionary', 'incredible', 'best', 'game-changing', 'intuitive']
        negative_words = ['inaccurate', 'better', 'worth', 'long', 'crashes', 'limited', 'expensive', 'work']
        
        def simple_sentiment(text):
            text_lower = str(text).lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            if pos_count > neg_count:
                return 0.5
            elif neg_count > pos_count:
                return -0.5
            else:
                return 0.0
        
        df['sentiment_score'] = df['user_feedback'].apply(simple_sentiment)
    
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
    
    return df

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 AI Tools Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        
        # Filters
        st.subheader("🔧 Filters")
        
        # Load data
        df = generate_sample_data()
        
        # Tool filter
        tools_list = ['All'] + sorted(df['tool_name'].unique().tolist())
        selected_tool = st.selectbox("Select AI Tool", tools_list)
        
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=(df['date'].min().date(), df['date'].max().date()),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
        
        # Sentiment filter
        sentiment_filter = st.selectbox(
            "Sentiment Filter",
            ['All', '😊 Positive', '😐 Neutral', '😞 Negative']
        )
        
        st.markdown("---")
        st.markdown("**📊 Sentiment Analysis Powered by:**")
        if SENTIMENT_LIB == "vader":
            st.info("VADER Sentiment")
        elif SENTIMENT_LIB == "textblob":
            st.info("TextBlob")
        else:
            st.warning("Basic Keyword Analysis")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_tool != 'All':
        filtered_df = filtered_df[filtered_df['tool_name'] == selected_tool]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter]
    
    # Main Dashboard
    if len(filtered_df) == 0:
        st.warning("⚠️ No data available for selected filters")
        return
    
    # Key Metrics Section
    st.markdown('<h2 class="section-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_adoption = filtered_df['adoption_rate'].mean()
        st.metric(
            label="📈 Avg Adoption Rate",
            value=f"{avg_adoption:.1f}%",
            delta=f"{avg_adoption - 65:.1f}%"
        )
    
    with col2:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        st.metric(
            label="💭 Avg Sentiment Score",
            value=f"{avg_sentiment:.3f}",
            delta=f"{avg_sentiment:.3f}"
        )
    
    with col3:
        total_users = filtered_df['users_count'].sum()
        st.metric(
            label="👥 Total Users",
            value=f"{total_users:,}",
            delta="Growing"
        )
    
    with col4:
        avg_satisfaction = filtered_df['satisfaction_rating'].mean()
        st.metric(
            label="⭐ Avg Satisfaction",
            value=f"{avg_satisfaction:.2f}/5",
            delta=f"{avg_satisfaction - 4:.2f}"
        )
    
    # Sentiment Analysis Section
    st.markdown('<h2 class="section-header">🧠 Sentiment Analysis Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Average sentiment per tool
        sentiment_by_tool = filtered_df.groupby('tool_name').agg({
            'sentiment_score': 'mean',
            'adoption_rate': 'mean'
        }).reset_index()
        
        fig = px.bar(
            sentiment_by_tool,
            x='tool_name',
            y='sentiment_score',
            title="📊 Average Sentiment Score by AI Tool",
            color='sentiment_score',
            color_continuous_scale=['red', 'yellow', 'green'],
            hover_data=['adoption_rate']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment distribution
        sentiment_dist = filtered_df['sentiment_label'].value_counts()
        
        fig = px.pie(
            values=sentiment_dist.values,
            names=sentiment_dist.index,
            title="🎯 Sentiment Distribution",
            color_discrete_map={
                '😊 Positive': '#27ae60',
                '😐 Neutral': '#f39c12',
                '😞 Negative': '#e74c3c'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bonus: Adoption Rate vs Sentiment Score
    st.markdown('<h2 class="section-header">📈 Adoption Rate vs Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    fig = px.scatter(
        filtered_df,
        x='sentiment_score',
        y='adoption_rate',
        color='tool_name',
        size='users_count',
        hover_data=['user_feedback'],
        title="🔍 Relationship: Sentiment Score vs Adoption Rate",
        labels={
            'sentiment_score': 'Sentiment Score',
            'adoption_rate': 'Adoption Rate (%)'
        }
    )
    fig.add_hline(y=filtered_df['adoption_rate'].mean(), line_dash="dash", annotation_text="Avg Adoption Rate")
    fig.add_vline(x=0, line_dash="dash", annotation_text="Neutral Sentiment")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Data Table with Sentiment
    st.markdown('<h2 class="section-header">📋 Detailed Data with Sentiment Scores</h2>', unsafe_allow_html=True)
    
    # Format the dataframe for display
    display_df = filtered_df[['tool_name', 'date', 'adoption_rate', 'user_feedback', 'sentiment_score', 'sentiment_label']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df = display_df.sort_values('sentiment_score', ascending=False)
    
    # Color code the sentiment scores
    def color_sentiment(val):
        color = get_sentiment_color(val)
        return f'background-color: {color}; color: white; font-weight: bold'
    
    styled_df = display_df.style.applymap(color_sentiment, subset=['sentiment_score'])
    
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download button for data
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data with Sentiment Analysis",
        data=csv,
        file_name=f"ai_tools_sentiment_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🧠 AI Tools Dashboard**")
        st.caption("Powered by Advanced Sentiment Analysis")
    
    with col2:
        st.markdown("**📊 Analytics Features:**")
        st.caption("• Real-time sentiment tracking")
        st.caption("• AI tool performance metrics")
        st.caption("• User feedback analysis")
    
    with col3:
        st.markdown("**🔗 Share Results:**")
        if st.button("📱 Generate Share Link"):
            st.success("Share link generated! 🎉")
            st.code("https://ai-tools-dashboard.com/share/abc123")

if __name__ == "__main__":
    main()
