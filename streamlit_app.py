import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import altair as alt
import io
import base64

# NEW DAY 10: Gemini AI Integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Export Libraries
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os

# Email functionality
try:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

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
    page_title="🚀 AI Tool Recommender - Complete Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DAY 11: Enhanced Custom CSS for Professional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: pulse 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        from { opacity: 0.8; }
        to { opacity: 1; }
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #6c757d;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.8rem;
        margin: 2rem 0 1.5rem 0;
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 50px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .ai-insight-card::before {
        content: '🤖';
        position: absolute;
        top: 15px;
        right: 15px;
        font-size: 1.5rem;
    }
    
    .trend-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
    }
    
    .export-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 2px solid #667eea;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .export-header {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .success-message {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(39, 174, 96, 0.3);
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .stMetric > div > div > div > div {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        padding: 0 25px;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        border: 2px solid transparent;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* DAY 14: Deployment Ready Styling */
    .deployment-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
        margin: 0.5rem;
    }
    
    .beta-banner {
        background: linear-gradient(135deg, #fd7e14 0%, #ffc107 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# DAY 10: Gemini AI Functions
def setup_gemini_api():
    """Setup Gemini API with API key from secrets or sidebar"""
    if GEMINI_AVAILABLE:
        # Try to get API key from Streamlit secrets first
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            return True, "Gemini API configured from secrets"
        except:
            # Fallback to sidebar input
            with st.sidebar:
                st.markdown("### 🤖 Gemini AI Configuration")
                api_key = st.text_input("Enter Gemini API Key:", type="password", 
                                       help="Get your API key from Google AI Studio")
                if api_key:
                    genai.configure(api_key=api_key)
                    return True, "Gemini API configured"
                else:
                    st.warning("⚠️ Enter Gemini API key to enable AI insights")
                    return False, "API key required"
    else:
        return False, "Gemini library not installed"

def generate_ai_summary(tool_name, adoption_rate, feedback_sample, users_count):
    """Generate AI-powered summary using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Analyze this AI tool data and provide a concise 1-2 sentence insight:
        
        Tool: {tool_name}
        Adoption Rate: {adoption_rate}%
        Sample User Feedback: "{feedback_sample}"
        User Base: {users_count:,} users
        
        Provide a professional insight about this tool's performance, market position, or user sentiment. 
        Be specific and actionable. Focus on what makes this tool stand out or areas for improvement.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"AI analysis unavailable: {str(e)[:50]}..."

def get_ai_recommendations(df, user_preferences=None):
    """Generate personalized AI tool recommendations"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare data summary
        tool_summary = df.groupby('tool_name').agg({
            'adoption_rate': 'mean',
            'sentiment_score': 'mean',
            'users_count': 'sum'
        }).round(2).to_dict()
        
        prompt = f"""
        Based on this AI tools performance data, recommend the top 3 tools for different use cases:
        
        {tool_summary}
        
        User preferences: {user_preferences if user_preferences else "General productivity and creativity"}
        
        Provide 3 recommendations in this format:
        1. [Tool Name] - [Use Case] - [Brief reason]
        2. [Tool Name] - [Use Case] - [Brief reason]  
        3. [Tool Name] - [Use Case] - [Brief reason]
        
        Be specific and practical.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return "AI recommendations unavailable. Please check your Gemini API configuration."

# Enhanced Data Generation with More Realistic Patterns
@st.cache_data
def generate_sample_data():
    """Generate sample data with realistic AI tool adoption patterns"""
    
    # DAY 12: Enhanced tools with more detailed categories
    tools_data = {
        "ChatGPT": {"category": "Conversational AI", "base": 85, "trend": 0.5, "volatility": 0.8},
        "Claude": {"category": "Conversational AI", "base": 72, "trend": 1.2, "volatility": 0.9},
        "Gemini": {"category": "Conversational AI", "base": 68, "trend": 1.8, "volatility": 1.1},
        "GitHub Copilot": {"category": "Code Assistant", "base": 75, "trend": 0.8, "volatility": 0.7},
        "Midjourney": {"category": "Image Generation", "base": 65, "trend": 0.6, "volatility": 1.2},
        "Stable Diffusion": {"category": "Image Generation", "base": 58, "trend": 0.7, "volatility": 1.0},
        "Notion AI": {"category": "Productivity", "base": 52, "trend": 1.1, "volatility": 0.8},
        "Jasper": {"category": "Content Writing", "base": 48, "trend": 0.4, "volatility": 0.9},
        "Copy.ai": {"category": "Content Writing", "base": 45, "trend": 0.5, "volatility": 0.8},
        "Grammarly": {"category": "Writing Assistant", "base": 82, "trend": 0.3, "volatility": 0.5},
        "DALL-E 3": {"category": "Image Generation", "base": 60, "trend": 0.9, "volatility": 1.1},
        "Runway ML": {"category": "Video Generation", "base": 42, "trend": 1.3, "volatility": 1.2}
    }
    
    # Enhanced feedback samples by category
    feedback_by_category = {
        "Conversational AI": [
            "Excellent for complex reasoning tasks",
            "Great conversational abilities, very natural",
            "Sometimes provides inconsistent answers",
            "Perfect for research and analysis",
            "Helps me write better emails and documents",
            "Can be slow during peak hours",
            "Impressive knowledge base and accuracy"
        ],
        "Code Assistant": [
            "Saves me hours of coding time daily",
            "Great at explaining complex algorithms",
            "Sometimes suggests outdated syntax",
            "Perfect pair programming companion",
            "Excellent for debugging and code review",
            "Needs better context understanding",
            "Revolutionary for development workflow"
        ],
        "Image Generation": [
            "Creates stunning artistic images",
            "Love the creative possibilities",
            "Sometimes misinterprets prompts",
            "Great for marketing materials",
            "Quality has improved significantly",
            "Can be expensive for commercial use",
            "Amazing detail and creativity"
        ],
        "Productivity": [
            "Streamlines my daily workflow perfectly",
            "Great integration with existing tools",
            "Learning curve is a bit steep",
            "Saves time on repetitive tasks",
            "Love the automation features",
            "Could use more customization options",
            "Essential for team collaboration"
        ],
        "Content Writing": [
            "Helps overcome writer's block",
            "Great for social media content",
            "Sometimes lacks originality",
            "Perfect for email marketing",
            "Good for brainstorming ideas",
            "Content quality varies",
            "Excellent for blog post outlines"
        ],
        "Writing Assistant": [
            "Catches errors I always miss",
            "Improved my writing significantly",
            "Sometimes overly pedantic",
            "Essential for professional writing",
            "Great grammar and style suggestions",
            "Free version is quite limited",
            "Reliable and accurate corrections"
        ],
        "Video Generation": [
            "Cutting-edge video creation capabilities",
            "Still in early stages but promising",
            "Quality is inconsistent",
            "Great for short video content",
            "Expensive but worth it for professionals",
            "Limited customization options",
            "Future of video content creation"
        ]
    }
    
    # Generate comprehensive dataset
    np.random.seed(42)
    data = []
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 1, 31)
    
    current_date = start_date
    month_counter = 0
    
    while current_date <= end_date:
        for tool_name, tool_info in tools_data.items():
            category = tool_info["category"]
            base_rate = tool_info["base"]
            trend_factor = tool_info["trend"]
            volatility = tool_info["volatility"]
            
            # Generate multiple records per tool per month
            for _ in range(np.random.randint(8, 20)):
                # Calculate adoption rate with trends and seasonality
                seasonal_factor = 1 + 0.1 * np.sin(month_counter * np.pi / 6)  # 6-month cycle
                adoption_rate = (base_rate + 
                               (month_counter * trend_factor * 0.5) + 
                               (seasonal_factor * volatility * 5) +
                               np.random.normal(0, 8 * volatility))
                adoption_rate = max(15, min(95, adoption_rate))
                
                # Random date within month
                random_day = np.random.randint(0, min(28, (current_date.replace(month=current_date.month+1) - current_date).days if current_date.month < 12 else 31))
                record_date = current_date + timedelta(days=random_day)
                
                # Select appropriate feedback
                user_feedback = np.random.choice(feedback_by_category[category])
                
                # User count based on adoption rate and tool popularity
                base_users = {"Conversational AI": 1000, "Code Assistant": 800, "Image Generation": 600, 
                             "Productivity": 500, "Content Writing": 400, "Writing Assistant": 900, 
                             "Video Generation": 300}
                users_count = int(base_users[category] * (adoption_rate / 70) * np.random.uniform(0.8, 1.5))
                
                # Satisfaction rating correlated with adoption
                satisfaction_base = 2.5 + (adoption_rate / 100) * 2
                satisfaction_rating = np.random.normal(satisfaction_base, 0.4)
                satisfaction_rating = max(1.0, min(5.0, satisfaction_rating))
                
                data.append({
                    'tool_name': tool_name,
                    'category': category,
                    'date': record_date,
                    'year': record_date.year,
                    'month': record_date.month,
                    'adoption_rate': round(adoption_rate, 1),
                    'user_feedback': user_feedback,
                    'users_count': users_count,
                    'satisfaction_rating': round(satisfaction_rating, 2),
                    'market_trend': 'Growing' if trend_factor > 0.8 else 'Stable' if trend_factor > 0.4 else 'Declining'
                })
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
        month_counter += 1
    
    df = pd.DataFrame(data)
    
    # Add sentiment analysis
    if SENTIMENT_LIB == "vader":
        df['sentiment_score'] = df['user_feedback'].apply(analyze_sentiment_vader)
    elif SENTIMENT_LIB == "textblob":
        df['sentiment_score'] = df['user_feedback'].apply(analyze_sentiment_textblob)
    else:
        # Enhanced keyword-based sentiment
        positive_words = ['excellent', 'amazing', 'great', 'love', 'perfect', 'outstanding', 'revolutionary', 
                         'incredible', 'best', 'essential', 'impressive', 'stunning', 'saves', 'improved']
        negative_words = ['inconsistent', 'slow', 'expensive', 'limited', 'steep', 'overly', 'outdated',
                         'misinterprets', 'varies', 'unavailable', 'crashes', 'lacks']
        
        def enhanced_sentiment(text):
            text_lower = str(text).lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return min(0.8, pos_count * 0.3)
            elif neg_count > pos_count:
                return max(-0.8, -neg_count * 0.3)
            else:
                return np.random.uniform(-0.1, 0.1)
        
        df['sentiment_score'] = df['user_feedback'].apply(enhanced_sentiment)
    
    df['sentiment_label'] = df['sentiment_score'].apply(get_sentiment_label)
    
    return df

# Sentiment Analysis Functions (enhanced)
def analyze_sentiment_vader(text):
    if not text or pd.isna(text):
        return 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    return scores['compound']

def analyze_sentiment_textblob(text):
    if not text or pd.isna(text):
        return 0.0
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

def get_sentiment_label(score):
    if score >= 0.1:
        return "😊 Positive"
    elif score <= -0.1:
        return "😞 Negative"
    else:
        return "😐 Neutral"

def get_sentiment_color(score):
    if score >= 0.1:
        return "#27ae60"
    elif score <= -0.1:
        return "#e74c3c"
    else:
        return "#f39c12"

# DAY 12: Enhanced Visualization Functions
def create_advanced_trend_chart(df):
    """Create advanced trend chart with multiple metrics"""
    trend_data = df.groupby(['year', 'month', 'tool_name', 'category']).agg({
        'adoption_rate': 'mean',
        'sentiment_score': 'mean',
        'users_count': 'sum'
    }).reset_index()
    
    trend_data['date'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('📈 Adoption Trends', '💭 Sentiment Trends', 
                       '👥 User Base Growth', '🏆 Category Performance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Top 5 tools for cleaner visualization
    top_tools = df.groupby('tool_name')['adoption_rate'].mean().nlargest(5).index
    trend_top = trend_data[trend_data['tool_name'].isin(top_tools)]
    
    colors = px.colors.qualitative.Set3[:len(top_tools)]
    
    # Adoption trends
    for i, tool in enumerate(top_tools):
        tool_data = trend_top[trend_top['tool_name'] == tool]
        fig.add_trace(
            go.Scatter(x=tool_data['date'], y=tool_data['adoption_rate'],
                      mode='lines+markers', name=tool, line=dict(color=colors[i]),
                      showlegend=True),
            row=1, col=1
        )
    
    # Sentiment trends
    for i, tool in enumerate(top_tools):
        tool_data = trend_top[trend_top['tool_name'] == tool]
        fig.add_trace(
            go.Scatter(x=tool_data['date'], y=tool_data['sentiment_score'],
                      mode='lines+markers', name=tool, line=dict(color=colors[i]),
                      showlegend=False),
            row=1, col=2
        )
    
    # User growth
    for i, tool in enumerate(top_tools):
        tool_data = trend_top[trend_top['tool_name'] == tool]
        fig.add_trace(
            go.Scatter(x=tool_data['date'], y=tool_data['users_count'],
                      mode='lines+markers', name=tool, line=dict(color=colors[i]),
                      showlegend=False),
            row=2, col=1
        )
    
    # Category performance
    category_data = trend_data.groupby(['date', 'category'])['adoption_rate'].mean().reset_index()
    for category in category_data['category'].unique():
        cat_data = category_data[category_data['category'] == category]
        fig.add_trace(
            go.Scatter(x=cat_data['date'], y=cat_data['adoption_rate'],
                      mode='lines+markers', name=category, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="📊 Comprehensive AI Tools Analysis Dashboard")
    return fig

def create_performance_heatmap(df):
    """Create performance heatmap by tool and time period"""
    # Monthly performance heatmap
    heatmap_data = df.groupby(['tool_name', 'year', 'month'])['adoption_rate'].mean().reset_index()
    heatmap_data['period'] = heatmap_data['year'].astype(str) + '-' + heatmap_data['month'].astype(str).str.zfill(2)
    
    heatmap_pivot = heatmap_data.pivot(index='tool_name', columns='period', values='adoption_rate')
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='RdYlGn',
        hoverongaps = False,
        colorbar=dict(title="Adoption Rate (%)")
    ))
    
    fig.update_layout(
        title="🔥 Tool Performance Heatmap Over Time",
        xaxis_title="Time Period",
        yaxis_title="AI Tools",
        height=600
    )
    
    return fig

# DAY 13: Enhanced Export Functions
def create_enhanced_csv_export(df, filters_info, include_ai_insights=False):
    """Create enhanced CSV export with AI insights"""
    export_df = df.copy()
    
    # Add AI insights column if requested and available
    if include_ai_insights and GEMINI_AVAILABLE:
        gemini_ready, _ = setup_gemini_api()
        if gemini_ready:
            export_df['ai_summary'] = "Generating..."
            # Note: In production, you'd generate summaries here
    
    # Format date
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    
    # Add metadata header
    metadata = f"""# 🚀 AI Tool Recommender Dashboard Export
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Filters Applied: {filters_info}
# Total Records: {len(export_df)}
# Date Range: {export_df['date'].min()} to {export_df['date'].max()}
# Categories: {', '.join(export_df['category'].unique())}
# AI Insights Included: {include_ai_insights and GEMINI_AVAILABLE}
#
"""
    
    csv_buffer = io.StringIO()
    csv_buffer.write(metadata)
    export_df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()

def create_executive_pdf_report(df, filters_info):
    """Create executive-level PDF report with insights"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ExecutiveTitle',
        parent=styles['Title'],
        fontSize=28,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#667eea'),
        fontName='Helvetica-Bold'
    )
    
    # Title page
    elements.append(Paragraph("🚀 AI Tool Recommender", title_style))
    elements.append(Paragraph("Executive Analysis Report", styles['Heading1']))
    elements.append(Spacer(1, 30))
    
    # Executive summary with key insights
    summary_data = {
        'total_records': len(df),
        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
        'avg_adoption': df['adoption_rate'].mean(),
        'avg_sentiment': df['sentiment_score'].mean(),
        'total_users': df['users_count'].sum(),
        'top_category': df.groupby('category')['adoption_rate'].mean().idxmax(),
        'fastest_growing': df.groupby('tool_name')['adoption_rate'].mean().idxmax()
    }
    
    executive_summary = f"""
    <b>📊 KEY FINDINGS:</b><br/><br/>
    • <b>Market Leader:</b> {summary_data['fastest_growing']} dominates with highest adoption<br/>
    • <b>Strongest Category:</b> {summary_data['top_category']} shows consistent growth<br/>
    • <b>Overall Sentiment:
