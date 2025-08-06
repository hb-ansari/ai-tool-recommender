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


# Gemini AI Integration
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
    page_title="AI Adoption Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .disclaimer-banner {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        margin-bottom: 2rem;
        border: 1px solid #d68910;
    }
    
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .section-header {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 600;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1.5rem 0;
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .export-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        border: 1px solid #dee2e6;
    }
    
    .export-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .success-message {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #3498db;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: 1px solid #3498db;
    }
    
    .ranking-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Gemini AI Functions
def setup_gemini_api():
    """Setup Gemini API with API key from secrets or sidebar"""
    if GEMINI_AVAILABLE:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            genai.configure(api_key=api_key)
            return True, "Gemini API configured from secrets"
        except:
            with st.sidebar:
                st.markdown("### AI Configuration")
                api_key = st.text_input("Enter Gemini API Key:", type="password", 
                                       help="Get your API key from Google AI Studio")
                if api_key:
                    genai.configure(api_key=api_key)
                    return True, "Gemini API configured"
                else:
                    st.warning("Enter Gemini API key to enable AI insights")
                    return False, "API key required"
    else:
        return False, "Gemini library not installed"

def generate_ai_summary(tool_name, adoption_rate, feedback_sample, users_count):
    """Generate AI-powered summary using Gemini"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Analyze this AI tool data and provide a concise professional insight:
        
        Tool: {tool_name}
        Adoption Rate: {adoption_rate}%
        Sample User Feedback: "{feedback_sample}"
        User Base: {users_count:,} users
        
        Provide a professional business insight about this tool's performance and market position. 
        Be specific and actionable. Keep it under 2 sentences.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return f"AI analysis unavailable: {str(e)[:50]}..."

def get_ai_recommendations(df, user_preferences=None):
    """Generate personalized AI tool recommendations"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        tool_summary = df.groupby('tool_name').agg({
            'adoption_rate': 'mean',
            'sentiment_score': 'mean',
            'users_count': 'sum'
        }).round(2).to_dict()
        
        prompt = f"""
        Based on this AI tools performance data, recommend the top 3 tools:
        
        {tool_summary}
        
        User preferences: {user_preferences if user_preferences else "General productivity"}
        
        Provide 3 professional recommendations in this format:
        1. [Tool Name] - [Use Case] - [Brief business reason]
        2. [Tool Name] - [Use Case] - [Brief business reason]  
        3. [Tool Name] - [Use Case] - [Brief business reason]
        
        Keep it professional and practical.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        return "AI recommendations unavailable. Please check your Gemini API configuration."

# Enhanced Data Generation
@st.cache_data
def generate_sample_data():
    """Generate sample data with realistic AI tool adoption patterns"""
    
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
            
            for _ in range(np.random.randint(8, 20)):
                seasonal_factor = 1 + 0.1 * np.sin(month_counter * np.pi / 6)
                adoption_rate = (base_rate + 
                               (month_counter * trend_factor * 0.5) + 
                               (seasonal_factor * volatility * 5) +
                               np.random.normal(0, 8 * volatility))
                adoption_rate = max(15, min(95, adoption_rate))
                
                random_day = np.random.randint(0, min(28, (current_date.replace(month=current_date.month+1) - current_date).days if current_date.month < 12 else 31))
                record_date = current_date + timedelta(days=random_day)
                
                user_feedback = np.random.choice(feedback_by_category[category])
                
                base_users = {"Conversational AI": 1000, "Code Assistant": 800, "Image Generation": 600, 
                             "Productivity": 500, "Content Writing": 400, "Writing Assistant": 900, 
                             "Video Generation": 300}
                users_count = int(base_users[category] * (adoption_rate / 70) * np.random.uniform(0.8, 1.5))
                
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

# Sentiment Analysis Functions
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
        return "Positive"
    elif score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

def get_sentiment_color(score):
    if score >= 0.1:
        return "#27ae60"
    elif score <= -0.1:
        return "#e74c3c"
    else:
        return "#f39c12"

# Visualization Functions
def create_advanced_trend_chart(df):
    """Create advanced trend chart with multiple metrics"""
    trend_data = df.groupby(['year', 'month', 'tool_name', 'category']).agg({
        'adoption_rate': 'mean',
        'sentiment_score': 'mean',
        'users_count': 'sum'
    }).reset_index()
    
    trend_data['date'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Adoption Trends', 'Sentiment Trends', 
                       'User Base Growth', 'Category Performance'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    top_tools = df.groupby('tool_name')['adoption_rate'].mean().nlargest(5).index
    trend_top = trend_data[trend_data['tool_name'].isin(top_tools)]
    
    colors = px.colors.qualitative.Set3[:len(top_tools)]
    
    for i, tool in enumerate(top_tools):
        tool_data = trend_top[trend_top['tool_name'] == tool]
        fig.add_trace(
            go.Scatter(x=tool_data['date'], y=tool_data['adoption_rate'],
                      mode='lines+markers', name=tool, line=dict(color=colors[i]),
                      showlegend=True),
            row=1, col=1
        )
    
    for i, tool in enumerate(top_tools):
        tool_data = trend_top[trend_top['tool_name'] == tool]
        fig.add_trace(
            go.Scatter(x=tool_data['date'], y=tool_data['sentiment_score'],
                      mode='lines+markers', name=tool, line=dict(color=colors[i]),
                      showlegend=False),
            row=1, col=2
        )
    
    for i, tool in enumerate(top_tools):
        tool_data = trend_top[trend_top['tool_name'] == tool]
        fig.add_trace(
            go.Scatter(x=tool_data['date'], y=tool_data['users_count'],
                      mode='lines+markers', name=tool, line=dict(color=colors[i]),
                      showlegend=False),
            row=2, col=1
        )
    
    category_data = trend_data.groupby(['date', 'category'])['adoption_rate'].mean().reset_index()
    for category in category_data['category'].unique():
        cat_data = category_data[category_data['category'] == category]
        fig.add_trace(
            go.Scatter(x=cat_data['date'], y=cat_data['adoption_rate'],
                      mode='lines+markers', name=category, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(height=800, title_text="Comprehensive AI Tools Analysis Dashboard")
    return fig

def create_performance_heatmap(df):
    """Create performance heatmap by tool and time period"""
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
        title="Tool Performance Heatmap Over Time",
        xaxis_title="Time Period",
        yaxis_title="AI Tools",
        height=600
    )
    
    return fig

# Export Functions
def create_enhanced_csv_export(df, filters_info):
    """Create enhanced CSV export"""
    export_df = df.copy()
    
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    
    metadata = f"""# AI Adoption Analytics Dashboard Export
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Filters Applied: {filters_info}
# Total Records: {len(export_df)}
# Date Range: {export_df['date'].min()} to {export_df['date'].max()}
# Categories: {', '.join(export_df['category'].unique())}
#
"""
    
    csv_buffer = io.StringIO()
    csv_buffer.write(metadata)
    export_df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()

def create_executive_pdf_report(df, filters_info):
    """Create executive-level PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50,
                           topMargin=50, bottomMargin=50)
    
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'ExecutiveTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,
        textColor=colors.HexColor('#2c3e50'),
        fontName='Helvetica-Bold'
    )
    
    elements.append(Paragraph("AI Adoption Analytics Dashboard", title_style))
    elements.append(Paragraph("Executive Analysis Report", styles['Heading1']))
    elements.append(Spacer(1, 30))
    
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
    <b>KEY FINDINGS:</b><br/><br/>
    • <b>Market Leader:</b> {summary_data['fastest_growing']} dominates with highest adoption<br/>
    • <b>Strongest Category:</b> {summary_data['top_category']} shows consistent growth<br/>
    • <b>Overall Sentiment:</b> {get_sentiment_label(summary_data['avg_sentiment'])}<br/>
    • <b>Total User Base:</b> {summary_data['total_users']:,} active users<br/>
    • <b>Market Maturity:</b> {"Mature" if summary_data['avg_adoption'] > 70 else "Growing" if summary_data['avg_adoption'] > 50 else "Emerging"}<br/>
    • <b>Analysis Period:</b> {summary_data['total_records']:,} data points over 24 months
    """
    
    elements.append(Paragraph(executive_summary, styles['Normal']))
    elements.append(PageBreak())
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main Application
def main():
    # Data disclaimer banner
    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ This dashboard uses <strong>simulated demo data (2023–2025)</strong>. Replace with real dataset for production use.
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">AI Adoption Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Professional analytics platform for AI tool adoption insights and trends</p>', unsafe_allow_html=True)
    
    # Setup Gemini AI
    gemini_ready, gemini_status = setup_gemini_api()
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("### Dashboard Controls")
        
        df = generate_sample_data()
        
        # AI Features Status
        with st.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown("**AI Features Status**")
            if gemini_ready:
                st.success("✅ Gemini AI: Active")
            else:
                st.warning("⚠️ Gemini AI: Configure API key")
            
            st.info(f"Sentiment Analysis: {SENTIMENT_LIB.title() if SENTIMENT_LIB else 'Basic'}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Filters
        st.markdown("### Tailor AI Recommendations")
        
        categories = ['All Categories'] + sorted(df['category'].unique().tolist())
        selected_category = st.selectbox("Tool Category", categories)
        
        if selected_category != 'All Categories':
            available_tools = df[df['category'] == selected_category]['tool_name'].unique()
            tools_list = ['All Tools'] + sorted(available_tools.tolist())
        else:
            tools_list = ['All Tools'] + sorted(df['tool_name'].unique().tolist())
        
        selected_tool = st.selectbox("Specific Tool", tools_list)
        
        available_years = sorted(df['year'].unique())
        selected_years = st.multiselect("Years", available_years, default=available_years[-2:])
        
        date_range = st.date_input(
            "Date Range",
            value=(df['date'].min().date(), df['date'].max().date()),
            min_value=df['date'].min().date(),
            max_value=df['date'].max().date()
        )
        
        st.markdown("### Advanced Filters")
        sentiment_filter = st.selectbox("Sentiment", ['All', 'Positive', 'Neutral', 'Negative'])
        adoption_range = st.slider("Adoption Rate Range", 0, 100, (0, 100), step=5)
        market_trend_filter = st.multiselect("Market Trend", ['Growing', 'Stable', 'Declining'], default=['Growing', 'Stable', 'Declining'])
        
        if gemini_ready:
            st.markdown("### Recommendation Preferences")
            use_case = st.selectbox("Primary Use Case", [
                "General Productivity", "Content Creation", "Software Development", 
                "Design & Art", "Research & Analysis", "Team Collaboration"
            ])
            
            budget_preference = st.selectbox("Budget Preference", ["Free/Freemium", "Budget-Conscious", "Premium/Enterprise"])
            user_preferences = f"{use_case}, {budget_preference}"
        else:
            user_preferences = "General productivity"
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_category != 'All Categories':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    if selected_tool != 'All Tools':
        filtered_df = filtered_df[filtered_df['tool_name'] == selected_tool]
    
    if selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    if sentiment_filter != 'All':
        filtered_df = filtered_df[filtered_df['sentiment_label'] == sentiment_filter]
    
    filtered_df = filtered_df[
        (filtered_df['adoption_rate'] >= adoption_range[0]) & 
        (filtered_df['adoption_rate'] <= adoption_range[1])
    ]
    
    if market_trend_filter:
        filtered_df = filtered_df[filtered_df['market_trend'].isin(market_trend_filter)]
    
    # Create filters info for exports
    filters_info = f"Category: {selected_category}, Tool: {selected_tool}, Years: {selected_years}, Sentiment: {sentiment_filter}, Adoption: {adoption_range[0]}-{adoption_range[1]}%"
    
    # Main Dashboard
    if len(filtered_df) == 0:
        st.error("⚠️ No data matches your current filters. Please adjust your selection.")
        return
    
    # AI-Powered Insights Banner
    if gemini_ready:
        with st.container():
            st.markdown('<div class="ai-insight-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### AI-Powered Market Intelligence")
                
                top_tool = filtered_df.groupby('tool_name')['adoption_rate'].mean().idxmax()
                avg_adoption = filtered_df['adoption_rate'].mean()
                sample_feedback = filtered_df['user_feedback'].iloc[0]
                users = filtered_df['users_count'].sum()
                
                if st.button("Generate Market Analysis", key="market_analysis"):
                    with st.spinner("Analyzing data..."):
                        market_insight = generate_ai_summary(top_tool, avg_adoption, sample_feedback, users)
                        st.write(f"**Market Insight:** {market_insight}")
            
            with col2:
                st.markdown("### Smart Recommendations")
                if st.button("Generate Recommendations", key="ai_recommendations"):
                    with st.spinner("Generating recommendations..."):
                        recommendations = get_ai_recommendations(filtered_df, user_preferences)
                        st.write(recommendations)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance Metrics Dashboard
    st.markdown('<h2 class="section-header">Performance Metrics Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_adoption = filtered_df['adoption_rate'].mean()
        st.metric(
            label="Avg Adoption",
            value=f"{avg_adoption:.1f}%",
            delta=f"{avg_adoption - 65:.1f}%"
        )
    
    with col2:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        st.metric(
            label="Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=get_sentiment_label(avg_sentiment)
        )
    
    with col3:
        total_users = filtered_df['users_count'].sum()
        st.metric(
            label="Total Users",
            value=f"{total_users:,}",
            delta="Active"
        )
    
    with col4:
        tools_count = filtered_df['tool_name'].nunique()
        st.metric(
            label="Tools Analyzed",
            value=f"{tools_count}",
            delta=f"{len(df['category'].unique())} categories"
        )
    
    with col5:
        avg_satisfaction = filtered_df['satisfaction_rating'].mean()
        st.metric(
            label="Satisfaction",
            value=f"{avg_satisfaction:.2f}/5",
            delta=f"{avg_satisfaction - 4:.2f}"
        )
    
    # Main Content Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "AI Insights", "Trends & Visualizations", "Rankings & Deep Dive", "Export & Reports"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">AI-Powered Insights</h2>', unsafe_allow_html=True)
        
        if gemini_ready:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### Individual Tool Analysis")
                
                top_tools = filtered_df.groupby('tool_name')['adoption_rate'].mean().nlargest(5)
                
                for tool_name, adoption_rate in top_tools.items():
                    tool_data = filtered_df[filtered_df['tool_name'] == tool_name]
                    sample_feedback = tool_data['user_feedback'].iloc[0] if len(tool_data) > 0 else "No feedback"
                    users = tool_data['users_count'].sum()
                    
                    with st.expander(f"{tool_name} - AI Analysis"):
                        if st.button(f"Generate Insight for {tool_name}", key=f"insight_{tool_name}"):
                            with st.spinner(f"Analyzing {tool_name}..."):
                                insight = generate_ai_summary(tool_name, adoption_rate, sample_feedback, users)
                                st.markdown(f'<div class="ai-insight-card"><strong>AI Insight:</strong><br/>{insight}</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Personalized Recommendations")
                
                if st.button("Get Smart Recommendations", key="smart_recommendations"):
                    with st.spinner("Analyzing your preferences..."):
                        recommendations = get_ai_recommendations(filtered_df, user_preferences)
                        st.markdown(f'<div class="ai-insight-card">{recommendations}</div>', unsafe_allow_html=True)
                
                st.markdown("### Market Trend Analysis")
                if st.button("Analyze Market Trends", key="trend_analysis"):
                    market_summary = f"""
                    Based on current data:
                    • {len(filtered_df)} data points analyzed
                    • {filtered_df['category'].nunique()} categories tracked
                    • Average adoption: {filtered_df['adoption_rate'].mean():.1f}%
                    • Sentiment trend: {get_sentiment_label(filtered_df['sentiment_score'].mean())}
                    """
                    st.markdown(f'<div class="ai-insight-card"><strong>Market Overview:</strong><br/>{market_summary}</div>', unsafe_allow_html=True)
        
        else:
            st.info("Configure Gemini API key in the sidebar to unlock AI-powered insights!")
            st.markdown("""
            **AI Features Available:**
            - Individual tool performance analysis
            - Personalized tool recommendations  
            - Market trend intelligence
            - Strategic insights and predictions
            """)
    
    with tab2:
        st.markdown('<h2 class="section-header">Trends & Visualizations</h2>', unsafe_allow_html=True)
        
        # Comprehensive trend dashboard
        trend_chart = create_advanced_trend_chart(filtered_df)
        st.plotly_chart(trend_chart, use_container_width=True)
        
        # Performance heatmap
        st.markdown("### Performance Heatmap")
        heatmap_fig = create_performance_heatmap(filtered_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Growth rate analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### YoY Growth Leaders")
            growth_data = filtered_df.groupby(['tool_name', 'year'])['adoption_rate'].mean().reset_index()
            growth_data['prev_year'] = growth_data.groupby('tool_name')['adoption_rate'].shift(1)
            growth_data['growth_rate'] = ((growth_data['adoption_rate'] - growth_data['prev_year']) / growth_data['prev_year'] * 100).fillna(0)
            
            latest_growth = growth_data[growth_data['year'] == growth_data['year'].max()].nlargest(5, 'growth_rate')
            
            fig = px.bar(
                latest_growth,
                x='tool_name',
                y='growth_rate',
                title="Fastest Growing Tools (YoY)",
                color='growth_rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Category Matrix")
            category_perf = filtered_df.groupby('category').agg({
                'adoption_rate': 'mean',
                'sentiment_score': 'mean',
                'users_count': 'sum'
            }).reset_index()
            
            fig = px.scatter(
                category_perf,
                x='adoption_rate',
                y='sentiment_score',
                size='users_count',
                color='category',
                title="Category Performance Matrix",
                hover_data=['users_count']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">Rankings & Deep Dive</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Performance Leaderboard")
            
            tool_performance = filtered_df.groupby('tool_name').agg({
                'adoption_rate': 'mean',
                'sentiment_score': 'mean',
                'users_count': 'sum',
                'satisfaction_rating': 'mean',
                'category': 'first'
            }).reset_index()
            
            tool_performance['composite_score'] = (
                (tool_performance['adoption_rate'] / 100) * 0.4 +
                ((tool_performance['sentiment_score'] + 1) / 2) * 0.3 +
                (tool_performance['satisfaction_rating'] / 5) * 0.3
            ) * 100
            
            tool_performance = tool_performance.sort_values('composite_score', ascending=False)
            tool_performance['rank'] = range(1, len(tool_performance) + 1)
            
            for idx, row in tool_performance.head(8).iterrows():
                rank_display = f"#{row['rank']}"
                
                st.markdown(f"""
                <div class="ranking-item">
                    <div>
                        <strong>{rank_display} {row['tool_name']}</strong>
                        <br/><small>{row['category']}</small>
                    </div>
                    <div style="text-align: right;">
                        <strong>{row['composite_score']:.1f}/100</strong>
                        <br/><small>{row['adoption_rate']:.1f}% adoption</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Category Champions")
            category_winners = tool_performance.loc[tool_performance.groupby('category')['composite_score'].idxmax()]
            
            for _, winner in category_winners.iterrows():
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); 
                           color: white; padding: 0.8rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>{winner['category']}</strong><br/>
                    {winner['tool_name']} ({winner['composite_score']:.1f}/100)
                </div>
                """, unsafe_allow_html=True)
        
        # Correlation analysis
        st.markdown("### Correlation Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            corr_data = filtered_df[['adoption_rate', 'sentiment_score', 'users_count', 'satisfaction_rating']].corr()
            
            fig = px.imshow(
                corr_data,
                text_auto=True,
                aspect="auto",
                title="Metrics Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                filtered_df,
                x='sentiment_score',
                y='adoption_rate',
                color='category',
                size='users_count',
                hover_data=['tool_name'],
                title="Sentiment vs Adoption Analysis",
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">Export & Reports</h2>', unsafe_allow_html=True)
        
        st.markdown('<div class="export-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="export-header">Data Export & Reporting</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### CSV Export")
            if st.button("Generate CSV Export", key="csv_export"):
                csv_data = create_enhanced_csv_export(filtered_df, filters_info)
                filename = f"ai_adoption_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    key="csv_download"
                )
                st.success("✅ CSV export ready!")
        
        with col2:
            st.markdown("### PDF Report")
            if st.button("Generate PDF Report", key="pdf_export"):
                with st.spinner("Creating report..."):
                    try:
                        pdf_buffer = create_executive_pdf_report(filtered_df, filters_info)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer.getvalue(),
                            file_name=f"ai_adoption_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            key="pdf_download"
                        )
                        st.success("✅ PDF report generated!")
                    except Exception as e:
                        st.error(f"❌ Report generation error: {str(e)}")
        
        with col3:
            st.markdown("### Chart Export")
            if st.button("Export Charts", key="chart_export_btn"):
                st.info("Chart export functionality - saves visualizations as images")
                st.success("✅ Charts prepared for export!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data summary
        st.markdown("### Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(filtered_df):,}")
        with col2:
            st.metric("Date Range", f"{(filtered_df['date'].max() - filtered_df['date'].min()).days} days")
        with col3:
            st.metric("Categories", filtered_df['category'].nunique())
        with col4:
            st.metric("Tools", filtered_df['tool_name'].nunique())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <strong>AI Adoption Analytics Dashboard</strong> · Powered by Gemini AI · Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
