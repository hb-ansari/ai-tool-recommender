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

# NEW: Export Libraries
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import os

# Optional: Email functionality
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
    page_title="AI Tools Dashboard - Trend Analysis & Export",
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
    
    .trend-subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .ranking-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .growth-positive {
        color: #27ae60;
        font-weight: bold;
    }
    
    .growth-negative {
        color: #e74c3c;
        font-weight: bold;
    }
    
    /* NEW: Export section styling */
    .export-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #667eea;
    }
    
    .export-header {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .export-button {
        margin: 0.5rem;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        border-radius: 10px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
    }
    
    .export-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .success-message {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    .stMetric > div > div > div > div {
        color: #667eea;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
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

# NEW: Export Functions
def create_csv_export(df, filters_info):
    """Create CSV export with metadata"""
    # Prepare export DataFrame
    export_df = df[['tool_name', 'date', 'year', 'month', 'adoption_rate', 
                   'user_feedback', 'sentiment_score', 'sentiment_label', 
                   'users_count', 'satisfaction_rating']].copy()
    
    # Format date for better readability
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
    
    # Add metadata as header comments
    metadata = f"""# AI Tools Dashboard Export
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Filters Applied: {filters_info}
# Total Records: {len(export_df)}
# Date Range: {export_df['date'].min()} to {export_df['date'].max()}
#
"""
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    csv_buffer.write(metadata)
    export_df.to_csv(csv_buffer, index=False)
    
    return csv_buffer.getvalue()

def create_pdf_report(df, filters_info, charts_data=None):
    """Create comprehensive PDF report"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center alignment
        textColor=colors.HexColor('#667eea')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#2c3e50')
    )
    
    # Title Page
    elements.append(Paragraph("🧠 AI Tools Dashboard", title_style))
    elements.append(Paragraph("Comprehensive Trend Analysis Report", styles['Heading2']))
    elements.append(Spacer(1, 20))
    
    # Report Info
    report_info = f"""
    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Filters Applied:</b> {filters_info}<br/>
    <b>Total Records:</b> {len(df):,}<br/>
    <b>Date Range:</b> {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}<br/>
    <b>Analysis Period:</b> {len(df['year'].unique())} years, {len(df['tool_name'].unique())} tools
    """
    elements.append(Paragraph(report_info, styles['Normal']))
    elements.append(Spacer(1, 30))
    
    # Executive Summary
    elements.append(Paragraph("📊 Executive Summary", heading_style))
    
    # Calculate key metrics
    avg_adoption = df['adoption_rate'].mean()
    avg_sentiment = df['sentiment_score'].mean()
    total_users = df['users_count'].sum()
    top_tool = df.groupby('tool_name')['adoption_rate'].mean().idxmax()
    best_sentiment_tool = df.groupby('tool_name')['sentiment_score'].mean().idxmax()
    
    summary_text = f"""
    • <b>Average Adoption Rate:</b> {avg_adoption:.1f}%<br/>
    • <b>Overall Sentiment Score:</b> {avg_sentiment:.3f} ({get_sentiment_label(avg_sentiment)})<br/>
    • <b>Total User Base:</b> {total_users:,} users<br/>
    • <b>Top Performing Tool:</b> {top_tool}<br/>
    • <b>Highest User Satisfaction:</b> {best_sentiment_tool}<br/>
    • <b>Market Trend:</b> {"Growing" if avg_adoption > 65 else "Stable" if avg_adoption > 50 else "Emerging"}
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Tool Performance Table
    elements.append(Paragraph("🏆 Tool Performance Rankings", heading_style))
    
    # Create performance summary table
    tool_summary = df.groupby('tool_name').agg({
        'adoption_rate': 'mean',
        'sentiment_score': 'mean',
        'users_count': 'sum',
        'satisfaction_rating': 'mean'
    }).round(2).reset_index()
    
    tool_summary = tool_summary.sort_values('adoption_rate', ascending=False)
    tool_summary['rank'] = range(1, len(tool_summary) + 1)
    
    # Prepare table data
    table_data = [['Rank', 'Tool Name', 'Adoption Rate', 'Sentiment', 'Users', 'Satisfaction']]
    for _, row in tool_summary.iterrows():
        table_data.append([
            str(row['rank']),
            row['tool_name'],
            f"{row['adoption_rate']:.1f}%",
            f"{row['sentiment_score']:.3f}",
            f"{row['users_count']:,}",
            f"{row['satisfaction_rating']:.2f}/5"
        ])
    
    # Create table
    table = Table(table_data, colWidths=[0.8*inch, 1.5*inch, 1.2*inch, 1*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Sentiment Analysis Section
    elements.append(Paragraph("💭 Sentiment Analysis Insights", heading_style))
    
    sentiment_dist = df['sentiment_label'].value_counts()
    sentiment_text = f"""
    <b>Sentiment Distribution:</b><br/>
    • Positive: {sentiment_dist.get('😊 Positive', 0)} records ({sentiment_dist.get('😊 Positive', 0)/len(df)*100:.1f}%)<br/>
    • Neutral: {sentiment_dist.get('😐 Neutral', 0)} records ({sentiment_dist.get('😐 Neutral', 0)/len(df)*100:.1f}%)<br/>
    • Negative: {sentiment_dist.get('😞 Negative', 0)} records ({sentiment_dist.get('😞 Negative', 0)/len(df)*100:.1f}%)<br/><br/>
    
    <b>Key Findings:</b><br/>
    • Overall sentiment is {get_sentiment_label(avg_sentiment).lower()}<br/>
    • {sentiment_dist.index[0]} feedback dominates user responses<br/>
    • Sentiment analysis powered by {SENTIMENT_LIB if SENTIMENT_LIB else "keyword-based analysis"}
    """
    elements.append(Paragraph(sentiment_text, styles['Normal']))
    elements.append(PageBreak())
    
    # Data Sample Section
    elements.append(Paragraph("📋 Data Sample (Latest Records)", heading_style))
    
    # Get latest 10 records
    latest_data = df.nlargest(10, 'date')[['tool_name', 'date', 'adoption_rate', 'sentiment_score', 'user_feedback']]
    
    sample_table_data = [['Tool', 'Date', 'Adoption Rate', 'Sentiment', 'User Feedback']]
    for _, row in latest_data.iterrows():
        feedback_preview = (row['user_feedback'][:40] + '...') if len(str(row['user_feedback'])) > 40 else str(row['user_feedback'])
        sample_table_data.append([
            row['tool_name'],
            row['date'].strftime('%Y-%m-%d'),
            f"{row['adoption_rate']:.1f}%",
            f"{row['sentiment_score']:.2f}",
            feedback_preview
        ])
    
    sample_table = Table(sample_table_data, colWidths=[1.2*inch, 1*inch, 1*inch, 0.8*inch, 2.5*inch])
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    
    elements.append(sample_table)
    elements.append(Spacer(1, 20))
    
    # Footer
    footer_text = f"""
    <br/><br/>
    <i>This report was generated by AI Tools Dashboard on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}.<br/>
    For more detailed analysis and interactive charts, visit the dashboard application.</i>
    """
    elements.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def send_email_report(to_email, subject, body, attachment_data=None, attachment_name=None):
    """Send email with optional attachment"""
    if not EMAIL_AVAILABLE:
        return False, "Email libraries not available"
    
    try:
        # Email configuration (in production, use environment variables)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "your-dashboard@example.com"  # Replace with actual email
        sender_password = "your-app-password"  # Replace with actual app password
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add body
        msg.attach(MIMEText(body, 'html'))
        
        # Add attachment if provided
        if attachment_data and attachment_name:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment_data)
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {attachment_name}'
            )
            msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, to_email, text)
        server.quit()
        
        return True, "Email sent successfully!"
        
    except Exception as e:
        return False, f"Failed to send email: {str(e)}"

# Enhanced Data Generation for Trends
@st.cache_data
def generate_sample_data():
    """Generate sample data with historical trends for better analysis"""
    
    # AI Tools data
    tools = [
        "ChatGPT", "Claude", "Gemini", "Copilot", "Midjourney", 
        "Stable Diffusion", "Notion AI", "Jasper", "Copy.ai", "Grammarly"
    ]
    
    # Sample feedback comments
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
    
    # Generate data with more structured time series
    np.random.seed(42)
    data = []
    
    # Generate data for the last 2 years with monthly progression
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 1, 31)
    
    # Create base trends for each tool
    tool_trends = {
        "ChatGPT": {"base": 75, "trend": 0.8},
        "Claude": {"base": 65, "trend": 1.2},
        "Gemini": {"base": 60, "trend": 1.5},
        "Copilot": {"base": 70, "trend": 0.5},
        "Midjourney": {"base": 55, "trend": 0.7},
        "Stable Diffusion": {"base": 50, "trend": 0.6},
        "Notion AI": {"base": 45, "trend": 1.0},
        "Jasper": {"base": 40, "trend": 0.3},
        "Copy.ai": {"base": 35, "trend": 0.4},
        "Grammarly": {"base": 80, "trend": 0.2}
    }
    
    current_date = start_date
    month_counter = 0
    
    while current_date <= end_date:
        for tool in tools:
            # Generate multiple records per tool per month
            for _ in range(np.random.randint(5, 15)):
                base_rate = tool_trends[tool]["base"]
                trend_factor = tool_trends[tool]["trend"]
                
                # Add trend over time + some randomness
                adoption_rate = (base_rate + 
                               (month_counter * trend_factor) + 
                               np.random.normal(0, 8))
                adoption_rate = max(10, min(95, adoption_rate))
                
                # Add some random days within the month
                random_day = np.random.randint(0, 28)
                record_date = current_date + timedelta(days=random_day)
                
                user_feedback = np.random.choice(feedback_samples)
                
                data.append({
                    'tool_name': tool,
                    'date': record_date,
                    'year': record_date.year,
                    'month': record_date.month,
                    'adoption_rate': round(adoption_rate, 1),
                    'user_feedback': user_feedback,
                    'users_count': np.random.randint(100, 5000),
                    'satisfaction_rating': np.random.uniform(2.5, 4.8)
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

def create_trend_chart(df):
    """Create interactive trend chart using Altair"""
    # Prepare data for trends (monthly averages)
    trend_data = df.groupby(['year', 'month', 'tool_name']).agg({
        'adoption_rate': 'mean',
        'sentiment_score': 'mean'
    }).reset_index()
    
    # Create date column for proper time series
    trend_data['date'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
    
    # Create the line chart
    line_chart = alt.Chart(trend_data).mark_line(point=True, strokeWidth=3).add_selection(
        alt.selection_multi(fields=['tool_name'])
    ).encode(
        x=alt.X('date:T', title='Date', axis=alt.Axis(format='%Y-%m')),
        y=alt.Y('adoption_rate:Q', title='Adoption Rate (%)', scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('tool_name:N', title='AI Tool', scale=alt.Scale(scheme='category10')),
        tooltip=['tool_name:N', 'date:T', 'adoption_rate:Q', 'sentiment_score:Q'],
        opacity=alt.condition(alt.datum.tool_name, alt.value(0.8), alt.value(0.2))
    ).properties(
        width=800,
        height=400,
        title="📈 Adoption Trends of AI Tools (2023-2025)"
    )
    
    return line_chart

def get_top_tools_ranking(df, year):
    """Get top 5 tools by adoption rate for a specific year"""
    year_data = df[df['year'] == year]
    ranking = year_data.groupby('tool_name').agg({
        'adoption_rate': 'mean',
        'sentiment_score': 'mean',
        'users_count': 'sum'
    }).reset_index()
    
    ranking = ranking.sort_values('adoption_rate', ascending=False).head(5)
    ranking['rank'] = range(1, len(ranking) + 1)
    
    # Calculate growth compared to previous year if available
    if year > 2023:
        prev_year_data = df[df['year'] == year - 1]
        prev_ranking = prev_year_data.groupby('tool_name')['adoption_rate'].mean()
        
        ranking['prev_adoption'] = ranking['tool_name'].map(prev_ranking)
        ranking['growth'] = ((ranking['adoption_rate'] - ranking['prev_adoption']) / ranking['prev_adoption'] * 100).fillna(0)
    else:
        ranking['growth'] = 0
    
    return ranking

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 AI Tools Dashboard - Trend Analysis & Export</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.subheader("🔧 Filters")
        
        # Load data
        df = generate_sample_data()
        
        # Tool filter
        tools_list = ['All'] + sorted(df['tool_name'].unique().tolist())
        selected_tool = st.selectbox("Select AI Tool", tools_list)
        
        # Year filter for trends
        available_years = sorted(df['year'].unique())
        selected_year = st.selectbox("Select Year for Rankings", available_years, index=len(available_years)-1)
        
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
    
    # Create filters info string for exports
    filters_info = f"Tool: {selected_tool}, Year: {selected_year}, Dates: {date_range[0] if len(date_range) > 0 else 'All'} to {date_range[1] if len(date_range) > 1 else 'All'}, Sentiment: {sentiment_filter}"
    
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
    
    # NEW: Export & Share Section
    st.markdown('<div class="export-section">', unsafe_allow_html=True)
    st.markdown('<h2 class="export-header">📦 Export & Share Results</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📂 Download CSV", key="csv_export", help="Export filtered data as CSV"):
            csv_data = create_csv_export(filtered_df, filters_info)
            st.download_button(
                label="📥 Download CSV File",
                data=csv_data,
                file_name=f"ai_tools_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="csv_download"
            )
            st.success("✅ CSV ready for download!")
    
    with col2:
        if st.button("📑 Export PDF Report", key="pdf_export", help="Generate comprehensive PDF report"):
            with st.spinner("🔄 Generating PDF report..."):
                try:
                    pdf_buffer = create_pdf_report(filtered_df, filters_info)
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"ai_tools_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="pdf_download"
                    )
                    st.success("✅ PDF report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
    
    with col3:
        if EMAIL_AVAILABLE:
            with st.popover("✉️ Email Report"):
                st.write("Send analysis via email")
                email_address = st.text_input("📧 Email Address", key="email_input")
                if st.button("📤 Send Report", key="email_send"):
                    if email_address and "@" in email_address:
                        # Note: This is a demo - in production, configure proper SMTP
                        st.info("📧 Email feature requires SMTP configuration")
                        st.code("Configure SMTP settings in the send_email_report function")
                    else:
                        st.error("❌ Please enter a valid email address")
        else:
            st.button("✉️ Email Report", disabled=True, help="Email libraries not installed")
    
    with col4:
        if st.button("🔗 Share Dashboard", key="share_dashboard", help="Generate shareable link"):
            st.success("🎉 Dashboard link generated!")
            share_url = f"https://ai-tools-dashboard.streamlit.app/?filters={hash(filters_info)}"
            st.code(share_url)
            st.caption("💡 Copy this link to share your filtered view")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Tabs Layout for organized content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Filtered Data & Sentiment", "📈 Trends & Rankings", "🔍 Detailed Analysis", "📋 Export Summary"])
    
    with tab1:
        # Original sentiment analysis content
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
        
        # Data Table with Sentiment
        st.markdown('<h2 class="section-header">📋 Detailed Data with Sentiment Scores</h2>', unsafe_allow_html=True)
        
        # Format the dataframe for display
        display_df = filtered_df[['tool_name', 'date', 'adoption_rate', 'user_feedback', 'sentiment_score', 'sentiment_label']].copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df = display_df.sort_values('sentiment_score', ascending=False)
        
        st.dataframe(display_df, use_container_width=True, height=400)
    
    with tab2:
        # Trends and Rankings Tab
        st.markdown('<p class="trend-subtitle">📈 Adoption Trends of AI Tools (2023–2025)</p>', unsafe_allow_html=True)
        
        # Trend Chart using Altair
        try:
            trend_chart = create_trend_chart(df)  # Use full dataset for trends
            st.altair_chart(trend_chart, use_container_width=True)
        except Exception as e:
            # Fallback to Plotly if Altair has issues
            st.warning("Using Plotly chart as fallback")
            trend_data = df.groupby(['year', 'month', 'tool_name']).agg({
                'adoption_rate': 'mean'
            }).reset_index()
            trend_data['date'] = pd.to_datetime(trend_data[['year', 'month']].assign(day=1))
            
            fig = px.line(
                trend_data,
                x='date',
                y='adoption_rate',
                color='tool_name',
                title="📈 Adoption Trends of AI Tools (2023-2025)",
                markers=True
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rankings Section
        st.markdown('<p class="trend-subtitle">🏆 Top Performing Tools by Year</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top 5 Tools Ranking
            ranking_df = get_top_tools_ranking(df, selected_year)
            
            st.markdown(f"### 🏆 Top 5 AI Tools - {selected_year}")
            
            for idx, row in ranking_df.iterrows():
                growth_color = "growth-positive" if row['growth'] > 0 else "growth-negative" if row['growth'] < 0 else ""
                growth_symbol = "📈" if row['growth'] > 0 else "📉" if row['growth'] < 0 else "➡️"
                
                with st.container():
                    st.markdown(f"""
                    <div class="ranking-card">
                        <h4>#{row['rank']} {row['tool_name']}</h4>
                        <p><strong>Adoption Rate:</strong> {row['adoption_rate']:.1f}%</p>
                        <p><strong>Sentiment Score:</strong> {row['sentiment_score']:.3f}</p>
                        <p><strong>Total Users:</strong> {row['users_count']:,}</p>
                        <p class="{growth_color}"><strong>YoY Growth:</strong> {growth_symbol} {row['growth']:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # Year-over-year comparison chart
            yearly_avg = df.groupby(['year', 'tool_name'])['adoption_rate'].mean().reset_index()
            top_tools = ranking_df['tool_name'].head(3).tolist()
            yearly_top = yearly_avg[yearly_avg['tool_name'].isin(top_tools)]
            
            fig = px.bar(
                yearly_top,
                x='year',
                y='adoption_rate',
                color='tool_name',
                title=f"📊 Top 3 Tools Performance by Year",
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Advanced Analysis
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
    
    with tab4:
        # NEW: Export Summary Tab
        st.markdown('<h2 class="section-header">📋 Export Summary & Data Insights</h2>', unsafe_allow_html=True)
        
        # Summary Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Current Dataset Summary")
            st.write(f"**Total Records:** {len(filtered_df):,}")
            st.write(f"**Date Range:** {filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Tools Analyzed:** {len(filtered_df['tool_name'].unique())}")
            st.write(f"**Average Adoption Rate:** {filtered_df['adoption_rate'].mean():.1f}%")
            st.write(f"**Overall Sentiment:** {get_sentiment_label(filtered_df['sentiment_score'].mean())}")
            
        with col2:
            st.markdown("### 🎯 Quick Export Options")
            
            # Quick CSV export
            csv_data = create_csv_export(filtered_df, filters_info)
            st.download_button(
                label="📥 Quick CSV Download",
                data=csv_data,
                file_name=f"ai_tools_quick_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="quick_csv"
            )
            
            # Data preview
            st.markdown("### 👀 Data Preview")
            st.dataframe(filtered_df.head(5)[['tool_name', 'adoption_rate', 'sentiment_score', 'sentiment_label']], use_container_width=True)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🧠 AI Tools Dashboard**")
        st.caption("Enhanced with Export & Automation Features")
    
    with col2:
        st.markdown("**📦 New Export Features:**")
        st.caption("• CSV data export with metadata")
        st.caption("• Professional PDF reports")
        st.caption("• Email automation (configurable)")
        st.caption("• Shareable dashboard links")
    
    with col3:
        st.markdown("**🎯 Day 9 Complete!**")
        if st.button("🎉 Celebrate Progress"):
            st.balloons()
            st.success("🚀 Congratulations! Your AI Tools Dashboard now has full export capabilities!")

if __name__ == "__main__":
    main()
