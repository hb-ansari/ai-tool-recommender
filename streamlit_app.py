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
    page_title="AI Tools Dashboard - Trend Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling (keeping your existing styles + new export section)
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
    
    .success-message {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sentiment Analysis Functions (keeping your existing functions)
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

# Keep all your existing data generation and analysis functions
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
    st.markdown('<h1 class="main-header">🧠 AI Tools Dashboard - Trend Analysis</h1>', unsafe_allow_html=True)
    
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
            value=f"{avg_sentiment:.3f
