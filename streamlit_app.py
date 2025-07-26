import streamlit as st
import pandas as pd
import altair as alt
from textblob import TextBlob
import nltk
import pdfkit
import requests
import gdown
import google.generativeai as genai

# Download required NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        with st.spinner("Downloading language models..."):
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)

# Download NLTK data
download_nltk_data()

# Page config
st.set_page_config(page_title="AI Tool Recommender", layout="wide")

# Load the dataset from Google Drive
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?id=14j9MWeeHn4v9ZNSnqIy6WGZdFVKdgFc1"
        output = "ai_tool_data.csv"
        gdown.download(url, output, quiet=False)
        return pd.read_csv(output)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data loaded successfully
if df.empty:
    st.stop()

# Sidebar filters
industry_filter = st.sidebar.selectbox("Select Industry", sorted(df["industry"].unique()))
year_filter = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()))
filtered_df = df[(df["industry"] == industry_filter) & (df["year"] == year_filter)]

# Title
st.title("🚀 AI Tool Recommender App – Smart Insights")
st.subheader(f"🔍 Filtered AI Tools for {industry_filter} in {year_filter}")
st.dataframe(filtered_df)

# 📈 Adoption Trend
st.markdown("### 📈 AI Tool Adoption Trend")
trend_df = df[df["industry"] == industry_filter].groupby("year")["adoption_rate"].mean().reset_index()
trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
    x='year:O',
    y='adoption_rate:Q'
).properties(
    title=f"{industry_filter} - Adoption Rate Over Years"
)
st.altair_chart(trend_chart, use_container_width=True)

# 💬 Sentiment Analysis
st.markdown("### 💬 Sentiment Analysis")
try:
    df["sentiment_score"] = df["user_feedback"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    sentiment_avg = df[df["industry"] == industry_filter].groupby("AI_tool")["sentiment_score"].mean().reset_index()
    sentiment_chart = alt.Chart(sentiment_avg).mark_bar().encode(
        x=alt.X("AI_tool:N", sort="-y"),
        y="sentiment_score:Q",
        color=alt.Color("sentiment_score:Q", scale=alt.Scale(scheme="redyellowgreen"))
    ).properties(
        width=700,
        height=400,
        title="Average Sentiment Score by Tool"
    )
    st.altair_chart(sentiment_chart, use_container_width=True)
except Exception as e:
    st.error(f"Error in sentiment analysis: {str(e)}")

# 📤 Export to PDF
st.markdown("### 📤 Export Filtered Results")
try:
    if st.button("Generate PDF"):
        with st.spinner("Generating PDF..."):
            html = filtered_df.to_html(index=False)
            pdf_file = "filtered_results.pdf"
            pdfkit.from_string(html, pdf_file)
            
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "📥 Download PDF", 
                    f, 
                    file_name="AI_Tool_Report.pdf",
                    mime="application/pdf"
                )
except Exception as e:
    st.error(f"PDF generation error: {str(e)}")
    st.info("PDF generation requires additional system packages. Try running locally or contact support.")

# 🤖 Google Gemini-Powered Summary
st.markdown("### 🤖 Google Gemini AI Summary")

def get_gemini_summary(industry, year):
    # Get API key from Streamlit secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        return "⚠️ Gemini API key not found. Please add GEMINI_API_KEY to your Streamlit secrets."
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Create the model
    model = genai.GenerativeModel('gemini-pro')
    
    # Create prompt with actual data context
    top_tools = filtered_df.nlargest(3, 'adoption_rate')['AI_tool'].tolist() if not filtered_df.empty else []
    
    prompt = f"""
    Based on the AI tools data for {industry} industry in {year}, provide a 3-line summary focusing on:
    
    Top tools by adoption rate: {', '.join(top_tools) if top_tools else 'No data available'}
    
    Please explain:
    1. Which AI tools are most popular in {industry} industry
    2. Why these tools are gaining adoption
    3. What this means for businesses in this sector
    
    Keep it concise and actionable.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Failed to get Gemini summary: {str(e)}"

if st.button("🔮 Generate AI Summary with Gemini"):
    with st.spinner("Generating AI insights with Google Gemini..."):
        summary = get_gemini_summary(industry_filter, year_filter)
        st.success(summary)

# Additional insights section
st.markdown("### 📊 Quick Insights")
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_adoption = filtered_df['adoption_rate'].mean()
        st.metric("Average Adoption Rate", f"{avg_adoption:.1f}%")
    
    with col2:
        top_tool = filtered_df.loc[filtered_df['adoption_rate'].idxmax(), 'AI_tool']
        st.metric("Top Tool", top_tool)
    
    with col3:
        total_tools = len(filtered_df)
        st.metric("Total Tools", total_tools)
else:
    st.info("No data available for the selected filters.")
