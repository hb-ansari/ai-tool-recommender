import streamlit as st
import pandas as pd
import altair as alt
import requests
import gdown
import os

# Safe imports with error handling
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ TextBlob import failed: {str(e)}")
    TEXTBLOB_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("⚠️ NLTK not available")

try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Enhanced NLTK download with SSL fix
@st.cache_resource
def download_nltk_data():
    if not NLTK_AVAILABLE:
        return False
    
    try:
        # Handle SSL certificate issues
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        
        # Check if data already exists
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        return True
    except LookupError:
        try:
            with st.spinner("Downloading language models..."):
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('brown', quiet=True)
                nltk.download('wordnet', quiet=True)
            return True
        except Exception as e:
            st.error(f"Failed to download NLTK data: {str(e)}")
            return False

# Download NLTK data
nltk_success = download_nltk_data()

# Page config
st.set_page_config(page_title="AI Tool Recommender", layout="wide")

# Enhanced data loading with sample data fallback
@st.cache_data
def load_data():
    try:
        url = "https://drive.google.com/uc?id=14j9MWeeHn4v9ZNSnqIy6WGZdFVKdgFc1"
        output = "ai_tool_data.csv"
        
        with st.spinner("Loading data from Google Drive..."):
            gdown.download(url, output, quiet=False)
        
        if not os.path.exists(output):
            raise FileNotFoundError("Failed to download the file")
            
        df = pd.read_csv(output)
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        st.success(f"✅ Successfully loaded {len(df)} records")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.info("Loading sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data if main data source fails"""
    import random
    
    industries = ['Healthcare', 'Finance', 'Manufacturing', 'Education', 'Retail']
    years = [2020, 2021, 2022, 2023, 2024]
    tools = ['ChatGPT', 'TensorFlow', 'AWS SageMaker', 'Azure ML', 'Google AI Platform']
    
    data = []
    for industry in industries:
        for year in years:
            for tool in tools:
                data.append({
                    'industry': industry,
                    'year': year,
                    'AI_tool': tool,
                    'adoption_rate': random.randint(15, 85),
                    'user_feedback': f"Excellent tool for {industry.lower()} applications. Very effective in {year}."
                })
    
    return pd.DataFrame(data)

# Helper function to find columns dynamically
def find_column(df, keywords):
    """Find column containing any of the keywords (case-insensitive)"""
    if df is None or df.empty:
        return None
    for col in df.columns:
        for keyword in keywords:
            if keyword.lower() in col.lower():
                return col
    return None

# Load data
df = load_data()

# Debug section - show data structure
with st.expander("🔍 Data Structure & Debug Info"):
    st.write("**Dataset Information:**")
    st.write(f"- Total rows: {len(df)}")
    st.write(f"- Total columns: {len(df.columns)}")
    st.write("**Column names:**")
    for i, col in enumerate(df.columns):
        st.write(f"  {i+1}. '{col}'")
    
    st.write("**Available Features:**")
    st.write(f"- TextBlob Sentiment Analysis: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
    st.write(f"- PDF Export: {'✅' if PDFKIT_AVAILABLE else '❌'}")
    st.write(f"- AI Summary: {'✅' if GEMINI_AVAILABLE else '❌'}")
    
    st.write("**Sample Data:**")
    st.dataframe(df.head())

# Check if data loaded successfully
if df.empty:
    st.error("No data available. Please check your data source.")
    st.stop()

# Sidebar filters with error handling
try:
    if 'industry' in df.columns and 'year' in df.columns:
        industry_options = sorted(df["industry"].dropna().unique())
        year_options = sorted(df["year"].dropna().unique())
        
        industry_filter = st.sidebar.selectbox("Select Industry", industry_options)
        year_filter = st.sidebar.selectbox("Select Year", year_options)
        filtered_df = df[(df["industry"] == industry_filter) & (df["year"] == year_filter)]
    else:
        st.error("Required filter columns ('industry', 'year') not found in dataset")
        st.write("Available columns:", list(df.columns))
        st.stop()
except Exception as e:
    st.error(f"Error setting up filters: {str(e)}")
    st.stop()

# Title
st.title("🚀 AI Tool Recommender App – Smart Insights")
st.subheader(f"🔍 Filtered AI Tools for {industry_filter} in {year_filter}")

if not filtered_df.empty:
    st.dataframe(filtered_df)
else:
    st.warning(f"No data found for {industry_filter} in {year_filter}")

# 📈 Adoption Trend
st.markdown("### 📈 AI Tool Adoption Trend")
try:
    adoption_col = find_column(df, ['adoption', 'rate'])
    if adoption_col and adoption_col in df.columns:
        trend_data = df[df["industry"] == industry_filter]
        if not trend_data.empty:
            trend_df = trend_data.groupby("year")[adoption_col].mean().reset_index()
            if not trend_df.empty and len(trend_df) > 0:
                trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                    x='year:O',
                    y=f'{adoption_col}:Q'
                ).properties(
                    title=f"{industry_filter} - Adoption Rate Over Years"
                )
                st.altair_chart(trend_chart, use_container_width=True)
            else:
                st.info("No trend data available for the selected industry.")
        else:
            st.info("No data available for trend analysis.")
    else:
        st.warning("Adoption rate column not found in dataset.")
        st.write("Available columns:", list(df.columns))
except Exception as e:
    st.error(f"Error creating trend chart: {str(e)}")
    st.write("Debug - Available columns:", list(df.columns) if not df.empty else "No data")

# 💬 Sentiment Analysis
st.markdown("### 💬 Sentiment Analysis")
if TEXTBLOB_AVAILABLE and nltk_success:
    try:
        # Find correct column names dynamically
        feedback_col = find_column(df, ['feedback', 'review', 'comment'])
        tool_col = find_column(df, ['tool', 'ai'])
        
        if feedback_col and tool_col and feedback_col in df.columns and tool_col in df.columns:
            # Apply sentiment analysis with null handling
            sentiment_data = df.copy()
            sentiment_data["sentiment_score"] = sentiment_data[feedback_col].apply(
                lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) and str(x).strip() != '' else 0
            )
            
            # Group by tool and calculate average sentiment for selected industry
            industry_data = sentiment_data[sentiment_data["industry"] == industry_filter]
            if not industry_data.empty:
                sentiment_avg = industry_data.groupby(tool_col)["sentiment_score"].mean().reset_index()
                
                if not sentiment_avg.empty and len(sentiment_avg) > 0:
                    sentiment_chart = alt.Chart(sentiment_avg).mark_bar().encode(
                        x=alt.X(f"{tool_col}:N", sort="-y"),
                        y="sentiment_score:Q",
                        color=alt.Color("sentiment_score:Q", scale=alt.Scale(scheme="redyellowgreen"))
                    ).properties(
                        width=700,
                        height=400,
                        title="Average Sentiment Score by Tool"
                    )
                    st.altair_chart(sentiment_chart, use_container_width=True)
                else:
                    st.info("No sentiment data available for the selected industry.")
            else:
                st.info("No data available for sentiment analysis with current filters.")
        else:
            missing_cols = []
            if not feedback_col:
                missing_cols.append("feedback/review column")
            if not tool_col:
                missing_cols.append("tool column")
            
            st.warning(f"Required columns not found: {', '.join(missing_cols)}")
            st.info("Available columns: " + ", ".join(df.columns))
            
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        st.write("Debug info:", str(e))
else:
    if not TEXTBLOB_AVAILABLE:
        st.info("💬 Sentiment Analysis disabled - TextBlob not available")
    elif not nltk_success:
        st.info("💬 Sentiment Analysis disabled - NLTK data download failed")

# 📤 Export to PDF
st.markdown("### 📤 Export Filtered Results")

# Check if wkhtmltopdf is available
def check_wkhtmltopdf():
    try:
        import subprocess
        result = subprocess.run(['which', 'wkhtmltopdf'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

if PDFKIT_AVAILABLE and check_wkhtmltopdf():
    try:
        if st.button("Generate PDF") and not filtered_df.empty:
            with st.spinner("Generating PDF..."):
                html = filtered_df.to_html(index=False, escape=False)
                pdf_file = "filtered_results.pdf"
                
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                    'no-outline': None,
                    'enable-local-file-access': None
                }
                
                pdfkit.from_string(html, pdf_file, options=options)
                
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "📥 Download PDF", 
                        f, 
                        file_name="AI_Tool_Report.pdf",
                        mime="application/pdf"
                    )
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        st.info("PDF generation requires system packages. Try CSV export instead.")
else:
    st.info("📄 PDF generation not available. Use CSV export instead.")

# Alternative CSV download
if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download as CSV",
        csv,
        file_name="AI_Tool_Report.csv",
        mime="text/csv"
    )

# 🤖 Google Gemini-Powered Summary
st.markdown("### 🤖 Google Gemini AI Summary")

def get_gemini_summary(industry, year):
    if not GEMINI_AVAILABLE:
        return "⚠️ Google Gemini not available. Please check requirements.txt"
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        return "⚠️ Gemini API key not found. Please add GEMINI_API_KEY to your Streamlit secrets."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Find correct columns dynamically
        tool_col = find_column(filtered_df, ['tool', 'ai'])
        adoption_col = find_column(filtered_df, ['adoption', 'rate'])
        
        # Get top tools if columns exist
        top_tools = []
        if tool_col and adoption_col and not filtered_df.empty:
            try:
                if tool_col in filtered_df.columns and adoption_col in filtered_df.columns:
                    top_tools = filtered_df.nlargest(3, adoption_col)[tool_col].tolist()
            except Exception:
                pass
        
        prompt = f"""
        Based on the AI tools data for {industry} industry in {year}, provide a 3-line summary focusing on:
        
        Dataset size: {len(filtered_df)} tools analyzed
        Top tools by adoption: {', '.join(top_tools) if top_tools else 'Analysis in progress'}
        
        Please provide insights about:
        1. AI adoption trends in {industry} industry for {year}
        2. Key tools and technologies being adopted
        3. Recommendations for businesses in this sector
        
        Keep it concise, actionable, and professional.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Failed to get Gemini summary: {str(e)}"

if GEMINI_AVAILABLE:
    if st.button("🔮 Generate AI Summary with Gemini"):
        with st.spinner("Generating AI insights with Google Gemini..."):
            summary = get_gemini_summary(industry_filter, year_filter)
            st.success(summary)
else:
    st.info("🤖 AI Summary disabled - Google Gemini not available")

# Additional insights section with robust error handling
st.markdown("### 📊 Quick Insights")
if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    
    try:
        # Find correct columns dynamically
        adoption_col = find_column(filtered_df, ['adoption', 'rate'])
        tool_col = find_column(filtered_df, ['tool', 'ai'])
        
        with col1:
            if adoption_col and adoption_col in filtered_df.columns:
                try:
                    numeric_data = pd.to_numeric(filtered_df[adoption_col], errors='coerce')
                    avg_adoption = numeric_data.mean()
                    if pd.notna(avg_adoption):
                        st.metric("Average Adoption Rate", f"{avg_adoption:.1f}%")
                    else:
                        st.metric("Average Adoption Rate", "No data")
                except Exception:
                    st.metric("Total Records", len(filtered_df))
            else:
                st.metric("Total Records", len(filtered_df))
        
        with col2:
            if tool_col and adoption_col and tool_col in filtered_df.columns and adoption_col in filtered_df.columns:
                try:
                    numeric_data = pd.to_numeric(filtered_df[adoption_col], errors='coerce')
                    if not numeric_data.isna().all():
                        top_tool_idx = numeric_data.idxmax()
                        top_tool = filtered_df.loc[top_tool_idx, tool_col]
                        display_tool = str(top_tool)[:20] + "..." if len(str(top_tool)) > 20 else str(top_tool)
                        st.metric("Top Tool", display_tool)
                    else:
                        st.metric("Top Tool", "No data")
                except Exception:
                    st.metric("Top Tool", "Analysis pending")
            else:
                st.metric("Data Columns", len(filtered_df.columns))
        
        with col3:
            total_tools = len(filtered_df)
            st.metric("Total Tools", total_tools)
            
        # Additional info about missing columns
        missing_cols = []
        if not adoption_col or adoption_col not in filtered_df.columns:
            missing_cols.append("adoption rate")
        if not tool_col or tool_col not in filtered_df.columns:
            missing_cols.append("tool name")
        
        if missing_cols:
            st.info(f"💡 Some metrics may be limited. Missing: {', '.join(missing_cols)}. Available columns: " + ", ".join(filtered_df.columns))
            
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        # Show debug info
        st.write("**Debug Info:**")
        st.write(f"- Filtered data shape: {filtered_df.shape}")
        st.write(f"- Available columns: {list(filtered_df.columns)}")
        
        # Fallback metrics
        try:
            col1.metric("Total Records", len(filtered_df))
            col2.metric("Columns", len(filtered_df.columns))
            col3.metric("Year", year_filter)
        except Exception:
            st.write("Unable to display fallback metrics")
        
else:
    st.info("📊 No data available for the selected filters. Try different industry/year combinations.")
    
    # Show available combinations
    if not df.empty and 'industry' in df.columns and 'year' in df.columns:
        try:
            available_combinations = df.groupby(['industry', 'year']).size().reset_index(name='count')
            st.write("**Available data combinations:**")
            st.dataframe(available_combinations)
        except Exception as e:
            st.write(f"Error showing available combinations: {str(e)}")

# Footer with helpful information
st.markdown("---")
st.markdown("### 💡 How to Use This App")
st.markdown("""
1. **Select filters** in the sidebar to focus on specific industry and year
2. **Explore visualizations** to understand adoption trends and sentiment
3. **Generate AI summary** for expert insights on your selected data
4. **Export data** as CSV or PDF for further analysis
5. **Check the debug section** if you encounter any issues
""")

# Error reporting section
st.markdown("### 🐛 Having Issues?")
st.markdown("""
If you encounter any errors:
1. Check the **Data Structure & Debug Info** section above
2. Try different industry/year combinations
3. Refresh the page to reload data
4. The app will show debug information to help identify issues
""")

# System status
with st.expander("🔧 System Status"):
    st.write("**Library Status:**")
    st.write(f"- Streamlit: ✅ {st.__version__}")
    st.write(f"- Pandas: ✅ {pd.__version__}")
    st.write(f"- TextBlob: {'✅' if TEXTBLOB_AVAILABLE else '❌'}")
    st.write(f"- NLTK: {'✅' if NLTK_AVAILABLE else '❌'}")
    st.write(f"- PDFKit: {'✅' if PDFKIT_AVAILABLE else '❌'}")
    st.write(f"- Google Gemini: {'✅' if GEMINI_AVAILABLE else '❌'}")
    
    st.write("**Data Status:**")
    st.write(f"- Dataset loaded: {'✅' if not df.empty else '❌'}")
    st.write(f"- Filtered data: {'✅' if not filtered_df.empty else '❌'}")
    st.write(f"- Total records: {len(df)}")
    st.write(f"- Filtered records: {len(filtered_df)}")
