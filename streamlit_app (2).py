import streamlit as st
import pandas as pd
import altair as alt
from textblob import TextBlob
import pdfkit
import requests

# Page configuration
st.set_page_config(page_title="AI Tool Recommender", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("your_dataset.csv")  # Replace with actual path if needed

df = load_data()

# Sidebar filters
industry_filter = st.sidebar.selectbox("Select Industry", sorted(df["industry"].unique()))
year_filter = st.sidebar.selectbox("Select Year", sorted(df["year"].unique()))

filtered_df = df[(df["industry"] == industry_filter) & (df["year"] == year_filter)]

st.title("🚀 AI Tool Recommender App – Smart Insights")

st.subheader(f"🔍 Filtered AI Tools for {industry_filter} in {year_filter}")
st.dataframe(filtered_df)

# ------------------------------------------
# 📈 Trend Chart: AI Adoption Over Years
# ------------------------------------------
st.markdown("### 📈 AI Tool Adoption Trend")

trend_df = df[df["industry"] == industry_filter].groupby("year")["adoption_rate"].mean().reset_index()

trend_chart = alt.Chart(trend_df).mark_line(point=True).encode(
    x='year:O',
    y='adoption_rate:Q'
).properties(
    title=f"{industry_filter} - Adoption Rate Over Years"
)

st.altair_chart(trend_chart, use_container_width=True)

# ------------------------------------------
# 💬 Sentiment Analysis: User Feedback
# ------------------------------------------
st.markdown("### 💬 Sentiment Analysis")

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

# ------------------------------------------
# 📤 Export Filtered Results to PDF
# ------------------------------------------
st.markdown("### 📤 Export Filtered Results")

html = filtered_df.to_html(index=False)
pdf_file = "filtered_results.pdf"
pdfkit.from_string(html, pdf_file)

with open(pdf_file, "rb") as f:
    st.download_button("📥 Download PDF", f, file_name="AI_Tool_Report.pdf")

# ------------------------------------------
# 🤖 GPT Summary (via OpenRouter)
# ------------------------------------------
st.markdown("### 🤖 GPT-Powered Summary")

def get_gpt_summary(industry, year):
    prompt = f"""Give a 3-line summary of the most adopted AI tools in the {industry} industry based on {year} data. Mention top tools and why they're popular."""

    headers = {
        "Authorization": "Bearer YOUR_OPENROUTER_API_KEY",  # 🔐 Replace with your key
        "Content-Type": "application/json"
    }

    body = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ Failed to get GPT summary. Check your API key or internet."

if st.button("Generate GPT Summary"):
    summary = get_gpt_summary(industry_filter, year_filter)
    st.success(summary)
