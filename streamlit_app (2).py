import streamlit as st
import pandas as pd

# Load dataset from Google Drive
file_id = "14j9MWeeHn4v9ZNSnqIy6WGZdFVKdgFc1"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url)

# Dropdown Filters
industry = st.selectbox("Select Industry", df['industry'].unique())
company_size = st.selectbox("Select Company Size", df['company_size'].unique())
year = st.selectbox("Select Year", sorted(df['year'].unique()))

# Apply filters
filtered_df = df[
    (df['industry'] == industry) &
    (df['company_size'] == company_size) &
    (df['year'] == year)
]

# Show Output
st.write("### Filtered AI Tools Based on Your Selection:")
st.dataframe(filtered_df[['ai_tool', 'adoption_rate', 'user_feedback']])

