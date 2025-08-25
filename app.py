import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job Recommendation System", layout="wide")
st.title("Content-Based Job Recommendation System")

# Load dataset directly from .pkl
df = pd.read_pickle("job_recommendation_dataset.pkl")
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# Sidebar settings
st.sidebar.header("Recommendation Settings")
top_n = st.sidebar.number_input("Top N Recommendations", min_value=1, max_value=10, value=5)
candidate_id = st.sidebar.selectbox("Select Candidate ID for Recommendation", df['Candidate_ID'].tolist())

# --- Preprocessing: Combine relevant features ---
feature_columns = ['Skills', 'Experience_Years', 'Course', 'Language_Proficiency']
df_features = df.copy()

# Convert numeric experience to string for vectorization
df_features['Experience_Years'] = df_features['Experience_Years'].astype(str)

# Combine features into a single string per candidate
df_features['Combined_Features'] = df_features[feature_columns].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df_features['Combined_Features'])

# Compute Similarity
candidate_index = df_features[df_features['Candidate_ID'] == candidate_id].index[0]
cosine_sim = cosine_similarity(feature_matrix[candidate_index], feature_matrix).flatten()

# Recommend Top-N
sim_scores = list(enumerate(cosine_sim))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = [x for x in sim_scores if x[0] != candidate_index]  # remove self

top_indices = [i[0] for i in sim_scores[:top_n]]
recommendations = df.iloc[top_indices][['Candidate_ID', 'Current_Role', 'Target_Role', 'Skills', 'Experience_Years']]

st.subheader(f"Top {top_n} Job Recommendations for Candidate ID {candidate_id}")
st.dataframe(recommendations)
