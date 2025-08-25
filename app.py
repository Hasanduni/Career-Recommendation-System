import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job Recommendation System", layout="wide")
st.title("Content-Based Job Recommendation System")

# --- Load dataset from pickle ---
df = pd.read_pickle("job_recommendation_dataset.pkl")
st.success("Dataset loaded successfully!")
st.dataframe(df.head())

# --- Preprocessing: Combine relevant features ---
feature_columns = ['Skills', 'Experience_Years', 'Course', 'Language_Proficiency']
df_features = df.copy()
df_features['Experience_Years'] = df_features['Experience_Years'].astype(str)
df_features['Combined_Features'] = df_features[feature_columns].apply(lambda x: ' '.join(x), axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df_features['Combined_Features'])

# -------------------------------
# Sidebar for recommendation options
# -------------------------------
st.sidebar.header("Recommendation Settings")
top_n = st.sidebar.number_input("Top N Recommendations", min_value=1, max_value=10, value=5)
option = st.sidebar.radio("Choose Candidate Type", ["Existing Candidate", "New Candidate"])

# -------------------------------
# Existing Candidate Recommendations
# -------------------------------
if option == "Existing Candidate":
    candidate_id = st.sidebar.selectbox("Select Candidate ID", df['Candidate_ID'].tolist())
    
    candidate_index = df_features[df_features['Candidate_ID'] == candidate_id].index[0]
    cosine_sim = cosine_similarity(feature_matrix[candidate_index], feature_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != candidate_index]  # remove self
    
    top_indices = [i[0] for i in sim_scores[:top_n]]
    recommendations = df.iloc[top_indices][['Candidate_ID', 'Current_Role', 'Target_Role', 'Skills', 'Experience_Years']]
    
    st.subheader(f"Top {top_n} Job Recommendations for Candidate ID {candidate_id}")
    st.dataframe(recommendations)

# -------------------------------
else:
    st.subheader("Enter New Candidate Details")
    new_skills = st.text_input("Skills (comma separated, e.g., Python, SQL, TensorFlow)")
    new_experience = st.number_input("Experience in Years", min_value=0.0, max_value=50.0, step=0.1, value=0.0)
    new_course = st.text_input("Course")
    new_languages = st.text_input("Language Proficiency (comma separated, e.g., English, Sinhala)")
    
    if st.button("Recommend Jobs for New Candidate"):
        # Combine features
        new_user_str = ' '.join([new_skills, str(new_experience), new_course, new_languages])
        new_user_vec = vectorizer.transform([new_user_str])
        cosine_sim_new = cosine_similarity(new_user_vec, feature_matrix).flatten()
        
        top_indices = cosine_sim_new.argsort()[::-1][:5]  # fixed top 5
        recommendations_new = df.iloc[top_indices][['Candidate_ID', 'Current_Role', 'Target_Role', 'Skills', 'Experience_Years']]
        
        st.subheader(f"Top 5 Job Recommendations for New Candidate")
        st.dataframe(recommendations_new)
