import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job Recommendation System", layout="wide")
st.title("Content-Based Job Recommendation System")

# === Load dataset from pickle ===
df = pd.read_pickle("job_recommendation_dataset.pkl")
st.success("Dataset loaded successfully!")

# === Preprocessing ===
feature_columns = ['Skills', 'Course_University', 'Language_Proficiency']
df_features = df.copy()

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    return re.sub(r'[^\w\s]', '', str(text).lower())

for col in feature_columns:
    df_features[col] = df_features[col].apply(preprocess_text)

df_features['Combined_Features'] = (
    df_features['Skills'] + ' ' + df_features['Skills'] + ' ' + df_features['Skills'] + ' ' +
    df_features['Course_University'] + ' ' +
    df_features['Language_Proficiency']
)

# Normalize Experience_Years
scaler = MinMaxScaler()
df_features['Experience_Years_Scaled'] = scaler.fit_transform(df_features[['Experience_Years']].astype(float))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(df_features['Combined_Features']).toarray()

# Combine text features with scaled experience
final_features = np.hstack((feature_matrix, df_features[['Experience_Years_Scaled']].values))

# === Sidebar ===
st.sidebar.header("Recommendation Settings")
top_n = st.sidebar.number_input("Top N Recommendations", min_value=1, max_value=10, value=5)
experience_tolerance = st.sidebar.slider("Experience Tolerance (Years)", 0, 10, 2)
option = st.sidebar.radio("Choose Candidate Type", ["Existing Candidate", "New Candidate"])

# === Function to display recommendations as HTML cards ===
def display_recommendation_cards(recommendations, title):
    st.subheader(title)
    
    if recommendations.empty:
        st.warning("No recommendations match the experience filter.")
        return
    
    cards_html = ""
    for _, row in recommendations.iterrows():
        cards_html += f"""
        <div style="
            background-color: #90D5FF;
            padding: 15px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            width: 300px;
            display: inline-block;
            vertical-align: top;
        ">
            <h4 style="color:#000000;">{row['Target_Role']}</h4>
            <p><strong>Candidate ID:</strong> {row['Candidate_ID']}</p>
            <p><strong>Current Role:</strong> {row['Current_Role']}</p>
            <p><strong>Skills:</strong> {row['Skills']}</p>
            <p><strong>Experience:</strong> {row['Experience_Years']} years</p>
        </div>
        """
    
    st.markdown(cards_html, unsafe_allow_html=True)

# === Existing Candidate ===
if option == "Existing Candidate":
    candidate_id = st.sidebar.selectbox("Select Candidate ID", df['Candidate_ID'].tolist())
    candidate_index = df_features[df_features['Candidate_ID'] == candidate_id].index[0]
    
    cosine_sim = cosine_similarity([final_features[candidate_index]], final_features).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != candidate_index]
    
    filtered_indices = [
        idx for idx, score in sim_scores
        if abs(df.iloc[idx]['Experience_Years'] - df.iloc[candidate_index]['Experience_Years']) <= experience_tolerance
    ]
    
    top_indices = filtered_indices[:top_n]
    recommendations = df.iloc[top_indices][['Candidate_ID', 'Current_Role', 'Target_Role', 'Skills', 'Experience_Years']]
    
    display_recommendation_cards(recommendations, f"Top {top_n} Job Recommendations for Candidate ID {candidate_id}")

# === New Candidate ===
else:
    st.subheader("Enter New Candidate Details")
    new_skills = st.text_input("Skills (comma separated, e.g., Python, SQL, TensorFlow)")
    new_experience = st.number_input("Experience in Years", min_value=0.0, max_value=50.0, step=0.1, value=0.0)
    new_course = st.text_input("Course")
    new_languages = st.text_input("Language Proficiency (comma separated, e.g., English, Sinhala)")
    
    if st.button("Recommend Jobs for New Candidate"):
        # Preprocess input
        new_skills_proc = preprocess_text(new_skills)
        new_course_proc = preprocess_text(new_course)
        new_languages_proc = preprocess_text(new_languages)
        
        # Scale experience
        new_exp_scaled = scaler.transform([[new_experience]])[0][0]
        
        # Combine features
        new_user_str = new_skills_proc + ' ' + new_skills_proc + ' ' + new_skills_proc + ' ' + new_course_proc + ' ' + new_languages_proc
        new_user_vec = vectorizer.transform([new_user_str]).toarray()
        new_user_vec = np.hstack((new_user_vec, [[new_exp_scaled]]))
        
        # Compute similarity
        cosine_sim_new = cosine_similarity(new_user_vec, final_features).flatten()
        sim_scores_new = list(enumerate(cosine_sim_new))
        sim_scores_new = sorted(sim_scores_new, key=lambda x: x[1], reverse=True)
        
        filtered_indices_new = [
            idx for idx, score in sim_scores_new
            if abs(df.iloc[idx]['Experience_Years'] - new_experience) <= experience_tolerance
        ]
        
        top_indices_new = filtered_indices_new[:top_n]
        recommendations_new = df.iloc[top_indices_new][['Candidate_ID', 'Current_Role', 'Target_Role', 'Skills', 'Experience_Years']]
        
        display_recommendation_cards(recommendations_new, f"Top {top_n} Job Recommendations for New Candidate")
