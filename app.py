import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# === Data Sources ===
courses = [
    "Arts - Information Technology", "Computer Science", "Computer Science", "Computer Science", "Computer Science",
    "Physical Science - ICT", "Physical Science - ICT", "Artificial Intelligence", "Electronics and Computer Science",
    "Information Systems", "Information Systems", "Information Systems", "Data Science", "Information Technology (IT)",
    "Management and Information Technology (MIT)", "Computer Science & Technology", "Information Communication Technology",
    "Information Communication Technology", "Information Communication Technology", "Information Communication Technology",
    "Information Communication Technology", "Information Communication Technology", "Information Communication Technology",
    "Information Communication Technology", "Information Communication Technology"
]

universities = [
    "University of Sri Jayewardenepura", "University of Colombo School of Computing (UCSC)", "University of Jaffna",
    "University of Ruhuna", "Trincomalee Campus, Eastern University, Sri Lanka", "University of Kelaniya",
    "University of Sri Jayewardenepura", "University of Moratuwa", "University of Kelaniya",
    "University of Colombo, School of Computing (UCSC)", "University of Sri Jayewardenepura",
    "Sabaragamuwa University of Sri Lanka", "Sabaragamuwa University of Sri Lanaka", "University of Moratuwa",
    "University of Kelaniya", "Uva Wellassa University of Sri Lanka", "University of Sri Jayewardenepura",
    "University of Kelaniya", "University of Vavuniya, Sri Lanka", "University of Ruhuna",
    "South Eastern University of Sri Lanka", "Rajarata University of Sri Lanka", "University of Colombo",
    "Uva Wellassa University of Sri Lanka", "Eastern University, Sri Lanka"
]

languages = ["English", "Sinhala", "Tamil"]

skills_list = [
    "Python", "Java", "SQL", "JavaScript", "TensorFlow", "Pandas", "Docker",
    "Kubernetes", "HTML/CSS", "Power BI", "Spark", "AWS", "Azure",
    "Linux", "Tableau", "React", "Node.js"
]

internships = [
    "Software Intern", "Data Analyst Intern", "ML Intern", "QA Intern",
    "BI Intern", "Cloud Intern", "Network Intern", "Cybersecurity Intern", "UI/UX Intern","None"
]


# --- Streamlit Page Config ---
st.set_page_config(page_title="Job Recommendation System", layout="wide")
st.title("Content-Based Job Recommendation System")

# --- Load dataset from pickle ---
df = pd.read_pickle("job_recommendation_dataset.pkl")
st.success("Dataset loaded successfully!")

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
option = st.sidebar.radio("Choose Candidate Type", ["Existing Candidate", "New Candidate"])

# -------------------------------
# Existing Candidate Recommendations
# -------------------------------
if option == "Existing Candidate":
    top_n = st.sidebar.number_input("Top N Recommendations", min_value=1, max_value=10, value=5)

    candidate_id = st.sidebar.selectbox("Select Candidate ID", df['Candidate_ID'].tolist())
    
    candidate_index = df_features[df_features['Candidate_ID'] == candidate_id].index[0]
    cosine_sim = cosine_similarity(feature_matrix[candidate_index], feature_matrix).flatten()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [x for x in sim_scores if x[0] != candidate_index]  # remove self
    
    top_indices = [i[0] for i in sim_scores[:top_n]]
    recommendations = df.iloc[top_indices]['Target_Role']
    
    st.subheader(f"Top {top_n} Recommended Roles for Candidate ID {candidate_id}")
    for idx, role in enumerate(recommendations, start=1):
        st.write(f"{idx}. {role}")

# -------------------------------
# New Candidate Recommendations
# -------------------------------
else:
    st.subheader("Enter New Candidate Details")

    # Dropdown for Course
    new_course = st.selectbox("Select Course", courses)

    # Dropdown for University
    new_university = st.selectbox("Select University", universities)

    # Multiselect for Skills
    new_skills = st.multiselect("Select Skills", skills_list)

    # Number input for Experience
    new_experience = st.number_input("Experience in Years", min_value=0.0, max_value=50.0, step=0.1, value=0.0)

    # Multiselect for Languages
    new_languages = st.multiselect("Select Language Proficiency", languages)

    # Dropdown for Internship
    new_internship = st.selectbox("Select Internship", internships)

    if st.button("Recommend Jobs for New Candidate"):
        # Combine features into a single string
        new_user_str = ' '.join([
            ' '.join(new_skills),
            str(new_experience),
            new_course,
            new_university,
            ' '.join(new_languages),
            new_internship
        ])

        # Vectorize and get recommendations
        new_user_vec = vectorizer.transform([new_user_str])
        cosine_sim_new = cosine_similarity(new_user_vec, feature_matrix).flatten()

        top_indices = cosine_sim_new.argsort()[::-1][:5]  # fixed top 5
        recommendations_new = df.iloc[top_indices]['Target_Role']

        st.subheader("Top 5 Recommended Roles for New Candidate")
        for idx, role in enumerate(recommendations_new, start=1):
            st.write(f"{idx}. {role}")
