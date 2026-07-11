import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model and the exact column order it was trained on
model = joblib.load('burnout_model.pkl')
model_features = joblib.load('model_features.pkl')

# Load the original dataset so we can compare a single student to the average
df = pd.read_csv('ai_student_impact_dataset (1).csv')

# Basic page setup - title shown in the browser tab
st.set_page_config(page_title="AI Usage & Student Burnout Risk", layout="centered")

st.title("🎓 GenAI Usage & Student Burnout Risk")
st.write("A tool to help identify students at risk of burnout based on their AI usage habits.")

# --- SIDEBAR: all user inputs live here, separate from the results area ---
st.sidebar.header("Enter Student Details")

weekly_genai_hours = st.sidebar.slider("Weekly GenAI Hours", 0.0, 40.0, 8.0)
traditional_study_hours = st.sidebar.slider("Traditional Study Hours (per week)", 0.0, 30.0, 12.0)
perceived_ai_dependency = st.sidebar.slider("Perceived AI Dependency (1 = low, 5 = high)", 1, 5, 3)
tool_diversity = st.sidebar.slider("Number of Different AI Tools Used", 1, 5, 2)
prompt_skill = st.sidebar.selectbox("Prompt Engineering Skill", ["Beginner", "Intermediate", "Advanced"])
policy = st.sidebar.selectbox("Institutional Policy", ["Actively_Encouraged", "Allowed_With_Citation", "Strict_Ban"])

# button returns True only on the run right after it's clicked
predict_clicked = st.sidebar.button("Predict Burnout Risk")

# Show a hint message until the user actually clicks predict
if not predict_clicked:
    st.info("👈 Enter student details in the sidebar and click 'Predict Burnout Risk' to see results.")

if predict_clicked:
    # Convert skill level text into the same 0/1/2 numbers used during training
    skill_order = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}

    # Build one row of data shaped exactly like the training data
    input_data = pd.DataFrame([{
        'Weekly_GenAI_Hours': weekly_genai_hours,
        'Traditional_Study_Hours': traditional_study_hours,
        'Perceived_AI_Dependency': perceived_ai_dependency,
        'Tool_Diversity': tool_diversity,
        'Prompt_Engineering_Skill': skill_order[prompt_skill],
        # Recreate one-hot encoding by hand: True/False for each policy option
        'Institutional_Policy_Allowed_With_Citation': policy == 'Allowed_With_Citation',
        'Institutional_Policy_Strict_Ban': policy == 'Strict_Ban'
    }])

    # Force columns into the exact order the model expects - prevents silent wrong predictions
    input_data = input_data[model_features]

    # Ask the model to predict: 0 = Low, 1 = Medium, 2 = High
    prediction = model.predict(input_data)[0]

    risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk_colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    risk = risk_labels[prediction]

    # --- Side-by-side layout: left = the answer, right = the explanation ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction")
        # Colored heading so risk level is instantly visible (red/orange/green)
        st.markdown(f"<h2 style='color:{risk_colors[risk]}'>Risk Level: {risk}</h2>", unsafe_allow_html=True)

        # Different message box depending on risk level
        if risk == 'High':
            st.warning("⚠️ Recommendation: Encourage this student to reduce weekly AI usage hours and check in on their overall workload and wellbeing.")
        elif risk == 'Medium':
            st.info("ℹ️ Recommendation: Monitor usage trends. No immediate action needed, but worth tracking over time.")
        else:
            st.success("✅ This student's current AI usage pattern is not flagged as a burnout risk.")

    with col2:
        st.subheader("Why this prediction?")
        # These 3.5 / 9.3 cutoffs match the Low/Medium/High usage bins from the EDA
        if weekly_genai_hours < 3.5:
            usage_note = "This student's weekly AI usage is in the **Low** range."
        elif weekly_genai_hours < 9.3:
            usage_note = "This student's weekly AI usage is in the **Medium** range."
        else:
            usage_note = "This student's weekly AI usage is in the **High** range."

        st.write(f"""
        {usage_note} Our analysis of 50,000 students found that **weekly AI usage hours 
        is the single strongest predictor of burnout risk**, accounting for over 90% of 
        what drives this prediction. Institutional policy, tool variety, and prompting 
        skill level had almost no measurable effect.
        """)

    # --- Comparison chart: shows this student's hours next to the dataset average ---
    st.subheader("How this compares to the average student")
    avg_hours = df['Weekly_GenAI_Hours'].mean()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(['This Student', 'Dataset Average'], [weekly_genai_hours, avg_hours],
            color=['#F2867A', '#6FD9C9'])
    ax.set_xlabel('Weekly GenAI Hours')
    ax.set_title('Weekly AI Usage: This Student vs. Average')
    st.pyplot(fig)