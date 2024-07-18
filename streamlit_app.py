import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

# Load Bootstrap CSS and custom CSS for background image
bootstrap = """
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
<style>
    .main {
        background-image: url('https://static.vecteezy.com/system/resources/previews/006/712/985/non_2x/abstract-health-medical-science-healthcare-icon-digital-technology-science-concept-modern-innovation-treatment-medicine-on-hi-tech-future-blue-background-for-wallpaper-template-web-design-vector.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 20px;
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: white;
    }
</style>
"""
st.markdown(bootstrap, unsafe_allow_html=True)

# Function to load the model
def load_model(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

svc_model = load_model('svc_model.pkl')
logreg_model = load_model('logreg_model.pkl')
xgb_model = load_model('xgb_model.pkl')
rf_model = load_model('rf_model.pkl')

# Title and Description
st.title('Autism Detection')
st.markdown("""
### Welcome to the Autism Detection Application
This app uses machine learning models to predict the likelihood of Autism Spectrum Disorder (ASD) based on various inputs.
Please fill in the information below to get a prediction.
""")

# User Inputs
input_data = {}
input_data['Social_Responsiveness_Scale'] = st.slider(
    "Social Responsiveness Scale", min_value=0, max_value=10, value=5,
    help="Rate the social communication and interaction difficulties."
)
input_data['Age_Years'] = st.number_input(
    'Age (in years)', min_value=1, max_value=100, value=1,
    help="Enter the age of the individual."
)
option = st.radio('Speech Delay or Language Disorder?', ("Yes", "No"))
input_data['Speech Delay/Language Disorder'] = 1 if option == "Yes" else 0
option = st.radio('Learning Disorder?', ("Yes", "No"))
input_data['Learning disorder'] = 1 if option == "Yes" else 0
option = st.radio('Genetic Disorders?', ("Yes", "No"))
input_data['Genetic_Disorders'] = 1 if option == "Yes" else 0
option = st.radio('Depression?', ("Yes", "No"))
input_data['Depression'] = 1 if option == "Yes" else 0
option = st.radio('Intellectual Disability?', ("Yes", "No"))
input_data['Global developoental delay/intellectual disability'] = 1 if option == "Yes" else 0
option = st.radio('Behavioural Issues?', ("Yes", "No"))
input_data['Social/Behavioural Issues'] = 1 if option == "Yes" else 0
input_data['Childhood Autism Rating Scale'] = st.slider(
    "Childhood Autism Rating Scale", min_value=0, max_value=4, value=2,
    help="Rate the childhood autism symptoms."
)
option = st.radio('Anxiety Disorder?', ("Yes", "No"))
input_data['Anxiety_disorder'] = 1 if option == "Yes" else 0
option = st.radio('Gender', ("Male", "Female"))
input_data['Sex'] = 1 if option == "Male" else 0
option = st.radio('Jaundice?', ("Yes", "No"))
input_data['Jaundice'] = 1 if option == "Yes" else 0
option = st.radio('Family member with ASD?', ("Yes", "No"))
input_data['Family_member_with_ASD'] = 1 if option == "Yes" else 0

# Prediction
if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    svc_pred = svc_model.predict(input_df)[0]
    logreg_pred = logreg_model.predict(input_df)[0]
    xgb_pred = xgb_model.predict(input_df)[0]
    
    # Prepare ensemble input
    ensemble_input = pd.DataFrame({'SVC': [svc_pred], 'LogisticRegression': [logreg_pred], 'XGBoost': [xgb_pred]})
    final_pred = rf_model.predict(ensemble_input)[0]
    
    # Display the prediction results
    st.subheader('Autism Detection Result:')
    if final_pred == 1:
        st.write('The model predicts that the person has Autism Spectrum Disorder.')
    else:
        st.write("The model predicts that the person does not have Autism Spectrum Disorder.")
    
    # Save Prediction and Inputs
    input_data['Prediction'] = 'Autism' if final_pred == 1 else 'No Autism'
    input_data['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('predictions.json', 'a') as f:
        f.write(json.dumps(input_data) + '\n')
    
    # Display Model Information
    st.markdown("""
    #### Models Used:
    - Support Vector Classifier (SVC)
    - Logistic Regression
    - XGBoost
    - Random Forest (for ensemble prediction)
    """)
    
    # Visualization
    pred_counts = pd.DataFrame({
        'Model': ['SVC', 'Logistic Regression', 'XGBoost'], 
        'Prediction': [svc_pred, logreg_pred, xgb_pred]
    })
    fig = px.bar(
        pred_counts, 
        x='Model', 
        y='Prediction', 
        title='Model Predictions',
        color='Prediction',
        labels={'Prediction': 'Prediction (0 = No Autism, 1 = Autism)'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(
        title_text='Model Predictions',
        title_x=0.5,
        xaxis_title='Model',
        yaxis_title='Prediction'
    )
    fig.update_traces(
        texttemplate='%{y}', 
        textposition='outside'
    )
    st.plotly_chart(fig)

# User Feedback
st.markdown("### Provide Feedback")
feedback = st.text_area("Please provide your feedback on the prediction or the application.")
if st.button('Submit Feedback'):
    feedback_data = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Feedback': feedback
    }
    with open('feedback.json', 'a') as f:
        f.write(json.dumps(feedback_data) + '\n')
    st.success('Thank you for your feedback!')
