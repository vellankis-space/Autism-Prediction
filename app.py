import streamlit as st
import pickle as pkl
import numpy as np

# Load the encoders and model pickle files
encoders_file = open("encoder.pkl",'rb')
encoders = pkl.load(encoders_file)

model_file = open("autism_model.pkl",'rb')
model = pkl.load(model_file)

# Function to predict autism based on the input data
def predict_autism(a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score,
                   age, gender, ethnicity, jaundice, autism, country_of_residence, app_usage, result, relations):
    # Encoding categorical features using saved encoders
    encoded_ethnicity = encoders['ethnicity'].transform([ethnicity])[0]
    encoded_gender = encoders['gender'].transform([gender])[0]
    encoded_jaundice = encoders['jaundice'].transform([jaundice])[0]
    encoded_autism = encoders['austim'].transform([autism])[0]
    encoded_country = encoders['Country_of_res'].transform([country_of_residence])[0]
    encoded_app_usage = encoders['used_app_before'].transform([app_usage])[0]
    encoded_relations = encoders['relation'].transform([relations])[0]
    
    # Prepare features for prediction (including the encoded features)
    features = [
        a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score,
        age, encoded_gender, encoded_ethnicity, encoded_jaundice, encoded_autism, encoded_country, encoded_app_usage, result, encoded_relations
    ]
    
    # Convert features to numpy array and reshape for prediction
    features_array = np.asarray(features)
    features_array_reshaped = features_array.reshape(1, -1)
    
    # Prepare dictionary for storing predictions from different models
    predictions = {}
    for model_name, model_values in model.items():
        predictions[model_name] = model[model_name].predict(features_array_reshaped)
    
    return predictions

def main():
    # Inject custom CSS for styling the Streamlit app
    st.markdown(""" 
        <style>
            .stApp {
                background-color: black;
            }
            .stColumn {
                background-color: #fc514e;
                padding: 10px;
                border-radius: 5px;
                color: black !important;
            }
            .stSelectbox label {
                color: white !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Set the app title and instructions
    st.title(":rainbow[Autism Spectrum Disorder Predictor]")
    st.write(":green[Please provide the required information below to predict Autism Spectrum Disorder (ASD)]:smile:")
    
    # Define column layouts for organizing input fields
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="small")
    col4, col5, col6 = st.columns([0.33, 0.33, 0.33], gap="small")
    col7, col8, col9 = st.columns([0.33, 0.33, 0.33], gap="small")
    col10, col11, col12 = st.columns([0.33, 0.33, 0.33], gap="small")
    col13, col14, col15 = st.columns([0.33, 0.33, 0.33], gap="small")
    col16, col17, col18 = st.columns([0.33, 0.33, 0.33], gap="small")
    
    # Predefined options for ethnicity, relations, and countries
    ethnicity_list = ['Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern', 'Pasifika', 'South Asian', 'Turkish', 'White-European', 'Others']
    relation_list = ['Self', 'Others']
    countries = ["Afghanistan", "AmericanSamoa", "Angola", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bangladesh", "Belgium", "Bolivia", "Brazil", "Burundi", "Canada", "China", "Colombia", "Czech Republic", "Denmark", "Egypt", "France", "Germany", "Greece", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Japan", "Kenya", "Malaysia", "Mexico", "Netherlands", "New Zealand", "Nigeria", "Norway", "Pakistan", "Philippines", "Poland", "Portugal", "Russia", "Saudi Arabia", "Singapore", "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", "Thailand", "Turkey", "Uganda", "United Kingdom", "United States", "Vietnam"]

    # Collect user inputs via Streamlit components
    with col1:
        a1_score = st.radio(":one: A1_Score", [0, 1], key=1)
    with col2:
        a2_score = st.radio(":two: A2_Score", [0, 1], key=2)
    with col3:
        a3_score = st.radio(":three: A3_Score", [0, 1], key=3)
    with col4:
        a4_score = st.radio(":four: A4_Score", [0, 1], key=4)
    with col5:
        a5_score = st.radio(":five: A5_Score", [0, 1], key=5)
    with col6:
        a6_score = st.radio(":six: A6_Score", [0, 1], key=6)
    with col7:
        a7_score = st.radio(":seven: A7_Score", [0, 1], key=7)
    with col8:
        a8_score = st.radio(":eight: A8_Score", [0, 1], key=8)
    with col9:
        a9_score = st.radio(":nine: A9_Score", [0, 1], key=9)
    with col10:
        a10_score = st.radio("A10_Score", [0, 1], key=10)
    with col11:
        age = st.number_input("Enter Your Age :seedling:", value=None, placeholder="Enter your age", key=11, step=1)
    with col12:
        gender = st.radio("Gender :boy::girl:", ['f', 'm'], key=12)
    with col13:
        ethnicity = st.selectbox("Ethnicity :bust_in_silhouette:", options=ethnicity_list, key=13)
    with col14:
        jaundice = st.radio("Jaundice", ['yes', 'no'], key=14)
    with col15:
        autism = st.radio("Previous Autism History :medical_symbol:", ['yes', 'no'], key=15)
    with col16:
        country_of_residence = st.selectbox("Country of Residence :earth_africa:", options=countries, key=16)
    with col17:
        app_usage = st.radio("Used Application Before", ['yes', 'no'], key=17)
    with col18:
        result = st.number_input('Enter the Result', value=None, placeholder='Result', key=18)
    relations = st.selectbox("Relation :family:", options=relation_list, key=19)
    
    # Add some spacing
    st.markdown('<br></br>', unsafe_allow_html=True)
    
    # Initialize diagnosis dictionary
    diagnosis = {}
    
    # Trigger prediction when the "Predict" button is clicked
    if st.button("Predict", type="primary", icon="💡"):
        diagnosis = predict_autism(a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score, age, gender, ethnicity, jaundice, autism, country_of_residence, app_usage, result, relations)
    
    # Display the predictions
    for model_name, prediction in diagnosis.items():
        st.success(f"Result: \n {model_name}:{prediction}")

# Run the app
if __name__ == '__main__':
    main()