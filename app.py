import streamlit as st
import pickle as pkl
import numpy as np

encoders_file = open("/Users/vellankisuryatejareddy/Desktop/Autism-Prediction/Autism-Prediction/encoder.pkl",'rb')
encoders = pkl.load(encoders_file)

model_file = open("/Users/vellankisuryatejareddy/Desktop/Autism-Prediction/Autism-Prediction/autism_model.pkl",'rb')
model = pkl.load(model_file)

def predict_autism(a1_score,a2_score,a3_score,a4_score,a5_score,a6_score,a7_score,a8_score,a9_score,a10_score,age,gender,ethnicity,jaundice,autism,country_of_residence,app_usage,result,relations):
    encoded_ethnicity = encoders['ethnicity'].transform([ethnicity])[0]
    encoded_gender = encoders['gender'].transform([gender])[0]
    encoded_jaundice = encoders['jaundice'].transform([jaundice])[0]
    encoded_autism = encoders['austim'].transform([autism])[0]
    encoded_country = encoders['Country_of_res'].transform([country_of_residence])[0]
    encoded_app_usage = encoders['used_app_before'].transform([app_usage])[0]
    encoded_relations = encoders['relation'].transform([relations])[0]
    features = [
        a1_score, a2_score, a3_score, a4_score, a5_score, a6_score, a7_score, a8_score, a9_score, a10_score,
        age, encoded_gender, encoded_ethnicity, encoded_jaundice, encoded_autism, encoded_country, encoded_app_usage, result, encoded_relations]
    features_array = np.asarray(features)
    features_array_reshaped = features_array.reshape(1,-1)
    predictions = {}
    for model_name , model_values in model.items():
        predictions[model_name] = model[model_name].predict(features_array_reshaped)
    return predictions

def main():
# Inject custom CSS for the background color of the columns and the container
    st.markdown(""" 
        <style>
            /* Background color for columns */
            .stColumn {
                background-color: #fc514e;
                padding: 10px;
                border-radius: 5px;
                color: black !important;
            }
        </style>
        """, unsafe_allow_html=True)
    st.title(":rainbow[Autism Spectrum Disorder Predictor]")
    st.write(":green[Please provide the required information below to predict Autism Spectrum Disorder (ASD)]:smile:")
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33], gap="small")
    col4, col5, col6 = st.columns([0.33, 0.33, 0.33], gap="small")
    col7, col8, col9 = st.columns([0.33, 0.33, 0.33], gap="small")
    col10, col11, col12 = st.columns([0.33, 0.33, 0.33], gap="small")
    col13, col14, col15 = st.columns([0.33, 0.33, 0.33], gap="small")
    col16, col17, col18 = st.columns([0.33, 0.33, 0.33], gap="small")
    ethnicity_list = ['Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern', 'Pasifika', 'South Asian', 'Turkish', 'White-European', 'Others']
    relation_list = ['Self', 'Others']
    countries = ["Afghanistan", "AmericanSamoa", "Angola", "Argentina", "Armenia", "Aruba", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bangladesh", "Belgium", "Bolivia", "Brazil", "Burundi", "Canada", "China", "Cyprus", "Czech Republic", "Egypt", "Ethiopia", "France", "Germany", "Hong Kong", "Iceland", "India", "Iran", "Iraq", "Ireland", "Italy", "Japan", "Jordan", "Kazakhstan", "Malaysia", "Mexico", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Oman", "Pakistan", "Romania", "Russia", "Saudi Arabia", "Serbia", "Sierra Leone", "South Africa", "Spain", "Sri Lanka", "Sweden", "Tonga", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Viet Nam"]
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
    st.markdown('<br></br>', unsafe_allow_html=True)
    diagnosis = {}
    if st.button("Predict", type="primary", icon="💡"):
        diagnosis=predict_autism(a1_score,a2_score,a3_score,a4_score,a5_score,a6_score,a7_score,a8_score,a9_score,a10_score,age,gender,ethnicity,jaundice,autism,country_of_residence,app_usage,result,relations)
    for model_name , prediction in diagnosis.items():
        st.success(f"Result: \n {model_name}:{prediction}")


if __name__ == '__main__':
    main()