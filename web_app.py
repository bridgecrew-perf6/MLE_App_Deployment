import streamlit as st
import keras
import ast
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def recode_diagnosis(user_input):
    categories = ['Schizophrenia spectrum and other psychotic disorders','Depressive disorders','Bipolar and related disorders',
                  'Other specified and unspecified mood disorders','Anxiety and fear related disorders',
                  'Obsessive-compulsive and related disorders','Trauma and stressor related disorders',
                  'Disruptive, impulse-control and conduct disorders','Personality disorders','Feeding and eating disorders',
                  'Somatic disorders','Suicidal ideation/attempt/intentional self-harm',
                  'Miscellaneous mental and behavioral disorders/conditions','Neurodevelopmental disorders','Alcohol-related disorders',
                  'Opioid-related disorders','Cannabis-related disorders','Sedative-related disorders','Stimulant-related disorders',
                  'Hallucinogen-related disorders','Tobacco-related disorders','Other specified substance-related disorders',
                  'Mental and substance use disorders in remission']
    Primary_dx = categories.index(user_input)
    return Primary_dx

def recode_mortality(user_input):
    categories = ['Minor likelihood of dying','Moderate likelihood of dying',
                  'Major likelihood of dying','Extreme likelihood of dying']
    APRDRG_Risk_Mortality = categories.index(user_input) + 1
    return APRDRG_Risk_Mortality

def recode_severity(user_input):
    categories = ['Minor loss of function','Moderate loss of function',
                  'Major loss of function','Extreme loss of function']
    APRDRG_Severity = categories.index(user_input) + 1
    return APRDRG_Severity

st.header("Predicting Hospital Readmission")
st.write("""
Created by Nate C. Carnes, PhD

This web app allows users to predict the likelihood of hospital readmission for patients with mental health diagnoses. 
A DL model was trained and validated on over 600,000 electronic health records and more than 100 features from the 
Healthcare Cost and Utilization Project (HCUP).

Use the sidebar to manipulate the input features. Each feature defaults to its mean or mode, as appropriate. 
Note that this prediction function makes some assumptions about the patient and hospital to simplify the user experience. 
(These assumptions can be seen in the DataFrame provided below.)
""")
st.sidebar.header("User Input Features")

def user_input_features():  
    Primary_dx = st.sidebar.selectbox('Primary Diagnosis', ('Schizophrenia spectrum and other psychotic disorders',
                                                            'Depressive disorders','Bipolar and related disorders',
                                                            'Other specified and unspecified mood disorders',
                                                            'Anxiety and fear related disorders',
                                                            'Obsessive-compulsive and related disorders',
                                                            'Trauma and stressor related disorders',
                                                            'Disruptive, impulse-control and conduct disorders',
                                                            'Personality disorders','Feeding and eating disorders',
                                                            'Somatic disorders','Suicidal ideation/attempt/intentional self-harm',
                                                            'Miscellaneous mental and behavioral disorders/conditions',
                                                            'Neurodevelopmental disorders','Alcohol-related disorders',
                                                            'Opioid-related disorders','Cannabis-related disorders',
                                                            'Sedative-related disorders','Stimulant-related disorders',
                                                            'Hallucinogen-related disorders','Tobacco-related disorders',
                                                            'Other specified substance-related disorders',
                                                            'Mental and substance use disorders in remission'), 1)
    APRDRG_Risk_Mortality = st.sidebar.selectbox('Risk of Mortality', ('Minor likelihood of dying','Moderate likelihood of dying',
                                                                       'Major likelihood of dying','Extreme likelihood of dying'), 0)
    APRDRG_Severity = st.sidebar.selectbox('Severity of Illness', ('Minor loss of function','Moderate loss of function',
                                                                   'Major loss of function','Extreme loss of function'), 1)
    AGE = st.sidebar.slider('Age', 0, 90, 40)
    FEMALE = st.sidebar.selectbox('Gender', ('Female','Male'), 1)
    ELECTIVE = st.sidebar.selectbox('Elective Admission', ('No','Yes'), 0)
    HCUP_ED = st.sidebar.selectbox('Emergency Department Services', ('No','Yes'), 1)

    Primary_dx = recode_diagnosis(Primary_dx)
    APRDRG_Risk_Mortality = recode_mortality(APRDRG_Risk_Mortality)
    APRDRG_Severity = recode_severity(APRDRG_Severity)
    FEMALE = np.where(FEMALE == 'Female', 1, 0)
    ELECTIVE = np.where(ELECTIVE == 'Yes', 1, 0)
    HCUP_ED = np.where(HCUP_ED == 'Yes', 1, 0)

    file = open("./Dependencies/defaults.txt", "r")
    defaults = file.read()
    data = ast.literal_eval(defaults)
    file.close()

    data.update({'Primary_dx':Primary_dx})
    data.update({'APRDRG_Risk_Mortality':APRDRG_Risk_Mortality})
    data.update({'APRDRG_Severity':APRDRG_Severity})
    data.update({'AGE':AGE})
    data.update({'FEMALE':FEMALE})
    data.update({'ELECTIVE':ELECTIVE})
    data.update({'HCUP_ED':HCUP_ED})
    data['Readmit'] = 0.0

    features = pd.DataFrame(data, index=[0])
    features = features.astype({'Primary_dx':'float64','APRDRG_Risk_Mortality':'float64','APRDRG_Severity':'float64',
                                'AGE':'float64','FEMALE':'float64','ELECTIVE':'float64','HCUP_ED':'float64'})
    return features, Primary_dx

input_df, Primary_dx = user_input_features()

st.subheader('User Input (DataFrame)')
st.write(input_df)

st.header('Prediction')
st.write("""
This is a binary classification model indicating the likelihood of readmission (for any reason) within 30 days. 
This web app will make a prediction using the user-provided input features. 
It will also present a visualization of the prediction function for patients with the same mental health diagnosis.
""")

def dataframe_to_dataset(dataframe, batch=1):
    df = dataframe.copy()
    labels = df.pop('Readmit')
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    ds = ds.batch(batch)
    ds = ds.prefetch(batch)
    return ds

def predict_from_file(file, data, batch=1):
    model = keras.models.load_model(file)
    prediction = model.predict(data, batch_size=batch, verbose=0)
    return prediction

input_ds = dataframe_to_dataset(input_df)
prediction = predict_from_file('./Model',input_ds)
if prediction >= 0.8:
    st.subheader("Yes, there is a very high likelihood of readmission.")
elif (prediction >= 0.6) and (prediction < 0.8):
    st.subheader("Yes, there is a high likelihood of readmission.")
elif (prediction >= 0.5) and (prediction < 0.6):
    st.subheader("Yes, there is an above-chance likelihood of readmission.")
elif (prediction >= 0.4) and (prediction < 0.5):
    st.subheader("No, there is a below-chance likelihood of readmission.")
elif (prediction >= 0.2) and (prediction < 0.4):
    st.subheader("No, there is a low likelihood of readmission.")  
else:
    st.subheader("No, there is a very low likelihood of readmission.")
st.subheader("{}% Predicted Probability of Readmission".format(round(prediction[0,0]*100,2)))

graph_df = pd.read_csv("./Dependencies/app_data.csv")
graph_df = graph_df[graph_df.Primary_dx == Primary_dx].drop(columns='Primary_dx')

fig, ax = plt.subplots()
ax.hist(graph_df, bins=50, color='red', alpha=0.5)
plt.xlabel('Probability of Readmission')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Probabilities for Similar Patients')
plt.xticks(ticks=np.linspace(0,1,11))
st.pyplot(fig)

group_avg = graph_df['Prob'].mean(axis=0)
if prediction >= group_avg:
    comparison1 = 'more'
    comparison2 = 'higher'
else:
    comparison1 = 'less'
    comparison2 = 'lower'
st.write("""
Patients with the same mental health diagnosis were readmitted within 30 days, on average, {} percent of the time. 
Relative to patients with similar diagnoses, the user input patient is {} likely to be readmitted 
and should be considered {} risk within this population.
""".format(round(group_avg*100,2), comparison1, comparison2))