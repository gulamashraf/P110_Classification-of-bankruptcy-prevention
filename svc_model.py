import pandas as pd
import streamlit as st
import numpy as np
from sklearn import svm
from pickle import dump
from pickle import load


st.title('Model Deployment: P110_Classification of bankruptcy-prevention')

st.sidebar.header('User Input Parameters')

def user_input_features():
    management_risk = st.sidebar.selectbox('Management_Risk',('1.0','0.5','0.0'))
    financial_flexibility = st.sidebar.selectbox('Financial_Flexibility',('1.0','0.5','0.0'))
    credibility	= st.sidebar.selectbox('Credibility',('1.0','0.5','0.0'))
    competitiveness = st.sidebar.selectbox('Competitiveness',('1.0','0.5','0.0'))
    operating_risk = st.sidebar.selectbox('Operating_Risk',('1.0','0.5','0.0'))
    data = {'management_risk':management_risk,
            ' financial_flexibility': financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
            'operating_risk':operating_risk}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

# load the model from disk
model = load(open('C:/Users/Ashraf/Downloads/files/P110_Model.pkl', 'rb'))
prediction = model.predict(df)
st.subheader('Predicted Result')
st.subheader('Detected As')

st.write('Yes' if prediction ==1.0 else 'No')
st.subheader('This Means')
st.write('Bankruptcy Detected' if prediction ==1.0 else 'Non-Bankruptcy Detected')