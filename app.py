import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('model.pkl')


# Function to preprocess input data
def preprocess_input(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    # Convert categorical variables to numerical
    sex = 1 if Sex == 'male' else 0

    # One-hot encode Embarked
    Embarked= 1 if Embarked == 'C' else 0
    Embarked = 2 if Embarked == 'Q' else 0
    Embarked = 3 if Embarked == 'S' else 0

    # Create a DataFrame with the preprocessed data
    input_data = pd.DataFrame({
        'Pclass': [Pclass],
        'Sex': [sex],
        'Age': [Age],
        'SibSp': [SibSp],
        'Parch': [Parch],
        'Fare': [Fare],
        'Embarked': [Embarked]
    })

    return input_data


# Streamlit app
st.title('Titanic Survival Prediction')

# Inputs from user
Pclass = st.sidebar.selectbox('Passenger’s class', [1, 2, 3])
Sex = st.sidebar.selectbox('Passenger’s sex', ['male', 'female'])
Age = st.sidebar.number_input('Passenger’s Age')
SibSp = st.sidebar.selectbox('Number of siblings',[1,0])
Parch = st.sidebar.selectbox('Number of parents/children aboard',[1,0])
Fare = st.sidebar.number_input('Fare')
Embarked = st.sidebar.selectbox('Port of embarkation', ['C', 'Q', 'S'])

if st.button('Predict'):
    input_data = preprocess_input(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

    prediction = model.predict(input_data)
    # Display prediction
    if prediction[0] == 1:
        st.write('Survived')
    else:
        st.write('Did not survive')