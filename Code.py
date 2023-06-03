import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import streamlit as st


#data.csv belom ada
data = pd.read_csv('Brain-Tumor.csv')

# preProcessing

data = data.drop(data[['Image']],axis=1)

# Memisahkan label dan fitur 
X = data.drop(columns='Class', axis=1)
Y = data['Class']

# standarisasi data
scaler = StandardScaler()
scaler.fit(X)
standarized_data = scaler.transform(X)
X = standarized_data
Y = data['Class']

# Pisah data training dan test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=8)

# Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Form

st.title("Brain Tumor Detection")
form = st.form(key='my-form')
Mean = form.number_input("Input Nilai Mean")
Variance = form.number_input("Input Nilai Variance")
Standard_Deviation = form.number_input("Input Nilai Standard Deviation")
Entropy = form.number_input("Input Nilai Entropy")
Skewness = form.number_input("Input Nilai Skewness")
Kurtosis = form.number_input("Input Nilai Kurtosis")
Contrast = form.number_input("Input Nilai Contrast")
Energy = form.number_input("Input Nilai Energy")
ASM = form.number_input("Input Nilai ASM")
Homogeneity = form.number_input("Input Nilai Homogeneity")
Dissimilarity = form.number_input("Input Nilai Dissimilarity")
Correlation = form.number_input("Input Nilai Correlation")
submit = form.form_submit_button('Submit')

completeData = np.array([Mean, Variance, Standard_Deviation, Entropy, Skewness, Kurtosis, Contrast, Energy, ASM, Homogeneity, Dissimilarity, Correlation ]).reshape(1, -1)
scaledData = scaler.transform(completeData)

if submit: 
    prediction = classifier.predict(scaledData)
    if prediction == 0:
        st.success('Pasien tidak terkena tumor')
    else :
        st.error('Pasien terkena tumor')
